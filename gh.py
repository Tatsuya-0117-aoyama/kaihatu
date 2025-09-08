import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import os
from pathlib import Path
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

# ================================
# 表示・フォント設定
# ================================
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 10

# ================================
# 設定クラス
# ================================
class Config:
    def __init__(self):
        # パス設定（必要に応じて変更）
        self.rgb_base_path = r"C:\Users\EyeBelow"
        self.signal_base_path = r"C:\Users\Data_signals_bp"
        self.base_save_path = r"D:\EPSCAN\001"

        # 日付時間フォルダを作成
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.base_save_path, self.timestamp)

        # 解析タイプ
        self.analysis_type = "cross"  # "individual"（個人内） or "cross"（個人間）

        # ★除外する被験者（例: ["bp001","bp005"]）
        self.exclude_subjects = []

        # モデルタイプ: "standard"(5block) / "deep"(8block;推奨) / "very_deep"(10block) / "resnet"
        self.model_type = "deep"

        # データ設定
        if self.analysis_type == "individual":
            self.subjects = ["bp001"]
            self.n_folds = 1
        else:
            all_subjects = [f"bp{i:03d}" for i in range(1, 33)]
            self.subjects = [s for s in all_subjects if s not in self.exclude_subjects]
            if self.exclude_subjects:
                print(f"\n【除外設定】\n  除外被験者: {', '.join(self.exclude_subjects)}")
                print(f"  使用被験者数: {len(self.subjects)}名（全{len(all_subjects)}名中）")
            else:
                print(f"\n【全被験者使用】\n  使用被験者数: {len(self.subjects)}名")

            if len(self.subjects) >= 8:
                self.n_folds = 8
            elif len(self.subjects) >= 4:
                self.n_folds = 4
            else:
                self.n_folds = min(len(self.subjects), 2)
                if self.n_folds < 2:
                    print(f"警告: 被験者数が少なすぎます（{len(self.subjects)}名）。最低2名必要です。")

        # タスク・時間（個人内用）
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60  # 各60秒

        # 入力チャンネル
        self.use_channel = 'B'  # 'R','G','B','RGB'
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)

        # 学習設定（モデルタイプで自動調整）
        if self.model_type == "standard":
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        elif self.model_type == "deep":
            self.batch_size = 32
            self.epochs = 200
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 40
        elif self.model_type == "very_deep":
            self.batch_size = 32
            self.epochs = 250
            self.learning_rate = 0.0005
            self.weight_decay = 1e-4
            self.patience = 50
        elif self.model_type == "resnet":
            self.batch_size = 32
            self.epochs = 200
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 40
        else:
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 損失関数
        self.loss_type = "combined"  # "mse","combined","huber_combined"
        self.loss_alpha = 0.7
        self.loss_beta = 0.3

        # スケジューラ
        self.scheduler_type = "cosine"  # "cosine","onecycle","plateau"
        self.scheduler_T0 = 30 if self.model_type != "standard" else 20
        self.scheduler_T_mult = 2 if self.model_type != "standard" else 1

        # データ分割（個人内）
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.random_split = True
        self.random_seed = 42

        # 表示
        self.verbose = True

        # ================================
        # ここから 事前学習（自己教師あり）設定
        # ================================
        self.pretrain = True
        self.pretrain_method = "simclr"
        self.pretrain_epochs = 60

        # 被験者バランス・バッチ設計（例: 16人×各8枚=128）
        self.subjects_per_batch = 16
        self.samples_per_subject = 8

        # SimCLRハイパラ
        self.temperature = 0.2
        self.projection_dim = 128

        # 時間的ポジティブ（同一被験者の±W秒を正例候補）
        self.pretrain_temporal_pos = True
        self.temporal_window = 5  # ±5秒（1Hzなら±5フレーム）

        # 負例から同一被験者を除外（IDバイアス抑制）
        self.exclude_same_subject_negatives = True

        # 逆学習（被験者IDを消す）※任意
        self.use_subject_adversary = True
        self.adversary_weight = 0.2

        # FT時の凍結
        self.freeze_backbone_epochs = 10

# ================================
# カスタム損失
# ================================
class CombinedLoss(nn.Module):
    """MSE + (1 - PearsonCorr)"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        numerator = torch.sum(pred_mean * target_mean)
        denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
        correlation = numerator / denominator
        corr_loss = 1 - correlation
        total = self.alpha * mse_loss + self.beta * corr_loss
        return total, mse_loss, corr_loss

class HuberCorrelationLoss(nn.Module):
    """Huber + (1 - PearsonCorr)"""
    def __init__(self, delta=1.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.huber = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        huber_loss = self.huber(pred, target)
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        numerator = torch.sum(pred_mean * target_mean)
        denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
        correlation = numerator / denominator
        corr_loss = 1 - correlation
        total = self.alpha * huber_loss + self.beta * corr_loss
        return total, huber_loss, corr_loss

# ================================
# データセット（CO回帰）
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, co_data, use_channel='B'):
        if use_channel == 'R':
            selected_data = rgb_data[:, :, :, 0:1]
        elif use_channel == 'G':
            selected_data = rgb_data[:, :, :, 1:2]
        elif use_channel == 'B':
            selected_data = rgb_data[:, :, :, 2:3]
        else:
            selected_data = rgb_data
        self.rgb_data = torch.FloatTensor(selected_data).permute(0, 3, 1, 2)  # (N,C,H,W)
        self.co_data = torch.FloatTensor(co_data)

    def __len__(self):
        return len(self.rgb_data)

    def __getitem__(self, idx):
        return self.rgb_data[idx], self.co_data[idx]

# ================================
# オリジナル PhysNet2DCNN（5ブロック）
# ================================
class PhysNet2DCNN(nn.Module):
    def __init__(self, input_shape):
        super(PhysNet2DCNN, self).__init__()
        in_channels = input_shape[2]
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ELU()
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU()
        )
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU()
        )
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU()
        )
        self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ELU()
        )
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward_features(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        x = x.view(batch_size, in_channels, -1)
        x = self.conv_block1(x); x = self.avgpool1(x); x = self.dropout(x)
        x = self.conv_block2(x); x = self.avgpool2(x); x = self.dropout(x)
        x = self.conv_block3(x); x = self.avgpool3(x); x = self.dropout(x)
        x = self.conv_block4(x); x = self.avgpool4(x); x = self.dropout(x)
        x = self.conv_block5(x)
        x = self.global_avgpool(x).squeeze(-1)  # (B,64)
        return x

    def forward(self, x):
        x = self.forward_features(x)               # (B,64)
        x = x.unsqueeze(-1)                        # (B,64,1)
        x = self.conv_final(x)                     # (B,1,1)
        x = x.squeeze()
        if x.ndim == 0: x = x.unsqueeze(0)
        return x

# ================================
# 深層 PhysNet2DCNN（8/10ブロック）
# ================================
class DeepPhysNet2DCNN(nn.Module):
    def __init__(self, input_shape, depth_level="deep"):
        super(DeepPhysNet2DCNN, self).__init__()
        in_channels = input_shape[2]
        if depth_level == "deep":
            self.num_blocks = 8
            self.channels = [32, 64, 64, 128, 128, 128, 256, 256]
        elif depth_level == "very_deep":
            self.num_blocks = 10
            self.channels = [32, 64, 64, 128, 128, 256, 256, 256, 512, 512]
        else:
            raise ValueError(f"Unknown depth_level: {depth_level}")
        self.use_residual = True

        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        prev_channels = in_channels

        for i in range(self.num_blocks):
            out_channels = self.channels[i]
            if i == 0:
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels), nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels), nn.ELU()
                )
            elif i < 4:
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels), nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels), nn.ELU()
                )
            else:
                mid_channels = out_channels
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, mid_channels, kernel_size=1),
                    nn.BatchNorm1d(mid_channels), nn.ELU(),
                    nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(mid_channels), nn.ELU(),
                    nn.Conv1d(mid_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels), nn.ELU()
                )
            self.conv_blocks.append(conv_block)

            if i < 2:
                self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            elif i < 4:
                if i % 2 == 0:
                    self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
                else:
                    self.pools.append(None)
            else:
                self.pools.append(None)

            if prev_channels != out_channels:
                self.residual_convs.append(nn.Conv1d(prev_channels, out_channels, kernel_size=1))
            else:
                self.residual_convs.append(None)

            prev_channels = out_channels

        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        if depth_level == "deep":
            hidden_dim = 128; self.dropout_rates = [0.3, 0.2]
        else:
            hidden_dim = 256; self.dropout_rates = [0.4, 0.3]

        self.fc = nn.Sequential(
            nn.Linear(prev_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(self.dropout_rates[0]),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ELU(), nn.Dropout(self.dropout_rates[1]),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.block_dropout = nn.Dropout(0.1)

    def forward_features(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        x = x.view(batch_size, in_channels, -1)
        for i in range(self.num_blocks):
            identity = x
            x = self.conv_blocks[i](x)
            if self.residual_convs[i] is not None:
                identity = self.residual_convs[i](identity)
            if self.pools[i] is not None:
                x = self.pools[i](x); identity = self.pools[i](identity)
            x = x + identity * 0.3
            if i >= 4:
                x = self.block_dropout(x)
        x = self.global_avgpool(x).squeeze(-1)  # (B, C_last)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x).squeeze(-1)
        if x.ndim == 0: x = x.unsqueeze(0)
        return x

# ================================
# ResNetスタイル
# ================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNetPhysNet(nn.Module):
    def __init__(self, input_shape):
        super(ResNetPhysNet, self).__init__()
        in_channels = input_shape[2]
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)
        x = self.conv1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)  # (B,512)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x).squeeze(-1)
        if x.ndim == 0: x = x.unsqueeze(0)
        return x

# ================================
# モデル作成
# ================================
def create_model(config):
    if config.model_type == "standard":
        model = PhysNet2DCNN(config.input_shape)
        model_name = "PhysNet2DCNN (Standard - 5 blocks)"
        feat_hint = 64
    elif config.model_type == "deep":
        model = DeepPhysNet2DCNN(config.input_shape, depth_level="deep")
        model_name = "DeepPhysNet2DCNN (Deep - 8 blocks)"
        feat_hint = 256
    elif config.model_type == "very_deep":
        model = DeepPhysNet2DCNN(config.input_shape, depth_level="very_deep")
        model_name = "DeepPhysNet2DCNN (Very Deep - 10 blocks)"
        feat_hint = 512
    elif config.model_type == "resnet":
        model = ResNetPhysNet(config.input_shape)
        model_name = "ResNetPhysNet"
        feat_hint = 512
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    print(f"\n選択モデル: {model_name}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    if config.model_type == "standard":
        print("【注意】10000データには 'deep' または 'very_deep' を推奨")
    elif config.model_type == "deep":
        print("【推奨】10000データに最適な設定です")
    return model

# ================================
# データ読み込み
# ================================
def load_data_single_subject(subject, config):
    rgb_path = os.path.join(config.rgb_base_path, subject, f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません")
        return None, None
    rgb_data = np.load(rgb_path)

    co_data_list = []
    for task in config.tasks:
        co_path = os.path.join(config.signal_base_path, subject, "CO", f"CO_s2_{task}.npy")
        if not os.path.exists(co_path):
            print(f"警告: {subject}の{task}のCOデータが見つかりません")
            return None, None
        co_data_list.append(np.load(co_path))
    co_data = np.concatenate(co_data_list)
    return rgb_data, co_data

def load_all_data(config):
    print("="*60)
    print("データ読み込み中...")
    print("="*60)
    all_rgb_data, all_co_data, all_subject_ids = [], [], []
    loaded_subjects, failed_subjects = [], []

    if config.analysis_type == "cross" and config.exclude_subjects:
        print(f"除外被験者: {', '.join(config.exclude_subjects)}\n")

    for subject in config.subjects:
        rgb_data, co_data = load_data_single_subject(subject, config)
        if rgb_data is not None and co_data is not None:
            all_rgb_data.append(rgb_data)
            all_co_data.append(co_data)
            all_subject_ids.extend([subject] * len(rgb_data))
            loaded_subjects.append(subject)
            print(f"✓ {subject}のデータ読み込み完了")
        else:
            failed_subjects.append(subject)

    if len(all_rgb_data) == 0:
        raise ValueError("データが読み込めませんでした")

    all_rgb_data = np.concatenate(all_rgb_data, axis=0)
    all_co_data = np.concatenate(all_co_data, axis=0)

    print(f"\n読み込み完了:")
    print(f"  成功: {len(loaded_subjects)}名")
    if failed_subjects:
        print(f"  失敗: {len(failed_subjects)}名 ({', '.join(failed_subjects)})")
    print(f"  データ形状: {all_rgb_data.shape}")
    print(f"  データ数: {len(all_rgb_data)}")
    print(f"  使用チャンネル: {config.use_channel}成分")

    return all_rgb_data, all_co_data, all_subject_ids

# ================================
# 個人内分割
# ================================
def split_data_individual(rgb_data, co_data, config):
    if config.random_split:
        print("\nデータ分割中（ランダム分割）...")
        np.random.seed(config.random_seed)
    else:
        print("\nデータ分割中（順番分割）...")

    train_rgb, train_co, val_rgb, val_co, test_rgb, test_co = [], [], [], [], [], []

    for i, task in enumerate(config.tasks):
        start_idx = i * config.task_duration
        end_idx = (i + 1) * config.task_duration
        task_rgb = rgb_data[start_idx:end_idx]
        task_co = co_data[start_idx:end_idx]
        train_size = int(config.task_duration * config.train_ratio)
        val_size = int(config.task_duration * config.val_ratio)

        if config.random_split:
            indices = np.arange(config.task_duration)
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            train_rgb.append(task_rgb[train_indices]); train_co.append(task_co[train_indices])
            val_rgb.append(task_rgb[val_indices]);     val_co.append(task_co[val_indices])
            test_rgb.append(task_rgb[test_indices]);   test_co.append(task_co[test_indices])
        else:
            train_rgb.append(task_rgb[:train_size]);            train_co.append(task_co[:train_size])
            val_rgb.append(task_rgb[train_size:train_size+val_size]); val_co.append(task_co[train_size:train_size+val_size])
            test_rgb.append(task_rgb[train_size+val_size:]);    test_co.append(task_co[train_size+val_size:])

    train_rgb = np.concatenate(train_rgb); train_co = np.concatenate(train_co)
    val_rgb = np.concatenate(val_rgb);     val_co = np.concatenate(val_co)
    test_rgb = np.concatenate(test_rgb);   test_co = np.concatenate(test_co)

    print(f"分割結果: 訓練{len(train_rgb)}, 検証{len(val_rgb)}, テスト{len(test_rgb)}")
    return (train_rgb, train_co), (val_rgb, val_co), (test_rgb, test_co)

# ================================
# 事前学習のための拡張・Dataset・Sampler・Loss
# ================================
def simple_augment(x):
    """14x16向けの軽量Aug（壊しすぎない）。x: (B?,C,H,W) or (C,H,W)"""
    single = (x.ndim == 3)
    if single:
        x = x.unsqueeze(0)
    # 1) 左右反転
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    # 2) 微小シフト
    if random.random() < 0.8:
        dx = random.randint(-1, 1); dy = random.randint(-1, 1)
        x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    # 3) 輝度・コントラスト
    if random.random() < 0.8:
        alpha = 1.0 + 0.15*torch.randn(1, device=x.device).clamp(-0.25, 0.25).item()
        beta  = 0.05*torch.randn(1, device=x.device).clamp(-0.1, 0.1).item()
        m = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - m)*alpha + m + beta
    # 4) 微小ノイズ
    if random.random() < 0.5:
        x = x + 0.01*torch.randn_like(x)
    if single:
        x = x.squeeze(0)
    return x

class SSLPretrainDataset(Dataset):
    """SimCLR用。view1: 同一フレームAug, view2: 同一or時間近傍フレームAug"""
    def __init__(self, rgb_data, subject_ids, use_channel='B', temporal_pos=True, window=5):
        if use_channel == 'R':
            selected = rgb_data[:, :, :, 0:1]
        elif use_channel == 'G':
            selected = rgb_data[:, :, :, 1:2]
        elif use_channel == 'B':
            selected = rgb_data[:, :, :, 2:3]
        else:
            selected = rgb_data
        self.x = torch.FloatTensor(selected).permute(0, 3, 1, 2)  # (N,C,H,W)

        uniq = sorted(list(set(subject_ids)))
        self.subj2idx = {s: i for i, s in enumerate(uniq)}
        self.sids = np.array([self.subj2idx[s] for s in subject_ids], dtype=np.int64)

        self.idx_by_subj = {}
        for sid in range(len(uniq)):
            self.idx_by_subj[sid] = np.where(self.sids == sid)[0]

        self.temporal_pos = temporal_pos
        self.window = window

    def __len__(self):
        return len(self.x)

    def _sample_temporal_neighbor(self, i, sid):
        arr = self.idx_by_subj[sid]
        pos = np.searchsorted(arr, i)
        candidates = []
        for d in range(1, self.window + 1):
            if pos - d >= 0: candidates.append(arr[pos - d])
            if pos + d < len(arr): candidates.append(arr[pos + d])
        if candidates:
            return int(np.random.choice(candidates))
        return int(i)

    def __getitem__(self, i):
        xi = self.x[i]
        sid = int(self.sids[i])
        v1 = simple_augment(xi)
        j = i
        if self.temporal_pos and np.random.rand() < 0.5:
            j = self._sample_temporal_neighbor(i, sid)
        xj = self.x[j]
        v2 = simple_augment(xj)
        return v1, v2, sid

class BalancedSubjectBatchSampler(Sampler):
    """
    各バッチ: subjects_per_batch 人 × samples_per_subject 枚
    例: 16×8=128。32人いれば1バッチで半数をカバー。
    """
    def __init__(self, subject_ids, subjects_per_batch=16, samples_per_subject=8, drop_last=True):
        self.sids_raw = np.array(subject_ids, dtype=object)  # 文字列のまま
        uniq = sorted(list(set(self.sids_raw)))
        self.subj2idx = {s: i for i, s in enumerate(uniq)}
        self.sids_norm = np.array([self.subj2idx[s] for s in self.sids_raw], dtype=np.int64)

        self.idx_by_subj = {k: np.where(self.sids_norm == k)[0].tolist() for k in range(len(uniq))}
        self.S = min(subjects_per_batch, len(uniq))  # 安全化
        self.M = samples_per_subject
        self.drop_last = drop_last

        per_subj_chunks = [len(v) // self.M for v in self.idx_by_subj.values()]
        self.chunks_per_subject = max(1, min(per_subj_chunks))
        self.groups_per_epoch = max(1, (len(uniq) // self.S) * self.chunks_per_subject)

    def __iter__(self):
        chunks = {sid: [] for sid in self.idx_by_subj}
        for sid, idxs in self.idx_by_subj.items():
            idxs = idxs.copy()
            np.random.shuffle(idxs)
            for k in range(self.chunks_per_subject):
                start = k * self.M
                seg = idxs[start:start + self.M]
                if len(seg) == self.M:
                    chunks[sid].append(seg)

        sids_all = list(chunks.keys())
        for _ in range(self.chunks_per_subject):
            np.random.shuffle(sids_all)
            for g in range(0, len(sids_all) // self.S):
                pick = sids_all[g * self.S:(g + 1) * self.S]
                batch = []
                ok = True
                for sid in pick:
                    if not chunks[sid]:
                        ok = False; break
                    batch.extend(chunks[sid].pop())
                if ok and len(batch) == self.S * self.M:
                    yield batch
                elif not self.drop_last and len(batch) > 0:
                    yield batch

    def __len__(self):
        return self.groups_per_epoch

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * proj_dim),
            nn.BatchNorm1d(2 * proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * proj_dim, proj_dim)
        )
    def forward(self, x): return self.net(x)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out): return -ctx.lambd * grad_out, None

class SubjectAdversary(nn.Module):
    def __init__(self, in_dim, n_subjects):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, n_subjects)
        )
    def forward(self, feat, lambd=1.0):
        rev = GradReverse.apply(feat, lambd)
        return self.clf(rev)

def info_nce_with_subject_mask(z1, z2, sids, temperature=0.2, exclude_same_subject_neg=True):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    B = z1.size(0); dev = z1.device
    z = torch.cat([z1, z2], dim=0)               # (2B,D)
    sim = (z @ z.T) / temperature                # (2B,2B)

    mask_self = torch.eye(2 * B, device=dev).bool()
    sim = sim.masked_fill(mask_self, -1e9)
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(dev)
    pos_sim = sim[torch.arange(2 * B, device=dev), pos]

    if exclude_same_subject_neg and sids is not None:
        s = torch.tensor(sids, device=dev, dtype=torch.long)
        s = torch.cat([s, s], dim=0)  # (2B,)
        same = (s.unsqueeze(0) == s.unsqueeze(1))
        same[torch.arange(2 * B), pos] = True
        sim = sim.masked_fill(same, -1e9)

    denom = torch.logsumexp(sim, dim=1)
    loss = -pos_sim + denom
    return loss.mean()

# ================================
# 事前学習（SimCLR）
# ================================
def infer_feature_dim(model, input_shape, device):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(2, input_shape[2], input_shape[0], input_shape[1], device=device)
    with torch.no_grad():
        f = model.forward_features(dummy)
    return f.shape[1]

def pretrain_encoder(rgb_data, subject_ids, config):
    print("\n=== Self-Supervised Pretraining (SimCLR, subject-balanced) ===")
    dataset = SSLPretrainDataset(
        rgb_data, subject_ids, use_channel=config.use_channel,
        temporal_pos=config.pretrain_temporal_pos, window=config.temporal_window
    )

    loader = DataLoader(
        dataset,
        batch_sampler=BalancedSubjectBatchSampler(
            subject_ids=subject_ids,
            subjects_per_batch=config.subjects_per_batch,
            samples_per_subject=config.samples_per_subject,
            drop_last=True
        )
    )

    backbone = create_model(config).to(config.device)
    feat_dim = infer_feature_dim(backbone, config.input_shape, config.device)
    projector = ProjectionHead(feat_dim, proj_dim=config.projection_dim).to(config.device)

    adversary = None
    if config.use_subject_adversary:
        n_subj = len(set(subject_ids))
        adversary = SubjectAdversary(feat_dim, n_subj).to(config.device)

    params = list(backbone.parameters()) + list(projector.parameters())
    if adversary is not None: params += list(adversary.parameters())
    opt = optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    for epoch in range(config.pretrain_epochs):
        backbone.train(); projector.train()
        if adversary is not None: adversary.train()
        total, t_con, t_adv = 0.0, 0.0, 0.0

        for v1, v2, sid in loader:
            v1, v2 = v1.to(config.device), v2.to(config.device)
            sid_np = sid.numpy()
            opt.zero_grad()

            f1 = backbone.forward_features(v1)
            f2 = backbone.forward_features(v2)

            z1 = projector(f1); z2 = projector(f2)
            con_loss = info_nce_with_subject_mask(
                z1, z2, sids=sid_np, temperature=config.temperature,
                exclude_same_subject_neg=config.exclude_same_subject_negatives
            )

            adv_loss = torch.tensor(0.0, device=config.device)
            if adversary is not None and config.adversary_weight > 0:
                ce = nn.CrossEntropyLoss()
                p1 = adversary(f1, lambd=1.0); p2 = adversary(f2, lambd=1.0)
                adv_loss = 0.5 * (ce(p1, sid.to(config.device)) + ce(p2, sid.to(config.device)))

            loss = con_loss + config.adversary_weight * adv_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            opt.step()

            total += loss.item(); t_con += con_loss.item()
            t_adv += (adv_loss.item() if isinstance(adv_loss, torch.Tensor) else 0.0)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [SSL] epoch {epoch + 1}/{config.pretrain_epochs}  "
                  f"loss={total / max(1, len(loader)):.4f}  "
                  f"contrast={t_con / max(1, len(loader)):.4f}  "
                  f"adv={t_adv / max(1, len(loader)):.4f}")

    save_dir = Path(config.save_path); save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), save_dir / "ssl_backbone.pth")
    print(f"  -> 事前学習済みbackboneを保存: {save_dir/'ssl_backbone.pth'}")
    return str(save_dir / "ssl_backbone.pth")

# ================================
# FT用ユーティリティ（重み読み込み/凍結）
# ================================
def load_backbone_weights_if_any(model, ckpt_path):
    if ckpt_path is None:
        return
    if not os.path.exists(ckpt_path):
        print(f"  [FT] 事前学習重みが見つかりません: {ckpt_path}")
        return
    sd = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    print(f"  [FT] 事前学習重みを読み込み（strict=False）: {ckpt_path}")

def freeze_backbone_until(model, current_epoch, unfreeze_at):
    def is_head(name):
        return any(k in name for k in ['fc', 'conv_final'])
    requires = current_epoch >= unfreeze_at
    for n, p in model.named_parameters():
        if is_head(n):
            p.requires_grad = True
        else:
            p.requires_grad = requires

# ================================
# 学習（CO回帰）
# ================================
def train_model(model, train_loader, val_loader, config, fold=None):
    fold_str = f"Fold {fold + 1}" if fold is not None else ""
    print(f"\n学習開始 {fold_str}")
    print(f"  モデル: {config.model_type}")
    print(f"  エポック数: {config.epochs}")
    print(f"  バッチサイズ: {config.batch_size}")

    model = model.to(config.device)

    if config.loss_type == "combined":
        print(f"  損失関数: CombinedLoss (α={config.loss_alpha}, β={config.loss_beta})")
        criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    elif config.loss_type == "huber_combined":
        print(f"  損失関数: HuberCorrelationLoss")
        criterion = HuberCorrelationLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    else:
        print("  損失関数: MSE")
        mse = nn.MSELoss()
        criterion = lambda pred, target: (mse(pred, target), mse(pred, target), torch.tensor(0.0))

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.scheduler_type == "cosine":
        print(f"  スケジューラー: CosineAnnealingWarmRestarts")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=1e-6
        )
        scheduler_per_batch = False
    elif config.scheduler_type == "onecycle":
        print("  スケジューラー: OneCycleLR")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate * 10,
            epochs=config.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos'
        )
        scheduler_per_batch = True
    else:
        print("  スケジューラー: ReduceLROnPlateau")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)
        scheduler_per_batch = False

    train_losses, val_losses = [], []
    train_correlations, val_correlations = [], []
    best_val_loss = float('inf')
    best_val_corr = -1
    patience_counter = 0

    train_preds_best = None
    train_targets_best = None

    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f'best_model_{config.model_type}_fold{fold + 1}.pth' if fold is not None else f'best_model_{config.model_type}.pth'

    for epoch in range(config.epochs):
        freeze_backbone_until(model, epoch, config.freeze_backbone_epochs)

        # Train
        model.train()
        train_loss = 0
        train_preds_all, train_targets_all = [], []

        for rgb, co in train_loader:
            rgb, co = rgb.to(config.device), co.to(config.device)
            optimizer.zero_grad()
            pred = model(rgb)
            loss, mse_loss, corr_loss = criterion(pred, co)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler_per_batch:
                scheduler.step()
            train_loss += loss.item()
            train_preds_all.extend(pred.detach().cpu().numpy().tolist())
            train_targets_all.extend(co.detach().cpu().numpy().tolist())

        # Val
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for rgb, co in val_loader:
                rgb, co = rgb.to(config.device), co.to(config.device)
                pred = model(rgb)
                loss, _, _ = criterion(pred, co)
                val_loss += loss.item()
                val_preds.extend(pred.cpu().numpy().tolist())
                val_targets.extend(co.cpu().numpy().tolist())

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        train_corr = np.corrcoef(train_preds_all, train_targets_all)[0, 1] if len(train_preds_all) > 1 else 0.0
        val_corr = np.corrcoef(val_preds, val_targets)[0, 1] if len(val_preds) > 1 else 0.0
        val_mae = mean_absolute_error(val_preds, val_targets) if len(val_preds) > 0 else np.nan

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_correlations.append(train_corr); val_correlations.append(val_corr)

        if not scheduler_per_batch:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best
        if val_loss < best_val_loss or (val_loss < best_val_loss * 1.1 and val_corr > best_val_corr):
            best_val_loss = val_loss; best_val_corr = val_corr; patience_counter = 0
            train_preds_best = np.array(train_preds_all); train_targets_best = np.array(train_targets_all)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_corr': best_val_corr,
                'model_type': config.model_type
            }, save_dir / model_name)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch + 1:3d}/{config.epochs}] LR: {current_lr:.2e}")
            print(f"    Train Loss: {train_loss:.4f}, Corr: {train_corr:.4f}")
            print(f"    Val   Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # ベストモデル読み込み
    if os.path.exists(save_dir / model_name):
        checkpoint = torch.load(save_dir / model_name, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  警告: ベストモデルが保存されていません。最後のエポックの重みを使用します。")

    return model, train_preds_best, train_targets_best

# ================================
# 評価
# ================================
def evaluate_model(model, test_loader, config):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for rgb, co in test_loader:
            rgb, co = rgb.to(config.device), co.to(config.device)
            pred = model(rgb)
            predictions.extend(pred.cpu().numpy().tolist())
            targets.extend(co.cpu().numpy().tolist())
    predictions = np.array(predictions); targets = np.array(targets)

    mae = mean_absolute_error(targets, predictions) if len(predictions) > 0 else np.nan
    rmse = np.sqrt(np.mean((targets - predictions) ** 2)) if len(predictions) > 0 else np.nan
    try:
        corr, p_value = pearsonr(targets, predictions)
    except Exception:
        corr, p_value = 0.0, 1.0
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2) + 1e-12
    r2 = 1 - (ss_res / ss_tot)
    return {'mae': mae, 'rmse': rmse, 'corr': corr, 'r2': r2, 'p_value': p_value,
            'predictions': predictions, 'targets': targets}

# ================================
# プロット（個人内）
# ================================
def plot_individual_results(test_results, train_preds, train_targets, config):
    save_dir = Path(config.save_path); save_dir.mkdir(parents=True, exist_ok=True)

    # テスト散布図
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(test_results['targets'], test_results['predictions'], alpha=0.5, s=20)
    min_val = min(test_results['targets'].min(), test_results['predictions'].min())
    max_val = max(test_results['targets'].max(), test_results['predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値 (CO)'); plt.ylabel('予測値 (CO)')
    plt.title(f"テスト [{config.model_type}] - MAE: {test_results['mae']:.3f}, Corr: {test_results['corr']:.3f}")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / f'test_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # トレーン散布図
    if train_preds is not None and train_targets is not None:
        fig = plt.figure(figsize=(10, 8))
        train_corr = np.corrcoef(train_targets, train_preds)[0, 1] if len(train_preds) > 1 else 0.0
        train_mae = mean_absolute_error(train_targets, train_preds) if len(train_preds) > 0 else np.nan
        plt.scatter(train_targets, train_preds, alpha=0.5, s=20)
        min_val = min(train_targets.min(), train_preds.min())
        max_val = max(train_targets.max(), train_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値 (CO)'); plt.ylabel('予測値 (CO)')
        plt.title(f"トレーニング [{config.model_type}] - MAE: {train_mae:.3f}, Corr: {train_corr:.3f}")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(save_dir / f'train_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 真値と予測の波形
    if train_preds is not None and train_targets is not None:
        fig = plt.figure(figsize=(16, 8))
        plt.subplot(2, 1, 1)
        plt.plot(test_results['targets'], 'b-', label='真値', alpha=0.7, linewidth=1)
        plt.plot(test_results['predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
        for i in range(1, len(config.tasks)):
            x_pos = i * len(test_results['targets']) // len(config.tasks)
            plt.axvline(x=x_pos, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('時間 (秒)'); plt.ylabel('CO値'); plt.title(f'テスト波形 [{config.model_type}]')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(train_targets, 'b-', label='真値', alpha=0.7, linewidth=1)
        plt.plot(train_preds, 'g-', label='予測', alpha=0.7, linewidth=1)
        for i in range(1, len(config.tasks)):
            x_pos = i * len(train_targets) // len(config.tasks)
            plt.axvline(x=x_pos, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('時間 (秒)'); plt.ylabel('CO値'); plt.title(f'トレーニング波形 [{config.model_type}]')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'waveforms_{config.model_type}.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n図を保存しました: {config.save_path}")
    print(f"  - test_scatter_{config.model_type}.png")
    if train_preds is not None:
        print(f"  - train_scatter_{config.model_type}.png")
        print(f"  - waveforms_{config.model_type}.png")

# ================================
# 交差検証（個人間）
# ================================
def cross_validation(rgb_data, co_data, subject_ids, config, ssl_ckpt=None):
    print("\n" + "=" * 60)
    print(f"{config.n_folds}分割交差検証開始 - モデル: {config.model_type}")
    if config.exclude_subjects:
        print(f"除外被験者: {', '.join(config.exclude_subjects)}")
    print("=" * 60)

    unique_subjects = sorted(list(set(subject_ids)))
    subject_indices = {subj: [] for subj in unique_subjects}
    for i, subj in enumerate(subject_ids):
        subject_indices[subj].append(i)

    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_subjects)):
        print(f"\nFold {fold + 1}/{config.n_folds}")
        train_subjects = [unique_subjects[i] for i in train_idx]
        test_subjects = [unique_subjects[i] for i in test_idx]
        print(f"  訓練被験者: {len(train_subjects)}名")
        print(f"  テスト被験者: {len(test_subjects)}名 ({', '.join(test_subjects[:3])}...)")

        train_indices = []
        for subj in train_subjects: train_indices.extend(subject_indices[subj])
        test_indices = []
        for subj in test_subjects:  test_indices.extend(subject_indices[subj])

        train_val_rgb = rgb_data[train_indices]; train_val_co = co_data[train_indices]
        test_rgb = rgb_data[test_indices];       test_co = co_data[test_indices]

        split_idx = int(len(train_val_rgb) * 0.8)
        train_rgb = train_val_rgb[:split_idx]; train_co = train_val_co[:split_idx]
        val_rgb   = train_val_rgb[split_idx:]; val_co   = train_val_co[split_idx:]

        train_loader = DataLoader(CODataset(train_rgb, train_co, config.use_channel),
                                  batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(CODataset(val_rgb,   val_co,   config.use_channel),
                                  batch_size=config.batch_size, shuffle=False)
        test_loader  = DataLoader(CODataset(test_rgb,  test_co,  config.use_channel),
                                  batch_size=config.batch_size, shuffle=False)

        model = create_model(config)
        load_backbone_weights_if_any(model, ssl_ckpt)
        model, train_preds, train_targets = train_model(model, train_loader, val_loader, config, fold)

        eval_results = evaluate_model(model, test_loader, config)
        print(f"  結果 - MAE: {eval_results['mae']:.4f}, Corr: {eval_results['corr']:.4f}")

        results.append({
            'fold': fold,
            'mae': eval_results['mae'],
            'corr': eval_results['corr'],
            'predictions': eval_results['predictions'],
            'targets': eval_results['targets'],
            'train_predictions': train_preds,
            'train_targets': train_targets
        })
    return results

# ================================
# プロット（個人間）
# ================================
def plot_cross_results(results, config):
    save_dir = Path(config.save_path); save_dir.mkdir(parents=True, exist_ok=True)

    # 各Fold散布図
    for r in results:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(r['targets'], r['predictions'], alpha=0.5, s=10)
        min_val = min(r['targets'].min(), r['predictions'].min())
        max_val = max(r['targets'].max(), r['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値'); plt.ylabel('予測値')
        plt.title(f"Fold {r['fold'] + 1} テスト [{config.model_type}] - MAE: {r['mae']:.3f}, Corr: {r['corr']:.3f}")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(save_dir / f'fold{r["fold"] + 1}_test_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 全Fold統合
    fig = plt.figure(figsize=(10, 8))
    all_test_targets = np.concatenate([r['targets'] for r in results])
    all_test_predictions = np.concatenate([r['predictions'] for r in results])
    overall_test_mae = mean_absolute_error(all_test_targets, all_test_predictions)
    overall_test_corr, _ = pearsonr(all_test_targets, all_test_predictions)

    plt.scatter(all_test_targets, all_test_predictions, alpha=0.5, s=10)
    min_val = min(all_test_targets.min(), all_test_predictions.min())
    max_val = max(all_test_targets.max(), all_test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値'); plt.ylabel('予測値')

    title = f'全Foldテストデータ [{config.model_type}] - MAE: {overall_test_mae:.3f}, Corr: {overall_test_corr:.3f}'
    if config.exclude_subjects:
        title += f'\n（除外: {", ".join(config.exclude_subjects)}）'
    plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_dir / f'all_test_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n図を保存しました: {config.save_path}")
    print(f"  - 各Foldの散布図")
    print(f"  - 全体の散布図")
    return overall_test_mae, overall_test_corr

# ================================
# メイン
# ================================
def main():
    config = Config()

    print("\n" + "=" * 60)
    print(" PhysNet2DCNN - CO推定モデル（自己教師あり事前学習付き）")
    print("=" * 60)
    print(f"解析: {'個人内' if config.analysis_type == 'individual' else '個人間'}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    print(f"損失関数: {config.loss_type}")
    print(f"スケジューラー: {config.scheduler_type}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")

    try:
        rgb_data, co_data, subject_ids = load_all_data(config)

        # 事前学習
        ssl_ckpt = None
        if config.pretrain:
            ssl_ckpt = pretrain_encoder(rgb_data, subject_ids, config)

        if config.analysis_type == "individual":
            # 個人内
            train_data, val_data, test_data = split_data_individual(rgb_data, co_data, config)

            train_loader = DataLoader(CODataset(train_data[0], train_data[1], config.use_channel),
                                      batch_size=config.batch_size, shuffle=True)
            val_loader   = DataLoader(CODataset(val_data[0],   val_data[1],   config.use_channel),
                                      batch_size=config.batch_size, shuffle=False)
            test_loader  = DataLoader(CODataset(test_data[0],  test_data[1],  config.use_channel),
                                      batch_size=config.batch_size, shuffle=False)

            model = create_model(config)
            load_backbone_weights_if_any(model, ssl_ckpt)

            model, train_preds, train_targets = train_model(model, train_loader, val_loader, config)

            print("\nテストデータで評価中...")
            eval_results = evaluate_model(model, test_loader, config)
            print(f"\n最終結果 [{config.model_type}]:")
            print(f"  MAE: {eval_results['mae']:.4f}")
            print(f"  RMSE: {eval_results['rmse']:.4f}")
            print(f"  相関係数: {eval_results['corr']:.4f}")
            print(f"  R²: {eval_results['r2']:.4f}")

            plot_individual_results(eval_results, train_preds, train_targets, config)

        else:
            # 個人間（交差検証）
            results = cross_validation(rgb_data, co_data, subject_ids, config, ssl_ckpt=ssl_ckpt)
            overall_mae, overall_corr = plot_cross_results(results, config)

            print(f"\n交差検証結果 [{config.model_type}]:")
            for r in results:
                print(f"  Fold {r['fold'] + 1}: MAE={r['mae']:.4f}, Corr={r['corr']:.4f}")
            print(f"\n全体: MAE={overall_mae:.4f}, Corr={overall_corr:.4f}")
            if config.exclude_subjects:
                print(f"\n※除外被験者: {', '.join(config.exclude_subjects)}")

        print("\n完了しました。")

    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
