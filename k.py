import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# フォント設定（メイリオ）
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 10

# ================================
# 設定クラス
# ================================
class Config:
    def __init__(self):
        # パス設定
        self.rgb_base_path = r"C:\Users\EyeBelow"
        self.signal_base_path = r"C:\Users\Data_signals_bp"
        self.base_save_path = r"D:\EPSCAN\001"
        
        # 日付時間フォルダを作成
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.base_save_path, self.timestamp)
        
        # 解析タイプ
        self.analysis_type = "cross"  # "individual"（個人内） or "cross"（個人間）
        
        # ★★★ 除外する被験者リスト（例："bp001", "bp005", "bp010"）★★★
        # 除外したい被験者IDをリストで指定（空リスト[]なら全員使用）
        self.exclude_subjects = []  # 例: ["bp001", "bp005"] または []
        
        # モデルタイプ選択
        # "standard": 元のモデル（5ブロック）
        # "deep": 深いモデル（8ブロック）- 10000データ推奨
        # "very_deep": とても深いモデル（10ブロック）
        # "resnet": ResNetスタイル
        self.model_type = "deep"  # ★10000データなら"deep"推奨
        
        # データ設定
        if self.analysis_type == "individual":
            self.subjects = ["bp001"]
            self.n_folds = 1
        else:
            # 全被験者リストを生成
            all_subjects = [f"bp{i:03d}" for i in range(1, 33)]
            
            # 除外対象を削除
            self.subjects = [s for s in all_subjects if s not in self.exclude_subjects]
            
            # 除外情報を表示
            if self.exclude_subjects:
                print(f"\n【除外設定】")
                print(f"  除外被験者: {', '.join(self.exclude_subjects)}")
                print(f"  使用被験者数: {len(self.subjects)}名（全{len(all_subjects)}名中）")
            else:
                print(f"\n【全被験者使用】")
                print(f"  使用被験者数: {len(self.subjects)}名")
            
            # 交差検証の分割数（被験者数に応じて調整）
            if len(self.subjects) >= 8:
                self.n_folds = 8
            elif len(self.subjects) >= 4:
                self.n_folds = 4
            else:
                self.n_folds = min(len(self.subjects), 2)
                if self.n_folds < 2:
                    print(f"警告: 被験者数が少なすぎます（{len(self.subjects)}名）。最低2名必要です。")
        
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # 使用チャンネル設定
        self.use_channel = 'B'  # 'R', 'G', 'B', 'RGB'
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)
        
        # 学習設定（モデルタイプに応じて自動調整）
        if self.model_type == "standard":
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        elif self.model_type == "deep":
            # 10000データ向け推奨設定
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
            # デフォルト設定
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7  # MSE/Huber損失の重み
        self.loss_beta = 0.3   # 相関損失の重み
        
        # 学習率スケジューラー設定
        self.scheduler_type = "cosine"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 30 if self.model_type != "standard" else 20
        self.scheduler_T_mult = 2 if self.model_type != "standard" else 1
        
        # データ分割設定
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.random_split = True
        self.random_seed = 42
        
        # 表示設定
        self.verbose = True

# ================================
# カスタム損失関数
# ================================
class CombinedLoss(nn.Module):
    """MSE損失と相関損失を組み合わせた複合損失関数"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE損失
        mse_loss = self.mse(pred, target)
        
        # 相関損失
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        
        numerator = torch.sum(pred_mean * target_mean)
        denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
        correlation = numerator / denominator
        corr_loss = 1 - correlation
        
        total_loss = self.alpha * mse_loss + self.beta * corr_loss
        
        return total_loss, mse_loss, corr_loss

class HuberCorrelationLoss(nn.Module):
    """Huber損失と相関損失の組み合わせ（外れ値にロバスト）"""
    def __init__(self, delta=1.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(self, pred, target):
        # Huber損失
        huber_loss = self.huber(pred, target)
        
        # 相関損失
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        
        numerator = torch.sum(pred_mean * target_mean)
        denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
        correlation = numerator / denominator
        corr_loss = 1 - correlation
        
        total_loss = self.alpha * huber_loss + self.beta * corr_loss
        
        return total_loss, huber_loss, corr_loss

# ================================
# データセット
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, co_data, use_channel='B'):
        # チャンネル選択
        if use_channel == 'R':
            selected_data = rgb_data[:, :, :, 0:1]
        elif use_channel == 'G':
            selected_data = rgb_data[:, :, :, 1:2]
        elif use_channel == 'B':
            selected_data = rgb_data[:, :, :, 2:3]
        else:
            selected_data = rgb_data
        
        self.rgb_data = torch.FloatTensor(selected_data).permute(0, 3, 1, 2)
        self.co_data = torch.FloatTensor(co_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.co_data[idx]

# ================================
# オリジナルのPhysNet2DCNNモデル
# ================================
class PhysNet2DCNN(nn.Module):
    """オリジナルのPhysNet2DCNN（5ブロック構成）"""
    def __init__(self, input_shape):
        super(PhysNet2DCNN, self).__init__()
        
        in_channels = input_shape[2]
        
        # ConvBlock 1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # ConvBlock 2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # ConvBlock 3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # ConvBlock 4
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # ConvBlock 5
        self.conv_block5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        
        # Global Average Pooling
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Final Convolution
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        
        # (B, C, H, W) -> (B, C, H*W=224)
        x = x.view(batch_size, in_channels, -1)
        
        # ConvBlocks with pooling and dropout
        x = self.conv_block1(x)
        x = self.avgpool1(x)
        x = self.dropout(x)
        
        x = self.conv_block2(x)
        x = self.avgpool2(x)
        x = self.dropout(x)
        
        x = self.conv_block3(x)
        x = self.avgpool3(x)
        x = self.dropout(x)
        
        x = self.conv_block4(x)
        x = self.avgpool4(x)
        x = self.dropout(x)
        
        x = self.conv_block5(x)
        
        # Global pooling and final output
        x = self.global_avgpool(x)
        x = self.conv_final(x)
        
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        
        return x

# ================================
# 深層PhysNet2DCNNモデル
# ================================
class DeepPhysNet2DCNN(nn.Module):
    """深層PhysNet2DCNN（8または10ブロック構成）"""
    def __init__(self, input_shape, depth_level="deep"):
        super(DeepPhysNet2DCNN, self).__init__()
        
        in_channels = input_shape[2]
        
        # 深さレベルの設定
        if depth_level == "deep":
            # 深いモデル（8ブロック）- 10000データ推奨
            self.num_blocks = 8
            self.channels = [32, 64, 64, 128, 128, 128, 256, 256]
        elif depth_level == "very_deep":
            # とても深いモデル（10ブロック）
            self.num_blocks = 10
            self.channels = [32, 64, 64, 128, 128, 256, 256, 256, 512, 512]
        else:
            raise ValueError(f"Unknown depth_level: {depth_level}")
        
        self.use_residual = True
        
        # ConvBlocks を動的に生成
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        
        prev_channels = in_channels
        
        for i in range(self.num_blocks):
            out_channels = self.channels[i]
            
            # ConvBlock構成
            if i == 0:
                # 最初のブロック
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU()
                )
            elif i < 4:
                # 前半のブロック（2層構成）
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU()
                )
            else:
                # 後半の深いブロック（3層構成）
                mid_channels = out_channels
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, mid_channels, kernel_size=1),
                    nn.BatchNorm1d(mid_channels),
                    nn.ELU(),
                    nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(mid_channels),
                    nn.ELU(),
                    nn.Conv1d(mid_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU()
                )
            
            self.conv_blocks.append(conv_block)
            
            # Pooling戦略
            if i < 2:
                self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            elif i < 4:
                if i % 2 == 0:
                    self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
                else:
                    self.pools.append(None)
            else:
                self.pools.append(None)
            
            # Residual connection用の1x1畳み込み
            if prev_channels != out_channels:
                self.residual_convs.append(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=1)
                )
            else:
                self.residual_convs.append(None)
            
            prev_channels = out_channels
        
        # Global Average Pooling
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 最終層
        if depth_level == "deep":
            hidden_dim = 128
            self.dropout_rates = [0.3, 0.2]
        else:  # very_deep
            hidden_dim = 256
            self.dropout_rates = [0.4, 0.3]
        
        self.fc = nn.Sequential(
            nn.Linear(prev_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout_rates[0]),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(self.dropout_rates[1]),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.block_dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        
        # (B, C, H, W) -> (B, C, H*W=224)
        x = x.view(batch_size, in_channels, -1)
        
        # ConvBlocks with residual connections
        for i in range(self.num_blocks):
            identity = x
            
            # Convolution block
            x = self.conv_blocks[i](x)
            
            # Residual connection
            if self.residual_convs[i] is not None:
                identity = self.residual_convs[i](identity)
            
            # Pooling
            if self.pools[i] is not None:
                x = self.pools[i](x)
                identity = self.pools[i](identity)
            
            # Residual追加
            x = x + identity * 0.3
            
            # ブロック間のDropout
            if i >= 4:
                x = self.block_dropout(x)
        
        # Global pooling
        x = self.global_avgpool(x)
        x = x.squeeze(-1)
        
        # Fully connected layers
        x = self.fc(x)
        x = x.squeeze(-1)
        
        if batch_size == 1:
            x = x.unsqueeze(0)
        
        return x

# ================================
# ResNetスタイルのPhysNetモデル
# ================================
class ResidualBlock(nn.Module):
    """基本的なResidualブロック"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ResNetPhysNet(nn.Module):
    """ResNetスタイルのPhysNet"""
    def __init__(self, input_shape):
        super(ResNetPhysNet, self).__init__()
        
        in_channels = input_shape[2]
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # 重み初期化
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        
        # 最初のブロック
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # 残りのブロック
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape
        x = x.view(batch_size, x.size(1), -1)
        
        # ResNet forward
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.squeeze(-1)
        
        # Classification
        x = self.fc(x)
        x = x.squeeze(-1)
        
        if batch_size == 1:
            x = x.unsqueeze(0)
        
        return x

# ================================
# モデル作成関数
# ================================
def create_model(config):
    """設定に基づいてモデルを作成"""
    if config.model_type == "standard":
        model = PhysNet2DCNN(config.input_shape)
        model_name = "PhysNet2DCNN (Standard - 5 blocks)"
    elif config.model_type == "deep":
        model = DeepPhysNet2DCNN(config.input_shape, depth_level="deep")
        model_name = "DeepPhysNet2DCNN (Deep - 8 blocks)"
    elif config.model_type == "very_deep":
        model = DeepPhysNet2DCNN(config.input_shape, depth_level="very_deep")
        model_name = "DeepPhysNet2DCNN (Very Deep - 10 blocks)"
    elif config.model_type == "resnet":
        model = ResNetPhysNet(config.input_shape)
        model_name = "ResNetPhysNet"
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    print(f"\n選択モデル: {model_name}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 10000データ向けのアドバイス
    if config.model_type == "standard":
        print("【注意】10000データには'deep'または'very_deep'モデルを推奨します")
    elif config.model_type == "deep":
        print("【推奨】10000データに最適な設定です")
    
    return model

# ================================
# データ読み込み
# ================================
def load_data_single_subject(subject, config):
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません")
        return None, None
    
    rgb_data = np.load(rgb_path)
    
    co_data_list = []
    for task in config.tasks:
        co_path = os.path.join(config.signal_base_path, subject, 
                              "CO", f"CO_s2_{task}.npy")
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
    
    all_rgb_data = []
    all_co_data = []
    all_subject_ids = []
    
    # 除外情報を再度表示（個人間解析の場合）
    if config.analysis_type == "cross" and config.exclude_subjects:
        print(f"除外被験者: {', '.join(config.exclude_subjects)}")
        print()
    
    loaded_subjects = []
    failed_subjects = []
    
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
# データ分割（個人内）
# ================================
def split_data_individual(rgb_data, co_data, config):
    if config.random_split:
        print("\nデータ分割中（ランダム分割）...")
        np.random.seed(config.random_seed)
    else:
        print("\nデータ分割中（順番分割）...")
    
    train_rgb, train_co = [], []
    val_rgb, val_co = [], []
    test_rgb, test_co = [], []
    
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
            
            train_rgb.append(task_rgb[train_indices])
            train_co.append(task_co[train_indices])
            val_rgb.append(task_rgb[val_indices])
            val_co.append(task_co[val_indices])
            test_rgb.append(task_rgb[test_indices])
            test_co.append(task_co[test_indices])
        else:
            train_rgb.append(task_rgb[:train_size])
            train_co.append(task_co[:train_size])
            val_rgb.append(task_rgb[train_size:train_size + val_size])
            val_co.append(task_co[train_size:train_size + val_size])
            test_rgb.append(task_rgb[train_size + val_size:])
            test_co.append(task_co[train_size + val_size:])
    
    train_rgb = np.concatenate(train_rgb)
    train_co = np.concatenate(train_co)
    val_rgb = np.concatenate(val_rgb)
    val_co = np.concatenate(val_co)
    test_rgb = np.concatenate(test_rgb)
    test_co = np.concatenate(test_co)
    
    print(f"分割結果: 訓練{len(train_rgb)}, 検証{len(val_rgb)}, テスト{len(test_rgb)}")
    
    return (train_rgb, train_co), (val_rgb, val_co), (test_rgb, test_co)

# ================================
# 学習
# ================================
def train_model(model, train_loader, val_loader, config, fold=None):
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    print(f"\n学習開始 {fold_str}")
    print(f"  モデル: {config.model_type}")
    print(f"  エポック数: {config.epochs}")
    print(f"  バッチサイズ: {config.batch_size}")
    
    model = model.to(config.device)
    
    # 損失関数の選択
    if config.loss_type == "combined":
        print(f"  損失関数: CombinedLoss (α={config.loss_alpha}, β={config.loss_beta})")
        criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    elif config.loss_type == "huber_combined":
        print(f"  損失関数: HuberCorrelationLoss")
        criterion = HuberCorrelationLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    else:
        print("  損失関数: MSE")
        criterion = lambda pred, target: (nn.MSELoss()(pred, target), 
                                         nn.MSELoss()(pred, target), 
                                         torch.tensor(0.0))
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # スケジューラーの選択
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=False
        )
        scheduler_per_batch = False
    
    train_losses = []
    val_losses = []
    train_correlations = []
    val_correlations = []
    best_val_loss = float('inf')
    best_val_corr = -1
    patience_counter = 0
    
    # 学習時の予測値と真値を保存
    train_preds_best = None
    train_targets_best = None
    
    for epoch in range(config.epochs):
        # 学習フェーズ
        model.train()
        train_loss = 0
        train_preds_all = []
        train_targets_all = []
        
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
            train_preds_all.extend(pred.detach().cpu().numpy())
            train_targets_all.extend(co.detach().cpu().numpy())
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for rgb, co in val_loader:
                rgb, co = rgb.to(config.device), co.to(config.device)
                pred = model(rgb)
                
                loss, mse_loss, corr_loss = criterion(pred, co)
                val_loss += loss.item()
                
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(co.cpu().numpy())
        
        # メトリクス計算
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_corr = np.corrcoef(train_preds_all, train_targets_all)[0, 1]
        val_corr = np.corrcoef(val_preds, val_targets)[0, 1]
        val_mae = mean_absolute_error(val_preds, val_targets)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_correlations.append(train_corr)
        val_correlations.append(val_corr)
        
        # スケジューラー更新
        if not scheduler_per_batch:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # モデル保存
        if val_loss < best_val_loss or (val_loss < best_val_loss * 1.1 and val_corr > best_val_corr):
            best_val_loss = val_loss
            best_val_corr = val_corr
            patience_counter = 0
            
            # 最良時の訓練データの予測値を保存
            train_preds_best = np.array(train_preds_all)
            train_targets_best = np.array(train_targets_all)
            
            save_dir = Path(config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            model_name = f'best_model_{config.model_type}_fold{fold+1}.pth' if fold is not None else f'best_model_{config.model_type}.pth'
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_corr': best_val_corr,
                'model_type': config.model_type
            }, save_dir / model_name)
        else:
            patience_counter += 1
        
        # ログ出力
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e}")
            print(f"    Train Loss: {train_loss:.4f}, Corr: {train_corr:.4f}")
            print(f"    Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")
        
        # Early Stopping
        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # ベストモデル読み込み
    checkpoint = torch.load(save_dir / model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_preds_best, train_targets_best

# ================================
# 評価
# ================================
def evaluate_model(model, test_loader, config):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for rgb, co in test_loader:
            rgb, co = rgb.to(config.device), co.to(config.device)
            pred = model(rgb)
            predictions.extend(pred.cpu().numpy())
            targets.extend(co.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    corr, p_value = pearsonr(targets, predictions)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae, 'rmse': rmse, 'corr': corr,
        'r2': r2, 'p_value': p_value,
        'predictions': predictions, 'targets': targets
    }

# ================================
# プロット（個人内）
# ================================
def plot_individual_results(test_results, train_preds, train_targets, config):
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. テストデータの散布図
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(test_results['targets'], test_results['predictions'], alpha=0.5, s=20)
    min_val = min(test_results['targets'].min(), test_results['predictions'].min())
    max_val = max(test_results['targets'].max(), test_results['predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値 (CO)')
    plt.ylabel('予測値 (CO)')
    plt.title(f"テストデータ [{config.model_type}] - MAE: {test_results['mae']:.3f}, Corr: {test_results['corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'test_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. トレーニングデータの散布図
    fig = plt.figure(figsize=(10, 8))
    train_corr = np.corrcoef(train_targets, train_preds)[0, 1]
    train_mae = mean_absolute_error(train_targets, train_preds)
    plt.scatter(train_targets, train_preds, alpha=0.5, s=20)
    min_val = min(train_targets.min(), train_preds.min())
    max_val = max(train_targets.max(), train_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値 (CO)')
    plt.ylabel('予測値 (CO)')
    plt.title(f"トレーニングデータ [{config.model_type}] - MAE: {train_mae:.3f}, Corr: {train_corr:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'train_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 真値と予測値の波形図
    fig = plt.figure(figsize=(16, 8))
    
    # テストデータの波形
    plt.subplot(2, 1, 1)
    plt.plot(test_results['targets'], 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(test_results['predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
    
    # タスクの境界に赤線を引く
    for i in range(1, len(config.tasks)):
        x_pos = i * len(test_results['targets']) // len(config.tasks)
        plt.axvline(x=x_pos, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('CO値')
    plt.title(f'テストデータ [{config.model_type}] - 真値と予測値の波形')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # トレーニングデータの波形
    plt.subplot(2, 1, 2)
    plt.plot(train_targets, 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(train_preds, 'g-', label='予測', alpha=0.7, linewidth=1)
    
    # タスクの境界に赤線を引く
    for i in range(1, len(config.tasks)):
        x_pos = i * len(train_targets) // len(config.tasks)
        plt.axvline(x=x_pos, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('CO値')
    plt.title(f'トレーニングデータ [{config.model_type}] - 真値と予測値の波形')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'waveforms_{config.model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n図を保存しました: {config.save_path}")
    print(f"  - test_scatter_{config.model_type}.png")
    print(f"  - train_scatter_{config.model_type}.png")
    print(f"  - waveforms_{config.model_type}.png")

# ================================
# 交差検証（個人間）
# ================================
def cross_validation(rgb_data, co_data, subject_ids, config):
    print("\n" + "="*60)
    print(f"{config.n_folds}分割交差検証開始 - モデル: {config.model_type}")
    if config.exclude_subjects:
        print(f"除外被験者: {', '.join(config.exclude_subjects)}")
    print("="*60)
    
    unique_subjects = sorted(list(set(subject_ids)))
    subject_indices = {subj: [] for subj in unique_subjects}
    for i, subj in enumerate(subject_ids):
        subject_indices[subj].append(i)
    
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_subjects)):
        print(f"\nFold {fold+1}/{config.n_folds}")
        
        train_subjects = [unique_subjects[i] for i in train_idx]
        test_subjects = [unique_subjects[i] for i in test_idx]
        
        print(f"  訓練被験者: {len(train_subjects)}名")
        print(f"  テスト被験者: {len(test_subjects)}名 ({', '.join(test_subjects[:3])}...)")
        
        # データ分割
        train_indices = []
        for subj in train_subjects:
            train_indices.extend(subject_indices[subj])
        test_indices = []
        for subj in test_subjects:
            test_indices.extend(subject_indices[subj])
        
        train_val_rgb = rgb_data[train_indices]
        train_val_co = co_data[train_indices]
        test_rgb = rgb_data[test_indices]
        test_co = co_data[test_indices]
        
        # 訓練・検証分割
        split_idx = int(len(train_val_rgb) * 0.8)
        train_rgb = train_val_rgb[:split_idx]
        train_co = train_val_co[:split_idx]
        val_rgb = train_val_rgb[split_idx:]
        val_co = train_val_co[split_idx:]
        
        # データローダー
        train_dataset = CODataset(train_rgb, train_co, config.use_channel)
        val_dataset = CODataset(val_rgb, val_co, config.use_channel)
        test_dataset = CODataset(test_rgb, test_co, config.use_channel)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # モデル作成と学習
        model = create_model(config)
        model, train_preds, train_targets = train_model(model, train_loader, val_loader, config, fold)
        
        # 評価
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
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 各Foldのテストデータ散布図
    for r in results:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(r['targets'], r['predictions'], alpha=0.5, s=10)
        min_val = min(r['targets'].min(), r['predictions'].min())
        max_val = max(r['targets'].max(), r['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"Fold {r['fold']+1} テスト [{config.model_type}] - MAE: {r['mae']:.3f}, Corr: {r['corr']:.3f}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'fold{r["fold"]+1}_test_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 全Foldのテストデータ統合散布図
    fig = plt.figure(figsize=(10, 8))
    all_test_targets = np.concatenate([r['targets'] for r in results])
    all_test_predictions = np.concatenate([r['predictions'] for r in results])
    overall_test_mae = mean_absolute_error(all_test_targets, all_test_predictions)
    overall_test_corr, _ = pearsonr(all_test_targets, all_test_predictions)
    
    plt.scatter(all_test_targets, all_test_predictions, alpha=0.5, s=10)
    min_val = min(all_test_targets.min(), all_test_predictions.min())
    max_val = max(all_test_targets.max(), all_test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    
    # 除外情報を追加
    title = f'全Foldテストデータ [{config.model_type}] - MAE: {overall_test_mae:.3f}, Corr: {overall_test_corr:.3f}'
    if config.exclude_subjects:
        title += f'\n（除外: {", ".join(config.exclude_subjects)}）'
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'all_test_scatter_{config.model_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n図を保存しました: {config.save_path}")
    print(f"  - 各Foldの散布図")
    print(f"  - 全体の散布図")
    
    return overall_test_mae, overall_test_corr

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - CO推定モデル")
    print("="*60)
    print(f"解析: {'個人内' if config.analysis_type == 'individual' else '個人間'}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    print(f"損失関数: {config.loss_type}")
    print(f"スケジューラー: {config.scheduler_type}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    
    try:
        # データ読み込み
        rgb_data, co_data, subject_ids = load_all_data(config)
        
        if config.analysis_type == "individual":
            # 個人内解析
            train_data, val_data, test_data = split_data_individual(rgb_data, co_data, config)
            
            # データローダー
            train_dataset = CODataset(train_data[0], train_data[1], config.use_channel)
            val_dataset = CODataset(val_data[0], val_data[1], config.use_channel)
            test_dataset = CODataset(test_data[0], test_data[1], config.use_channel)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            
            # モデル作成
            model = create_model(config)
            
            # モデル学習
            model, train_preds, train_targets = train_model(
                model, train_loader, val_loader, config
            )
            
            # 評価
            print("\nテストデータで評価中...")
            eval_results = evaluate_model(model, test_loader, config)
            
            print(f"\n最終結果 [{config.model_type}]:")
            print(f"  MAE: {eval_results['mae']:.4f}")
            print(f"  RMSE: {eval_results['rmse']:.4f}")
            print(f"  相関係数: {eval_results['corr']:.4f}")
            print(f"  R²: {eval_results['r2']:.4f}")
            
            # プロット
            plot_individual_results(eval_results, train_preds, train_targets, config)
            
        else:
            # 個人間解析
            results = cross_validation(rgb_data, co_data, subject_ids, config)
            overall_mae, overall_corr = plot_cross_results(results, config)
            
            print(f"\n交差検証結果 [{config.model_type}]:")
            for r in results:
                print(f"  Fold {r['fold']+1}: MAE={r['mae']:.4f}, Corr={r['corr']:.4f}")
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
