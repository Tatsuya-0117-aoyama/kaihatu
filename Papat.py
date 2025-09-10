import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
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
        
        # ★血行動態信号タイプ設定（ユーザー変更可能）
        self.signal_type = "CO"  # "CO", "HbO", "HbR", "HbT" など
        self.signal_prefix = "CO_s2"  # ファイル名のプレフィックス
        
        # 被験者設定（bp001～bp032）
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定（6分割交差検証用）
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # モデルタイプ選択
        # "standard": 元のモデル（5ブロック）
        # "deep": 深いモデル（8ブロック）- 10000データ推奨
        # "very_deep": とても深いモデル（10ブロック）
        # "resnet": ResNetスタイル
        self.model_type = "deep"  # ★10000データなら"deep"推奨
        
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
        
        # 表示設定
        self.verbose = True
        self.random_seed = 42

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
    def __init__(self, rgb_data, signal_data, use_channel='B'):
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
        self.signal_data = torch.FloatTensor(signal_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.signal_data[idx]

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
    
    if config.verbose:
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
    """単一被験者のデータを読み込み"""
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません")
        return None, None
    
    rgb_data = np.load(rgb_path)
    
    signal_data_list = []
    for task in config.tasks:
        signal_path = os.path.join(config.signal_base_path, subject, 
                                  config.signal_type, 
                                  f"{config.signal_prefix}_{task}.npy")
        if not os.path.exists(signal_path):
            print(f"警告: {subject}の{task}の{config.signal_type}データが見つかりません")
            return None, None
        signal_data_list.append(np.load(signal_path))
    
    signal_data = np.concatenate(signal_data_list)
    return rgb_data, signal_data

# ================================
# 学習関数
# ================================
def train_model(model, train_loader, val_loader, config, fold=None, subject=None):
    """モデルの学習"""
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    subject_str = f"{subject}" if subject is not None else ""
    
    if config.verbose:
        print(f"\n  学習開始 {subject_str} {fold_str}")
        print(f"    モデル: {config.model_type}")
        print(f"    エポック数: {config.epochs}")
        print(f"    バッチサイズ: {config.batch_size}")
    
    model = model.to(config.device)
    
    # 損失関数の選択
    if config.loss_type == "combined":
        if config.verbose:
            print(f"    損失関数: CombinedLoss (α={config.loss_alpha}, β={config.loss_beta})")
        criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    elif config.loss_type == "huber_combined":
        if config.verbose:
            print(f"    損失関数: HuberCorrelationLoss")
        criterion = HuberCorrelationLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    else:
        if config.verbose:
            print("    損失関数: MSE")
        criterion = lambda pred, target: (nn.MSELoss()(pred, target), 
                                         nn.MSELoss()(pred, target), 
                                         torch.tensor(0.0))
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # スケジューラーの選択
    if config.scheduler_type == "cosine":
        if config.verbose:
            print(f"    スケジューラー: CosineAnnealingWarmRestarts")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=1e-6
        )
        scheduler_per_batch = False
    elif config.scheduler_type == "onecycle":
        if config.verbose:
            print("    スケジューラー: OneCycleLR")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate * 10,
            epochs=config.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos'
        )
        scheduler_per_batch = True
    else:
        if config.verbose:
            print("    スケジューラー: ReduceLROnPlateau")
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
        
        for rgb, sig in train_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            
            optimizer.zero_grad()
            pred = model(rgb)
            
            loss, mse_loss, corr_loss = criterion(pred, sig)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler_per_batch:
                scheduler.step()
            
            train_loss += loss.item()
            train_preds_all.extend(pred.detach().cpu().numpy())
            train_targets_all.extend(sig.detach().cpu().numpy())
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for rgb, sig in val_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                pred = model(rgb)
                
                loss, mse_loss, corr_loss = criterion(pred, sig)
                val_loss += loss.item()
                
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(sig.cpu().numpy())
        
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
            
            # モデル保存先の決定
            save_dir = Path(config.save_path)
            if subject is not None:
                save_dir = save_dir / subject
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if fold is not None:
                model_name = f'best_model_fold{fold+1}.pth'
            else:
                model_name = f'best_model_{config.model_type}.pth'
            
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
        if config.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e}")
            print(f"      Train Loss: {train_loss:.4f}, Corr: {train_corr:.4f}")
            print(f"      Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")
        
        # Early Stopping
        if patience_counter >= config.patience:
            if config.verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # ベストモデル読み込み
    checkpoint = torch.load(save_dir / model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, train_preds_best, train_targets_best

# ================================
# 評価関数
# ================================
def evaluate_model(model, test_loader, config):
    """モデルの評価"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for rgb, sig in test_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            pred = model(rgb)
            predictions.extend(pred.cpu().numpy())
            targets.extend(sig.cpu().numpy())
    
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
# 6分割交差検証
# ================================
def task_cross_validation(rgb_data, signal_data, config, subject, subject_save_dir):
    """タスクごとの6分割交差検証"""
    
    fold_results = []
    all_test_predictions = []
    all_test_targets = []
    all_test_task_indices = []
    
    for fold, test_task in enumerate(config.tasks):
        if config.verbose:
            print(f"\n  Fold {fold+1}/6 - テストタスク: {test_task}")
        
        # タスクごとにデータを分割
        train_rgb_list = []
        train_signal_list = []
        test_rgb_list = []
        test_signal_list = []
        
        for i, task in enumerate(config.tasks):
            start_idx = i * config.task_duration
            end_idx = (i + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signal = signal_data[start_idx:end_idx]
            
            if task == test_task:
                test_rgb_list.append(task_rgb)
                test_signal_list.append(task_signal)
                all_test_task_indices.extend([i] * config.task_duration)
            else:
                train_rgb_list.append(task_rgb)
                train_signal_list.append(task_signal)
        
        # データ結合
        train_rgb = np.concatenate(train_rgb_list)
        train_signal = np.concatenate(train_signal_list)
        test_rgb = np.concatenate(test_rgb_list)
        test_signal = np.concatenate(test_signal_list)
        
        # 訓練・検証分割（8:2）
        split_idx = int(len(train_rgb) * 0.8)
        val_rgb = train_rgb[split_idx:]
        val_signal = train_signal[split_idx:]
        train_rgb = train_rgb[:split_idx]
        train_signal = train_signal[:split_idx]
        
        # データローダー作成
        train_dataset = CODataset(train_rgb, train_signal, config.use_channel)
        val_dataset = CODataset(val_rgb, val_signal, config.use_channel)
        test_dataset = CODataset(test_rgb, test_signal, config.use_channel)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # モデル作成
        model = create_model(config)
        
        # モデル学習
        model, train_preds, train_targets = train_model(
            model, train_loader, val_loader, config, fold, subject
        )
        
        # 評価
        test_results = evaluate_model(model, test_loader, config)
        
        if config.verbose:
            print(f"    Train: MAE={mean_absolute_error(train_targets, train_preds):.4f}, Corr={np.corrcoef(train_targets, train_preds)[0, 1]:.4f}")
            print(f"    Test:  MAE={test_results['mae']:.4f}, Corr={test_results['corr']:.4f}")
        
        # 結果保存
        fold_results.append({
            'fold': fold + 1,
            'test_task': test_task,
            'train_predictions': train_preds,
            'train_targets': train_targets,
            'test_predictions': test_results['predictions'],
            'test_targets': test_results['targets'],
            'train_mae': mean_absolute_error(train_targets, train_preds),
            'train_corr': np.corrcoef(train_targets, train_preds)[0, 1],
            'test_mae': test_results['mae'],
            'test_corr': test_results['corr']
        })
        
        # 全体のテストデータ集約
        all_test_predictions.extend(test_results['predictions'])
        all_test_targets.extend(test_results['targets'])
        
        # 各Foldのプロット
        plot_fold_results(fold_results[-1], subject_save_dir)
    
    # テスト予測を元の順序に並び替え
    sorted_indices = np.argsort(all_test_task_indices)
    all_test_predictions = np.array(all_test_predictions)[sorted_indices]
    all_test_targets = np.array(all_test_targets)[sorted_indices]
    
    return fold_results, all_test_predictions, all_test_targets

# ================================
# プロット関数
# ================================
def plot_fold_results(result, save_dir):
    """各Foldの結果をプロット"""
    fold = result['fold']
    
    # 訓練データ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['train_targets'], result['train_predictions'], alpha=0.5, s=10)
    min_val = min(result['train_targets'].min(), result['train_predictions'].min())
    max_val = max(result['train_targets'].max(), result['train_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"Fold {fold} 訓練データ - MAE: {result['train_mae']:.3f}, Corr: {result['train_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # テストデータ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['test_targets'], result['test_predictions'], alpha=0.5, s=10)
    min_val = min(result['test_targets'].min(), result['test_predictions'].min())
    max_val = max(result['test_targets'].max(), result['test_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"Fold {fold} テストデータ ({result['test_task']}) - MAE: {result['test_mae']:.3f}, Corr: {result['test_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 波形比較
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(result['train_targets'], 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(result['train_predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'Fold {fold} 訓練データ波形')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(result['test_targets'], 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(result['test_predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'Fold {fold} テストデータ波形 ({result["test_task"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_waveforms.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_subject_summary(fold_results, all_test_predictions, all_test_targets, 
                         subject, subject_save_dir):
    """被験者の全体結果をプロット"""
    
    # 全訓練データ統合
    all_train_predictions = np.concatenate([r['train_predictions'] for r in fold_results])
    all_train_targets = np.concatenate([r['train_targets'] for r in fold_results])
    all_train_mae = mean_absolute_error(all_train_targets, all_train_predictions)
    all_train_corr, _ = pearsonr(all_train_targets, all_train_predictions)
    
    # 全テストデータメトリクス
    all_test_mae = mean_absolute_error(all_test_targets, all_test_predictions)
    all_test_corr, _ = pearsonr(all_test_targets, all_test_predictions)
    
    # 全訓練データ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(all_train_targets, all_train_predictions, alpha=0.5, s=10)
    min_val = min(all_train_targets.min(), all_train_predictions.min())
    max_val = max(all_train_targets.max(), all_train_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"{subject} 全訓練データ - MAE: {all_train_mae:.3f}, Corr: {all_train_corr:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 全テストデータ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(all_test_targets, all_test_predictions, alpha=0.5, s=10)
    min_val = min(all_test_targets.min(), all_test_predictions.min())
    max_val = max(all_test_targets.max(), all_test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"{subject} 全テストデータ - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 全テストデータ連結波形
    plt.figure(figsize=(20, 6))
    plt.plot(all_test_targets, 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(all_test_predictions, 'g-', label='予測', alpha=0.7, linewidth=1)
    
    # タスク境界に縦線
    for i in range(1, 6):
        plt.axvline(x=i*60, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'{subject} 全テストデータ連結波形 - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_waveform.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return all_train_mae, all_train_corr, all_test_mae, all_test_corr

def plot_all_subjects_summary(all_subjects_results, config):
    """全被験者のサマリープロット"""
    save_dir = Path(config.save_path)
    
    # 被験者ごとのメトリクスを抽出
    subjects = []
    train_maes = []
    train_corrs = []
    test_maes = []
    test_corrs = []
    
    for result in all_subjects_results:
        subjects.append(result['subject'])
        train_maes.append(result['train_mae'])
        train_corrs.append(result['train_corr'])
        test_maes.append(result['test_mae'])
        test_corrs.append(result['test_corr'])
    
    # 全被験者の訓練データサマリー
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.ravel()
    
    for i, result in enumerate(all_subjects_results[:32]):
        ax = axes[i]
        ax.text(0.5, 0.5, f"{result['subject']}\nMAE: {result['train_mae']:.3f}\nCorr: {result['train_corr']:.3f}", 
                ha='center', va='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#e8f4f8' if result['train_corr'] > 0.7 else '#f8e8e8')
    
    fig.suptitle(f'全被験者 訓練データ結果 - 平均MAE: {np.mean(train_maes):.3f}, 平均Corr: {np.mean(train_corrs):.3f}', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_train_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 全被験者のテストデータサマリー
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.ravel()
    
    for i, result in enumerate(all_subjects_results[:32]):
        ax = axes[i]
        ax.text(0.5, 0.5, f"{result['subject']}\nMAE: {result['test_mae']:.3f}\nCorr: {result['test_corr']:.3f}", 
                ha='center', va='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#e8f4f8' if result['test_corr'] > 0.7 else '#f8e8e8')
    
    fig.suptitle(f'全被験者 テストデータ結果 - 平均MAE: {np.mean(test_maes):.3f}, 平均Corr: {np.mean(test_corrs):.3f}', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_test_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - 個人内解析（6分割交差検証）")
    print("="*60)
    print(f"血行動態信号: {config.signal_type}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    print(f"損失関数: {config.loss_type}")
    print(f"スケジューラー: {config.scheduler_type}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    print(f"被験者数: {len(config.subjects)}")
    
    all_subjects_results = []
    
    for subj_idx, subject in enumerate(config.subjects):
        print(f"\n{'='*60}")
        print(f"被験者 {subject} ({subj_idx+1}/{len(config.subjects)})")
        print(f"{'='*60}")
        
        # 被験者用フォルダ作成
        subject_save_dir = Path(config.save_path) / subject
        subject_save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # データ読み込み
            rgb_data, signal_data = load_data_single_subject(subject, config)
            
            if rgb_data is None or signal_data is None:
                print(f"  {subject}のデータ読み込み失敗。スキップします。")
                continue
            
            print(f"  データ形状: RGB={rgb_data.shape}, Signal={signal_data.shape}")
            
            # 6分割交差検証実行
            fold_results, all_test_predictions, all_test_targets = task_cross_validation(
                rgb_data, signal_data, config, subject, subject_save_dir
            )
            
            # 被験者全体のサマリープロット
            train_mae, train_corr, test_mae, test_corr = plot_subject_summary(
                fold_results, all_test_predictions, all_test_targets, 
                subject, subject_save_dir
            )
            
            # 結果保存
            all_subjects_results.append({
                'subject': subject,
                'train_mae': train_mae,
                'train_corr': train_corr,
                'test_mae': test_mae,
                'test_corr': test_corr,
                'fold_results': fold_results
            })
            
            print(f"\n  {subject} 完了:")
            print(f"    全体訓練: MAE={train_mae:.4f}, Corr={train_corr:.4f}")
            print(f"    全体テスト: MAE={test_mae:.4f}, Corr={test_corr:.4f}")
            
        except Exception as e:
            print(f"  {subject}でエラー発生: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 全被験者のサマリープロット
    if all_subjects_results:
        print(f"\n{'='*60}")
        print("全被験者サマリー作成中...")
        plot_all_subjects_summary(all_subjects_results, config)
        
        # 統計サマリー
        avg_train_mae = np.mean([r['train_mae'] for r in all_subjects_results])
        avg_train_corr = np.mean([r['train_corr'] for r in all_subjects_results])
        avg_test_mae = np.mean([r['test_mae'] for r in all_subjects_results])
        avg_test_corr = np.mean([r['test_corr'] for r in all_subjects_results])
        
        print(f"\n全被験者平均結果:")
        print(f"  訓練: MAE={avg_train_mae:.4f}, Corr={avg_train_corr:.4f}")
        print(f"  テスト: MAE={avg_test_mae:.4f}, Corr={avg_test_corr:.4f}")
    
    print(f"\n{'='*60}")
    print("処理完了")
    print(f"結果保存先: {config.save_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
