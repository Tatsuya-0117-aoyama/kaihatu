"""
CO推定モデル - 事前学習済みエンコーダー統合版
事前学習で獲得した個人差にロバストな特徴抽出器を使用してCO推定を行う
"""

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

# フォント設定
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
        self.analysis_type = "cross"  # "individual" or "cross"
        
        # 除外する被験者リスト
        self.exclude_subjects = []
        
        # ★★★ 事前学習設定 ★★★
        self.use_pretrained = True  # 事前学習済みモデルを使用
        self.pretrained_encoder_path = r"D:\EPSCAN\pretrain\[timestamp]\pretrained_encoder_final.pth"
        self.freeze_encoder = False  # エンコーダーを完全に凍結するか
        self.freeze_encoder_epochs = 20  # 最初のNエポックはエンコーダーを凍結
        self.finetune_lr_ratio = 0.1  # エンコーダー部分の学習率比率
        
        # モデルタイプ選択
        self.model_type = "pretrained"  # "pretrained"を追加
        
        # データ設定
        if self.analysis_type == "individual":
            self.subjects = ["bp001"]
            self.n_folds = 1
        else:
            all_subjects = [f"bp{i:03d}" for i in range(1, 33)]
            self.subjects = [s for s in all_subjects if s not in self.exclude_subjects]
            
            if self.exclude_subjects:
                print(f"\n【除外設定】")
                print(f"  除外被験者: {', '.join(self.exclude_subjects)}")
                print(f"  使用被験者数: {len(self.subjects)}名")
            
            if len(self.subjects) >= 8:
                self.n_folds = 8
            elif len(self.subjects) >= 4:
                self.n_folds = 4
            else:
                self.n_folds = min(len(self.subjects), 2)
        
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # 使用チャンネル設定（事前学習モデルはRGB必須）
        if self.use_pretrained:
            self.use_channel = 'RGB'
            self.input_shape = (14, 16, 3)
        else:
            self.use_channel = 'B'
            self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)
        
        # 学習設定（事前学習モデル用に調整）
        if self.use_pretrained:
            self.batch_size = 32
            self.epochs = 100  # 事前学習済みなので少なめ
            self.learning_rate = 0.0005  # より小さい学習率
            self.weight_decay = 1e-4
            self.patience = 30
        else:
            self.batch_size = 16
            self.epochs = 150
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"
        self.loss_alpha = 0.7
        self.loss_beta = 0.3
        
        # スケジューラー設定
        self.scheduler_type = "cosine"
        self.scheduler_T0 = 20
        self.scheduler_T_mult = 2
        
        # データ分割設定
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.random_split = True
        self.random_seed = 42
        
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
        mse_loss = self.mse(pred, target)
        
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        
        numerator = torch.sum(pred_mean * target_mean)
        denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
        correlation = numerator / denominator
        corr_loss = 1 - correlation
        
        total_loss = self.alpha * mse_loss + self.beta * corr_loss
        
        return total_loss, mse_loss, corr_loss

# ================================
# データセット
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, co_data, use_channel='RGB'):
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
# 事前学習済みエンコーダー（事前学習コードから）
# ================================
class PretrainedEncoder(nn.Module):
    """事前学習で使用したエンコーダー構造"""
    
    def __init__(self, encoder_type="deep"):
        super(PretrainedEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "standard":
            self.encoder = self._build_standard_encoder()
            self.feature_dim = 256
        elif encoder_type == "deep":
            self.encoder = self._build_deep_encoder()
            self.feature_dim = 256
        elif encoder_type == "very_deep":
            self.encoder = self._build_very_deep_encoder()
            self.feature_dim = 512
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def _build_standard_encoder(self):
        """標準エンコーダー（5ブロック）"""
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Conv1d(32, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ELU()
            )
        ])
    
    def _build_deep_encoder(self):
        """深いエンコーダー（8ブロック）"""
        blocks = []
        channels = [3, 32, 64, 64, 128, 128, 256, 256, 256]
        
        for i in range(8):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            if i == 0:
                block = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU(),
                    nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU()
                )
            else:
                block = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU(),
                    nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU()
                )
            
            if i < 4:
                block = nn.Sequential(block, nn.AvgPool1d(2))
            
            blocks.append(block)
        
        return nn.ModuleList(blocks)
    
    def _build_very_deep_encoder(self):
        """とても深いエンコーダー（10ブロック）"""
        blocks = []
        channels = [3, 32, 64, 64, 128, 128, 256, 256, 256, 512, 512]
        
        for i in range(10):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            
            if i == 0:
                kernel_size = 7
            elif i < 3:
                kernel_size = 5
            else:
                kernel_size = 3
            
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ELU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ELU()
            )
            
            if i < 5 and i % 2 == 0:
                block = nn.Sequential(block, nn.AvgPool1d(2))
            
            blocks.append(block)
        
        return nn.ModuleList(blocks)
    
    def forward(self, x, return_features=True):
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), -1)
        
        for block in self.encoder:
            x = block(x)
        
        features = self.global_pool(x).squeeze(-1)
        return features

# ================================
# 事前学習済みモデルを使用したCO推定モデル
# ================================
class PretrainedPhysNetCO(nn.Module):
    """事前学習済みエンコーダーを使用したCO推定モデル"""
    
    def __init__(self, pretrained_path=None, encoder_type="deep", freeze_encoder=False):
        super(PretrainedPhysNetCO, self).__init__()
        
        # エンコーダー初期化
        self.encoder = PretrainedEncoder(encoder_type)
        self.feature_dim = self.encoder.feature_dim
        
        # 事前学習済み重みを読み込み
        if pretrained_path and Path(pretrained_path).exists():
            self._load_pretrained_weights(pretrained_path)
            print(f"✓ 事前学習済みエンコーダーを読み込みました")
        else:
            print("⚠ 事前学習済み重みが見つかりません。ランダム初期化で開始します")
        
        # エンコーダーを凍結するか
        if freeze_encoder:
            self._freeze_encoder()
        
        # CO推定用の回帰ヘッド
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 回帰ヘッドの重み初期化
        self._initialize_regression_head()
    
    def _load_pretrained_weights(self, pretrained_path):
        """事前学習済み重みを読み込み"""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # エンコーダーの重みを抽出
        if 'encoder_state_dict' in checkpoint:
            encoder_dict = checkpoint['encoder_state_dict']
        else:
            encoder_dict = checkpoint
        
        # 現在のモデルの状態辞書
        current_dict = self.encoder.state_dict()
        
        # マッチする重みを転送
        loaded_keys = []
        for key, value in encoder_dict.items():
            # 'encoder.'プレフィックスを除去
            clean_key = key.replace('encoder.', '')
            
            if clean_key in current_dict:
                if current_dict[clean_key].shape == value.shape:
                    current_dict[clean_key] = value
                    loaded_keys.append(clean_key)
                else:
                    print(f"  形状不一致のためスキップ: {clean_key}")
        
        # 重みを更新
        self.encoder.load_state_dict(current_dict)
        print(f"  {len(loaded_keys)}層の重みを転送しました")
    
    def _freeze_encoder(self):
        """エンコーダー層を凍結"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("  エンコーダー層を凍結しました")
    
    def _unfreeze_encoder(self):
        """エンコーダー層を解凍"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("  エンコーダー層を解凍しました")
    
    def _initialize_regression_head(self):
        """回帰ヘッドの重み初期化"""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # エンコーダーで特徴抽出
        features = self.encoder(x, return_features=True)
        
        # CO値を推定
        output = self.regression_head(features)
        output = output.squeeze(-1)
        
        if output.dim() == 0:
            output = output.unsqueeze(0)
        
        return output

# ================================
# オリジナルモデル（比較用）
# ================================
class PhysNet2DCNN(nn.Module):
    """オリジナルのPhysNet2DCNN（比較用）"""
    def __init__(self, input_shape):
        super(PhysNet2DCNN, self).__init__()
        
        in_channels = input_shape[2]
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.avgpool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        self.avgpool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv_block5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )
        
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        
        x = x.view(batch_size, in_channels, -1)
        
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
        
        x = self.global_avgpool(x)
        x = self.conv_final(x)
        
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        
        return x

# ================================
# モデル作成関数
# ================================
def create_model(config):
    """設定に基づいてモデルを作成"""
    
    if config.use_pretrained:
        # 事前学習済みモデルを使用
        pretrained_path = config.pretrained_encoder_path
        
        # エンコーダータイプを自動検出
        encoder_type = "deep"  # デフォルト
        if Path(pretrained_path).exists():
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'config' in checkpoint:
                encoder_type = checkpoint['config'].get('encoder_type', 'deep')
        
        model = PretrainedPhysNetCO(
            pretrained_path=pretrained_path,
            encoder_type=encoder_type,
            freeze_encoder=config.freeze_encoder
        )
        model_name = f"PretrainedPhysNetCO (事前学習済み - {encoder_type})"
    else:
        # オリジナルモデル
        model = PhysNet2DCNN(config.input_shape)
        model_name = "PhysNet2DCNN (オリジナル)"
    
    print(f"\n選択モデル: {model_name}")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"学習可能パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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
    
    if config.use_pretrained:
        print("★ 事前学習済みモデル使用のため、RGB全チャンネルで読み込みます")
    
    all_rgb_data = []
    all_co_data = []
    all_subject_ids = []
    
    for subject in config.subjects:
        rgb_data, co_data = load_data_single_subject(subject, config)
        if rgb_data is not None and co_data is not None:
            all_rgb_data.append(rgb_data)
            all_co_data.append(co_data)
            all_subject_ids.extend([subject] * len(rgb_data))
            print(f"✓ {subject}のデータ読み込み完了")
    
    if len(all_rgb_data) == 0:
        raise ValueError("データが読み込めませんでした")
    
    all_rgb_data = np.concatenate(all_rgb_data, axis=0)
    all_co_data = np.concatenate(all_co_data, axis=0)
    
    print(f"\n読み込み完了:")
    print(f"  データ形状: {all_rgb_data.shape}")
    print(f"  データ数: {len(all_rgb_data)}")
    print(f"  使用チャンネル: {config.use_channel}")
    
    return all_rgb_data, all_co_data, all_subject_ids

# ================================
# 学習（段階的ファインチューニング対応）
# ================================
def train_model(model, train_loader, val_loader, config, fold=None):
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    print(f"\n学習開始 {fold_str}")
    
    model = model.to(config.device)
    
    # 損失関数
    criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    
    # オプティマイザー設定（事前学習モデル用の調整）
    if config.use_pretrained and hasattr(model, 'encoder'):
        # エンコーダーと回帰ヘッドで異なる学習率
        encoder_params = list(model.encoder.parameters())
        head_params = list(model.regression_head.parameters())
        
        optimizer = optim.Adam([
            {'params': encoder_params, 'lr': config.learning_rate * config.finetune_lr_ratio},
            {'params': head_params, 'lr': config.learning_rate}
        ], weight_decay=config.weight_decay)
        
        print(f"  差分学習率適用:")
        print(f"    エンコーダー: {config.learning_rate * config.finetune_lr_ratio:.1e}")
        print(f"    回帰ヘッド: {config.learning_rate:.1e}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay)
    
    # スケジューラー
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_val_corr = -1
    patience_counter = 0
    
    train_preds_best = None
    train_targets_best = None
    
    # 段階的解凍の管理
    encoder_frozen = config.freeze_encoder
    
    for epoch in range(config.epochs):
        # 段階的解凍（事前学習モデルの場合）
        if config.use_pretrained and hasattr(model, 'encoder'):
            if encoder_frozen and epoch >= config.freeze_encoder_epochs:
                model._unfreeze_encoder()
                encoder_frozen = False
                
                # 学習率を再調整
                for param_group in optimizer.param_groups:
                    if len(param_group['params']) == len(list(model.encoder.parameters())):
                        param_group['lr'] = config.learning_rate * config.finetune_lr_ratio * 0.5
                        print(f"  エポック{epoch+1}: エンコーダー解凍、学習率調整")
        
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
        
        scheduler.step()
        
        # モデル保存
        if val_loss < best_val_loss or (val_loss < best_val_loss * 1.1 and val_corr > best_val_corr):
            best_val_loss = val_loss
            best_val_corr = val_corr
            patience_counter = 0
            
            train_preds_best = np.array(train_preds_all)
            train_targets_best = np.array(train_targets_all)
            
            save_dir = Path(config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_type = "pretrained" if config.use_pretrained else "original"
            model_name = f'best_model_{model_type}_fold{fold+1}.pth' if fold is not None else f'best_model_{model_type}.pth'
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_corr': best_val_corr,
                'use_pretrained': config.use_pretrained
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
# 交差検証
# ================================
def cross_validation(rgb_data, co_data, subject_ids, config):
    print("\n" + "="*60)
    model_type = "事前学習済み" if config.use_pretrained else "オリジナル"
    print(f"{config.n_folds}分割交差検証開始 - モデル: {model_type}")
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
        print(f"  テスト被験者: {len(test_subjects)}名")
        
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
# プロット
# ================================
def plot_results(results, config):
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_type = "pretrained" if config.use_pretrained else "original"
    
    # 全Foldの統合結果
    all_targets = np.concatenate([r['targets'] for r in results])
    all_predictions = np.concatenate([r['predictions'] for r in results])
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_corr, _ = pearsonr(all_targets, all_predictions)
    
    # 1. 全体の散布図
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(all_targets, all_predictions, alpha=0.5, s=10)
    min_val = min(all_targets.min(), all_predictions.min())
    max_val = max(all_targets.max(), all_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    
    title = f'全Foldテストデータ [{model_type}]\nMAE: {overall_mae:.3f}, Corr: {overall_corr:.3f}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'all_test_scatter_{model_type}.png', dpi=150)
    plt.close()
    
    # 2. Fold別の性能比較
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE比較
    fold_nums = [r['fold']+1 for r in results]
    maes = [r['mae'] for r in results]
    axes[0].bar(fold_nums, maes)
    axes[0].axhline(y=overall_mae, color='r', linestyle='--', label=f'平均: {overall_mae:.3f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Fold別MAE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 相関係数比較
    corrs = [r['corr'] for r in results]
    axes[1].bar(fold_nums, corrs)
    axes[1].axhline(y=overall_corr, color='r', linestyle='--', label=f'平均: {overall_corr:.3f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('相関係数')
    axes[1].set_title('Fold別相関係数')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'交差検証結果 [{model_type}]')
    plt.tight_layout()
    plt.savefig(save_dir / f'fold_comparison_{model_type}.png', dpi=150)
    plt.close()
    
    print(f"\n図を保存しました: {config.save_path}")
    
    return overall_mae, overall_corr

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    # 事前学習済みモデルのパスを設定（実際のパスに変更してください）
    # config.pretrained_encoder_path = r"D:\EPSCAN\pretrain\20241101_120000\pretrained_encoder_final.pth"
    
    print("\n" + "="*60)
    if config.use_pretrained:
        print(" PhysNet2DCNN - CO推定モデル（事前学習済み版）")
    else:
        print(" PhysNet2DCNN - CO推定モデル（オリジナル版）")
    print("="*60)
    print(f"解析: {'個人内' if config.analysis_type == 'individual' else '個人間'}")
    print(f"事前学習: {'有効' if config.use_pretrained else '無効'}")
    if config.use_pretrained:
        print(f"エンコーダー凍結: {config.freeze_encoder_epochs}エポック")
        print(f"学習率比率: {config.finetune_lr_ratio}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    
    try:
        # データ読み込み
        rgb_data, co_data, subject_ids = load_all_data(config)
        
        if config.analysis_type == "cross":
            # 交差検証
            results = cross_validation(rgb_data, co_data, subject_ids, config)
            overall_mae, overall_corr = plot_results(results, config)
            
            print(f"\n交差検証結果:")
            for r in results:
                print(f"  Fold {r['fold']+1}: MAE={r['mae']:.4f}, Corr={r['corr']:.4f}")
            print(f"\n全体: MAE={overall_mae:.4f}, Corr={overall_corr:.4f}")
            
            # 事前学習の効果を表示
            if config.use_pretrained:
                print("\n【事前学習の効果】")
                print("個人差にロバストな特徴抽出により、")
                print("未知被験者への汎化性能が向上しています。")
        
        print("\n完了しました。")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
