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
        self.save_path = r"D:\EPSCAN\001"
        
        # 解析タイプ
        self.analysis_type = "individual"  # "individual" or "cross"
        
        # データ設定
        if self.analysis_type == "individual":
            self.subjects = ["bp001"]
            self.n_folds = 1
        else:
            self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
            self.n_folds = 8
        
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # 時系列設定
        self.temporal_window = 5  # 時間窓（前後のフレーム数）
        self.use_temporal = True  # 時系列情報を使用するか
        self.temporal_type = "lstm"  # "lstm", "gru", "attention", "conv1d"
        
        # 使用チャンネル
        self.use_channel = 'B'  # 'R', 'G', 'B', 'RGB'
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)
        
        # 学習設定
        self.batch_size = 8  # 時系列処理のためバッチサイズを小さく
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7
        self.loss_beta = 0.3
        
        # スケジューラー設定
        self.scheduler_type = "cosine"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 20
        self.scheduler_T_mult = 1
        
        # データ分割
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.random_split = True
        self.random_seed = 42
        
        # Early Stopping
        self.patience = 20
        
        # 表示設定
        self.verbose = True

# ================================
# カスタム損失関数
# ================================
class CombinedLoss(nn.Module):
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

class HuberCorrelationLoss(nn.Module):
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
        
        total_loss = self.alpha * huber_loss + self.beta * corr_loss
        
        return total_loss, huber_loss, corr_loss

# ================================
# データセット（時系列対応）
# ================================
class TemporalCODataset(Dataset):
    def __init__(self, rgb_data, co_data, temporal_window=5, use_channel='B'):
        self.temporal_window = temporal_window
        
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
        
        # 時系列データの作成
        self.temporal_indices = []
        for i in range(len(self.rgb_data)):
            # 時間窓内のインデックスを取得（境界処理あり）
            indices = []
            for j in range(-temporal_window, temporal_window + 1):
                idx = i + j
                # 境界処理：クリッピング
                idx = max(0, min(len(self.rgb_data) - 1, idx))
                indices.append(idx)
            self.temporal_indices.append(indices)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        # 時間窓内のフレームを取得
        indices = self.temporal_indices[idx]
        temporal_rgb = self.rgb_data[indices]  # (T, C, H, W)
        
        # 中心フレームのCO値
        co = self.co_data[idx]
        
        return temporal_rgb, co

# ================================
# Temporal PhysNet2DCNNモデル
# ================================
class TemporalPhysNet2DCNN(nn.Module):
    def __init__(self, input_shape, temporal_window=5, temporal_type="lstm"):
        super(TemporalPhysNet2DCNN, self).__init__()
        
        self.temporal_window = temporal_window
        self.temporal_type = temporal_type
        self.temporal_size = 2 * temporal_window + 1  # 前後+中心
        
        in_channels = input_shape[2]
        
        # Spatial Feature Extraction (PhysNet部分)
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
        
        # Temporal Processing
        self.feature_dim = 64
        
        if temporal_type == "lstm":
            self.temporal = nn.LSTM(self.feature_dim, 32, 
                                   num_layers=2, batch_first=True, 
                                   bidirectional=True, dropout=0.2)
            self.temporal_out_dim = 64  # bidirectional
        elif temporal_type == "gru":
            self.temporal = nn.GRU(self.feature_dim, 32, 
                                  num_layers=2, batch_first=True, 
                                  bidirectional=True, dropout=0.2)
            self.temporal_out_dim = 64
        elif temporal_type == "attention":
            self.temporal = TemporalAttention(self.feature_dim, num_heads=4)
            self.temporal_out_dim = self.feature_dim
        else:  # conv1d
            self.temporal = nn.Sequential(
                nn.Conv1d(self.temporal_size, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 8, kernel_size=3, padding=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Conv1d(8, 1, kernel_size=1)
            )
            self.temporal_out_dim = self.feature_dim
        
        # Final layers
        if temporal_type in ["lstm", "gru", "attention"]:
            self.final_fc = nn.Sequential(
                nn.Linear(self.temporal_out_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)
            )
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1)
            )
        
        self.dropout = nn.Dropout(0.2)
        
    def extract_features(self, x):
        """単一フレームから特徴抽出"""
        batch_size = x.size(0)
        in_channels = x.size(1)
        
        # (B, C, H, W) -> (B, C, H*W)
        x = x.view(batch_size, in_channels, -1)
        
        # PhysNet feature extraction
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
        
        # (B, 64, 1) -> (B, 64)
        x = x.squeeze(-1)
        
        return x
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        batch_size = x.size(0)
        temporal_size = x.size(1)
        
        # 各時間ステップで特徴抽出
        features = []
        for t in range(temporal_size):
            feat = self.extract_features(x[:, t, :, :, :])
            features.append(feat)
        
        features = torch.stack(features, dim=1)  # (B, T, 64)
        
        # Temporal processing
        if self.temporal_type in ["lstm", "gru"]:
            temporal_out, _ = self.temporal(features)
            # 中心フレームの出力を使用
            center_idx = temporal_size // 2
            out = temporal_out[:, center_idx, :]
        elif self.temporal_type == "attention":
            out = self.temporal(features)
            # 中心フレームの出力
            center_idx = temporal_size // 2
            out = out[:, center_idx, :]
        else:  # conv1d
            # (B, T, F) -> (B, T, F) for conv1d
            features = features.permute(0, 1, 2)  # Keep shape
            out = self.temporal(features)  # (B, 1, F)
            out = out.squeeze(1)  # (B, F)
        
        # Final output
        out = self.final_fc(out)
        out = out.squeeze()
        
        if batch_size == 1:
            out = out.unsqueeze(0)
        
        return out

# ================================
# Temporal Attention Module
# ================================
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, 
                                              dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

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
    print(f"  被験者数: {len(config.subjects)}")
    print(f"  データ形状: {all_rgb_data.shape}")
    print(f"  使用チャンネル: {config.use_channel}成分")
    print(f"  時間窓: ±{config.temporal_window}フレーム")
    
    return all_rgb_data, all_co_data, all_subject_ids

# ================================
# データ分割
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
    
    model = model.to(config.device)
    
    # 損失関数
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
    
    # スケジューラー
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
    
    for epoch in range(config.epochs):
        # 学習
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
        
        # 検証
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
            
            save_dir = Path(config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            model_name = f'best_model_fold{fold+1}.pth' if fold is not None else 'best_model.pth'
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_corr': best_val_corr
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
    
    return model, train_losses, val_losses, train_correlations, val_correlations

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
# プロット
# ================================
def plot_results(eval_results, train_losses, val_losses, 
                train_correlations, val_correlations, config):
    fig = plt.figure(figsize=(20, 12))
    
    predictions = eval_results['predictions']
    targets = eval_results['targets']
    
    # 1. 損失曲線
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('損失の学習曲線')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 相関曲線
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(train_correlations, label='Train Corr', alpha=0.8, color='green')
    ax2.plot(val_correlations, label='Val Corr', alpha=0.8, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('相関係数の推移')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 予測vs真値
    ax3 = plt.subplot(3, 4, 3)
    ax3.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax3.set_xlabel('真値 (CO)')
    ax3.set_ylabel('予測値 (CO)')
    ax3.set_title(f"MAE: {eval_results['mae']:.3f}, Corr: {eval_results['corr']:.3f}")
    ax3.grid(True, alpha=0.3)
    
    # 4. 残差プロット
    ax4 = plt.subplot(3, 4, 4)
    residuals = targets - predictions
    ax4.scatter(predictions, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('予測値')
    ax4.set_ylabel('残差')
    ax4.set_title(f'平均: {residuals.mean():.3f}, STD: {residuals.std():.3f}')
    ax4.grid(True, alpha=0.3)
    
    # 5. 時系列比較（長期）
    ax5 = plt.subplot(3, 4, 5)
    sample_range = min(180, len(targets))
    ax5.plot(range(sample_range), targets[:sample_range], 'b-', label='真値', alpha=0.7, linewidth=0.8)
    ax5.plot(range(sample_range), predictions[:sample_range], 'r-', label='予測', alpha=0.7, linewidth=0.8)
    ax5.set_xlabel('時間 (秒)')
    ax5.set_ylabel('CO値')
    ax5.set_title('時系列比較（180秒）')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 時系列比較（短期・詳細）
    ax6 = plt.subplot(3, 4, 6)
    start = 60
    end = 90
    ax6.plot(range(start, end), targets[start:end], 'b-', marker='o', markersize=3, label='真値', alpha=0.7)
    ax6.plot(range(start, end), predictions[start:end], 'r-', marker='^', markersize=3, label='予測', alpha=0.7)
    ax6.set_xlabel('時間 (秒)')
    ax6.set_ylabel('CO値')
    ax6.set_title('時系列詳細比較（60-90秒）')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 誤差分布
    ax7 = plt.subplot(3, 4, 7)
    errors = np.abs(targets - predictions)
    ax7.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax7.axvline(x=eval_results['mae'], color='r', linestyle='--', lw=2, 
                label=f"MAE: {eval_results['mae']:.3f}")
    ax7.set_xlabel('絶対誤差')
    ax7.set_ylabel('頻度')
    ax7.set_title('絶対誤差分布')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. タスク別性能
    ax8 = plt.subplot(3, 4, 8)
    task_maes = []
    task_corrs = []
    for i in range(6):
        start = i * len(targets) // 6
        end = (i + 1) * len(targets) // 6 if i < 5 else len(targets)
        task_mae = mean_absolute_error(targets[start:end], predictions[start:end])
        task_corr, _ = pearsonr(targets[start:end], predictions[start:end])
        task_maes.append(task_mae)
        task_corrs.append(task_corr)
    
    x = np.arange(6)
    width = 0.35
    ax8.bar(x - width/2, task_maes, width, label='MAE', alpha=0.7, color='blue')
    ax8_twin = ax8.twinx()
    ax8_twin.bar(x + width/2, task_corrs, width, label='Corr', color='orange', alpha=0.7)
    ax8.set_xlabel('タスク')
    ax8.set_ylabel('MAE', color='blue')
    ax8_twin.set_ylabel('相関係数', color='orange')
    ax8.set_title('タスク別性能')
    ax8.set_xticks(x)
    ax8.set_xticklabels(config.tasks)
    ax8.grid(True, alpha=0.3)
    
    # 9. Bland-Altmanプロット
    ax9 = plt.subplot(3, 4, 9)
    mean_vals = (targets + predictions) / 2
    diff_vals = targets - predictions
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    
    ax9.scatter(mean_vals, diff_vals, alpha=0.5, s=20)
    ax9.axhline(y=mean_diff, color='red', linestyle='-', label=f'平均差: {mean_diff:.3f}')
    ax9.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--', 
                label=f'±1.96SD')
    ax9.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--')
    ax9.set_xlabel('平均値 (真値+予測値)/2')
    ax9.set_ylabel('差分 (真値-予測値)')
    ax9.set_title('Bland-Altmanプロット')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. 相関の時間変化
    ax10 = plt.subplot(3, 4, 10)
    window_size = 30
    rolling_corrs = []
    for i in range(len(targets) - window_size):
        window_corr, _ = pearsonr(targets[i:i+window_size], 
                                  predictions[i:i+window_size])
        rolling_corrs.append(window_corr)
    ax10.plot(rolling_corrs, alpha=0.7)
    ax10.set_xlabel('時間窓の開始位置')
    ax10.set_ylabel('相関係数')
    ax10.set_title(f'相関係数の時間変化（窓サイズ: {window_size}秒）')
    ax10.axhline(y=eval_results['corr'], color='r', linestyle='--', 
                 label=f'全体相関: {eval_results["corr"]:.3f}')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Q-Qプロット
    ax11 = plt.subplot(3, 4, 11)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax11)
    ax11.set_title('Q-Qプロット（残差の正規性）')
    ax11.grid(True, alpha=0.3)
    
    # 12. メトリクスサマリー
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    summary_text = f"""
    評価メトリクス
    
    MAE:     {eval_results['mae']:.4f}
    RMSE:    {eval_results['rmse']:.4f}
    相関係数: {eval_results['corr']:.4f}
    R²:      {eval_results['r2']:.4f}
    p値:     {eval_results['p_value']:.2e}
    
    モデル設定
    時系列: {config.temporal_type}
    時間窓: ±{config.temporal_window}
    損失: {config.loss_type}
    """
    ax12.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace')
    
    plt.suptitle(f'Temporal PhysNet2DCNN - CO推定結果（{config.temporal_type.upper()}）', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'results_temporal_{config.temporal_type}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    print("\n" + "="*60)
    print(" Temporal PhysNet2DCNN - CO推定モデル")
    print("="*60)
    print(f"時系列処理: {config.temporal_type.upper()}")
    print(f"時間窓: ±{config.temporal_window}フレーム")
    print(f"チャンネル: {config.use_channel}")
    print(f"損失関数: {config.loss_type}")
    print(f"デバイス: {config.device}")
    
    try:
        # データ読み込み
        rgb_data, co_data, subject_ids = load_all_data(config)
        
        if config.analysis_type == "individual":
            # 個人内解析
            train_data, val_data, test_data = split_data_individual(rgb_data, co_data, config)
            
            # 時系列データセット
            train_dataset = TemporalCODataset(train_data[0], train_data[1], 
                                             config.temporal_window, config.use_channel)
            val_dataset = TemporalCODataset(val_data[0], val_data[1], 
                                           config.temporal_window, config.use_channel)
            test_dataset = TemporalCODataset(test_data[0], test_data[1], 
                                            config.temporal_window, config.use_channel)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                                  shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                   shuffle=False, num_workers=0)
            
            # モデル作成
            model = TemporalPhysNet2DCNN(config.input_shape, 
                                        config.temporal_window, 
                                        config.temporal_type)
            print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
            
            # 学習
            model, train_losses, val_losses, train_corrs, val_corrs = train_model(
                model, train_loader, val_loader, config
            )
            
            # 評価
            print("\nテストデータで評価中...")
            eval_results = evaluate_model(model, test_loader, config)
            
            print("\n" + "="*60)
            print(" 最終結果")
            print("="*60)
            print(f"MAE:     {eval_results['mae']:.4f}")
            print(f"RMSE:    {eval_results['rmse']:.4f}")
            print(f"相関係数: {eval_results['corr']:.4f}")
            print(f"R²:      {eval_results['r2']:.4f}")
            print(f"p値:     {eval_results['p_value']:.2e}")
            
            # プロット
            plot_results(eval_results, train_losses, val_losses, 
                       train_corrs, val_corrs, config)
            
        print("\n完了しました。")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
