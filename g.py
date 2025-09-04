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
        self.analysis_type = "individual"  # "individual"（個人内） or "cross"（個人間）
        
        # データ設定
        if self.analysis_type == "individual":
            self.subjects = ["bp001"]
            self.n_folds = 1
        else:
            self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
            self.n_folds = 8
        
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # 使用チャンネル設定
        self.use_channel = 'B'  # 'R', 'G', 'B', 'RGB'
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)
        
        # 学習設定
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7  # MSE/Huber損失の重み
        self.loss_beta = 0.3   # 相関損失の重み
        
        # 学習率スケジューラー設定
        self.scheduler_type = "cosine"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 20  # CosineAnnealingWarmRestartsの初期周期
        self.scheduler_T_mult = 1  # 周期の倍率
        
        # データ分割設定
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
# PhysNet2DCNNモデル
# ================================
class PhysNet2DCNN(nn.Module):
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
# プロット（個人内）
# ================================
def plot_individual_results(eval_results, train_losses, val_losses, 
                           train_correlations, val_correlations, config):
    fig = plt.figure(figsize=(18, 12))
    
    predictions = eval_results['predictions']
    targets = eval_results['targets']
    
    # 1. 損失曲線
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('損失の学習曲線')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 相関曲線
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(train_correlations, label='Train Corr', alpha=0.8, color='green')
    ax2.plot(val_correlations, label='Val Corr', alpha=0.8, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Correlation')
    ax2.set_title('相関係数の推移')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 予測vs真値
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax3.set_xlabel('真値 (CO)')
    ax3.set_ylabel('予測値 (CO)')
    ax3.set_title(f"MAE: {eval_results['mae']:.3f}, Corr: {eval_results['corr']:.3f}")
    ax3.grid(True, alpha=0.3)
    
    # 4. 残差プロット
    ax4 = plt.subplot(3, 3, 4)
    residuals = targets - predictions
    ax4.scatter(predictions, residuals, alpha=0.5, s=20)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('予測値')
    ax4.set_ylabel('残差')
    ax4.set_title(f'平均: {residuals.mean():.3f}, STD: {residuals.std():.3f}')
    ax4.grid(True, alpha=0.3)
    
    # 5. 時系列比較
    ax5 = plt.subplot(3, 3, 5)
    sample_range = min(120, len(targets))
    ax5.plot(range(sample_range), targets[:sample_range], 'b-', label='真値', alpha=0.7)
    ax5.plot(range(sample_range), predictions[:sample_range], 'r-', label='予測', alpha=0.7)
    ax5.set_xlabel('時間 (秒)')
    ax5.set_ylabel('CO値')
    ax5.set_title('時系列比較（120秒）')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 誤差分布
    ax6 = plt.subplot(3, 3, 6)
    errors = np.abs(targets - predictions)
    ax6.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax6.axvline(x=eval_results['mae'], color='r', linestyle='--', lw=2)
    ax6.set_xlabel('絶対誤差')
    ax6.set_ylabel('頻度')
    ax6.set_title('絶対誤差分布')
    ax6.grid(True, alpha=0.3)
    
    # 7. タスク別性能
    ax7 = plt.subplot(3, 3, 7)
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
    ax7.bar(x - width/2, task_maes, width, label='MAE', alpha=0.7)
    ax7_twin = ax7.twinx()
    ax7_twin.bar(x + width/2, task_corrs, width, label='Corr', color='orange', alpha=0.7)
    ax7.set_xlabel('タスク')
    ax7.set_ylabel('MAE')
    ax7_twin.set_ylabel('相関係数')
    ax7.set_title('タスク別性能')
    ax7.set_xticks(x)
    ax7.set_xticklabels(config.tasks)
    ax7.grid(True, alpha=0.3)
    
    # 8. Bland-Altman
    ax8 = plt.subplot(3, 3, 8)
    mean_vals = (targets + predictions) / 2
    diff_vals = targets - predictions
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    
    ax8.scatter(mean_vals, diff_vals, alpha=0.5, s=20)
    ax8.axhline(y=mean_diff, color='red', linestyle='-', label=f'平均: {mean_diff:.3f}')
    ax8.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--')
    ax8.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--')
    ax8.set_xlabel('平均値')
    ax8.set_ylabel('差分')
    ax8.set_title('Bland-Altmanプロット')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. メトリクスサマリー
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
    評価メトリクス
    
    MAE:     {eval_results['mae']:.4f}
    RMSE:    {eval_results['rmse']:.4f}
    相関係数: {eval_results['corr']:.4f}
    R²:      {eval_results['r2']:.4f}
    p値:     {eval_results['p_value']:.2e}
    
    設定
    損失関数: {config.loss_type}
    スケジューラー: {config.scheduler_type}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center')
    
    plt.suptitle('PhysNet2DCNN - CO推定結果（改良版）', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'results_individual.png', dpi=150, bbox_inches='tight')
    plt.show()

# ================================
# 交差検証（個人間）
# ================================
def cross_validation(rgb_data, co_data, subject_ids, config):
    print("\n" + "="*60)
    print(f"{config.n_folds}分割交差検証開始")
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
        model = PhysNet2DCNN(config.input_shape)
        model, _, _, _, _ = train_model(model, train_loader, val_loader, config, fold)
        
        # 評価
        eval_results = evaluate_model(model, test_loader, config)
        print(f"  結果 - MAE: {eval_results['mae']:.4f}, Corr: {eval_results['corr']:.4f}")
        
        results.append({
            'fold': fold,
            'mae': eval_results['mae'],
            'corr': eval_results['corr'],
            'predictions': eval_results['predictions'],
            'targets': eval_results['targets']
        })
    
    return results

# ================================
# プロット（個人間）
# ================================
def plot_cross_results(results, config):
    fig = plt.figure(figsize=(20, 12))
    
    for i in range(min(8, config.n_folds)):
        ax = plt.subplot(3, 3, i+1)
        r = results[i]
        ax.scatter(r['targets'], r['predictions'], alpha=0.5, s=10)
        min_val = min(r['targets'].min(), r['predictions'].min())
        max_val = max(r['targets'].max(), r['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel('真値')
        ax.set_ylabel('予測値')
        ax.set_title(f"Fold {i+1}\nMAE: {r['mae']:.3f}, Corr: {r['corr']:.3f}")
        ax.grid(True, alpha=0.3)
    
    # 全体結果
    ax = plt.subplot(3, 3, 9)
    all_targets = np.concatenate([r['targets'] for r in results])
    all_predictions = np.concatenate([r['predictions'] for r in results])
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_corr, _ = pearsonr(all_targets, all_predictions)
    
    ax.scatter(all_targets, all_predictions, alpha=0.5, s=10)
    min_val = min(all_targets.min(), all_predictions.min())
    max_val = max(all_targets.max(), all_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel('真値')
    ax.set_ylabel('予測値')
    ax.set_title(f'全体結果\nMAE: {overall_mae:.3f}, Corr: {overall_corr:.3f}')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PhysNet2DCNN - 交差検証結果（{config.n_folds}分割）', fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'results_cross.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return overall_mae, overall_corr

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - CO推定モデル（改良版）")
    print("="*60)
    print(f"解析: {'個人内' if config.analysis_type == 'individual' else '個人間'}")
    print(f"チャンネル: {config.use_channel}")
    print(f"損失関数: {config.loss_type}")
    print(f"スケジューラー: {config.scheduler_type}")
    print(f"デバイス: {config.device}")
    
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
            
            # モデル学習
            model = PhysNet2DCNN(config.input_shape)
            print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
            
            model, train_losses, val_losses, train_corrs, val_corrs = train_model(
                model, train_loader, val_loader, config
            )
            
            # 評価
            print("\nテストデータで評価中...")
            eval_results = evaluate_model(model, test_loader, config)
            
            print("\n最終結果:")
            print(f"  MAE: {eval_results['mae']:.4f}")
            print(f"  RMSE: {eval_results['rmse']:.4f}")
            print(f"  相関係数: {eval_results['corr']:.4f}")
            print(f"  R²: {eval_results['r2']:.4f}")
            
            # プロット
            plot_individual_results(eval_results, train_losses, val_losses, 
                                   train_corrs, val_corrs, config)
            
        else:
            # 個人間解析
            results = cross_validation(rgb_data, co_data, subject_ids, config)
            overall_mae, overall_corr = plot_cross_results(results, config)
            
            print("\n交差検証結果:")
            for r in results:
                print(f"  Fold {r['fold']+1}: MAE={r['mae']:.4f}, Corr={r['corr']:.4f}")
            print(f"\n全体: MAE={overall_mae:.4f}, Corr={overall_corr:.4f}")
        
        print("\n完了しました。")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
