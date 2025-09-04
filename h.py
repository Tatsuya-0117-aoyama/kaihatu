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
        
        # データ拡張設定
        self.use_augmentation = True  # データ拡張を使用するか
        self.aug_noise_level = 0.01  # ノイズレベル
        self.aug_scale_range = (0.95, 1.05)  # スケーリング範囲
        self.aug_brightness_range = (-0.05, 0.05)  # 明度変化範囲
        self.aug_mixup_alpha = 0.2  # Mixupの強度
        self.aug_cutmix_prob = 0.3  # CutMixの確率
        
        # 使用チャンネル
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
# データ拡張関数
# ================================
class DataAugmentation:
    """生理信号用のデータ拡張"""
    
    @staticmethod
    def add_noise(rgb_data, noise_level=0.01):
        """ガウシアンノイズを追加"""
        noise = np.random.normal(0, noise_level, rgb_data.shape)
        augmented = rgb_data + noise
        return np.clip(augmented, 0, 1) if rgb_data.max() <= 1 else augmented
    
    @staticmethod
    def scale_amplitude(rgb_data, co_data, scale_range=(0.95, 1.05)):
        """振幅スケーリング（個人差をシミュレート）"""
        scale_rgb = np.random.uniform(scale_range[0], scale_range[1])
        scale_co = np.random.uniform(scale_range[0], scale_range[1])
        return rgb_data * scale_rgb, co_data * scale_co
    
    @staticmethod
    def brightness_adjust(rgb_data, brightness_range=(-0.05, 0.05)):
        """明度調整（照明条件の変化をシミュレート）"""
        brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        augmented = rgb_data + brightness
        return np.clip(augmented, 0, 1) if rgb_data.max() <= 1 else augmented
    
    @staticmethod
    def channel_shuffle(rgb_data):
        """チャンネルシャッフル（RGBの順序をランダムに）"""
        if rgb_data.shape[-1] == 3:
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            return rgb_data[:, :, :, channels]
        return rgb_data
    
    @staticmethod
    def mixup(rgb1, co1, rgb2, co2, alpha=0.2):
        """Mixup augmentation"""
        lam = np.random.beta(alpha, alpha)
        mixed_rgb = lam * rgb1 + (1 - lam) * rgb2
        mixed_co = lam * co1 + (1 - lam) * co2
        return mixed_rgb, mixed_co
    
    @staticmethod
    def cutmix(rgb1, co1, rgb2, co2, prob=0.5):
        """CutMix augmentation"""
        if np.random.rand() > prob:
            return rgb1, co1
        
        H, W = rgb1.shape[0], rgb1.shape[1]
        lam = np.random.beta(1.0, 1.0)
        
        cut_rat = np.sqrt(1. - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_rgb = rgb1.copy()
        mixed_rgb[bby1:bby2, bbx1:bbx2] = rgb2[bby1:bby2, bbx1:bbx2]
        
        # CO値も面積比に応じて混合
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        mixed_co = lam * co1 + (1 - lam) * co2
        
        return mixed_rgb, mixed_co
    
    @staticmethod
    def random_erasing(rgb_data, prob=0.3, area_range=(0.02, 0.1)):
        """Random Erasing（一部領域をマスク）"""
        if np.random.rand() > prob:
            return rgb_data
        
        H, W = rgb_data.shape[0], rgb_data.shape[1]
        area = H * W
        
        target_area = np.random.uniform(area_range[0], area_range[1]) * area
        aspect_ratio = np.random.uniform(0.3, 3.3)
        
        h = int(np.sqrt(target_area * aspect_ratio))
        w = int(np.sqrt(target_area / aspect_ratio))
        
        if h < H and w < W:
            y = np.random.randint(0, H - h)
            x = np.random.randint(0, W - w)
            
            augmented = rgb_data.copy()
            # ランダム値で埋める
            augmented[y:y+h, x:x+w] = np.random.uniform(0, 1, (h, w, rgb_data.shape[2]))
            return augmented
        
        return rgb_data

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
# データセット（データ拡張対応）
# ================================
class AugmentedCODataset(Dataset):
    def __init__(self, rgb_data, co_data, use_channel='B', 
                 is_training=False, config=None):
        self.is_training = is_training
        self.config = config
        self.augmentor = DataAugmentation()
        
        # チャンネル選択
        if use_channel == 'R':
            self.rgb_data = rgb_data[:, :, :, 0:1]
        elif use_channel == 'G':
            self.rgb_data = rgb_data[:, :, :, 1:2]
        elif use_channel == 'B':
            self.rgb_data = rgb_data[:, :, :, 2:3]
        else:
            self.rgb_data = rgb_data
        
        self.co_data = co_data
        self.indices = list(range(len(self.rgb_data)))
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        rgb = self.rgb_data[idx].copy()
        co = self.co_data[idx].copy()
        
        # 訓練時のみデータ拡張を適用
        if self.is_training and self.config and self.config.use_augmentation:
            # ランダムにデータ拡張を適用
            if np.random.rand() < 0.5:
                rgb = self.augmentor.add_noise(rgb, self.config.aug_noise_level)
            
            if np.random.rand() < 0.3:
                rgb, co = self.augmentor.scale_amplitude(rgb, co, self.config.aug_scale_range)
            
            if np.random.rand() < 0.3:
                rgb = self.augmentor.brightness_adjust(rgb, self.config.aug_brightness_range)
            
            if np.random.rand() < 0.2:
                rgb = self.augmentor.random_erasing(rgb, prob=0.5)
            
            # Mixup or CutMix（別のサンプルとの混合）
            if np.random.rand() < 0.3 and len(self.indices) > 1:
                # ランダムに別のサンプルを選択
                idx2 = np.random.choice([i for i in self.indices if i != idx])
                rgb2 = self.rgb_data[idx2]
                co2 = self.co_data[idx2]
                
                if np.random.rand() < 0.5:
                    rgb, co = self.augmentor.mixup(rgb, co, rgb2, co2, 
                                                  self.config.aug_mixup_alpha)
                else:
                    rgb, co = self.augmentor.cutmix(rgb, co, rgb2, co2, 
                                                   self.config.aug_cutmix_prob)
        
        # Tensorに変換
        rgb_tensor = torch.FloatTensor(rgb).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        co_tensor = torch.FloatTensor([co])
        
        return rgb_tensor, co_tensor.squeeze()

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
        
        # PhysNet blocks
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
        
        # Global pooling and output
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
    # 正規化（0-1の範囲に）
    if rgb_data.max() > 1:
        rgb_data = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())
    
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
    print(f"  データ拡張: {'有効' if config.use_augmentation else '無効'}")
    
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
    
    print(f"  データ拡張: {'有効' if config.use_augmentation else '無効'}")
    
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
    ax5.plot(range(sample_range), targets[:sample_range], 'b-', label='真値', alpha=0.7)
    ax5.plot(range(sample_range), predictions[:sample_range], 'r-', label='予測', alpha=0.7)
    ax5.set_xlabel('時間 (秒)')
    ax5.set_ylabel('CO値')
    ax5.set_title('時系列比較（180秒）')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 時系列比較（短期）
    ax6 = plt.subplot(3, 4, 6)
    start = 60
    end = 90
    ax6.plot(range(start, end), targets[start:end], 'b-', marker='o', markersize=3, label='真値')
    ax6.plot(range(start, end), predictions[start:end], 'r-', marker='^', markersize=3, label='予測')
    ax6.set_xlabel('時間 (秒)')
    ax6.set_ylabel('CO値')
    ax6.set_title('時系列詳細（60-90秒）')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 誤差分布
    ax7 = plt.subplot(3, 4, 7)
    errors = np.abs(targets - predictions)
    ax7.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax7.axvline(x=eval_results['mae'], color='r', linestyle='--', lw=2)
    ax7.set_xlabel('絶対誤差')
    ax7.set_ylabel('頻度')
    ax7.set_title(f"誤差分布 (MAE: {eval_results['mae']:.3f})")
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
    ax8.bar(x - width/2, task_maes, width, label='MAE', alpha=0.7)
    ax8_twin = ax8.twinx()
    ax8_twin.bar(x + width/2, task_corrs, width, label='Corr', color='orange', alpha=0.7)
    ax8.set_xlabel('タスク')
    ax8.set_ylabel('MAE')
    ax8_twin.set_ylabel('相関係数')
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
    ax9.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--')
    ax9.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--')
    ax9.set_xlabel('平均値')
    ax9.set_ylabel('差分')
    ax9.set_title('Bland-Altmanプロット')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. 相対誤差分布
    ax10 = plt.subplot(3, 4, 10)
    relative_errors = np.abs((targets - predictions) / (targets + 1e-8)) * 100
    ax10.hist(relative_errors, bins=30, edgecolor='black', alpha=0.7)
    ax10.axvline(x=np.mean(relative_errors), color='r', linestyle='--', lw=2,
                 label=f'平均: {np.mean(relative_errors):.1f}%')
    ax10.set_xlabel('相対誤差 (%)')
    ax10.set_ylabel('頻度')
    ax10.set_title('相対誤差分布')
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
    
    データ拡張設定
    使用: {'有効' if config.use_augmentation else '無効'}
    ノイズ: {config.aug_noise_level}
    スケール: {config.aug_scale_range}
    Mixup α: {config.aug_mixup_alpha}
    """
    ax12.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace')
    
    aug_status = "with_aug" if config.use_augmentation else "no_aug"
    plt.suptitle(f'PhysNet2DCNN - CO推定結果（データ拡張: {"有効" if config.use_augmentation else "無効"}）', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'results_{aug_status}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - CO推定モデル（データ拡張版）")
    print("="*60)
    print(f"解析: {'個人内' if config.analysis_type == 'individual' else '個人間'}")
    print(f"チャンネル: {config.use_channel}")
    print(f"データ拡張: {'有効' if config.use_augmentation else '無効'}")
    if config.use_augmentation:
        print(f"  - ノイズレベル: {config.aug_noise_level}")
        print(f"  - スケール範囲: {config.aug_scale_range}")
        print(f"  - Mixup α: {config.aug_mixup_alpha}")
        print(f"  - CutMix確率: {config.aug_cutmix_prob}")
    print(f"損失関数: {config.loss_type}")
    print(f"デバイス: {config.device}")
    
    try:
        # データ読み込み
        rgb_data, co_data, subject_ids = load_all_data(config)
        
        if config.analysis_type == "individual":
            # 個人内解析
            train_data, val_data, test_data = split_data_individual(rgb_data, co_data, config)
            
            # データセット（訓練時のみデータ拡張を適用）
            train_dataset = AugmentedCODataset(
                train_data[0], train_data[1], 
                config.use_channel, is_training=True, config=config
            )
            val_dataset = AugmentedCODataset(
                val_data[0], val_data[1], 
                config.use_channel, is_training=False, config=config
            )
            test_dataset = AugmentedCODataset(
                test_data[0], test_data[1], 
                config.use_channel, is_training=False, config=config
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                                  shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                   shuffle=False, num_workers=0)
            
            # モデル作成
            model = PhysNet2DCNN(config.input_shape)
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
            
            # データ拡張なしでも評価（比較用）
            if config.use_augmentation:
                print("\n" + "="*60)
                print(" データ拡張なしでの評価（比較用）")
                print("="*60)
                config_no_aug = Config()
                config_no_aug.use_augmentation = False
                test_dataset_no_aug = AugmentedCODataset(
                    test_data[0], test_data[1], 
                    config.use_channel, is_training=False, config=config_no_aug
                )
                test_loader_no_aug = DataLoader(test_dataset_no_aug, 
                                               batch_size=config.batch_size, 
                                               shuffle=False, num_workers=0)
                eval_results_no_aug = evaluate_model(model, test_loader_no_aug, config_no_aug)
                print(f"MAE（拡張なし）: {eval_results_no_aug['mae']:.4f}")
                print(f"相関係数（拡張なし）: {eval_results_no_aug['corr']:.4f}")
            
        print("\n完了しました。")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
