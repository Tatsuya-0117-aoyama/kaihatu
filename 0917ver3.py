import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # AMP用
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.ndimage import rotate as scipy_rotate
from scipy.interpolate import interp1d
import cv2
import os
from pathlib import Path
from datetime import datetime
import warnings
import random
import pandas as pd
warnings.filterwarnings('ignore')

# フォント設定（メイリオ）
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 10

# ================================
# 完全な再現性のための乱数シード設定関数
# ================================
def set_all_seeds(seed=42):
    """完全な再現性のための乱数シード設定"""
    # Python標準の乱数
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNNの決定的動作（高速化版では一部変更）
    torch.backends.cudnn.deterministic = False  # 高速化のためFalseに変更
    torch.backends.cudnn.benchmark = True  # 高速化のためTrueに変更
    
    # データローダーのワーカー初期化
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker

# ================================
# 設定クラス（複数指標対応＋モード切り替え）
# ================================
class Config:
    def __init__(self):
        # ================================
        # モード選択（個人内 or 被験者間）
        # ================================
        self.model_mode = 'cross_subject'  # 'within_subject' or 'cross_subject'
        
        # ================================
        # 信号正規化設定
        # ================================
        self.signal_normalization = 'z-score'  # 'none', 'min-max', 'z-score', 'robust'
        self.signal_norm_per_subject = True  # True: 被験者ごと, False: 全体で正規化
        self.signal_norm_per_task = False  # True: タスクごと, False: 全タスクまとめて
        
        # 正規化パラメータ保存用（推論時に必要）
        self.norm_params = {}
        
        # ================================
        # 血行動態信号タイプ設定（複数選択可能）
        # ================================
        # 利用可能な指標
        self.available_signals = ['CO', 'SV', 'HR_CO_SV', 'Cwk', 'Rp', 'Zao', 'I0', 'LIET', 'reDIA', 'reSYS']
        
        # 実行する指標のリスト（Cross-Subjectモードで複数選択可能）
        if self.model_mode == 'cross_subject':
            self.signal_types = ['CO', 'SV', 'HR_CO_SV']  # 複数指標を選択
        else:
            # Within-Subjectモードでは単一指標のみ
            self.signal_types = ['CO']  # 単一指標
        
        # 現在処理中の指標（内部的に使用）
        self.current_signal_type = self.signal_types[0] if self.signal_types else 'CO'
        self.current_signal_prefix = f"{self.current_signal_type}_s2"
        
        # Within-Subject用の設定（後方互換性のため）
        self.signal_type = self.current_signal_type
        self.signal_prefix = self.current_signal_prefix
        
        # パス設定
        self.rgb_base_path = r"C:\Users\EyeBelow"
        self.signal_base_path = r"C:\Users\Data_signals_bp"
        self.base_save_path = r"D:\EPSCAN\001"
        
        # 日付時間フォルダを作成
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.base_save_path, self.timestamp)
        
        # ================================
        # 高速化設定
        # ================================
        self.use_amp = True  # Automatic Mixed Precision使用
        self.use_compile = False  # torch.compile使用（PyTorch 2.0+）
        self.num_workers = 0  # DataLoaderのワーカー数
        self.pin_memory = True  # GPU転送の高速化
        self.persistent_workers = True  # ワーカーの再利用
        self.prefetch_factor = None  # 先読みバッチ数
        
        # ================================
        # LAB変換データ使用設定
        # ================================
        self.use_lab = True  # LABデータを使用するか（True: RGB+LAB, False: RGBのみ）
        self.lab_filename = "_downsampled_1Hzver2.npy"  # LABデータのファイル名
        
        # ================================
        # Cross-Subject モード設定
        # ================================
        if self.model_mode == 'cross_subject':
            self.n_folds = 5  # 5分割交差検証
            self.ensure_subject_in_each_fold = True  # 各foldに各被験者のデータを含める
            
        # ================================
        # Within-Subject モード設定
        # ================================
        elif self.model_mode == 'within_subject':
            self.train_val_split_ratio = 0.9  # 訓練データの割合（90%）
            self.validation_split_strategy = 'stratified'  # 'sequential' または 'stratified'
            self.n_strata = 5  # 層の数
            self.stratification_method = 'quantile'  # 'equal_range' or 'quantile'
        
        # 被験者設定（bp001～bp032）
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60  # 各タスクの長さ（秒）
        
        # タスクごとの色設定（Within-Subject用）
        self.task_colors = {
            "t1-1": "#FF6B6B",  # 赤系
            "t2": "#4ECDC4",    # 青緑系
            "t1-2": "#45B7D1",  # 青系
            "t4": "#96CEB4",    # 緑系
            "t1-3": "#FECA57",  # 黄系
            "t5": "#DDA0DD"     # 紫系
        }
        
        # Fold用の色設定（Cross-Subject用）
        self.fold_colors = {
            1: "#FF6B6B",  # 赤系
            2: "#4ECDC4",  # 青緑系
            3: "#45B7D1",  # 青系
            4: "#96CEB4",  # 緑系
            5: "#FECA57"   # 黄系
        }
        
        # 被験者ごとの色設定（32人用）- Cross-Subject用
        self.subject_colors = {}
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.subjects)))
        for i, subject in enumerate(self.subjects):
            self.subject_colors[subject] = colors[i]
        
        # ================================
        # モデル設定
        # ================================
        # モデルタイプ選択（"3d", "2d"）※軽量化版を削除
        self.model_type = "2d"
        
        # 使用チャンネル設定（LAB使用時は自動設定）
        if self.use_lab:
            self.use_channel = 'RGB+LAB'  # LAB使用時はRGB+LAB（6チャンネル）
        else:
            self.use_channel = 'RGB'
        
        # チャンネル数の自動計算
        channel_map = {
            'R': 1, 'G': 1, 'B': 1,
            'RG': 2, 'GB': 2, 'RB': 2,
            'RGB': 3, 'LAB': 3, 'RGB+LAB': 6
        }
        self.num_channels = channel_map.get(self.use_channel, 3)
        
        # データ形状設定（モードによって変更）
        if self.model_mode == 'cross_subject':
            # Cross-Subject: 1タスク60フレーム
            self.time_frames = 60
        else:
            # Within-Subject: 360フレーム
            self.time_frames = 360
            
        self.height = 36  # 画像の高さ
        self.width = 36   # 画像の幅
        self.input_shape = (self.time_frames, self.height, self.width, self.num_channels)
        
        # ================================
        # データ拡張設定
        # ================================
        self.use_augmentation = False  # データ拡張を使用するか
        
        # データ拡張のパラメータ
        if self.use_augmentation:
            self.crop_enabled = True
            self.crop_size_ratio = 0.9
            self.rotation_enabled = True
            self.rotation_range = 5
            self.time_stretch_enabled = True
            self.time_stretch_range = (0.9, 1.1)
            self.brightness_contrast_enabled = True
            self.brightness_range = 0.2
            self.contrast_range = 0.2
            self.aug_probability = 0.5
        else:
            self.crop_enabled = False
            self.rotation_enabled = False
            self.time_stretch_enabled = False
            self.brightness_contrast_enabled = False
        
        # ================================
        # 学習設定（モデルタイプに応じて自動調整）
        # ================================
        if self.model_type == "3d":
            self.batch_size = 8
            self.epochs = 150
            self.learning_rate = 0.001
        elif self.model_type == "2d":
            self.batch_size = 32 if self.use_amp else 16
            self.epochs = 150
            self.learning_rate = 0.001
        else:
            self.batch_size = 16
            self.epochs = 150
            self.learning_rate = 0.001
        
        self.weight_decay = 1e-4
        self.patience = 30
        self.gradient_clip_val = 0.5
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ================================
        # Warmup設定
        # ================================
        self.warmup_epochs = 5
        self.warmup_lr_factor = 0.1
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7  # MSE/Huber損失の重み
        self.loss_beta = 0.3   # 相関損失の重み
        
        # 学習率スケジューラー設定
        self.scheduler_type = "cosine"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 30
        self.scheduler_T_mult = 2
        
        # Early Stopping改善設定
        self.patience_improvement_threshold = 0.995
        self.min_delta = 0.0001
        
        # 表示設定
        self.verbose = True
        self.random_seed = 42
    
    def set_current_signal(self, signal_type):
        """現在処理する信号タイプを設定"""
        if signal_type not in self.available_signals:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        self.current_signal_type = signal_type
        self.current_signal_prefix = f"{signal_type}_s2"
        # Within-Subject互換性のため
        self.signal_type = signal_type
        self.signal_prefix = f"{signal_type}_s2"


# ================================
# データ拡張関数
# ================================
class DataAugmentation:
    """データ拡張を管理するクラス"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)
    
    def random_crop(self, data):
        """ランダムクロップ"""
        if np.random.random() > self.config.aug_probability or not self.config.crop_enabled:
            return data
        
        if data.ndim == 4:  # (T, H, W, C)
            t, h, w, c = data.shape
            new_h = int(h * self.config.crop_size_ratio)
            new_w = int(w * self.config.crop_size_ratio)
            
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            
            cropped = data[:, top:top+new_h, left:left+new_w, :]
            
            resized = np.zeros_like(data)
            for i in range(t):
                for j in range(c):
                    resized[i, :, :, j] = cv2.resize(cropped[i, :, :, j], (w, h))
            
            return resized
        
        elif data.ndim == 3:  # (H, W, C)
            h, w, c = data.shape
            new_h = int(h * self.config.crop_size_ratio)
            new_w = int(w * self.config.crop_size_ratio)
            
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            
            cropped = data[top:top+new_h, left:left+new_w, :]
            resized = cv2.resize(cropped, (w, h))
            
            return resized
        
        return data
    
    def random_rotation(self, data):
        """ランダム回転"""
        if np.random.random() > self.config.aug_probability or not self.config.rotation_enabled:
            return data
        
        angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
        
        if data.ndim == 4:  # (T, H, W, C)
            rotated = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[3]):
                    rotated[i, :, :, j] = scipy_rotate(data[i, :, :, j], angle, 
                                                       reshape=False, mode='reflect')
            return rotated
        
        elif data.ndim == 3:  # (H, W, C)
            rotated = np.zeros_like(data)
            for j in range(data.shape[2]):
                rotated[:, :, j] = scipy_rotate(data[:, :, j], angle, 
                                               reshape=False, mode='reflect')
            return rotated
        
        return data
    
    def time_stretch_fast(self, rgb_np, factor):
        """高速な時間軸ストレッチング（PyTorch F.interpolate使用）"""
        x = torch.from_numpy(rgb_np).permute(3,0,1,2).unsqueeze(0).float()  # (1,C,T,H,W)
        T = x.shape[2]
        T2 = max(1, int(T*factor))
        
        with torch.no_grad():
            x = F.interpolate(x, size=(T2, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
            x = F.interpolate(x, size=(T, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        
        out = x.squeeze(0).permute(1,2,3,0).cpu().numpy()  # (T,H,W,C)
        return out
    
    def time_stretch(self, rgb_data, signal_data=None):
        """時間軸ストレッチング（高速版）"""
        if np.random.random() > self.config.aug_probability or not self.config.time_stretch_enabled:
            return rgb_data, signal_data
        
        if rgb_data.ndim != 4:
            return rgb_data, signal_data
        
        stretch_factor = np.random.uniform(*self.config.time_stretch_range)
        
        rgb_stretched = self.time_stretch_fast(rgb_data, stretch_factor)
        
        if signal_data is not None and signal_data.ndim == 1:
            t_original = len(signal_data)
            t_stretched = int(t_original * stretch_factor)
            
            f_signal = interp1d(np.arange(t_original), signal_data, 
                              kind='linear', fill_value='extrapolate')
            signal_stretched = f_signal(np.linspace(0, t_original-1, t_stretched))
            
            f_signal_back = interp1d(np.arange(len(signal_stretched)), signal_stretched,
                                    kind='linear', fill_value='extrapolate')
            signal_resampled = f_signal_back(np.linspace(0, len(signal_stretched)-1, t_original))
            
            signal_resampled = signal_resampled * stretch_factor
            
            return rgb_stretched, signal_resampled
        
        return rgb_stretched, signal_data
    
    def brightness_contrast_adjust(self, data):
        """明度・コントラスト調整"""
        if np.random.random() > self.config.aug_probability or not self.config.brightness_contrast_enabled:
            return data
        
        brightness_delta = np.random.uniform(-self.config.brightness_range, 
                                            self.config.brightness_range)
        contrast_factor = np.random.uniform(1 - self.config.contrast_range, 
                                           1 + self.config.contrast_range)
        
        data_adjusted = data.copy()
        
        mean = np.mean(data_adjusted, axis=tuple(range(data_adjusted.ndim-1)), keepdims=True)
        data_adjusted = (data_adjusted - mean) * contrast_factor + mean
        data_adjusted = data_adjusted + brightness_delta
        data_adjusted = np.clip(data_adjusted, 0, 1)
        
        return data_adjusted
    
    def apply_augmentation(self, rgb_data, signal_data=None, is_training=True):
        """すべてのデータ拡張を適用"""
        if not is_training or not self.config.use_augmentation:
            return rgb_data, signal_data
        
        rgb_data = self.random_crop(rgb_data)
        rgb_data = self.random_rotation(rgb_data)
        
        if rgb_data.ndim == 4 and self.config.time_stretch_enabled:
            rgb_data, signal_data = self.time_stretch(rgb_data, signal_data)
        
        rgb_data = self.brightness_contrast_adjust(rgb_data)
        
        return rgb_data, signal_data

# ================================
# カスタム損失関数（高速化版 - ベクトル化）
# ================================
class CombinedLoss(nn.Module):
    """MSE損失と相関損失を組み合わせた複合損失関数（ベクトル化版）"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        if pred.dim() == 2 and pred.size(1) > 1:  # (B, T)形状の場合
            pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
            target_centered = target - torch.mean(target, dim=1, keepdim=True)
            
            numerator = torch.sum(pred_centered * target_centered, dim=1)
            denominator = torch.sqrt(
                torch.sum(pred_centered**2, dim=1) * 
                torch.sum(target_centered**2, dim=1) + 1e-8
            )
            
            correlation = numerator / denominator
            corr_loss = torch.mean(1 - correlation)
        else:
            pred_mean = pred - pred.mean()
            target_mean = target - target.mean()
            
            numerator = torch.sum(pred_mean * target_mean)
            denominator = torch.sqrt(
                torch.sum(pred_mean ** 2) * 
                torch.sum(target_mean ** 2) + 1e-8
            )
            correlation = numerator / denominator
            corr_loss = 1 - correlation
        
        total_loss = self.alpha * mse_loss + self.beta * corr_loss
        
        return total_loss, mse_loss, corr_loss

class HuberCorrelationLoss(nn.Module):
    """Huber損失と相関損失の組み合わせ（外れ値にロバスト、ベクトル化版）"""
    def __init__(self, delta=1.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(self, pred, target):
        huber_loss = self.huber(pred, target)
        
        if pred.dim() == 2 and pred.size(1) > 1:  # (B, T)形状の場合
            pred_centered = pred - torch.mean(pred, dim=1, keepdim=True)
            target_centered = target - torch.mean(target, dim=1, keepdim=True)
            
            numerator = torch.sum(pred_centered * target_centered, dim=1)
            denominator = torch.sqrt(
                torch.sum(pred_centered**2, dim=1) * 
                torch.sum(target_centered**2, dim=1) + 1e-8
            )
            
            correlation = numerator / denominator
            corr_loss = torch.mean(1 - correlation)
        else:
            pred_mean = pred - pred.mean()
            target_mean = target - target.mean()
            
            numerator = torch.sum(pred_mean * target_mean)
            denominator = torch.sqrt(
                torch.sum(pred_mean ** 2) * 
                torch.sum(target_mean ** 2) + 1e-8
            )
            correlation = numerator / denominator
            corr_loss = 1 - correlation
        
        total_loss = self.alpha * huber_loss + self.beta * corr_loss
        
        return total_loss, huber_loss, corr_loss

# ================================
# チャンネル選択ユーティリティ
# ================================
def select_channels(data, use_channel):
    """データから指定されたチャンネルを選択"""
    if use_channel == 'R':
        return data[..., 0:1]
    elif use_channel == 'G':
        return data[..., 1:2]
    elif use_channel == 'B':
        return data[..., 2:3]
    elif use_channel == 'RG':
        return data[..., 0:2]
    elif use_channel == 'GB':
        return data[..., 1:3]
    elif use_channel == 'RB':
        return np.stack([data[..., 0], data[..., 2]], axis=-1)
    elif use_channel == 'RGB':
        return data[..., 0:3] if data.shape[-1] > 3 else data
    elif use_channel == 'LAB':
        if data.shape[-1] == 6:
            return data[..., 3:6]
        else:
            raise ValueError(f"LAB channels not available in data with shape {data.shape}")
    elif use_channel == 'RGB+LAB':
        if data.shape[-1] == 6:
            return data
        else:
            raise ValueError(f"RGB+LAB requires 6 channels, got {data.shape[-1]}")
    else:
        raise ValueError(f"Unknown channel type: {use_channel}")

# ================================
# 信号正規化関数（ここに追加）
# ================================
def normalize_signal(signal_data, method='z-score', return_params=False):
    """信号データの正規化"""
    import warnings
    from scipy import stats
    
    original_shape = signal_data.shape
    signal_flat = signal_data.flatten()
    
    if method == 'none':
        if return_params:
            return signal_data, {}
        return signal_data
    
    elif method == 'min-max':
        # Min-Max正規化 (0-1範囲)
        min_val = np.min(signal_flat)
        max_val = np.max(signal_flat)
        
        if max_val - min_val < 1e-8:
            warnings.warn(f"信号の範囲が非常に小さいです: [{min_val}, {max_val}]")
            normalized = np.zeros_like(signal_flat)
        else:
            normalized = (signal_flat - min_val) / (max_val - min_val)
        
        params = {'method': 'min-max', 'min': min_val, 'max': max_val}
    
    elif method == 'z-score':
        # Z-score正規化 (平均0、標準偏差1)
        mean_val = np.mean(signal_flat)
        std_val = np.std(signal_flat)
        
        if std_val < 1e-8:
            warnings.warn(f"信号の標準偏差が非常に小さいです: {std_val}")
            normalized = signal_flat - mean_val
        else:
            normalized = (signal_flat - mean_val) / std_val
        
        params = {'method': 'z-score', 'mean': mean_val, 'std': std_val}
    
    elif method == 'robust':
        # Robust正規化 (中央値と四分位範囲を使用、外れ値に強い)
        median_val = np.median(signal_flat)
        q75, q25 = np.percentile(signal_flat, [75, 25])
        iqr = q75 - q25
        
        if iqr < 1e-8:
            warnings.warn(f"信号のIQRが非常に小さいです: {iqr}")
            normalized = signal_flat - median_val
        else:
            normalized = (signal_flat - median_val) / iqr
        
        params = {'method': 'robust', 'median': median_val, 'iqr': iqr, 'q25': q25, 'q75': q75}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 元の形状に戻す
    normalized = normalized.reshape(original_shape)
    
    if return_params:
        return normalized, params
    return normalized

def denormalize_signal(normalized_data, params):
    """正規化された信号データを元に戻す"""
    method = params.get('method', 'none')
    
    if method == 'none':
        return normalized_data
    
    elif method == 'min-max':
        min_val = params['min']
        max_val = params['max']
        return normalized_data * (max_val - min_val) + min_val
    
    elif method == 'z-score':
        mean_val = params['mean']
        std_val = params['std']
        return normalized_data * std_val + mean_val
    
    elif method == 'robust':
        median_val = params['median']
        iqr = params['iqr']
        return normalized_data * iqr + median_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# ================================
# 層化サンプリング関数（Within-Subject用）
# ================================
def stratified_sampling_split(task_rgb, task_signal, val_ratio=0.1, n_strata=5, method='quantile'):
    """信号値の分布を保持したまま訓練・検証データを分割"""
    
    n_samples = len(task_signal)
    
    signal_min = task_signal.min()
    signal_max = task_signal.max()
    signal_mean = task_signal.mean()
    signal_std = task_signal.std()
    
    if method == 'equal_range':
        bin_edges = np.linspace(signal_min, signal_max + 1e-10, n_strata + 1)
    elif method == 'quantile':
        quantiles = np.linspace(0, 1, n_strata + 1)
        bin_edges = np.quantile(task_signal, quantiles)
        bin_edges[-1] += 1e-10
    else:
        raise ValueError(f"Unknown stratification method: {method}")
    
    strata_assignment = np.digitize(task_signal, bin_edges) - 1
    
    train_indices = []
    val_indices = []
    strata_info = []
    
    for stratum_id in range(n_strata):
        stratum_mask = (strata_assignment == stratum_id)
        stratum_indices = np.where(stratum_mask)[0]
        
        if len(stratum_indices) == 0:
            continue
        
        stratum_signals = task_signal[stratum_indices]
        stratum_min = stratum_signals.min()
        stratum_max = stratum_signals.max()
        
        n_val_from_stratum = max(1, int(len(stratum_indices) * val_ratio))
        n_train_from_stratum = len(stratum_indices) - n_val_from_stratum
        
        np.random.shuffle(stratum_indices)
        val_from_stratum = stratum_indices[:n_val_from_stratum]
        train_from_stratum = stratum_indices[n_val_from_stratum:]
        
        val_indices.extend(val_from_stratum)
        train_indices.extend(train_from_stratum)
        
        strata_info.append({
            'stratum_id': stratum_id + 1,
            'signal_range': (stratum_min, stratum_max),
            'n_total': len(stratum_indices),
            'n_train': n_train_from_stratum,
            'n_val': n_val_from_stratum
        })
    
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    
    if len(strata_info) > 0:
        print(f"      層化サンプリング詳細 (方法: {method}):")
        for info in strata_info:
            print(f"        層{info['stratum_id']}: "
                  f"信号値[{info['signal_range'][0]:.3f}, {info['signal_range'][1]:.3f}] "
                  f"計{info['n_total']}個 → 訓練{info['n_train']}個, 検証{info['n_val']}個")
    
    val_signals = task_signal[val_indices]
    train_signals = task_signal[train_indices]
    
    print(f"      信号値の分布確認:")
    print(f"        元データ: 平均={signal_mean:.3f}, 標準偏差={signal_std:.3f}")
    print(f"        訓練データ: 平均={train_signals.mean():.3f}, 標準偏差={train_signals.std():.3f}")
    print(f"        検証データ: 平均={val_signals.mean():.3f}, 標準偏差={val_signals.std():.3f}")
    
    return (task_rgb[train_indices], task_signal[train_indices],
            task_rgb[val_indices], task_signal[val_indices])

# ================================
# データセット（データ拡張対応）
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, signal_data, model_type='3d', 
                 use_channel='RGB', config=None, is_training=True):
        """
        rgb_data: (N, T, H, W, C) 形状
        signal_data: (N, T) または (N,) 形状
        """
        self.model_type = model_type
        self.use_channel = use_channel
        self.is_training = is_training
        
        self.augmentation = DataAugmentation(config) if config else None
        
        self.rgb_data_raw = rgb_data
        self.signal_data_raw = signal_data
        
        rgb_data_selected = select_channels(rgb_data, use_channel)
        self.rgb_data = torch.FloatTensor(rgb_data_selected)
        
        # signal_dataが1次元の場合、時間次元に拡張
        if signal_data.ndim == 1:
            signal_data = np.repeat(signal_data[:, np.newaxis], rgb_data.shape[1], axis=1)
        
        self.signal_data = torch.FloatTensor(signal_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        rgb = self.rgb_data_raw[idx]
        signal = self.signal_data_raw[idx] if self.signal_data_raw.ndim > 1 else self.signal_data_raw[idx:idx+1].squeeze()
        
        if self.augmentation and self.is_training:
            rgb, signal = self.augmentation.apply_augmentation(rgb, signal, self.is_training)
        
        rgb = select_channels(rgb, self.use_channel)
        
        rgb_tensor = torch.FloatTensor(rgb)
        signal_tensor = torch.FloatTensor(signal if isinstance(signal, np.ndarray) else [signal])
        if signal_tensor.dim() == 0:
            signal_tensor = signal_tensor.unsqueeze(0)
        if signal_tensor.size(0) == 1 and rgb_tensor.size(0) > 1:
            signal_tensor = signal_tensor.repeat(rgb_tensor.size(0))
        
        return rgb_tensor, signal_tensor

# ================================
# CalibrationPhys準拠 PhysNet2DCNN (3D版)
# ================================
class PhysNet2DCNN_3D(nn.Module):
    """CalibrationPhys論文準拠のPhysNet2DCNN（3D畳み込み版）"""
    def __init__(self, input_shape=None):
        super(PhysNet2DCNN_3D, self).__init__()
        
        if input_shape is not None:
            in_channels = input_shape[-1]
            height = input_shape[1] if len(input_shape) >= 3 else 36
            width = input_shape[2] if len(input_shape) >= 3 else 36
        else:
            in_channels = 3
            height = 36
            width = 36
        
        self.adaptive_pooling = height < 36 or width < 36
        
        # ConvBlock 1: 32 filters
        self.conv1_1 = nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1_1 = nn.BatchNorm3d(32, momentum=0.01, eps=1e-5)
        self.elu1_1 = nn.ELU(inplace=True)
        
        self.conv1_2 = nn.Conv3d(32, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1_2 = nn.BatchNorm3d(32, momentum=0.01, eps=1e-5)
        self.elu1_2 = nn.ELU(inplace=True)
        
        if height <= 16 or width <= 16:
            self.pool1 = nn.Identity()
        else:
            self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # ConvBlock 2: 64 filters
        self.conv2_1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu2_1 = nn.ELU(inplace=True)
        
        self.conv2_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu2_2 = nn.ELU(inplace=True)
        
        if height <= 16 or width <= 16:
            self.pool2 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        else:
            self.pool2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # ConvBlock 3: 64 filters
        self.conv3_1 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu3_1 = nn.ELU(inplace=True)
        
        self.conv3_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu3_2 = nn.ELU(inplace=True)
        
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # ConvBlock 4: 64 filters
        self.conv4_1 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu4_1 = nn.ELU(inplace=True)
        
        self.conv4_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu4_2 = nn.ELU(inplace=True)
        
        self.pool4 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # ConvBlock 5: 64 filters with upsampling
        self.conv5_1 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu5_1 = nn.ELU(inplace=True)
        
        self.conv5_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu5_2 = nn.ELU(inplace=True)
        
        self.upsample = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)
        
        # ConvBlock 6: 64 filters
        self.conv6_1 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn6_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu6_1 = nn.ELU(inplace=True)
        
        self.conv6_2 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn6_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu6_2 = nn.ELU(inplace=True)
        
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.conv_final = nn.Conv3d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He初期化で重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """入力: (B, T, H, W, C) → 出力: (B, T)"""
        batch_size = x.size(0)
        time_frames = x.size(1)
        
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.elu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.elu4_2(x)
        x = self.pool4(x)
        x = self.dropout(x)
        
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.elu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.elu5_2(x)
        x = self.upsample(x)
        
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.elu6_1(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.elu6_2(x)
        
        x = self.spatial_pool(x)
        x = self.conv_final(x)
        
        x = x.squeeze(1).squeeze(-1).squeeze(-1)
        
        if x.size(-1) != time_frames:
            x = F.interpolate(x.unsqueeze(1), size=time_frames, mode='linear', align_corners=False)
            x = x.squeeze(1)
        
        return x

# ================================
# CalibrationPhys準拠 PhysNet2DCNN (2D版)
# ================================
class PhysNet2DCNN_2D(nn.Module):
    """2D畳み込みを使用したPhysNet2DCNN（効率的な実装）"""
    def __init__(self, input_shape=None):
        super(PhysNet2DCNN_2D, self).__init__()
        
        if input_shape is not None:
            in_channels = input_shape[-1]
            height = input_shape[1] if len(input_shape) >= 3 else 36
            width = input_shape[2] if len(input_shape) >= 3 else 36
        else:
            in_channels = 3
            height = 36
            width = 36
        
        self.small_input = height <= 16 or width <= 16
        
        # ConvBlock 1: 32 filters
        if self.small_input:
            self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
            self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        
        self.bn1_1 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-5)
        self.elu1_1 = nn.ELU(inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-5)
        self.elu1_2 = nn.ELU(inplace=True)
        
        if self.small_input:
            self.pool1 = nn.Identity()
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 2: 64 filters
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu2_1 = nn.ELU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu2_2 = nn.ELU(inplace=True)
        
        if self.small_input:
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 3: 64 filters
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu3_1 = nn.ELU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu3_2 = nn.ELU(inplace=True)
        
        if self.small_input:
            self.pool3 = nn.Identity()
        else:
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 4: 64 filters（小さい入力では省略可能）
        if not self.small_input:
            self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4_1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
            self.elu4_1 = nn.ELU(inplace=True)
            
            self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4_2 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
            self.elu4_2 = nn.ELU(inplace=True)
            
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 5: 64 filters
        self.conv5_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu5_1 = nn.ELU(inplace=True)
        
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu5_2 = nn.ELU(inplace=True)
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal processing
        self.temporal_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(64, momentum=0.01, eps=1e-5)
        self.temporal_elu1 = nn.ELU(inplace=True)
        
        self.temporal_conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.temporal_bn2 = nn.BatchNorm1d(32, momentum=0.01, eps=1e-5)
        self.temporal_elu2 = nn.ELU(inplace=True)
        
        self.fc = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He初期化で重みを初期化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """入力: (B, T, H, W, C) → 出力: (B, T)"""
        batch_size, time_frames = x.size(0), x.size(1)
        
        x = x.view(batch_size * time_frames, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 3, 1, 2)  # (B*T, C, H, W)
        
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        if not self.small_input:
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = self.elu4_1(x)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x = self.elu4_2(x)
            x = self.pool4(x)
            x = self.dropout(x)
        
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.elu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.elu5_2(x)
        
        x = self.spatial_pool(x)
        x = x.view(batch_size, time_frames, 64)
        
        x = x.permute(0, 2, 1)
        x = self.temporal_conv1(x)
        x = self.temporal_bn1(x)
        x = self.temporal_elu1(x)
        
        x = self.temporal_conv2(x)
        x = self.temporal_bn2(x)
        x = self.temporal_elu2(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.fc(x)
        x = x.squeeze(-1)
        
        return x

# ================================
# モデル作成関数（torch.compile対応）
# ================================
def create_model(config):
    """設定に基づいてモデルを作成（torch.compile対応）"""
    if config.model_type == "3d":
        model = PhysNet2DCNN_3D(config.input_shape)
        model_name = "PhysNet2DCNN_3D (CalibrationPhys準拠)"
    elif config.model_type == "2d":
        model = PhysNet2DCNN_2D(config.input_shape)
        model_name = "PhysNet2DCNN_2D (CalibrationPhys準拠 効率版)"
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    if config.use_compile and torch.__version__.startswith("2."):
        if config.verbose:
            print(f"  PyTorch 2.0+検出: torch.compileでモデルを最適化します")
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            if config.verbose:
                print(f"  torch.compileに失敗しました: {e}")
                print("  通常モードで続行します")
    
    if config.verbose and config.model_mode == 'within_subject':
        print(f"\n選択モデル: {model_name}")
        print(f"モード: {config.model_mode}")
        print(f"使用チャンネル: {config.use_channel}")
        if config.use_lab:
            print(f"LABデータ: 使用（RGB+LAB = {config.num_channels}チャンネル）")
        else:
            print(f"LABデータ: 未使用（RGB = {config.num_channels}チャンネル）")
        
        print(f"訓練:検証データ比率: {config.train_val_split_ratio*100:.0f}:{(1-config.train_val_split_ratio)*100:.0f}")
        print(f"検証データ分割戦略: {config.validation_split_strategy}")
        if config.validation_split_strategy == 'stratified':
            print(f"  層化サンプリング設定:")
            print(f"    層の数: {config.n_strata}")
            print(f"    分割方法: {config.stratification_method}")
        
        print(f"\n【高速化設定】")
        print(f"  AMP (自動混合精度): {'有効' if config.use_amp else '無効'}")
        print(f"  torch.compile: {'有効' if config.use_compile else '無効'}")
        print(f"  DataLoader workers: {config.num_workers}")
        print(f"  Pin memory: {config.pin_memory}")
        print(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}")
        
        print(f"\nデータ拡張: {'有効' if config.use_augmentation else '無効'}")
        if config.use_augmentation:
            print(f"  - ランダムクロップ: {'有効' if config.crop_enabled else '無効'}")
            print(f"  - 回転: {'有効' if config.rotation_enabled else '無効'}")
            print(f"  - 時間軸ストレッチング: {'有効 (高速版)' if config.time_stretch_enabled else '無効'}")
            print(f"  - 明度・コントラスト調整: {'有効' if config.brightness_contrast_enabled else '無効'}")
        
        print(f"Warmup: {config.warmup_epochs}エポック (初期学習率×{config.warmup_lr_factor})")
        print(f"勾配クリッピング: {config.gradient_clip_val}")
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        if config.model_type == "3d":
            print("【注意】3Dモデルはメモリを多く使用します。バッチサイズの調整を推奨します。")
        elif config.model_type == "2d":
            print("【推奨】2Dモデルは計算効率が良く、大きなバッチサイズでも動作します。")
    
    return model

# ================================
# データ読み込み関数（複数指標対応）
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み（LABデータ対応、現在の指標用）"""
    
    # RGBデータの読み込み
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません: {rgb_path}")
        return None, None
    
    rgb_data = np.load(rgb_path)
    
    # データのリサイズ（14x16 → 36x36）
    resized_rgb = np.zeros((rgb_data.shape[0], 36, 36, rgb_data.shape[-1]))
    for i in range(rgb_data.shape[0]):
        for c in range(rgb_data.shape[-1]):
            resized_rgb[i, :, :, c] = cv2.resize(rgb_data[i, :, :, c], (36, 36))
    rgb_data = resized_rgb
    
    # LABデータの読み込み（オプション）
    if config.use_lab:
        lab_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_downsampled_1Hzver2.npy")
        
        if not os.path.exists(lab_path):
            print(f"警告: {subject}のLABデータが見つかりません: {lab_path}")
            print(f"  LABデータなしで処理を続行します。")
            config.use_lab = False
            config.use_channel = 'RGB'
            config.num_channels = 3
        else:
            lab_data = np.load(lab_path)
            # LABデータもリサイズ
            resized_lab = np.zeros((lab_data.shape[0], 36, 36, lab_data.shape[-1]))
            for i in range(lab_data.shape[0]):
                for c in range(lab_data.shape[-1]):
                    resized_lab[i, :, :, c] = cv2.resize(lab_data[i, :, :, c], (36, 36))
            lab_data = resized_lab
            
            if rgb_data.shape == lab_data.shape:
                combined_data = np.concatenate([rgb_data, lab_data], axis=-1)
                
                if lab_data.max() > 1.0:
                    combined_data[..., 3:] = combined_data[..., 3:] / 255.0
                
                rgb_data = combined_data
    
    # 信号データの読み込み
    signal_data_list = []
    for task in config.tasks:
        # Cross-Subjectモードでは現在の指標を使用
        if config.model_mode == 'cross_subject':
            signal_path = os.path.join(config.signal_base_path, subject, 
                                      config.current_signal_type, 
                                      f"{config.current_signal_prefix}_{task}.npy")
        else:
            # Within-Subjectモードでは従来通り
            signal_path = os.path.join(config.signal_base_path, subject, 
                                      config.signal_type, 
                                      f"{config.signal_prefix}_{task}.npy")
        
        if not os.path.exists(signal_path):
            if config.model_mode == 'cross_subject':
                print(f"警告: {subject}の{task}の{config.current_signal_type}データが見つかりません")
            else:
                print(f"警告: {subject}の{task}の{config.signal_type}データが見つかりません")
            return None, None
        signal_data_list.append(np.load(signal_path))
    
    # ここで signal_data を定義（重要！）
    signal_data = np.concatenate(signal_data_list)
    
    # 信号データの正規化（signal_dataが定義された後）
    if config.signal_normalization != 'none':
        if config.signal_norm_per_subject:
            # 被験者ごとの正規化
            signal_data, norm_params = normalize_signal(
                signal_data, 
                method=config.signal_normalization,
                return_params=True
            )
            # 正規化パラメータを保存（推論時に必要）
            config.norm_params[subject] = norm_params
            
            if config.verbose and config.model_mode == 'within_subject':
                print(f"  {subject}の信号正規化 ({config.signal_normalization}):")
                if config.signal_normalization == 'min-max':
                    print(f"    元の範囲: [{norm_params['min']:.3f}, {norm_params['max']:.3f}]")
                elif config.signal_normalization == 'z-score':
                    print(f"    元の平均: {norm_params['mean']:.3f}, 標準偏差: {norm_params['std']:.3f}")
                elif config.signal_normalization == 'robust':
                    print(f"    元の中央値: {norm_params['median']:.3f}, IQR: {norm_params['iqr']:.3f}")
    
    # データの正規化（RGBは0-1の範囲に）
    if rgb_data[..., :3].max() > 1.0:  # RGBチャンネルのみチェック
        rgb_data[..., :3] = rgb_data[..., :3] / 255.0
    
    return rgb_data, signal_data

# ================================
# データ読み込み関数（複数指標対応）
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み（LABデータ対応、現在の指標用）"""
    
    # すべてのモデルで同じRGBデータファイルを使用
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません: {rgb_path}")
        return None, None
    
    rgb_data = np.load(rgb_path)  # Shape: (360, 14, 16, 3)
    
    # データのリサイズ（14x16 → 36x36）
    resized_rgb = np.zeros((rgb_data.shape[0], 36, 36, rgb_data.shape[-1]))
    for i in range(rgb_data.shape[0]):
        for c in range(rgb_data.shape[-1]):
            resized_rgb[i, :, :, c] = cv2.resize(rgb_data[i, :, :, c], (36, 36))
    rgb_data = resized_rgb
    
    # LABデータの読み込み（オプション）
    if config.use_lab:
        lab_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_downsampled_1Hzver2.npy")
        
        if not os.path.exists(lab_path):
            print(f"警告: {subject}のLABデータが見つかりません: {lab_path}")
            print(f"  LABデータなしで処理を続行します。")
            config.use_lab = False
            config.use_channel = 'RGB'
            config.num_channels = 3
        else:
            lab_data = np.load(lab_path)
            # LABデータもリサイズ
            resized_lab = np.zeros((lab_data.shape[0], 36, 36, lab_data.shape[-1]))
            for i in range(lab_data.shape[0]):
                for c in range(lab_data.shape[-1]):
                    resized_lab[i, :, :, c] = cv2.resize(lab_data[i, :, :, c], (36, 36))
            lab_data = resized_lab
            
            if rgb_data.shape == lab_data.shape:
                combined_data = np.concatenate([rgb_data, lab_data], axis=-1)
                
                if lab_data.max() > 1.0:
                    combined_data[..., 3:] = combined_data[..., 3:] / 255.0
                
                rgb_data = combined_data
    
    # 信号データの読み込み（現在の指標）
    signal_data_list = []
    for task in config.tasks:
        # Cross-Subjectモードでは現在の指標を使用
        if config.model_mode == 'cross_subject':
            signal_path = os.path.join(config.signal_base_path, subject, 
                                      config.current_signal_type, 
                                      f"{config.current_signal_prefix}_{task}.npy")
        else:
            # Within-Subjectモードでは従来通り
            signal_path = os.path.join(config.signal_base_path, subject, 
                                      config.signal_type, 
                                      f"{config.signal_prefix}_{task}.npy")
        
        if not os.path.exists(signal_path):
            if config.model_mode == 'cross_subject':
                print(f"警告: {subject}の{task}の{config.current_signal_type}データが見つかりません")
            else:
                print(f"警告: {subject}の{task}の{config.signal_type}データが見つかりません")
            return None, None
        signal_data_list.append(np.load(signal_path))
    
    # ================================
    # 信号データの正規化（ここに追加）
    # ================================
    if config.signal_normalization != 'none':
        if config.signal_norm_per_subject:
            # 被験者ごとの正規化
            signal_data, norm_params = normalize_signal(
                signal_data, 
                method=config.signal_normalization,
                return_params=True
            )
            # 正規化パラメータを保存（推論時に必要）
            config.norm_params[subject] = norm_params
            
            if config.verbose and config.model_mode == 'within_subject':
                print(f"  {subject}の信号正規化 ({config.signal_normalization}):")
                if config.signal_normalization == 'min-max':
                    print(f"    元の範囲: [{norm_params['min']:.3f}, {norm_params['max']:.3f}]")
                elif config.signal_normalization == 'z-score':
                    print(f"    元の平均: {norm_params['mean']:.3f}, 標準偏差: {norm_params['std']:.3f}")
                elif config.signal_normalization == 'robust':
                    print(f"    元の中央値: {norm_params['median']:.3f}, IQR: {norm_params['iqr']:.3f}")
    
    # データの正規化（RGBは0-1の範囲に）
    if rgb_data[..., :3].max() > 1.0:  # RGBチャンネルのみチェック
        rgb_data[..., :3] = rgb_data[..., :3] / 255.0
    
    return rgb_data, signal_data
    
    # データの正規化（0-1の範囲に）
    if rgb_data[..., :3].max() > 1.0:  # RGBチャンネルのみチェック
        rgb_data[..., :3] = rgb_data[..., :3] / 255.0
    
    return rgb_data, signal_data

def load_all_subjects_data(config):
    """全被験者のデータを読み込み（Cross-Subject用、現在の指標用）"""
    all_rgb_data = []
    all_signal_data = []
    subject_task_info = []
    
    print(f"\n全被験者データ読み込み中（{config.current_signal_type}）...")
    
    for subject in config.subjects:
        rgb_data, signal_data = load_data_single_subject(subject, config)
        
        if rgb_data is None or signal_data is None:
            print(f"  {subject}のデータ読み込み失敗。スキップします。")
            continue
        
        # タスクごとに分割（各タスク60フレーム）
        for task_idx, task in enumerate(config.tasks):
            start_idx = task_idx * config.task_duration
            end_idx = (task_idx + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signal = signal_data[start_idx:end_idx]
            
            all_rgb_data.append(task_rgb)
            all_signal_data.append(task_signal)
            subject_task_info.append({
                'subject': subject,
                'task': task,
                'subject_idx': config.subjects.index(subject),
                'task_idx': task_idx
            })
    
    all_rgb_data = np.array(all_rgb_data)  # (N_samples, 60, H, W, C)
    all_signal_data = np.array(all_signal_data)  # (N_samples, 60)
    
    print(f"  読み込み完了: {len(all_rgb_data)}タスク分のデータ")
    print(f"  データ形状: RGB={all_rgb_data.shape}, Signal={all_signal_data.shape}")
    
    return all_rgb_data, all_signal_data, subject_task_info
    
# ================================
# 学習関数（AMP対応高速化版）
# ================================
def train_model(model, train_loader, val_loader, config, fold=None):
    """モデルの学習（AMP対応、高速化版）"""
    
    if config.verbose and fold is not None:
        print(f"\n  Fold {fold+1} 学習開始")
    
    model = model.to(config.device)
    
    # 損失関数の選択
    if config.loss_type == "combined":
        criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    elif config.loss_type == "huber_combined":
        criterion = HuberCorrelationLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    else:
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # AMP用のGradScaler
    scaler = GradScaler() if config.use_amp else None
    
    # Warmup設定
    initial_lr = config.learning_rate
    if config.warmup_epochs > 0:
        warmup_lr = initial_lr * config.warmup_lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
    # 学習率スケジューラー
    if config.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=1e-6
        )
        scheduler_per_batch = False
    elif config.scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate * 10, 
            epochs=config.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=10000
        )
        scheduler_per_batch = True
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=False
        )
        scheduler_per_batch = False
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_preds_best = None
    train_targets_best = None
    
    import time
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        # Warmup処理
        if epoch < config.warmup_epochs and config.warmup_epochs > 0:
            current_warmup_lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_warmup_lr
        elif epoch == config.warmup_epochs and config.warmup_epochs > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        # 学習フェーズ
        model.train()
        train_loss = 0
        train_preds_all = []
        train_targets_all = []
        
        for rgb, sig in train_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            
            optimizer.zero_grad()
            
            if config.use_amp:
                with autocast():
                    pred = model(rgb)
                    if hasattr(criterion, 'alpha'):
                        loss, mse_loss, corr_loss = criterion(pred, sig)
                    else:
                        loss = criterion(pred, sig)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(rgb)
                if hasattr(criterion, 'alpha'):
                    loss, mse_loss, corr_loss = criterion(pred, sig)
                else:
                    loss = criterion(pred, sig)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
                optimizer.step()
            
            if scheduler_per_batch and epoch >= config.warmup_epochs:
                scheduler.step()
            
            train_loss += loss.item()
            
            if pred.dim() == 2:
                train_preds_all.append(pred.detach().cpu().numpy())
                train_targets_all.append(sig.detach().cpu().numpy())
            else:
                train_preds_all.extend(pred.detach().cpu().numpy())
                train_targets_all.extend(sig.detach().cpu().numpy())
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for rgb, sig in val_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                
                if config.use_amp:
                    with autocast():
                        pred = model(rgb)
                        if hasattr(criterion, 'alpha'):
                            loss, mse_loss, corr_loss = criterion(pred, sig)
                        else:
                            loss = criterion(pred, sig)
                else:
                    pred = model(rgb)
                    if hasattr(criterion, 'alpha'):
                        loss, mse_loss, corr_loss = criterion(pred, sig)
                    else:
                        loss = criterion(pred, sig)
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if not scheduler_per_batch and epoch >= config.warmup_epochs:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # モデル保存判定
        improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss > 0 else 1
        if improvement > config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            if isinstance(train_preds_all[0], np.ndarray) and train_preds_all[0].ndim > 0:
                train_preds_best = np.concatenate(train_preds_all, axis=0)
                train_targets_best = np.concatenate(train_targets_all, axis=0)
            else:
                train_preds_best = np.array(train_preds_all)
                train_targets_best = np.array(train_targets_all)
            
            # モデル保存先を決定
            if config.model_mode == 'cross_subject':
                save_dir = Path(config.save_path) / config.current_signal_type / f'fold{fold+1}'
            else:
                save_dir = Path(config.save_path) / 'models'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_name = f'best_model_fold{fold+1}.pth' if fold is not None else 'best_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'model_type': config.model_type,
                'signal_type': config.current_signal_type if config.model_mode == 'cross_subject' else config.signal_type
            }, save_dir / model_name)
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        
        if config.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"    Epoch [{epoch+1:3d}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.1f}s")
        
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
    """モデルの評価（AMP対応）"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for rgb, sig in test_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            
            if config.use_amp:
                with autocast():
                    pred = model(rgb)
            else:
                pred = model(rgb)
            
            predictions.append(pred.cpu().numpy())
            targets.append(sig.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(np.mean((targets.flatten() - predictions.flatten()) ** 2))
    corr, p_value = pearsonr(targets.flatten(), predictions.flatten())
    
    # R-squared
    ss_res = np.sum((targets.flatten() - predictions.flatten()) ** 2)
    ss_tot = np.sum((targets.flatten() - np.mean(targets.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'corr': corr,
        'r2': r2,
        'p_value': p_value,
        'predictions': predictions,
        'targets': targets
    }

# ================================
# Within-Subject: タスク別6分割交差検証
# ================================
def task_cross_validation(rgb_data, signal_data, config, subject, save_dir):
    """タスク別6分割交差検証（1タスクずつ検証用）"""
    n_tasks = len(config.tasks)
    fold_results = []
    
    all_test_predictions = []
    all_test_targets = []
    all_test_tasks = []
    
    for test_task_idx, test_task in enumerate(config.tasks):
        print(f"    Fold {test_task_idx+1}/{n_tasks}: テスト={test_task}")
        
        # フォルダ作成
        fold_dir = save_dir / f'fold{test_task_idx+1}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # データ分割
        test_start = test_task_idx * config.task_duration
        test_end = (test_task_idx + 1) * config.task_duration
        
        test_rgb = rgb_data[test_start:test_end]
        test_signal = signal_data[test_start:test_end]
        
        train_rgb = np.concatenate([
            rgb_data[:test_start],
            rgb_data[test_end:]
        ])
        train_signal = np.concatenate([
            signal_data[:test_start],
            signal_data[test_end:]
        ])
        
        print(f"      元のデータサイズ - 訓練: {len(train_rgb)}, テスト: {len(test_rgb)}")
        
        # 訓練データから検証データを分離
        if config.validation_split_strategy == 'stratified':
            train_rgb, train_signal, val_rgb, val_signal = stratified_sampling_split(
                train_rgb, train_signal,
                val_ratio=(1 - config.train_val_split_ratio),
                n_strata=config.n_strata,
                method=config.stratification_method
            )
        else:
            val_size = int(len(train_rgb) * (1 - config.train_val_split_ratio))
            val_indices = np.random.choice(len(train_rgb), val_size, replace=False)
            train_indices = np.setdiff1d(np.arange(len(train_rgb)), val_indices)
            
            val_rgb = train_rgb[val_indices]
            val_signal = train_signal[val_indices]
            train_rgb = train_rgb[train_indices]
            train_signal = train_signal[train_indices]
        
        print(f"      最終データサイズ - 訓練: {len(train_rgb)}, 検証: {len(val_rgb)}, テスト: {len(test_rgb)}")
        
        # データローダー作成
        seed_worker = set_all_seeds(config.random_seed)
        
        train_dataset = CODataset(train_rgb[np.newaxis, ...], train_signal, 
                                 config.model_type, config.use_channel, config, is_training=True)
        val_dataset = CODataset(val_rgb[np.newaxis, ...], val_signal, 
                               config.model_type, config.use_channel, config, is_training=False)
        test_dataset = CODataset(test_rgb[np.newaxis, ...], test_signal, 
                                config.model_type, config.use_channel, config, is_training=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            worker_init_fn=seed_worker, persistent_workers=(config.persistent_workers and config.num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            worker_init_fn=seed_worker, persistent_workers=(config.persistent_workers and config.num_workers > 0)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            worker_init_fn=seed_worker, persistent_workers=(config.persistent_workers and config.num_workers > 0)
        )
        
        # モデル作成・学習
        model = create_model(config)
        model, train_preds, train_targets = train_model(model, train_loader, val_loader, config, fold=test_task_idx)
        
        # 評価
        train_results = evaluate_model(model, train_loader, config)
        test_results = evaluate_model(model, test_loader, config)
        
        print(f"      訓練結果 - MAE: {train_results['mae']:.4f}, RMSE: {train_results['rmse']:.4f}, " +
              f"Corr: {train_results['corr']:.4f}, R²: {train_results['r2']:.4f}")
        print(f"      テスト結果 - MAE: {test_results['mae']:.4f}, RMSE: {test_results['rmse']:.4f}, " +
              f"Corr: {test_results['corr']:.4f}, R²: {test_results['r2']:.4f}")
        
        # 結果を保存
        fold_results.append({
            'fold': test_task_idx + 1,
            'test_task': test_task,
            'train_mae': train_results['mae'],
            'train_rmse': train_results['rmse'],
            'train_corr': train_results['corr'],
            'train_r2': train_results['r2'],
            'test_mae': test_results['mae'],
            'test_rmse': test_results['rmse'],
            'test_corr': test_results['corr'],
            'test_r2': test_results['r2']
        })
        
        all_test_predictions.extend(test_results['predictions'].flatten())
        all_test_targets.extend(test_results['targets'].flatten())
        all_test_tasks.extend([test_task] * len(test_results['predictions'].flatten()))
        
        torch.cuda.empty_cache()
    
    all_test_predictions = np.array(all_test_predictions)
    all_test_targets = np.array(all_test_targets)
    all_test_tasks = np.array(all_test_tasks)
    
    return fold_results, all_test_predictions, all_test_targets, all_test_tasks

# ================================
# Cross-Subject: 5分割交差検証（単一指標用）
# ================================
def cross_subject_cv_single_signal(config):
    """単一指標の被験者間5分割交差検証"""
    
    signal_type = config.current_signal_type
    signal_dir = Path(config.save_path) / signal_type
    signal_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"指標: {signal_type} の5分割交差検証開始")
    print(f"{'='*60}")
    
    # 全データ読み込み
    all_rgb_data, all_signal_data, subject_task_info = load_all_subjects_data(config)
    
    n_samples = len(all_rgb_data)
    
    # 完全ランダムシャッフル
    np.random.seed(config.random_seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 基本的な5分割
    fold_size = n_samples // config.n_folds
    fold_indices = []
    for i in range(config.n_folds):
        start_idx = i * fold_size
        if i == config.n_folds - 1:
            fold_idx = indices[start_idx:]
        else:
            fold_idx = indices[start_idx:start_idx + fold_size]
        fold_indices.append(list(fold_idx))
    
    # 各foldに各被験者のデータが最低1つ含まれるように調整
    if config.ensure_subject_in_each_fold:
        for fold_idx in range(config.n_folds):
            current_fold = fold_indices[fold_idx]
            subjects_in_fold = set()
            for idx in current_fold:
                subjects_in_fold.add(subject_task_info[idx]['subject'])
            
            missing_subjects = set(config.subjects) - subjects_in_fold
            
            if missing_subjects:
                for missing_subject in missing_subjects:
                    for other_fold_idx in range(config.n_folds):
                        if other_fold_idx == fold_idx:
                            continue
                        
                        other_fold = fold_indices[other_fold_idx]
                        subject_indices_in_other = []
                        for idx in other_fold:
                            if subject_task_info[idx]['subject'] == missing_subject:
                                subject_indices_in_other.append(idx)
                        
                        if len(subject_indices_in_other) >= 2:
                            idx_to_move = np.random.choice(subject_indices_in_other)
                            fold_indices[other_fold_idx].remove(idx_to_move)
                            fold_indices[fold_idx].append(idx_to_move)
                            break
    
    for i in range(config.n_folds):
        fold_indices[i] = np.array(fold_indices[i])
    
    # データ分割情報を保存
    split_info_file = signal_dir / 'data_split_info.txt'
    with open(split_info_file, 'w', encoding='utf-8') as f:
        f.write(f"="*60 + "\n")
        f.write(f"{signal_type} - 5分割交差検証 データ分割情報\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"総サンプル数: {n_samples}\n")
        f.write(f"被験者数: {len(config.subjects)}\n")
        f.write(f"タスク数/被験者: {len(config.tasks)}\n\n")
        
        for fold_idx, test_idx in enumerate(fold_indices):
            f.write(f"Fold {fold_idx + 1}: {len(test_idx)}サンプル\n")
            test_subjects = {}
            for idx in test_idx:
                subj = subject_task_info[idx]['subject']
                task = subject_task_info[idx]['task']
                if subj not in test_subjects:
                    test_subjects[subj] = []
                test_subjects[subj].append(task)
            f.write(f"  被験者数: {len(test_subjects)}\n")
            f.write(f"  被験者ごとの詳細:\n")
            for subj in sorted(test_subjects.keys()):
                tasks = test_subjects[subj]
                f.write(f"    {subj}: {len(tasks)}タスク ({', '.join(sorted(tasks))})\n")
            f.write("\n")
    
    # 交差検証実行
    fold_results = []
    
    for fold_idx in range(config.n_folds):
        print(f"\nFold {fold_idx + 1}/{config.n_folds}")
        
        fold_dir = signal_dir / f'fold{fold_idx + 1}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        test_idx = fold_indices[fold_idx]
        train_idx = np.concatenate([fold_indices[i] for i in range(config.n_folds) if i != fold_idx])
        
        # データ分割
        train_rgb = all_rgb_data[train_idx]
        train_signal = all_signal_data[train_idx]
        test_rgb = all_rgb_data[test_idx]
        test_signal = all_signal_data[test_idx]
        
        # 検証データ分離
        val_size = int(len(train_rgb) * 0.1)
        val_indices = np.random.choice(len(train_rgb), val_size, replace=False)
        train_indices = np.setdiff1d(np.arange(len(train_rgb)), val_indices)
        
        val_rgb = train_rgb[val_indices]
        val_signal = train_signal[val_indices]
        train_rgb = train_rgb[train_indices]
        train_signal = train_signal[train_indices]
        
        print(f"  訓練: {len(train_rgb)}, 検証: {len(val_rgb)}, テスト: {len(test_rgb)}")
        
        # データローダー作成
        seed_worker = set_all_seeds(config.random_seed)
        
        train_dataset = CODataset(train_rgb, train_signal, config.model_type, 
                                 config.use_channel, config, is_training=True)
        val_dataset = CODataset(val_rgb, val_signal, config.model_type, 
                               config.use_channel, config, is_training=False)
        test_dataset = CODataset(test_rgb, test_signal, config.model_type, 
                                config.use_channel, config, is_training=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            worker_init_fn=seed_worker
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            worker_init_fn=seed_worker
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            worker_init_fn=seed_worker
        )
        
        # モデル作成・学習
        model = create_model(config)
        model, train_preds, train_targets = train_model(model, train_loader, val_loader, config, fold=fold_idx)
        
        # 評価
        train_results = evaluate_model(model, train_loader, config)
        test_results = evaluate_model(model, test_loader, config)
        
        print(f"  結果 - 訓練MAE: {train_results['mae']:.4f}, Corr: {train_results['corr']:.4f}")
        print(f"        テストMAE: {test_results['mae']:.4f}, Corr: {test_results['corr']:.4f}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_mae': train_results['mae'],
            'train_corr': train_results['corr'],
            'test_mae': test_results['mae'],
            'test_corr': test_results['corr'],
            'train_predictions': train_results['predictions'],
            'train_targets': train_results['targets'],
            'test_predictions': test_results['predictions'],
            'test_targets': test_results['targets']
        })
        
        # プロット（被験者ごとに色分け）
        plot_fold_results_cross_subject(
            fold_results[-1], fold_dir, config, 
            subject_task_info, train_idx, test_idx
        )
        
        torch.cuda.empty_cache()
    
    # 全Fold統合プロット
    plot_all_folds_summary(fold_results, signal_dir, config, subject_task_info, fold_indices)
    
    # 結果サマリー
    avg_train_mae = np.mean([r['train_mae'] for r in fold_results])
    avg_train_corr = np.mean([r['train_corr'] for r in fold_results])
    avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
    avg_test_corr = np.mean([r['test_corr'] for r in fold_results])
    std_test_mae = np.std([r['test_mae'] for r in fold_results])
    std_test_corr = np.std([r['test_corr'] for r in fold_results])
    
    print(f"\n{signal_type} 最終結果:")
    print(f"  訓練平均: MAE={avg_train_mae:.4f}, Corr={avg_train_corr:.4f}")
    print(f"  テスト平均: MAE={avg_test_mae:.4f}±{std_test_mae:.4f}, Corr={avg_test_corr:.4f}±{std_test_corr:.4f}")
    
    # 結果をCSVに保存
    results_df = pd.DataFrame([{
        'Signal': signal_type,
        'Fold': r['fold'],
        'Train_MAE': r['train_mae'],
        'Train_Corr': r['train_corr'],
        'Test_MAE': r['test_mae'],
        'Test_Corr': r['test_corr']
    } for r in fold_results])
    
    mean_row = pd.DataFrame({
        'Signal': [signal_type],
        'Fold': ['Mean'],
        'Train_MAE': [avg_train_mae],
        'Train_Corr': [avg_train_corr],
        'Test_MAE': [avg_test_mae],
        'Test_Corr': [avg_test_corr]
    })
    
    std_row = pd.DataFrame({
        'Signal': [signal_type],
        'Fold': ['Std'],
        'Train_MAE': [np.std([r['train_mae'] for r in fold_results])],
        'Train_Corr': [np.std([r['train_corr'] for r in fold_results])],
        'Test_MAE': [std_test_mae],
        'Test_Corr': [std_test_corr]
    })
    
    results_df = pd.concat([results_df, mean_row, std_row], ignore_index=True)
    results_df.to_csv(signal_dir / 'cross_validation_results.csv', index=False)
    
    return {
        'signal_type': signal_type,
        'avg_train_mae': avg_train_mae,
        'avg_train_corr': avg_train_corr,
        'avg_test_mae': avg_test_mae,
        'avg_test_corr': avg_test_corr,
        'std_test_mae': std_test_mae,
        'std_test_corr': std_test_corr
    }

# ================================
# プロット関数
# ================================
def plot_fold_results_cross_subject(result, fold_dir, config, subject_task_info, train_idx, test_idx):
    """各Foldの結果をプロット（被験者ごとに色分け）"""
    fold = result['fold']
    
    # 訓練データ散布図
    plt.figure(figsize=(14, 10))
    
    train_subjects = [subject_task_info[idx]['subject'] for idx in train_idx]
    unique_train_subjects = list(set(train_subjects))
    
    for subject in sorted(unique_train_subjects):
        subject_mask = np.array([s == subject for s in train_subjects])
        subject_indices = np.where(subject_mask)[0]
        
        if len(subject_indices) > 0:
            subject_preds = result['train_predictions'][subject_indices].flatten()
            subject_targets = result['train_targets'][subject_indices].flatten()
            
            plt.scatter(subject_targets, subject_preds, 
                       alpha=0.3, s=5, color=config.subject_colors[subject],
                       label=subject)
    
    all_vals = np.concatenate([result['train_predictions'].flatten(), 
                              result['train_targets'].flatten()])
    min_val = all_vals.min()
    max_val = all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"{config.current_signal_type} - Fold {fold} 訓練データ - MAE: {result['train_mae']:.3f}, Corr: {result['train_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=6)
    plt.tight_layout()
    plt.savefig(fold_dir / f'train_scatter_by_subject.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # テストデータ散布図
    plt.figure(figsize=(14, 10))
    
    test_subjects = [subject_task_info[idx]['subject'] for idx in test_idx]
    unique_test_subjects = list(set(test_subjects))
    
    for subject in sorted(unique_test_subjects):
        subject_mask = np.array([s == subject for s in test_subjects])
        subject_indices = np.where(subject_mask)[0]
        
        if len(subject_indices) > 0:
            subject_preds = result['test_predictions'][subject_indices].flatten()
            subject_targets = result['test_targets'][subject_indices].flatten()
            
            plt.scatter(subject_targets, subject_preds, 
                       alpha=0.4, s=8, color=config.subject_colors[subject],
                       label=f"{subject} ({len(subject_indices)})")
    
    all_vals = np.concatenate([result['test_predictions'].flatten(), 
                              result['test_targets'].flatten()])
    min_val = all_vals.min()
    max_val = all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"{config.current_signal_type} - Fold {fold} テストデータ - MAE: {result['test_mae']:.3f}, Corr: {result['test_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=6)
    plt.tight_layout()
    plt.savefig(fold_dir / f'test_scatter_by_subject.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_all_folds_summary(fold_results, save_dir, config, subject_task_info, all_fold_indices):
    """全Foldの結果を統合してプロット（被験者ごとに色分け）"""
    
    # 訓練データ統合プロット
    plt.figure(figsize=(16, 12))
    
    for fold_idx, result in enumerate(fold_results):
        test_idx = all_fold_indices[fold_idx]
        train_idx = np.concatenate([all_fold_indices[i] for i in range(len(all_fold_indices)) if i != fold_idx])
        train_subjects = [subject_task_info[idx]['subject'] for idx in train_idx]
        
        plotted_subjects = set()
        for subject in sorted(set(train_subjects)):
            subject_mask = np.array([s == subject for s in train_subjects])
            subject_indices = np.where(subject_mask)[0]
            
            if len(subject_indices) > 0:
                subject_preds = result['train_predictions'][subject_indices].flatten()
                subject_targets = result['train_targets'][subject_indices].flatten()
                
                if subject not in plotted_subjects and fold_idx == 0:
                    plt.scatter(subject_targets, subject_preds, 
                               alpha=0.2, s=3, color=config.subject_colors[subject],
                               label=subject)
                    plotted_subjects.add(subject)
                else:
                    plt.scatter(subject_targets, subject_preds, 
                               alpha=0.2, s=3, color=config.subject_colors[subject])
    
    all_train_targets = np.concatenate([r['train_targets'].flatten() for r in fold_results])
    all_train_preds = np.concatenate([r['train_predictions'].flatten() for r in fold_results])
    min_val = min(all_train_targets.min(), all_train_preds.min())
    max_val = max(all_train_targets.max(), all_train_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('真値')
    plt.ylabel('予測値')
    
    avg_train_mae = np.mean([r['train_mae'] for r in fold_results])
    avg_train_corr = np.mean([r['train_corr'] for r in fold_results])
    plt.title(f'{config.current_signal_type} - 全Fold 訓練データ - 平均MAE: {avg_train_mae:.3f}, 平均Corr: {avg_train_corr:.3f}')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3, fontsize=5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_folds_train_scatter_by_subject.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # テストデータ統合プロット
    plt.figure(figsize=(16, 12))
    
    for fold_idx, result in enumerate(fold_results):
        test_idx = all_fold_indices[fold_idx]
        test_subjects = [subject_task_info[idx]['subject'] for idx in test_idx]
        
        plotted_subjects = set()
        for subject in sorted(set(test_subjects)):
            subject_mask = np.array([s == subject for s in test_subjects])
            subject_indices = np.where(subject_mask)[0]
            
            if len(subject_indices) > 0:
                subject_preds = result['test_predictions'][subject_indices].flatten()
                subject_targets = result['test_targets'][subject_indices].flatten()
                
                if subject not in plotted_subjects and fold_idx == 0:
                    plt.scatter(subject_targets, subject_preds, 
                               alpha=0.3, s=5, color=config.subject_colors[subject],
                               label=subject)
                    plotted_subjects.add(subject)
                else:
                    plt.scatter(subject_targets, subject_preds, 
                               alpha=0.3, s=5, color=config.subject_colors[subject])
    
    all_test_targets = np.concatenate([r['test_targets'].flatten() for r in fold_results])
    all_test_preds = np.concatenate([r['test_predictions'].flatten() for r in fold_results])
    min_val = min(all_test_targets.min(), all_test_preds.min())
    max_val = max(all_test_targets.max(), all_test_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('真値')
    plt.ylabel('予測値')
    
    avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
    avg_test_corr = np.mean([r['test_corr'] for r in fold_results])
    plt.title(f'{config.current_signal_type} - 全Fold テストデータ - 平均MAE: {avg_test_mae:.3f}, 平均Corr: {avg_test_corr:.3f}')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3, fontsize=5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_folds_test_scatter_by_subject.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Fold別パフォーマンス比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    folds = [f"Fold {r['fold']}" for r in fold_results]
    mae_values = [r['test_mae'] for r in fold_results]
    corr_values = [r['test_corr'] for r in fold_results]
    
    bars1 = ax1.bar(folds, mae_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax1.axhline(y=np.mean(mae_values), color='r', linestyle='--', 
                label=f'平均: {np.mean(mae_values):.3f}')
    ax1.set_ylabel('MAE')
    ax1.set_title('各FoldのMAE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    bars2 = ax2.bar(folds, corr_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax2.axhline(y=np.mean(corr_values), color='r', linestyle='--', 
                label=f'平均: {np.mean(corr_values):.3f}')
    ax2.set_ylabel('相関係数')
    ax2.set_title('各Foldの相関係数')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'fold_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_subject_summary_colored(fold_results, all_test_predictions, all_test_targets, 
                               all_test_tasks, subject, save_dir, config):
    """被験者の結果サマリープロット（タスクごとに色分け）"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 訓練データ散布図（結合）
    all_train_preds = []
    all_train_targets = []
    for result in fold_results:
        task = result['test_task']
        
        test_start = config.tasks.index(task) * config.task_duration
        test_end = test_start + config.task_duration
        
        train_preds = np.concatenate([
            result['train_predictions'][:test_start],
            result['train_predictions'][test_end-config.task_duration:]
        ]) if test_start > 0 else result['train_predictions'][test_end-config.task_duration:]
        
        train_targets = np.concatenate([
            result['train_targets'][:test_start],
            result['train_targets'][test_end-config.task_duration:]
        ]) if test_start > 0 else result['train_targets'][test_end-config.task_duration:]
        
        all_train_preds.extend(train_preds)
        all_train_targets.extend(train_targets)
    
    all_train_preds = np.array(all_train_preds)
    all_train_targets = np.array(all_train_targets)
    
    ax1.scatter(all_train_targets, all_train_preds, alpha=0.5, s=3, color='blue')
    ax1.plot([all_train_targets.min(), all_train_targets.max()], 
             [all_train_targets.min(), all_train_targets.max()], 'r--', lw=1)
    
    train_mae = mean_absolute_error(all_train_targets, all_train_preds)
    train_corr = pearsonr(all_train_targets, all_train_preds)[0]
    
    ax1.set_xlabel('真値')
    ax1.set_ylabel('予測値')
    ax1.set_title(f'訓練データ全体\nMAE: {train_mae:.3f}, Corr: {train_corr:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. テストデータ散布図（タスクごとに色分け）
    for task in config.tasks:
        task_mask = (all_test_tasks == task)
        task_preds = all_test_predictions[task_mask]
        task_targets = all_test_targets[task_mask]
        
        ax2.scatter(task_targets, task_preds, alpha=0.6, s=10, 
                   color=config.task_colors[task], label=task)
    
    ax2.plot([all_test_targets.min(), all_test_targets.max()], 
             [all_test_targets.min(), all_test_targets.max()], 'r--', lw=1)
    
    test_mae = mean_absolute_error(all_test_targets, all_test_predictions)
    test_corr = pearsonr(all_test_targets, all_test_predictions)[0]
    
    ax2.set_xlabel('真値')
    ax2.set_ylabel('予測値')
    ax2.set_title(f'テストデータ全体（タスク別）\nMAE: {test_mae:.3f}, Corr: {test_corr:.3f}')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Fold別MAE比較
    folds = [f"Fold{i+1}\n({result['test_task']})" for i, result in enumerate(fold_results)]
    train_mae_values = [result['train_mae'] for result in fold_results]
    test_mae_values = [result['test_mae'] for result in fold_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, train_mae_values, width, label='訓練', alpha=0.7)
    bars2 = ax3.bar(x + width/2, test_mae_values, width, label='テスト', alpha=0.7)
    
    for i, (bar1, bar2, result) in enumerate(zip(bars1, bars2, fold_results)):
        task = result['test_task']
        bar2.set_color(config.task_colors[task])
    
    ax3.set_xlabel('Fold (テストタスク)')
    ax3.set_ylabel('MAE')
    ax3.set_title('Fold別MAE比較')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds, fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    avg_train_mae = np.mean(train_mae_values)
    avg_test_mae = np.mean(test_mae_values)
    ax3.axhline(y=avg_test_mae, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # 4. Fold別相関係数比較
    train_corr_values = [result['train_corr'] for result in fold_results]
    test_corr_values = [result['test_corr'] for result in fold_results]
    
    bars3 = ax4.bar(x - width/2, train_corr_values, width, label='訓練', alpha=0.7)
    bars4 = ax4.bar(x + width/2, test_corr_values, width, label='テスト', alpha=0.7)
    
    for i, (bar3, bar4, result) in enumerate(zip(bars3, bars4, fold_results)):
        task = result['test_task']
        bar4.set_color(config.task_colors[task])
    
    ax4.set_xlabel('Fold (テストタスク)')
    ax4.set_ylabel('相関係数')
    ax4.set_title('Fold別相関係数比較')
    ax4.set_xticks(x)
    ax4.set_xticklabels(folds, fontsize=8)
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    avg_test_corr = np.mean(test_corr_values)
    ax4.axhline(y=avg_test_corr, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle(f'{subject} - 6分割交差検証結果（タスク別）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f'{subject}_summary_colored.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return train_mae, train_corr, test_mae, test_corr

def plot_all_subjects_summary_unified(all_subjects_results, config):
    """全被験者の統合サマリープロット"""
    save_dir = Path(config.save_path)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # データ準備
    subjects = [r['subject'] for r in all_subjects_results]
    train_maes = [r['train_mae'] for r in all_subjects_results]
    test_maes = [r['test_mae'] for r in all_subjects_results]
    train_corrs = [r['train_corr'] for r in all_subjects_results]
    test_corrs = [r['test_corr'] for r in all_subjects_results]
    
    x = np.arange(len(subjects))
    
    # 1. MAE比較（被験者別）
    width = 0.35
    bars1 = ax1.bar(x - width/2, train_maes, width, label='訓練', alpha=0.7, color='blue')
    bars2 = ax1.bar(x + width/2, test_maes, width, label='テスト', alpha=0.7, color='orange')
    
    ax1.set_xlabel('被験者')
    ax1.set_ylabel('MAE')
    ax1.set_title('被験者別MAE比較')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, rotation=45, fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    avg_train_mae = np.mean(train_maes)
    avg_test_mae = np.mean(test_maes)
    ax1.axhline(y=avg_train_mae, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_test_mae, color='orange', linestyle='--', alpha=0.5)
    
    # 2. 相関係数比較（被験者別）
    bars3 = ax2.bar(x - width/2, train_corrs, width, label='訓練', alpha=0.7, color='blue')
    bars4 = ax2.bar(x + width/2, test_corrs, width, label='テスト', alpha=0.7, color='orange')
    
    ax2.set_xlabel('被験者')
    ax2.set_ylabel('相関係数')
    ax2.set_title('被験者別相関係数比較')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects, rotation=45, fontsize=8)
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    avg_train_corr = np.mean(train_corrs)
    avg_test_corr = np.mean(test_corrs)
    ax2.axhline(y=avg_train_corr, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=avg_test_corr, color='orange', linestyle='--', alpha=0.5)
    
    # 3. MAE分布（ヒストグラム）
    ax3.hist(train_maes, bins=10, alpha=0.5, label='訓練', color='blue')
    ax3.hist(test_maes, bins=10, alpha=0.5, label='テスト', color='orange')
    ax3.axvline(x=avg_train_mae, color='blue', linestyle='--', label=f'訓練平均: {avg_train_mae:.3f}')
    ax3.axvline(x=avg_test_mae, color='orange', linestyle='--', label=f'テスト平均: {avg_test_mae:.3f}')
    ax3.set_xlabel('MAE')
    ax3.set_ylabel('頻度')
    ax3.set_title('MAE分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 相関係数分布（ヒストグラム）
    ax4.hist(train_corrs, bins=10, alpha=0.5, label='訓練', color='blue')
    ax4.hist(test_corrs, bins=10, alpha=0.5, label='テスト', color='orange')
    ax4.axvline(x=avg_train_corr, color='blue', linestyle='--', label=f'訓練平均: {avg_train_corr:.3f}')
    ax4.axvline(x=avg_test_corr, color='orange', linestyle='--', label=f'テスト平均: {avg_test_corr:.3f}')
    ax4.set_xlabel('相関係数')
    ax4.set_ylabel('頻度')
    ax4.set_title('相関係数分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('全被験者サマリー（Within-Subject 6分割交差検証）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_summary_unified.png', dpi=150, bbox_inches='tight')
    plt.close()

# ================================
# メイン実行
# ================================
def main():
    """メイン実行関数"""
    config = Config()
    
    # 乱数シード設定
    set_all_seeds(config.random_seed)
    
    import time
    total_start_time = time.time()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - 高速化版")
    print("="*60)
    print(f"モード: {config.model_mode}")
    
    if config.model_mode == 'cross_subject':
        print(f"解析指標: {', '.join(config.signal_types)}")
    else:
        print(f"信号タイプ: {config.signal_type}")
    
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    
    if config.model_mode == 'cross_subject':
        print(f"交差検証: {config.n_folds}分割")
    else:
        print(f"交差検証: 6分割（タスク別）")
    
    print(f"デバイス: {config.device}")
    
    if torch.cuda.is_available():
        print(f"\nGPU情報:")
        print(f"  デバイス名: {torch.cuda.get_device_name(0)}")
        print(f"  メモリ容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  PyTorchバージョン: {torch.__version__}")
        print(f"  CUDAバージョン: {torch.version.cuda}")
    
    print(f"\n保存先: {config.save_path}")
    print(f"乱数シード: {config.random_seed}")
    
    # ================================
    # Cross-Subject モード
    # ================================
    if config.model_mode == 'cross_subject':
        print(f"\n{'='*60}")
        print("Cross-Subject モード: 複数指標5分割交差検証")
        print(f"{'='*60}")
        
        # 全指標の結果を保存
        all_signals_results = []
        
        # 各指標について順次実行
        for signal_type in config.signal_types:
            signal_start_time = time.time()
            
            # 現在の指標を設定
            config.set_current_signal(signal_type)
            
            # 5分割交差検証実行
            result = cross_subject_cv_single_signal(config)
            all_signals_results.append(result)
            
            signal_time = time.time() - signal_start_time
            print(f"\n{signal_type} 処理時間: {signal_time:.1f}秒 ({signal_time/60:.1f}分)")
        
        # 全指標のサマリーを作成
        save_dir = Path(config.save_path)
        summary_df = pd.DataFrame(all_signals_results)
        summary_df.to_csv(save_dir / 'all_signals_summary.csv', index=False)
        
        # 全指標の比較グラフ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        signals = [r['signal_type'] for r in all_signals_results]
        test_maes = [r['avg_test_mae'] for r in all_signals_results]
        test_corrs = [r['avg_test_corr'] for r in all_signals_results]
        test_mae_stds = [r['std_test_mae'] for r in all_signals_results]
        test_corr_stds = [r['std_test_corr'] for r in all_signals_results]
        
        x = np.arange(len(signals))
        
        # MAE比較
        ax1.bar(x, test_maes, yerr=test_mae_stds, capsize=5)
        ax1.set_xlabel('指標')
        ax1.set_ylabel('MAE')
        ax1.set_title('各指標のMAE比較')
        ax1.set_xticks(x)
        ax1.set_xticklabels(signals, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 相関係数比較
        ax2.bar(x, test_corrs, yerr=test_corr_stds, capsize=5)
        ax2.set_xlabel('指標')
        ax2.set_ylabel('相関係数')
        ax2.set_title('各指標の相関係数比較')
        ax2.set_xticks(x)
        ax2.set_xticklabels(signals, rotation=45)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'all_signals_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        total_time = time.time() - total_start_time
        
        # 設定情報を保存
        with open(save_dir / 'config_summary.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("実験設定（Cross-Subject）\n")
            f.write("="*60 + "\n")
            f.write(f"実行日時: {config.timestamp}\n")
            f.write(f"モード: {config.model_mode}\n")
            f.write(f"解析指標: {', '.join(config.signal_types)}\n")
            f.write(f"モデルタイプ: {config.model_type}\n")
            f.write(f"使用チャンネル: {config.use_channel}\n")
            f.write(f"LABデータ使用: {config.use_lab}\n")
            f.write(f"交差検証: {config.n_folds}分割\n")
            f.write(f"バッチサイズ: {config.batch_size}\n")
            f.write(f"エポック数: {config.epochs}\n")
            f.write(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)\n")
            f.write("\n結果サマリー:\n")
            for result in all_signals_results:
                f.write(f"\n{result['signal_type']}:\n")
                f.write(f"  テストMAE: {result['avg_test_mae']:.4f}±{result['std_test_mae']:.4f}\n")
                f.write(f"  テスト相関: {result['avg_test_corr']:.4f}±{result['std_test_corr']:.4f}\n")
        
        print(f"\n{'='*60}")
        print("全指標の処理完了")
        print(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
        print(f"結果保存先: {config.save_path}")
        print(f"{'='*60}")
    
    # ================================
    # Within-Subject モード
    # ================================
    elif config.model_mode == 'within_subject':
        print(f"\n{'='*60}")
        print("Within-Subject モード: 個人内6分割交差検証")
        print(f"{'='*60}")
        
        all_subjects_results = []
        
        for subj_idx, subject in enumerate(config.subjects):
            subject_start_time = time.time()
            
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
                if config.use_lab and rgb_data.shape[-1] == 6:
                    print(f"  チャンネル構成: RGB(3) + LAB(3) = {rgb_data.shape[-1]}チャンネル")
                
                # 6分割交差検証実行
                fold_results, all_test_predictions, all_test_targets, all_test_tasks = task_cross_validation(
                    rgb_data, signal_data, config, subject, subject_save_dir
                )
                
                # 被験者全体のサマリープロット（色分け対応）
                train_mae, train_corr, test_mae, test_corr = plot_subject_summary_colored(
                    fold_results, all_test_predictions, all_test_targets, all_test_tasks,
                    subject, subject_save_dir, config
                )
                
                # 結果保存
                all_subjects_results.append({
                    'subject': subject,
                    'train_mae': train_mae,
                    'train_corr': train_corr,
                    'test_mae': test_mae,
                    'test_corr': test_corr,
                    'fold_results': fold_results,
                    'all_test_predictions': all_test_predictions,
                    'all_test_targets': all_test_targets,
                    'all_test_tasks': all_test_tasks
                })
                
                subject_time = time.time() - subject_start_time
                
                print(f"\n  {subject} 完了:")
                print(f"    全体訓練: MAE={train_mae:.4f}, Corr={train_corr:.4f}")
                print(f"    全体テスト: MAE={test_mae:.4f}, Corr={test_corr:.4f}")
                print(f"    処理時間: {subject_time:.1f}秒 ({subject_time/60:.1f}分)")
                
                # 結果をCSVファイルに保存
                results_df = pd.DataFrame({
                    'Subject': [subject],
                    'Train_MAE': [train_mae],
                    'Train_Corr': [train_corr],
                    'Test_MAE': [test_mae],
                    'Test_Corr': [test_corr],
                    'Processing_Time_Sec': [subject_time]
                })
                
                csv_path = subject_save_dir / 'results_summary.csv'
                results_df.to_csv(csv_path, index=False)
                
                # 各Foldの結果も保存
                fold_df = pd.DataFrame(fold_results)
                fold_csv_path = subject_save_dir / 'fold_results.csv'
                fold_df.to_csv(fold_csv_path, index=False)
                
            except Exception as e:
                print(f"  {subject}でエラー発生: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 全被験者のサマリープロット（統合版）
        if all_subjects_results:
            print(f"\n{'='*60}")
            print("全被験者サマリー作成中...")
            plot_all_subjects_summary_unified(all_subjects_results, config)
            
            # 統計サマリー
            avg_train_mae = np.mean([r['train_mae'] for r in all_subjects_results])
            avg_train_corr = np.mean([r['train_corr'] for r in all_subjects_results])
            avg_test_mae = np.mean([r['test_mae'] for r in all_subjects_results])
            avg_test_corr = np.mean([r['test_corr'] for r in all_subjects_results])
            
            std_train_mae = np.std([r['train_mae'] for r in all_subjects_results])
            std_train_corr = np.std([r['train_corr'] for r in all_subjects_results])
            std_test_mae = np.std([r['test_mae'] for r in all_subjects_results])
            std_test_corr = np.std([r['test_corr'] for r in all_subjects_results])
            
            print(f"\n全被験者平均結果:")
            print(f"  訓練: MAE={avg_train_mae:.4f}±{std_train_mae:.4f}, Corr={avg_train_corr:.4f}±{std_train_corr:.4f}")
            print(f"  テスト: MAE={avg_test_mae:.4f}±{std_test_mae:.4f}, Corr={avg_test_corr:.4f}±{std_test_corr:.4f}")
            
            # 全体結果をCSVファイルに保存
            all_results_df = pd.DataFrame([{
                'Subject': r['subject'],
                'Train_MAE': r['train_mae'],
                'Train_Corr': r['train_corr'],
                'Test_MAE': r['test_mae'],
                'Test_Corr': r['test_corr']
            } for r in all_subjects_results])
            
            # 平均と標準偏差を追加
            mean_row = pd.DataFrame({
                'Subject': ['Mean'],
                'Train_MAE': [avg_train_mae],
                'Train_Corr': [avg_train_corr],
                'Test_MAE': [avg_test_mae],
                'Test_Corr': [avg_test_corr]
            })
            
            std_row = pd.DataFrame({
                'Subject': ['Std'],
                'Train_MAE': [std_train_mae],
                'Train_Corr': [std_train_corr],
                'Test_MAE': [std_test_mae],
                'Test_Corr': [std_test_corr]
            })
            
            all_results_df = pd.concat([all_results_df, mean_row, std_row], ignore_index=True)
            
            save_dir = Path(config.save_path)
            all_results_df.to_csv(save_dir / 'all_subjects_results.csv', index=False)
            
            total_time = time.time() - total_start_time
            
            # 設定情報も保存
            with open(save_dir / 'config_summary.txt', 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("実験設定（Within-Subject）\n")
                f.write("="*60 + "\n")
                f.write(f"実行日時: {config.timestamp}\n")
                f.write(f"モード: {config.model_mode}\n")
                f.write(f"モデルタイプ: {config.model_type}\n")
                f.write(f"信号タイプ: {config.signal_type}\n")
                f.write(f"使用チャンネル: {config.use_channel}\n")
                f.write(f"LABデータ使用: {config.use_lab}\n")
                f.write(f"訓練:検証比率: {config.train_val_split_ratio}:{1-config.train_val_split_ratio}\n")
                f.write(f"検証分割戦略: {config.validation_split_strategy}\n")
                if config.validation_split_strategy == 'stratified':
                    f.write(f"  層の数: {config.n_strata}\n")
                    f.write(f"  分割方法: {config.stratification_method}\n")
                f.write(f"データ拡張: {config.use_augmentation}\n")
                f.write(f"バッチサイズ: {config.batch_size}\n")
                f.write("\n高速化設定:\n")
                f.write(f"  AMP (自動混合精度): {config.use_amp}\n")
                f.write(f"  torch.compile: {config.use_compile}\n")
                f.write(f"  DataLoader workers: {config.num_workers}\n")
                f.write(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)\n")
                f.write(f"処理被験者数: {len(all_subjects_results)}/{len(config.subjects)}\n")
                f.write(f"平均訓練MAE: {avg_train_mae:.4f}±{std_train_mae:.4f}\n")
                f.write(f"平均訓練相関: {avg_train_corr:.4f}±{std_train_corr:.4f}\n")
                f.write(f"平均テストMAE: {avg_test_mae:.4f}±{std_test_mae:.4f}\n")
                f.write(f"平均テスト相関: {avg_test_corr:.4f}±{std_test_corr:.4f}\n")
                f.write(f"平均処理時間/被験者: {total_time/len(all_subjects_results):.1f}秒\n")
    
    print(f"\n{'='*60}")
    print("処理完了")
    print(f"結果保存先: {config.save_path}")
    print(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
