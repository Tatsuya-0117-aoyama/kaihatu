import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
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
from collections import defaultdict
import json
import time
from torch.amp import GradScaler, autocast
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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker

# ================================
# 設定クラス（拡張版）
# ================================
class Config:
    def __init__(self):
        # ================================
        # 解析タイプ選択（新機能）
        # ================================
        self.analysis_type = "inter_subject"  # "inter_subject"（個人間） or "intra_subject"（個人内）
        
        # ================================
        # モデルアーキテクチャ選択（新機能）
        # ================================
        self.model_architecture = "auto_balance"  # "auto_balance" or "cross_stitch"
        
        # ================================
        # 使用する指標の選択（新機能）
        # ================================
        # 利用可能な指標: ['CO', 'SV', 'HR_CO_SV', 'Cwk', 'Rp', 'Zao', 'I0', 'LIET', 'reDIA', 'reSYS']
        self.target_indicators = ['CO']  # デフォルトはCOのみ
        # self.target_indicators = ['CO', 'SV', 'HR_CO_SV']  # 複数指標の例
        
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
        self.use_amp = True
        self.use_compile = False
        self.num_workers = 0
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = None
        
        # ================================
        # LAB変換データ使用設定
        # ================================
        self.use_lab = False
        self.lab_filename = "_downsampled_1Hzver2.npy"
        
        # ================================
        # 個人間解析用設定（新機能）
        # ================================
        self.inter_subject_folds = 5  # 個人間解析の分割数
        
        # ================================
        # 個人内解析用設定（既存）
        # ================================
        self.train_val_split_ratio = 0.9
        self.validation_split_strategy = 'stratified'
        self.n_strata = 5
        self.stratification_method = 'quantile'
        
        # 被験者設定
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # タスクごとの色設定
        self.task_colors = {
            "t1-1": "#FF6B6B",
            "t2": "#4ECDC4",
            "t1-2": "#45B7D1",
            "t4": "#96CEB4",
            "t1-3": "#FECA57",
            "t5": "#DDA0DD"
        }
        
        # Fold毎の色設定（個人間解析用）
        self.fold_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        # ================================
        # モデル設定
        # ================================
        self.model_type = "2d"  # "3d" or "2d"
        
        # 使用チャンネル設定
        if self.use_lab:
            self.use_channel = 'RGB+LAB'
        else:
            self.use_channel = 'RGB'
        
        # チャンネル数の自動計算
        channel_map = {
            'R': 1, 'G': 1, 'B': 1,
            'RG': 2, 'GB': 2, 'RB': 2,
            'RGB': 3, 'LAB': 3, 'RGB+LAB': 6
        }
        self.num_channels = channel_map.get(self.use_channel, 3)
        
        # データ形状設定
        self.time_frames = 360
        self.height = 36
        self.width = 36
        self.input_shape = (self.time_frames, self.height, self.width, self.num_channels)
        
        # ================================
        # データ拡張設定
        # ================================
        self.use_augmentation = False
        
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
        # 学習設定
        # ================================
        if self.model_type == "3d":
            self.batch_size = 8
            self.epochs = 150
        elif self.model_type == "2d":
            self.batch_size = 32 if self.use_amp else 16
            self.epochs = 200
        else:
            self.batch_size = 16
            self.epochs = 150
        
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.patience = 30
        self.gradient_clip_val = 0.5
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Warmup設定
        self.warmup_epochs = 5
        self.warmup_lr_factor = 0.1
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.8
        self.loss_beta = 0.2
        
        # 学習率スケジューラー設定
        self.scheduler_type = "plateau"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 30
        self.scheduler_T_mult = 2
        
        # Early Stopping改善設定
        self.patience_improvement_threshold = 0.995
        self.min_delta = 0.0001
        
        # 表示設定
        self.verbose = True
        self.random_seed = 42

# ================================
# データ拡張関数
# ================================
class DataAugmentation:
    """データ拡張を管理するクラス"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)
    
    def random_crop(self, data):
        if np.random.random() > self.config.aug_probability or not self.config.crop_enabled:
            return data
        
        if data.ndim == 4:
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
        
        elif data.ndim == 3:
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
        if np.random.random() > self.config.aug_probability or not self.config.rotation_enabled:
            return data
        
        angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
        
        if data.ndim == 4:
            rotated = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[3]):
                    rotated[i, :, :, j] = scipy_rotate(data[i, :, :, j], angle, 
                                                       reshape=False, mode='reflect')
            return rotated
        
        elif data.ndim == 3:
            rotated = np.zeros_like(data)
            for j in range(data.shape[2]):
                rotated[:, :, j] = scipy_rotate(data[:, :, j], angle, 
                                               reshape=False, mode='reflect')
            return rotated
        
        return data
    
    def time_stretch_fast(self, rgb_np, factor):
        x = torch.from_numpy(rgb_np).permute(3,0,1,2).unsqueeze(0).float()
        T = x.shape[2]
        T2 = max(1, int(T*factor))
        
        with torch.no_grad():
            x = F.interpolate(x, size=(T2, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
            x = F.interpolate(x, size=(T, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        
        out = x.squeeze(0).permute(1,2,3,0).cpu().numpy()
        return out
    
    def time_stretch(self, rgb_data, signal_data=None):
        if np.random.random() > self.config.aug_probability or not self.config.time_stretch_enabled:
            return rgb_data, signal_data
        
        if rgb_data.ndim != 4:
            return rgb_data, signal_data
        
        stretch_factor = np.random.uniform(*self.config.time_stretch_range)
        
        rgb_stretched = self.time_stretch_fast(rgb_data, stretch_factor)
        
        if signal_data is not None and isinstance(signal_data, dict):
            signal_stretched = {}
            for key, sig in signal_data.items():
                if sig.ndim == 1:
                    t_original = len(sig)
                    t_stretched = int(t_original * stretch_factor)
                    
                    f_signal = interp1d(np.arange(t_original), sig, 
                                      kind='linear', fill_value='extrapolate')
                    sig_stretched = f_signal(np.linspace(0, t_original-1, t_stretched))
                    
                    f_signal_back = interp1d(np.arange(len(sig_stretched)), sig_stretched,
                                            kind='linear', fill_value='extrapolate')
                    sig_resampled = f_signal_back(np.linspace(0, len(sig_stretched)-1, t_original))
                    
                    sig_resampled = sig_resampled * stretch_factor
                    signal_stretched[key] = sig_resampled
                else:
                    signal_stretched[key] = sig
            return rgb_stretched, signal_stretched
        elif signal_data is not None and signal_data.ndim == 1:
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
        if not is_training or not self.config.use_augmentation:
            return rgb_data, signal_data
        
        rgb_data = self.random_crop(rgb_data)
        rgb_data = self.random_rotation(rgb_data)
        
        if rgb_data.ndim == 4 and self.config.time_stretch_enabled:
            rgb_data, signal_data = self.time_stretch(rgb_data, signal_data)
        
        rgb_data = self.brightness_contrast_adjust(rgb_data)
        
        return rgb_data, signal_data

# ================================
# 自動重み調整損失関数（Uncertainty Weighting）
# ================================
class AutoWeightedLoss(nn.Module):
    """各タスクの重みを自動学習する損失関数"""
    def __init__(self, num_tasks):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses_dict):
        weighted_losses = []
        weights_info = {}
        
        for i, (key, loss) in enumerate(losses_dict.items()):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            weights_info[key] = precision.item()
        
        total_loss = sum(weighted_losses)
        
        return total_loss, weights_info

# ================================
# カスタム損失関数（ベクトル化版）
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
        
        if pred.dim() == 2 and pred.size(1) > 1:
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
        
        if pred.dim() == 2 and pred.size(1) > 1:
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
# 層化サンプリング関数
# ================================
def stratified_sampling_split(task_rgb, task_signal, val_ratio=0.1, n_strata=5, method='quantile'):
    """信号値の分布を保持したまま訓練・検証データを分割"""
    
    # 信号が辞書の場合は最初の指標を使用
    if isinstance(task_signal, dict):
        first_key = list(task_signal.keys())[0]
        signal_for_stratification = task_signal[first_key]
    else:
        signal_for_stratification = task_signal
    
    n_samples = len(signal_for_stratification)
    
    signal_min = signal_for_stratification.min()
    signal_max = signal_for_stratification.max()
    signal_mean = signal_for_stratification.mean()
    signal_std = signal_for_stratification.std()
    
    if method == 'equal_range':
        bin_edges = np.linspace(signal_min, signal_max + 1e-10, n_strata + 1)
    elif method == 'quantile':
        quantiles = np.linspace(0, 1, n_strata + 1)
        bin_edges = np.quantile(signal_for_stratification, quantiles)
        bin_edges[-1] += 1e-10
    else:
        raise ValueError(f"Unknown stratification method: {method}")
    
    strata_assignment = np.digitize(signal_for_stratification, bin_edges) - 1
    
    train_indices = []
    val_indices = []
    
    strata_info = []
    
    for stratum_id in range(n_strata):
        stratum_mask = (strata_assignment == stratum_id)
        stratum_indices = np.where(stratum_mask)[0]
        
        if len(stratum_indices) == 0:
            continue
        
        stratum_signals = signal_for_stratification[stratum_indices]
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
    
    val_signals = signal_for_stratification[val_indices]
    train_signals = signal_for_stratification[train_indices]
    
    print(f"      信号値の分布確認:")
    print(f"        元データ: 平均={signal_mean:.3f}, 標準偏差={signal_std:.3f}")
    print(f"        訓練データ: 平均={train_signals.mean():.3f}, 標準偏差={train_signals.std():.3f}")
    print(f"        検証データ: 平均={val_signals.mean():.3f}, 標準偏差={val_signals.std():.3f}")
    
    # 複数指標の場合の処理
    if isinstance(task_signal, dict):
        train_signal = {k: v[train_indices] for k, v in task_signal.items()}
        val_signal = {k: v[val_indices] for k, v in task_signal.items()}
    else:
        train_signal = task_signal[train_indices]
        val_signal = task_signal[val_indices]
    
    return (task_rgb[train_indices], train_signal,
            task_rgb[val_indices], val_signal)

# ================================
# データセット（複数指標対応）
# ================================
class MultiIndicatorDataset(Dataset):
    """複数指標に対応したデータセット"""
    def __init__(self, rgb_data, signal_data_dict, model_type='3d', 
                 use_channel='RGB', config=None, is_training=True):
        """
        rgb_data: (N, T, H, W, C) 形状
        signal_data_dict: {indicator_name: (N, T) or (N,)} の辞書 or 単一配列
        """
        self.model_type = model_type
        self.use_channel = use_channel
        self.is_training = is_training
        self.config = config
        
        # データ拡張の初期化
        self.augmentation = DataAugmentation(config) if config else None
        
        # データを保存
        self.rgb_data_raw = rgb_data
        
        # signal_data_dictが辞書でない場合（単一指標）の処理
        if not isinstance(signal_data_dict, dict):
            signal_data_dict = {config.target_indicators[0]: signal_data_dict}
        
        self.signal_data_dict_raw = signal_data_dict
        
        # チャンネル選択を適用
        rgb_data_selected = select_channels(rgb_data, use_channel)
        self.rgb_data = torch.FloatTensor(rgb_data_selected)
        
        # 各指標のデータを処理
        self.signal_data_dict = {}
        for key, signal_data in signal_data_dict.items():
            if signal_data.ndim == 1:
                signal_data = np.repeat(signal_data[:, np.newaxis], rgb_data.shape[1], axis=1)
            self.signal_data_dict[key] = torch.FloatTensor(signal_data)
        
        self.indicator_names = list(signal_data_dict.keys())
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        # 元データを取得
        rgb = self.rgb_data_raw[idx]
        
        # 信号データを辞書として取得
        signals = {}
        for key in self.indicator_names:
            if self.signal_data_dict_raw[key].ndim > 1:
                signals[key] = self.signal_data_dict_raw[key][idx]
            else:
                signals[key] = self.signal_data_dict_raw[key][idx:idx+1].squeeze()
        
        # データ拡張を適用
        if self.augmentation and self.is_training:
            rgb, signals = self.augmentation.apply_augmentation(rgb, signals, self.is_training)
        
        # チャンネル選択
        rgb = select_channels(rgb, self.use_channel)
        
        # Tensorに変換
        rgb_tensor = torch.FloatTensor(rgb)
        
        signal_tensors = {}
        for key, signal in signals.items():
            if isinstance(signal, np.ndarray):
                signal_tensor = torch.FloatTensor(signal)
            else:
                signal_tensor = torch.FloatTensor([signal])
            
            if signal_tensor.dim() == 0:
                signal_tensor = signal_tensor.unsqueeze(0)
            if signal_tensor.size(0) == 1 and rgb_tensor.size(0) > 1:
                signal_tensor = signal_tensor.repeat(rgb_tensor.size(0))
            
            signal_tensors[key] = signal_tensor
        
        return rgb_tensor, signal_tensors

# ================================
# Cross-stitch層
# ================================
class CrossStitchLayer(nn.Module):
    """特徴を相互に交換・強化するCross-stitch層"""
    def __init__(self, num_branches):
        super().__init__()
        # 学習可能な結合行列
        self.stitch_matrix = nn.Parameter(torch.eye(num_branches) * 0.9 + 0.1 / num_branches)
    
    def forward(self, features_list):
        """
        features_list: [(B, T, D), ...] 各ブランチの特徴のリスト
        """
        # 特徴を結合
        stacked_features = torch.stack(features_list, dim=0)  # (num_branches, B, T, D)
        
        # Cross-stitch変換
        stitched_features = []
        for i in range(len(features_list)):
            weighted_sum = sum(self.stitch_matrix[i, j] * stacked_features[j] 
                             for j in range(len(features_list)))
            stitched_features.append(weighted_sum)
        
        return stitched_features

# ================================
# Self-Attention層
# ================================
class SelfAttention(nn.Module):
    """Self-Attention層"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
    
    def forward(self, x):
        """x: (B, T, D)"""
        B, T, D = x.shape
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attention, v)
        
        return out

# ================================
# 指標ブランチ
# ================================
class IndicatorBranch(nn.Module):
    """各指標用の1D CNN特徴抽出器"""
    def __init__(self, output_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, output_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """x: (B, T) -> (B, T, output_dim)"""
        # (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # (B, output_dim, T) -> (B, T, output_dim)
        x = x.transpose(1, 2)
        
        return x

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
        
        # Adaptive Spatial Global Average Pooling
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Final Conv
        self.conv_final = nn.Conv3d(64, 1, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        time_frames = x.size(1)
        
        x = x.permute(0, 4, 1, 2, 3)
        
        # ConvBlocks
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
        x = x.squeeze(-1).squeeze(-1)
        
        # 特徴を返す（統合モデル用）
        return x.transpose(1, 2)  # (B, T, 64)

# ================================
# CalibrationPhys準拠 PhysNet2DCNN (2D版)
# ================================
class PhysNet2DCNN_2D(nn.Module):
    """2D畳み込みを使用したPhysNet2DCNN"""
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
        
        # ConvBlock 4: 64 filters
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
        
        # Final layer
        self.fc = nn.Linear(32, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        batch_size, time_frames = x.size(0), x.size(1)
        
        x = x.view(batch_size * time_frames, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 3, 1, 2)
        
        # ConvBlocks
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
        
        # 時間次元の処理
        x = x.permute(0, 2, 1)
        x = self.temporal_conv1(x)
        x = self.temporal_bn1(x)
        x = self.temporal_elu1(x)
        
        x = self.temporal_conv2(x)
        x = self.temporal_bn2(x)
        x = self.temporal_elu2(x)
        
        x = x.permute(0, 2, 1)  # (B, T, 32)
        
        # 特徴を返す（統合モデル用）
        return x

# ================================
# 統合モデル（自動損失バランシング版）
# ================================
class MultiIndicatorModel_AutoBalance(nn.Module):
    """自動損失バランシングを使用する統合モデル"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.indicator_names = config.target_indicators
        self.num_indicators = len(self.indicator_names)
        
        # RGB特徴抽出器
        if config.model_type == "3d":
            self.rgb_branch = PhysNet2DCNN_3D(config.input_shape)
        else:
            self.rgb_branch = PhysNet2DCNN_2D(config.input_shape)
        
        # 各指標用のブランチ
        self.indicator_branches = nn.ModuleDict({
            name: IndicatorBranch(output_dim=32) 
            for name in self.indicator_names
        })
        
        # 統合層
        rgb_feature_dim = 64 if config.model_type == "3d" else 32
        feature_dim = rgb_feature_dim + 32 * self.num_indicators
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(128)
        )
        
        # 各指標用の出力層
        self.output_heads = nn.ModuleDict({
            name: nn.Linear(128, 1) 
            for name in self.indicator_names
        })
        
        # 自動重み学習
        self.auto_loss = AutoWeightedLoss(self.num_indicators)
    
    def forward(self, rgb_input, indicator_inputs):
        """
        rgb_input: (B, T, H, W, C)
        indicator_inputs: {name: (B, T)} の辞書
        """
        # RGB特徴抽出
        rgb_features = self.rgb_branch(rgb_input)  # (B, T, 64 or 32)
        
        # 各指標の特徴抽出
        indicator_features = []
        for name in self.indicator_names:
            if name in indicator_inputs:
                features = self.indicator_branches[name](indicator_inputs[name])  # (B, T, 32)
                indicator_features.append(features)
        
        # 特徴結合
        all_features = torch.cat([rgb_features] + indicator_features, dim=-1)
        
        # 統合層
        fused_features = self.fusion_layer(all_features)  # (B, T, 128)
        
        # 各指標の予測
        outputs = {}
        for name in self.indicator_names:
            output = self.output_heads[name](fused_features).squeeze(-1)  # (B, T)
            outputs[name] = output
        
        return outputs

# ================================
# 統合モデル（Cross-stitch版）
# ================================
class MultiIndicatorModel_CrossStitch(nn.Module):
    """Cross-stitchとAttentionを使用する統合モデル"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.indicator_names = config.target_indicators
        self.num_indicators = len(self.indicator_names)
        
        # RGB特徴抽出器
        if config.model_type == "3d":
            self.rgb_branch = PhysNet2DCNN_3D(config.input_shape)
        else:
            self.rgb_branch = PhysNet2DCNN_2D(config.input_shape)
        
        # 各指標用のブランチ
        self.indicator_branches = nn.ModuleDict({
            name: IndicatorBranch(output_dim=32) 
            for name in self.indicator_names
        })
        
        # Cross-stitch層
        self.cross_stitch = CrossStitchLayer(1 + self.num_indicators)
        
        # Self-Attention層
        rgb_feature_dim = 64 if config.model_type == "3d" else 32
        feature_dim = rgb_feature_dim + 32 * self.num_indicators
        self.self_attention = SelfAttention(feature_dim)
        
        # 統合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(128)
        )
        
        # 各指標用の出力層
        self.output_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for name in self.indicator_names
        })
        
        # 自動重み学習
        self.auto_loss = AutoWeightedLoss(self.num_indicators)
    
    def forward(self, rgb_input, indicator_inputs):
        """
        rgb_input: (B, T, H, W, C)
        indicator_inputs: {name: (B, T)} の辞書
        """
        # RGB特徴抽出
        rgb_features = self.rgb_branch(rgb_input)  # (B, T, 64 or 32)
        
        # 各指標の特徴抽出
        indicator_features_list = []
        for name in self.indicator_names:
            if name in indicator_inputs:
                features = self.indicator_branches[name](indicator_inputs[name])  # (B, T, 32)
                indicator_features_list.append(features)
        
        # Cross-stitch適用
        all_features_list = [rgb_features] + indicator_features_list
        stitched_features = self.cross_stitch(all_features_list)
        
        # 特徴結合
        all_features = torch.cat(stitched_features, dim=-1)
        
        # Self-Attention適用
        attended_features = self.self_attention(all_features)
        all_features = all_features + attended_features  # Residual connection
        
        # 統合層
        fused_features = self.fusion_layer(all_features)  # (B, T, 128)
        
        # 各指標の予測
        outputs = {}
        for name in self.indicator_names:
            output = self.output_heads[name](fused_features).squeeze(-1)  # (B, T)
            outputs[name] = output
        
        return outputs
# ================================
# モデル作成関数
# ================================
def create_model(config):
    """設定に基づいてモデルを作成"""
    
    if len(config.target_indicators) == 1:
        # 単一指標の場合は既存のモデルを使用
        if config.model_type == "3d":
            class SingleIndicatorModel3D(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.base_model = PhysNet2DCNN_3D(config.input_shape)
                    self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
                
                def forward(self, x):
                    features = self.base_model(x)  # (B, T, 64)
                    x = features.transpose(1, 2)  # (B, 64, T)
                    x = self.conv_final(x)  # (B, 1, T)
                    return x.squeeze(1)  # (B, T)
            
            model = SingleIndicatorModel3D(config)
            model_name = "PhysNet2DCNN_3D (単一指標)"
        else:
            class SingleIndicatorModel2D(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.base_model = PhysNet2DCNN_2D(config.input_shape)
                    self.fc = nn.Linear(32, 1)
                
                def forward(self, x):
                    features = self.base_model(x)  # (B, T, 32)
                    x = self.fc(features)  # (B, T, 1)
                    return x.squeeze(-1)  # (B, T)
            
            model = SingleIndicatorModel2D(config)
            model_name = "PhysNet2DCNN_2D (単一指標)"
    else:
        # 複数指標の場合
        if config.model_architecture == "auto_balance":
            model = MultiIndicatorModel_AutoBalance(config)
            model_name = "MultiIndicatorModel (自動損失バランシング)"
        elif config.model_architecture == "cross_stitch":
            model = MultiIndicatorModel_CrossStitch(config)
            model_name = "MultiIndicatorModel (Cross-stitch + Attention)"
        else:
            raise ValueError(f"Unknown model_architecture: {config.model_architecture}")
    
    # torch.compile対応
    if config.use_compile and torch.__version__.startswith("2."):
        if config.verbose:
            print(f"  PyTorch 2.0+検出: torch.compileでモデルを最適化します")
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            if config.verbose:
                print(f"  torch.compileに失敗しました: {e}")
                print("  通常モードで続行します")
    
    if config.verbose:
        print(f"\n選択モデル: {model_name}")
        print(f"対象指標: {config.target_indicators}")
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

# ================================
# データ読み込み関数（複数指標対応）
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み（複数指標対応）"""
    
    # RGBデータ読み込み
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません: {rgb_path}")
        return None, None
    
    rgb_data = np.load(rgb_path)
    print(f"  RGBデータ読み込み成功: {rgb_data.shape}")
    
    # LABデータの読み込み
    if config.use_lab:
        lab_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_downsampled_1Hzver2.npy")
        
        if os.path.exists(lab_path):
            lab_data = np.load(lab_path)
            print(f"  LABデータ読み込み成功: {lab_data.shape}")
            
            if rgb_data.shape == lab_data.shape:
                combined_data = np.concatenate([rgb_data, lab_data], axis=-1)
                print(f"  RGB+LAB結合データ: {combined_data.shape}")
                
                if lab_data.max() > 1.0:
                    combined_data[..., 3:] = combined_data[..., 3:] / 255.0
                
                rgb_data = combined_data
            else:
                print(f"警告: {subject}のRGBとLABデータの形状が一致しません")
                config.use_lab = False
                config.use_channel = 'RGB'
                config.num_channels = 3
        else:
            print(f"警告: {subject}のLABデータが見つかりません: {lab_path}")
            config.use_lab = False
            config.use_channel = 'RGB'
            config.num_channels = 3
    
    # データ形状の調整
    if rgb_data.ndim == 4:
        rgb_data = np.expand_dims(rgb_data, axis=1)
        rgb_data = np.repeat(rgb_data, config.time_frames, axis=1)
    elif rgb_data.ndim == 5:
        pass
    
    # 各指標のデータ読み込み
    signal_data_dict = {}
    
    for indicator in config.target_indicators:
        signal_data_list = []
        
        # 指標名からプレフィックスを生成
        if indicator == "HR_CO_SV":
            prefix = "HR_CO_SV_s2"
        else:
            prefix = f"{indicator}_s2"
        
        for task in config.tasks:
            signal_path = os.path.join(config.signal_base_path, subject, 
                                      indicator, 
                                      f"{prefix}_{task}.npy")
            if not os.path.exists(signal_path):
                print(f"警告: {subject}の{task}の{indicator}データが見つかりません")
                return None, None
            signal_data_list.append(np.load(signal_path))
        
        signal_data_dict[indicator] = np.concatenate(signal_data_list)
        print(f"  {indicator}データ読み込み成功: {signal_data_dict[indicator].shape}")
    
    # データの正規化
    if rgb_data[..., :3].max() > 1.0:
        rgb_data[..., :3] = rgb_data[..., :3] / 255.0
    
    return rgb_data, signal_data_dict

# ================================
# 個人間解析用データ準備関数
# ================================
def prepare_inter_subject_data(config):
    """個人間解析用にすべての被験者のデータを準備"""
    
    all_data = []
    subject_task_info = []
    
    for subject in config.subjects:
        rgb_data, signal_data_dict = load_data_single_subject(subject, config)
        
        if rgb_data is None or signal_data_dict is None:
            print(f"{subject}のデータ読み込み失敗。スキップします。")
            continue
        
        # タスクごとにデータを分割
        for task_idx, task in enumerate(config.tasks):
            start_idx = task_idx * config.task_duration
            end_idx = (task_idx + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signals = {
                key: signals[start_idx:end_idx] 
                for key, signals in signal_data_dict.items()
            }
            
            all_data.append({
                'rgb': task_rgb,
                'signals': task_signals,
                'subject': subject,
                'task': task,
                'subject_task': f"{subject}_{task}"
            })
            
            subject_task_info.append({
                'subject': subject,
                'task': task,
                'index': len(all_data) - 1
            })
    
    return all_data, subject_task_info

def create_balanced_folds(subject_task_info, num_folds=5):
    """各被験者のデータが均等に分散するようにfoldを作成"""
    
    # 被験者ごとにタスクをグループ化
    subject_tasks = defaultdict(list)
    for info in subject_task_info:
        subject_tasks[info['subject']].append(info['index'])
    
    # 各foldに割り当てるインデックス
    folds = [[] for _ in range(num_folds)]
    
    # 各被験者のタスクを各foldに均等に分配
    for subject, task_indices in subject_tasks.items():
        # タスクをシャッフル
        np.random.shuffle(task_indices)
        
        # 各foldに順番に割り当て
        for i, idx in enumerate(task_indices):
            fold_id = i % num_folds
            folds[fold_id].append(idx)
    
    # 各foldをシャッフル
    for fold in folds:
        np.random.shuffle(fold)
    
    return folds

# ================================
# 学習関数（複数指標対応）
# ================================
def train_model_multi_indicator(model, train_loader, val_loader, config, fold=None, save_name="model"):
    """複数指標対応の学習関数"""
    
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
    
    # スケジューラーの選択
    if config.scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=1e-6
        )
        scheduler_per_batch = False
    elif config.scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate * 10,
            epochs=config.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos'
        )
        scheduler_per_batch = True
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=False
        )
        scheduler_per_batch = False
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 各指標の最良予測値を保存
    best_predictions = {}
    best_targets = {}
    
    # 時間計測
    epoch_times = []
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
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
        train_predictions = defaultdict(list)
        train_targets = defaultdict(list)
        
        for rgb, signals_dict in train_loader:
            rgb = rgb.to(config.device)
            signals_dict = {k: v.to(config.device) for k, v in signals_dict.items()}
            
            optimizer.zero_grad()
            
            if config.use_amp:
                with autocast():
                    # 複数指標モデルの場合
                    if hasattr(model, 'auto_loss'):
                        outputs = model(rgb, signals_dict)
                        
                        # 各指標の損失を計算
                        losses_dict = {}
                        for name in config.target_indicators:
                            if name in outputs and name in signals_dict:
                                loss, _, _ = criterion(outputs[name], signals_dict[name])
                                losses_dict[name] = loss
                        
                        # 自動重み付け損失
                        total_loss, weights_info = model.auto_loss(losses_dict)
                    else:
                        # 単一指標モデル
                        output = model(rgb)
                        first_indicator = config.target_indicators[0]
                        total_loss, _, _ = criterion(output, signals_dict[first_indicator])
                        outputs = {first_indicator: output}
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 通常の学習（AMP無効時）
                if hasattr(model, 'auto_loss'):
                    outputs = model(rgb, signals_dict)
                    losses_dict = {}
                    for name in config.target_indicators:
                        if name in outputs and name in signals_dict:
                            loss, _, _ = criterion(outputs[name], signals_dict[name])
                            losses_dict[name] = loss
                    total_loss, weights_info = model.auto_loss(losses_dict)
                else:
                    output = model(rgb)
                    first_indicator = config.target_indicators[0]
                    total_loss, _, _ = criterion(output, signals_dict[first_indicator])
                    outputs = {first_indicator: output}
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
                optimizer.step()
            
            if scheduler_per_batch and epoch >= config.warmup_epochs:
                scheduler.step()
            
            train_loss += total_loss.item()
            
            # 予測値と真値を保存
            for name in config.target_indicators:
                if name in outputs:
                    if outputs[name].dim() == 2:
                        train_predictions[name].extend(outputs[name].mean(dim=1).detach().cpu().numpy())
                        train_targets[name].extend(signals_dict[name].mean(dim=1).detach().cpu().numpy())
                    else:
                        train_predictions[name].extend(outputs[name].detach().cpu().numpy())
                        train_targets[name].extend(signals_dict[name].detach().cpu().numpy())
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_predictions = defaultdict(list)
        val_targets = defaultdict(list)
        
        with torch.no_grad():
            for rgb, signals_dict in val_loader:
                rgb = rgb.to(config.device)
                signals_dict = {k: v.to(config.device) for k, v in signals_dict.items()}
                
                if config.use_amp:
                    with autocast():
                        if hasattr(model, 'auto_loss'):
                            outputs = model(rgb, signals_dict)
                            losses_dict = {}
                            for name in config.target_indicators:
                                if name in outputs and name in signals_dict:
                                    loss, _, _ = criterion(outputs[name], signals_dict[name])
                                    losses_dict[name] = loss
                            total_loss, _ = model.auto_loss(losses_dict)
                        else:
                            output = model(rgb)
                            first_indicator = config.target_indicators[0]
                            total_loss, _, _ = criterion(output, signals_dict[first_indicator])
                            outputs = {first_indicator: output}
                else:
                    if hasattr(model, 'auto_loss'):
                        outputs = model(rgb, signals_dict)
                        losses_dict = {}
                        for name in config.target_indicators:
                            if name in outputs and name in signals_dict:
                                loss, _, _ = criterion(outputs[name], signals_dict[name])
                                losses_dict[name] = loss
                        total_loss, _ = model.auto_loss(losses_dict)
                    else:
                        output = model(rgb)
                        first_indicator = config.target_indicators[0]
                        total_loss, _, _ = criterion(output, signals_dict[first_indicator])
                        outputs = {first_indicator: output}
                
                val_loss += total_loss.item()
                
                # 予測値と真値を保存
                for name in config.target_indicators:
                    if name in outputs:
                        if outputs[name].dim() == 2:
                            val_predictions[name].extend(outputs[name].mean(dim=1).cpu().numpy())
                            val_targets[name].extend(signals_dict[name].mean(dim=1).cpu().numpy())
                        else:
                            val_predictions[name].extend(outputs[name].cpu().numpy())
                            val_targets[name].extend(signals_dict[name].cpu().numpy())
        
        # メトリクス計算
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # スケジューラー更新（Warmup終了後）
        if not scheduler_per_batch and epoch >= config.warmup_epochs:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # モデル保存（改善判定を厳密化）
        improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss > 0 else 1
        if improvement > config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 最良時の予測値を保存
            best_predictions = {k: np.array(v) for k, v in val_predictions.items()}
            best_targets = {k: np.array(v) for k, v in val_targets.items()}
            
            # モデル保存
            save_dir = Path(config.save_path)
            if fold is not None:
                save_dir = save_dir / f"fold{fold+1}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'config': config
            }, save_dir / f'{save_name}.pth')
        else:
            patience_counter += 1
        
        # エポック時間計算
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # ログ出力
        if config.verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch < config.warmup_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            if epoch < config.warmup_epochs:
                print(f"    [Warmup] Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e} Time: {epoch_time:.1f}s")
            else:
                print(f"    Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e} Time: {epoch_time:.1f}s")
            print(f"      Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 各指標のメトリクス表示
            for name in config.target_indicators:
                if name in train_predictions and name in val_predictions:
                    train_corr = np.corrcoef(train_predictions[name], train_targets[name])[0, 1]
                    val_corr = np.corrcoef(val_predictions[name], val_targets[name])[0, 1]
                    val_mae = mean_absolute_error(val_targets[name], val_predictions[name])
                    print(f"      {name}: Train Corr={train_corr:.4f}, Val MAE={val_mae:.4f}, Val Corr={val_corr:.4f}")
            
            # AMP使用時はスケール値も表示
            if config.use_amp and scaler is not None:
                print(f"      AMP Scale: {scaler.get_scale():.1f}")
        
        # Early Stopping
        if patience_counter >= config.patience:
            if config.verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # ベストモデル読み込み
    checkpoint = torch.load(save_dir / f'{save_name}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, best_predictions, best_targets

# ================================
# 評価関数（複数指標対応）
# ================================
def evaluate_model_multi_indicator(model, test_loader, config):
    """複数指標対応の評価関数"""
    model.eval()
    
    predictions = defaultdict(list)
    targets = defaultdict(list)
    waveform_predictions = defaultdict(list)
    waveform_targets = defaultdict(list)
    
    with torch.no_grad():
        for rgb, signals_dict in test_loader:
            rgb = rgb.to(config.device)
            signals_dict = {k: v.to(config.device) for k, v in signals_dict.items()}
            
            if config.use_amp:
                with autocast():
                    if hasattr(model, 'auto_loss'):
                        outputs = model(rgb, signals_dict)
                    else:
                        output = model(rgb)
                        first_indicator = config.target_indicators[0]
                        outputs = {first_indicator: output}
            else:
                if hasattr(model, 'auto_loss'):
                    outputs = model(rgb, signals_dict)
                else:
                    output = model(rgb)
                    first_indicator = config.target_indicators[0]
                    outputs = {first_indicator: output}
            
            # 予測値と真値を保存
            for name in config.target_indicators:
                if name in outputs:
                    if outputs[name].dim() == 2:
                        # 波形全体を保存
                        waveform_predictions[name].append(outputs[name].cpu().numpy())
                        waveform_targets[name].append(signals_dict[name].cpu().numpy())
                        # 平均値も計算
                        predictions[name].extend(outputs[name].mean(dim=1).cpu().numpy())
                        targets[name].extend(signals_dict[name].mean(dim=1).cpu().numpy())
                    else:
                        predictions[name].extend(outputs[name].cpu().numpy())
                        targets[name].extend(signals_dict[name].cpu().numpy())
    
    # 各指標のメトリクス計算
    results = {}
    for name in config.target_indicators:
        if name in predictions:
            pred_array = np.array(predictions[name])
            target_array = np.array(targets[name])
            
            mae = mean_absolute_error(target_array, pred_array)
            rmse = np.sqrt(np.mean((target_array - pred_array) ** 2))
            corr, p_value = pearsonr(target_array, pred_array)
            
            ss_res = np.sum((target_array - pred_array) ** 2)
            ss_tot = np.sum((target_array - np.mean(target_array)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'corr': corr,
                'r2': r2,
                'p_value': p_value,
                'predictions': pred_array,
                'targets': target_array
            }
            
            # 波形データがある場合は追加
            if name in waveform_predictions and waveform_predictions[name]:
                results[name]['waveform_predictions'] = np.concatenate(waveform_predictions[name], axis=0)
                results[name]['waveform_targets'] = np.concatenate(waveform_targets[name], axis=0)
    
    return results

# ================================
# 個人間5分割交差検証
# ================================
def inter_subject_cross_validation(config):
    """個人間5分割交差検証"""
    
    print("\n" + "="*60)
    print("個人間5分割交差検証開始")
    print("="*60)
    
    # 全体の処理時間計測
    cv_start_time = time.time()
    
    # 全データ準備
    all_data, subject_task_info = prepare_inter_subject_data(config)
    
    if len(all_data) == 0:
        print("データが読み込めませんでした。")
        return
    
    print(f"総データ数: {len(all_data)}個 ({len(set([info['subject'] for info in subject_task_info]))}人 × {len(config.tasks)}タスク)")
    
    # バランスの取れたfold作成
    folds = create_balanced_folds(subject_task_info, config.inter_subject_folds)
    
    # fold情報を保存
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fold_info = {}
    for fold_id, fold_indices in enumerate(folds):
        fold_info[f"fold{fold_id+1}"] = []
        for idx in fold_indices:
            data_info = all_data[idx]
            fold_info[f"fold{fold_id+1}"].append({
                'subject': data_info['subject'],
                'task': data_info['task'],
                'index': idx
            })
    
    # fold情報をJSONとして保存
    with open(save_dir / 'fold_division.json', 'w', encoding='utf-8') as f:
        json.dump(fold_info, f, ensure_ascii=False, indent=2)
    
    # fold情報をテキストファイルにも保存
    with open(save_dir / 'fold_division.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("個人間5分割交差検証 - データ分割情報\n")
        f.write("="*60 + "\n\n")
        
        for fold_id, fold_indices in enumerate(folds):
            f.write(f"【Fold {fold_id+1}】 ({len(fold_indices)}個のデータ)\n")
            
            # 被験者ごとにグループ化して表示
            subject_tasks_in_fold = defaultdict(list)
            for idx in fold_indices:
                data_info = all_data[idx]
                subject_tasks_in_fold[data_info['subject']].append(data_info['task'])
            
            for subject in sorted(subject_tasks_in_fold.keys()):
                tasks = subject_tasks_in_fold[subject]
                f.write(f"  {subject}: {', '.join(tasks)} ({len(tasks)}タスク)\n")
            f.write("\n")
    
    # 交差検証実行
    fold_results = []
    all_test_predictions = defaultdict(list)
    all_test_targets = defaultdict(list)
    all_test_fold_indices = defaultdict(list)
    
    seed_worker = set_all_seeds(config.random_seed)
    
    for fold_id in range(config.inter_subject_folds):
        fold_start_time = time.time()
        
        print(f"\n【Fold {fold_id+1}/{config.inter_subject_folds}】")
        
        # テストデータのインデックス
        test_indices = folds[fold_id]
        
        # 訓練データのインデックス（他のすべてのfold）
        train_indices = []
        for other_fold_id in range(config.inter_subject_folds):
            if other_fold_id != fold_id:
                train_indices.extend(folds[other_fold_id])
        
        # 訓練データから検証データを分離（10%）
        np.random.shuffle(train_indices)
        val_size = int(len(train_indices) * 0.1)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        print(f"  データ分割: 訓練={len(train_indices)}, 検証={len(val_indices)}, テスト={len(test_indices)}")
        
        # データセット作成
        train_rgb = np.concatenate([all_data[i]['rgb'] for i in train_indices])
        train_signals = defaultdict(list)
        for i in train_indices:
            for key, value in all_data[i]['signals'].items():
                train_signals[key].append(value)
        train_signals = {k: np.concatenate(v) for k, v in train_signals.items()}
        
        val_rgb = np.concatenate([all_data[i]['rgb'] for i in val_indices])
        val_signals = defaultdict(list)
        for i in val_indices:
            for key, value in all_data[i]['signals'].items():
                val_signals[key].append(value)
        val_signals = {k: np.concatenate(v) for k, v in val_signals.items()}
        
        test_rgb = np.concatenate([all_data[i]['rgb'] for i in test_indices])
        test_signals = defaultdict(list)
        for i in test_indices:
            for key, value in all_data[i]['signals'].items():
                test_signals[key].append(value)
        test_signals = {k: np.concatenate(v) for k, v in test_signals.items()}
        
        # データローダー作成
        train_dataset = MultiIndicatorDataset(
            train_rgb, train_signals, config.model_type,
            config.use_channel, config, is_training=True
        )
        val_dataset = MultiIndicatorDataset(
            val_rgb, val_signals, config.model_type,
            config.use_channel, config, is_training=False
        )
        test_dataset = MultiIndicatorDataset(
            test_rgb, test_signals, config.model_type,
            config.use_channel, config, is_training=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
        )
        
        # モデル作成と学習
        model = create_model(config)
        model, val_preds, val_targets = train_model_multi_indicator(
            model, train_loader, val_loader, config, fold_id, f"model_fold{fold_id+1}"
        )
        
        # テスト評価
        test_results = evaluate_model_multi_indicator(model, test_loader, config)
        
        fold_time = time.time() - fold_start_time
        
        # 結果保存
        fold_result = {
            'fold': fold_id + 1,
            'val_predictions': val_preds,
            'val_targets': val_targets,
            'test_results': test_results
        }
        fold_results.append(fold_result)
        
        # 全テストデータ集約
        for name in config.target_indicators:
            if name in test_results:
                all_test_predictions[name].extend(test_results[name]['predictions'])
                all_test_targets[name].extend(test_results[name]['targets'])
                all_test_fold_indices[name].extend([fold_id] * len(test_results[name]['predictions']))
        
        # Fold結果のプロット
        plot_fold_results_multi_indicator(fold_result, fold_id, save_dir, config)
        
        # メトリクス表示
        print(f"  Fold {fold_id+1} 結果 (処理時間: {fold_time:.1f}秒):")
        for name in config.target_indicators:
            if name in test_results:
                print(f"    {name}: MAE={test_results[name]['mae']:.4f}, RMSE={test_results[name]['rmse']:.4f}, Corr={test_results[name]['corr']:.4f}, R²={test_results[name]['r2']:.4f}")
        
        # CSV保存
        fold_metrics = []
        for name in config.target_indicators:
            if name in test_results:
                fold_metrics.append({
                    'Fold': fold_id + 1,
                    'Indicator': name,
                    'MAE': test_results[name]['mae'],
                    'RMSE': test_results[name]['rmse'],
                    'Correlation': test_results[name]['corr'],
                    'R2': test_results[name]['r2']
                })
        fold_df = pd.DataFrame(fold_metrics)
        fold_df.to_csv(save_dir / f'fold{fold_id+1}' / 'metrics.csv', index=False)
    
    # 全体結果のプロット
    plot_overall_results_multi_indicator(
        fold_results, all_test_predictions, all_test_targets, 
        all_test_fold_indices, save_dir, config
    )
    
    cv_total_time = time.time() - cv_start_time
    
    # 最終結果の表示とCSV保存
    print("\n" + "="*60)
    print("個人間5分割交差検証 - 最終結果")
    print("="*60)
    
    final_results = []
    for name in config.target_indicators:
        if name in all_test_predictions:
            all_preds = np.array(all_test_predictions[name])
            all_targets = np.array(all_test_targets[name])
            
            final_mae = mean_absolute_error(all_targets, all_preds)
            final_rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
            final_corr = np.corrcoef(all_targets, all_preds)[0, 1]
            
            ss_res = np.sum((all_targets - all_preds) ** 2)
            ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
            final_r2 = 1 - (ss_res / ss_tot)
            
            print(f"{name}:")
            print(f"  全体MAE: {final_mae:.4f}")
            print(f"  全体RMSE: {final_rmse:.4f}")
            print(f"  全体相関: {final_corr:.4f}")
            print(f"  全体R²: {final_r2:.4f}")
            
            final_results.append({
                'Indicator': name,
                'MAE': final_mae,
                'RMSE': final_rmse,
                'Correlation': final_corr,
                'R2': final_r2
            })
    
    print(f"\n処理時間: {cv_total_time:.1f}秒 ({cv_total_time/60:.1f}分)")
    
    # 最終結果をCSVに保存
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(save_dir / 'final_results.csv', index=False)
    
    # 設定情報を保存
    save_config_summary(config, save_dir, cv_total_time, "個人間5分割交差検証")

# ================================
# プロット関数（複数指標対応）
# ================================
def plot_fold_results_multi_indicator(fold_result, fold_id, save_dir, config):
    """Fold結果のプロット（複数指標対応）"""
    
    fold_dir = save_dir / f"fold{fold_id+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # 各指標ごとにプロット
    for indicator_name in config.target_indicators:
        if indicator_name not in fold_result['test_results']:
            continue
        
        test_result = fold_result['test_results'][indicator_name]
        
        # 散布図
        plt.figure(figsize=(10, 8))
        color = config.fold_colors[fold_id % len(config.fold_colors)]
        plt.scatter(test_result['targets'], test_result['predictions'],
                   alpha=0.6, s=20, color=color, label=f'Fold {fold_id+1}')
        
        min_val = min(test_result['targets'].min(), test_result['predictions'].min())
        max_val = max(test_result['targets'].max(), test_result['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f'{indicator_name}推定モデル - Fold {fold_id+1} - MAE: {test_result["mae"]:.3f}, Corr: {test_result["corr"]:.3f}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fold_dir / f'{indicator_name}_test_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 訓練データもプロット
        if indicator_name in fold_result['val_predictions']:
            plt.figure(figsize=(10, 8))
            plt.scatter(fold_result['val_targets'][indicator_name], 
                       fold_result['val_predictions'][indicator_name],
                       alpha=0.5, s=10, color='gray', label='検証データ')
            
            min_val = min(fold_result['val_targets'][indicator_name].min(), 
                         fold_result['val_predictions'][indicator_name].min())
            max_val = max(fold_result['val_targets'][indicator_name].max(), 
                         fold_result['val_predictions'][indicator_name].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            val_mae = mean_absolute_error(fold_result['val_targets'][indicator_name], 
                                         fold_result['val_predictions'][indicator_name])
            val_corr = np.corrcoef(fold_result['val_targets'][indicator_name], 
                                  fold_result['val_predictions'][indicator_name])[0, 1]
            
            plt.xlabel('真値')
            plt.ylabel('予測値')
            plt.title(f'{indicator_name}推定モデル - Fold {fold_id+1} 検証データ - MAE: {val_mae:.3f}, Corr: {val_corr:.3f}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fold_dir / f'{indicator_name}_val_scatter.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 波形プロット（もしあれば）
        if 'waveform_predictions' in test_result and test_result['waveform_predictions'] is not None:
            # 最初の5サンプルの波形を表示
            n_samples_to_plot = min(5, test_result['waveform_predictions'].shape[0])
            
            fig, axes = plt.subplots(n_samples_to_plot, 1, figsize=(16, 3*n_samples_to_plot))
            if n_samples_to_plot == 1:
                axes = [axes]
            
            for i in range(n_samples_to_plot):
                axes[i].plot(test_result['waveform_targets'][i], 'b-', label='真値', alpha=0.7, linewidth=1)
                axes[i].plot(test_result['waveform_predictions'][i], 
                           color=color, linestyle='-', label='予測', alpha=0.7, linewidth=1)
                axes[i].set_xlabel('時間 (秒)')
                axes[i].set_ylabel('信号値')
                axes[i].set_title(f'{indicator_name} - サンプル {i+1}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(fold_dir / f'{indicator_name}_waveforms.png', dpi=150, bbox_inches='tight')
            plt.close()

def plot_overall_results_multi_indicator(fold_results, all_predictions, all_targets, 
                                        all_fold_indices, save_dir, config):
    """全体結果のプロット（複数指標対応）"""
    
    # 各指標ごとにプロット
    for indicator_name in config.target_indicators:
        if indicator_name not in all_predictions:
            continue
        
        # 全体散布図（Foldごとに色分け）
        plt.figure(figsize=(14, 10))
        
        # 各Foldのデータを色分けしてプロット
        for fold_id in range(config.inter_subject_folds):
            fold_mask = np.array(all_fold_indices[indicator_name]) == fold_id
            if np.any(fold_mask):
                fold_preds = np.array(all_predictions[indicator_name])[fold_mask]
                fold_targets = np.array(all_targets[indicator_name])[fold_mask]
                color = config.fold_colors[fold_id % len(config.fold_colors)]
                plt.scatter(fold_targets, fold_preds, alpha=0.4, s=8, 
                          color=color, label=f'Fold {fold_id+1}')
        
        # 対角線
        all_min = np.array(all_targets[indicator_name]).min()
        all_max = np.array(all_targets[indicator_name]).max()
        plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.7)
        
        # メトリクス計算
        all_preds = np.array(all_predictions[indicator_name])
        all_targs = np.array(all_targets[indicator_name])
        overall_mae = mean_absolute_error(all_targs, all_preds)
        overall_corr = np.corrcoef(all_targs, all_preds)[0, 1]
        overall_rmse = np.sqrt(np.mean((all_targs - all_preds) ** 2))
        
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f'{indicator_name}推定モデル - 全体結果 - MAE: {overall_mae:.3f}, RMSE: {overall_rmse:.3f}, Corr: {overall_corr:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{indicator_name}_overall.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Foldごとのパフォーマンス比較
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        fold_maes = []
        fold_corrs = []
        
        for fold_id in range(config.inter_subject_folds):
            fold_mask = np.array(all_fold_indices[indicator_name]) == fold_id
            if np.any(fold_mask):
                fold_preds = np.array(all_predictions[indicator_name])[fold_mask]
                fold_targets = np.array(all_targets[indicator_name])[fold_mask]
                
                fold_mae = mean_absolute_error(fold_targets, fold_preds)
                fold_corr = np.corrcoef(fold_targets, fold_preds)[0, 1]
                
                fold_maes.append(fold_mae)
                fold_corrs.append(fold_corr)
        
        x = np.arange(len(fold_maes))
        colors = [config.fold_colors[i % len(config.fold_colors)] for i in range(len(fold_maes))]
        
        # MAE比較
        bars1 = ax1.bar(x, fold_maes, color=colors)
        ax1.axhline(y=overall_mae, color='r', linestyle='--', label=f'平均: {overall_mae:.3f}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('MAE')
        ax1.set_title(f'{indicator_name} - Foldごとの MAE')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_maes))])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 相関係数比較
        bars2 = ax2.bar(x, fold_corrs, color=colors)
        ax2.axhline(y=overall_corr, color='r', linestyle='--', label=f'平均: {overall_corr:.3f}')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('相関係数')
        ax2.set_title(f'{indicator_name} - Foldごとの相関係数')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_corrs))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{indicator_name}_fold_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

def save_config_summary(config, save_dir, total_time, analysis_name):
    """設定情報をテキストファイルに保存"""
    
    with open(save_dir / 'config_summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"{analysis_name} - 実験設定\n")
        f.write("="*60 + "\n")
        f.write(f"実行日時: {config.timestamp}\n")
        f.write(f"解析タイプ: {config.analysis_type}\n")
        f.write(f"モデルアーキテクチャ: {config.model_architecture}\n")
        f.write(f"対象指標: {config.target_indicators}\n")
        f.write(f"モデルタイプ: {config.model_type}\n")
        f.write(f"使用チャンネル: {config.use_channel}\n")
        f.write(f"LABデータ使用: {config.use_lab}\n")
        
        if config.analysis_type == "intra_subject":
            f.write(f"訓練:検証比率: {config.train_val_split_ratio}:{1-config.train_val_split_ratio}\n")
            f.write(f"検証分割戦略: {config.validation_split_strategy}\n")
        else:
            f.write(f"交差検証Fold数: {config.inter_subject_folds}\n")
        
        f.write(f"データ拡張: {config.use_augmentation}\n")
        f.write(f"バッチサイズ: {config.batch_size}\n")
        
        f.write("\n高速化設定:\n")
        f.write(f"  AMP (自動混合精度): {config.use_amp}\n")
        f.write(f"  torch.compile: {config.use_compile}\n")
        f.write(f"  DataLoader workers: {config.num_workers}\n")
        f.write(f"  Pin memory: {config.pin_memory}\n")
        f.write(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}\n")
        
        f.write(f"\n学習設定:\n")
        f.write(f"  損失関数: {config.loss_type}\n")
        f.write(f"  Warmupエポック: {config.warmup_epochs}\n")
        f.write(f"  学習率: {config.learning_rate}\n")
        f.write(f"  エポック数: {config.epochs}\n")
        f.write(f"  勾配クリッピング: {config.gradient_clip_val}\n")
        f.write(f"  乱数シード: {config.random_seed}\n")
        
        f.write(f"\n処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)\n")
        
        # GPU情報
        if torch.cuda.is_available():
            f.write(f"\nGPU情報:\n")
            f.write(f"  デバイス名: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  メモリ容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
            f.write(f"  PyTorchバージョン: {torch.__version__}\n")
            f.write(f"  CUDAバージョン: {torch.version.cuda}\n")

# ================================
# 個人内6分割交差検証（完全版）
# ================================
def intra_subject_cross_validation(config):
    """個人内6分割交差検証（完全実装）"""
    
    print("\n" + "="*60)
    print("個人内6分割交差検証開始")
    print("="*60)
    
    # 全体の処理時間計測
    total_start_time = time.time()
    
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
            rgb_data, signal_data_dict = load_data_single_subject(subject, config)
            
            if rgb_data is None or signal_data_dict is None:
                print(f"  {subject}のデータ読み込み失敗。スキップします。")
                continue
            
            print(f"  データ形状: RGB={rgb_data.shape}, Signal={list(signal_data_dict.values())[0].shape}")
            
            # 6分割交差検証実行
            fold_results, all_test_predictions, all_test_targets, all_test_tasks = task_cross_validation_multi_indicator(
                rgb_data, signal_data_dict, config, subject, subject_save_dir
            )
            
            # 被験者全体のサマリープロット（複数指標対応）
            metrics = plot_subject_summary_multi_indicator(
                fold_results, all_test_predictions, all_test_targets, all_test_tasks,
                subject, subject_save_dir, config
            )
            
            # 結果保存
            subject_result = {
                'subject': subject,
                'fold_results': fold_results,
                'all_test_predictions': all_test_predictions,
                'all_test_targets': all_test_targets,
                'all_test_tasks': all_test_tasks,
                'metrics': metrics
            }
            all_subjects_results.append(subject_result)
            
            subject_time = time.time() - subject_start_time
            
            print(f"\n  {subject} 完了 (処理時間: {subject_time:.1f}秒):")
            for indicator in config.target_indicators:
                if indicator in metrics:
                    print(f"    {indicator}: 訓練MAE={metrics[indicator]['train_mae']:.4f}, "
                          f"訓練Corr={metrics[indicator]['train_corr']:.4f}, "
                          f"テストMAE={metrics[indicator]['test_mae']:.4f}, "
                          f"テストCorr={metrics[indicator]['test_corr']:.4f}")
            
            # 結果をCSVファイルに保存
            results_data = []
            for indicator in config.target_indicators:
                if indicator in metrics:
                    results_data.append({
                        'Subject': subject,
                        'Indicator': indicator,
                        'Train_MAE': metrics[indicator]['train_mae'],
                        'Train_Corr': metrics[indicator]['train_corr'],
                        'Test_MAE': metrics[indicator]['test_mae'],
                        'Test_Corr': metrics[indicator]['test_corr'],
                        'Processing_Time_Sec': subject_time
                    })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(subject_save_dir / 'results_summary.csv', index=False)
            
            # 各Foldの結果も保存
            fold_data = []
            for fold_result in fold_results:
                for indicator in config.target_indicators:
                    if indicator in fold_result['test_results']:
                        fold_data.append({
                            'Fold': fold_result['fold'],
                            'Test_Task': fold_result['test_task'],
                            'Indicator': indicator,
                            'Train_MAE': fold_result['train_mae'].get(indicator, np.nan),
                            'Train_Corr': fold_result['train_corr'].get(indicator, np.nan),
                            'Test_MAE': fold_result['test_results'][indicator]['mae'],
                            'Test_Corr': fold_result['test_results'][indicator]['corr']
                        })
            
            fold_df = pd.DataFrame(fold_data)
            fold_df.to_csv(subject_save_dir / 'fold_results.csv', index=False)
            
        except Exception as e:
            print(f"  {subject}でエラー発生: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 全被験者のサマリープロット（統合版）
    if all_subjects_results:
        print(f"\n{'='*60}")
        print("全被験者サマリー作成中...")
        plot_all_subjects_summary_unified_multi_indicator(all_subjects_results, config)
        
        # 統計サマリー
        for indicator in config.target_indicators:
            mae_list = []
            corr_list = []
            
            for subject_result in all_subjects_results:
                if indicator in subject_result['metrics']:
                    mae_list.append(subject_result['metrics'][indicator]['test_mae'])
                    corr_list.append(subject_result['metrics'][indicator]['test_corr'])
            
            if mae_list:
                avg_test_mae = np.mean(mae_list)
                std_test_mae = np.std(mae_list)
                avg_test_corr = np.mean(corr_list)
                std_test_corr = np.std(corr_list)
                
                print(f"\n{indicator} - 全被験者平均結果:")
                print(f"  テストMAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")
                print(f"  テスト相関: {avg_test_corr:.4f} ± {std_test_corr:.4f}")
        
        # 全体結果をCSVファイルに保存
        all_results_data = []
        for subject_result in all_subjects_results:
            for indicator in config.target_indicators:
                if indicator in subject_result['metrics']:
                    all_results_data.append({
                        'Subject': subject_result['subject'],
                        'Indicator': indicator,
                        'Train_MAE': subject_result['metrics'][indicator]['train_mae'],
                        'Train_Corr': subject_result['metrics'][indicator]['train_corr'],
                        'Test_MAE': subject_result['metrics'][indicator]['test_mae'],
                        'Test_Corr': subject_result['metrics'][indicator]['test_corr']
                    })
        
        all_results_df = pd.DataFrame(all_results_data)
        
        # 平均と標準偏差を追加
        for indicator in config.target_indicators:
            indicator_df = all_results_df[all_results_df['Indicator'] == indicator]
            if not indicator_df.empty:
                mean_row = pd.DataFrame({
                    'Subject': [f'Mean_{indicator}'],
                    'Indicator': [indicator],
                    'Train_MAE': [indicator_df['Train_MAE'].mean()],
                    'Train_Corr': [indicator_df['Train_Corr'].mean()],
                    'Test_MAE': [indicator_df['Test_MAE'].mean()],
                    'Test_Corr': [indicator_df['Test_Corr'].mean()]
                })
                
                std_row = pd.DataFrame({
                    'Subject': [f'Std_{indicator}'],
                    'Indicator': [indicator],
                    'Train_MAE': [indicator_df['Train_MAE'].std()],
                    'Train_Corr': [indicator_df['Train_Corr'].std()],
                    'Test_MAE': [indicator_df['Test_MAE'].std()],
                    'Test_Corr': [indicator_df['Test_Corr'].std()]
                })
                
                all_results_df = pd.concat([all_results_df, mean_row, std_row], ignore_index=True)
        
        save_dir = Path(config.save_path)
        all_results_df.to_csv(save_dir / 'all_subjects_results.csv', index=False)
        
        # 総処理時間
        total_time = time.time() - total_start_time
        
        # 設定情報も保存
        save_config_summary(config, save_dir, total_time, "個人内6分割交差検証")
        
        print(f"\n処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")

def task_cross_validation_multi_indicator(rgb_data, signal_data_dict, config, subject, subject_save_dir):
    """タスクごとの6分割交差検証（複数指標対応）"""
    
    fold_results = []
    all_test_predictions = defaultdict(list)
    all_test_targets = defaultdict(list)
    all_test_task_indices = []
    all_test_tasks = []
    
    # 乱数シードを設定
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    seed_worker = set_all_seeds(config.random_seed)
    
    # 交差検証開始時間
    cv_start_time = time.time()
    
    for fold, test_task in enumerate(config.tasks):
        fold_start_time = time.time()
        
        if config.verbose:
            print(f"\n  Fold {fold+1}/6 - テストタスク: {test_task}")
            print(f"    検証データ分割戦略: {config.validation_split_strategy}")
        
        # タスクごとにデータを分割
        train_rgb_list = []
        train_signal_dict_list = defaultdict(list)
        val_rgb_list = []
        val_signal_dict_list = defaultdict(list)
        test_rgb_list = []
        test_signal_dict_list = defaultdict(list)
        
        for i, task in enumerate(config.tasks):
            start_idx = i * config.task_duration
            end_idx = (i + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signals = {
                key: signals[start_idx:end_idx]
                for key, signals in signal_data_dict.items()
            }
            
            if task == test_task:
                # テストデータ
                test_rgb_list.append(task_rgb)
                for key, value in task_signals.items():
                    test_signal_dict_list[key].append(value)
                all_test_task_indices.extend([i] * config.task_duration)
                all_test_tasks.extend([test_task] * config.task_duration)
            else:
                # 分割戦略に応じて訓練・検証データを分離
                if config.validation_split_strategy == 'stratified':
                    # 層化サンプリング
                    train_rgb, train_signals, val_rgb, val_signals = stratified_sampling_split(
                        task_rgb, 
                        task_signals,
                        val_ratio=(1 - config.train_val_split_ratio),
                        n_strata=config.n_strata,
                        method=config.stratification_method
                    )
                    train_rgb_list.append(train_rgb)
                    for key, value in train_signals.items():
                        train_signal_dict_list[key].append(value)
                    val_rgb_list.append(val_rgb)
                    for key, value in val_signals.items():
                        val_signal_dict_list[key].append(value)
                else:
                    # 既存の方法（各タスクの最後10%）
                    val_size = int(config.task_duration * (1 - config.train_val_split_ratio))
                    val_start_idx = config.task_duration - val_size
                    
                    train_rgb_list.append(task_rgb[:val_start_idx])
                    val_rgb_list.append(task_rgb[val_start_idx:])
                    for key, value in task_signals.items():
                        train_signal_dict_list[key].append(value[:val_start_idx])
                        val_signal_dict_list[key].append(value[val_start_idx:])
        
        # データ結合
        train_rgb = np.concatenate(train_rgb_list)
        train_signals = {k: np.concatenate(v) for k, v in train_signal_dict_list.items()}
        val_rgb = np.concatenate(val_rgb_list)
        val_signals = {k: np.concatenate(v) for k, v in val_signal_dict_list.items()}
        test_rgb = np.concatenate(test_rgb_list)
        test_signals = {k: np.concatenate(v) for k, v in test_signal_dict_list.items()}
        
        if config.verbose:
            print(f"    データサイズ - 訓練: {len(train_rgb)}, 検証: {len(val_rgb)}, テスト: {len(test_rgb)}")
        
        # データローダー作成
        train_dataset = MultiIndicatorDataset(
            train_rgb, train_signals, config.model_type,
            config.use_channel, config, is_training=True
        )
        val_dataset = MultiIndicatorDataset(
            val_rgb, val_signals, config.model_type,
            config.use_channel, config, is_training=False
        )
        test_dataset = MultiIndicatorDataset(
            test_rgb, test_signals, config.model_type,
            config.use_channel, config, is_training=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
        )
        
        # モデル作成
        model = create_model(config)
        
        # モデル学習
        model, train_preds, train_targets = train_model_multi_indicator(
            model, train_loader, val_loader, config, fold, subject
        )
        
        # 評価
        test_results = evaluate_model_multi_indicator(model, test_loader, config)
        
        fold_time = time.time() - fold_start_time
        
        if config.verbose:
            print(f"    Fold処理時間: {fold_time:.1f}秒")
            for indicator in config.target_indicators:
                if indicator in test_results:
                    print(f"    {indicator}: Test MAE={test_results[indicator]['mae']:.4f}, Corr={test_results[indicator]['corr']:.4f}")
        
        # 訓練メトリクス計算
        train_mae_dict = {}
        train_corr_dict = {}
        for indicator in config.target_indicators:
            if indicator in train_preds:
                train_mae_dict[indicator] = mean_absolute_error(train_targets[indicator], train_preds[indicator])
                train_corr_dict[indicator] = np.corrcoef(train_targets[indicator], train_preds[indicator])[0, 1]
        
        # 結果保存
        fold_results.append({
            'fold': fold + 1,
            'test_task': test_task,
            'train_predictions': train_preds,
            'train_targets': train_targets,
            'train_mae': train_mae_dict,
            'train_corr': train_corr_dict,
            'test_results': test_results
        })
        
        # 全体のテストデータ集約
        for indicator in config.target_indicators:
            if indicator in test_results:
                all_test_predictions[indicator].extend(test_results[indicator]['predictions'])
                all_test_targets[indicator].extend(test_results[indicator]['targets'])
        
        # 各Foldのプロット（色分け対応）
        plot_fold_results_colored_multi_indicator(fold_results[-1], subject_save_dir, config)
    
    cv_total_time = time.time() - cv_start_time
    if config.verbose:
        print(f"\n  交差検証総処理時間: {cv_total_time:.1f}秒 ({cv_total_time/60:.1f}分)")
    
    # テスト予測を元の順序に並び替え
    sorted_indices = np.argsort(all_test_task_indices)
    for indicator in all_test_predictions:
        all_test_predictions[indicator] = np.array(all_test_predictions[indicator])[sorted_indices]
        all_test_targets[indicator] = np.array(all_test_targets[indicator])[sorted_indices]
    all_test_tasks = np.array(all_test_tasks)[sorted_indices]
    
    return fold_results, all_test_predictions, all_test_targets, all_test_tasks

def plot_fold_results_colored_multi_indicator(result, save_dir, config):
    """各Foldの結果をプロット（複数指標・色分け対応）"""
    fold = result['fold']
    test_task = result['test_task']
    task_color = config.task_colors[test_task]
    
    for indicator in config.target_indicators:
        if indicator not in result['test_results']:
            continue
        
        # 訓練データ散布図
        if indicator in result['train_predictions']:
            plt.figure(figsize=(10, 8))
            plt.scatter(result['train_targets'][indicator], result['train_predictions'][indicator], 
                        alpha=0.5, s=10, color='gray', label='訓練データ')
            min_val = min(result['train_targets'][indicator].min(), result['train_predictions'][indicator].min())
            max_val = max(result['train_targets'][indicator].max(), result['train_predictions'][indicator].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('真値')
            plt.ylabel('予測値')
            plt.title(f"{indicator} - Fold {fold} 訓練データ - MAE: {result['train_mae'][indicator]:.3f}, Corr: {result['train_corr'][indicator]:.3f}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f'fold{fold}_{indicator}_train_scatter.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # テストデータ散布図（色分け）
        test_result = result['test_results'][indicator]
        plt.figure(figsize=(10, 8))
        plt.scatter(test_result['targets'], test_result['predictions'], 
                    alpha=0.6, s=20, color=task_color, label=f'テストタスク: {test_task}')
        min_val = min(test_result['targets'].min(), test_result['predictions'].min())
        max_val = max(test_result['targets'].max(), test_result['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"{indicator} - Fold {fold} テストデータ ({test_task}) - MAE: {test_result['mae']:.3f}, Corr: {test_result['corr']:.3f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f'fold{fold}_{indicator}_test_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 波形比較（色分け）
        if 'waveform_predictions' in test_result and test_result['waveform_predictions'] is not None:
            plt.figure(figsize=(16, 8))
            
            # 最初のサンプルの波形を表示
            if test_result['waveform_predictions'].shape[0] > 0:
                plt.subplot(2, 1, 1)
                if indicator in result['train_predictions']:
                    # 訓練データの最初の数サンプルを表示
                    n_samples = min(5, len(result['train_targets'][indicator]))
                    for i in range(n_samples):
                        plt.plot(result['train_targets'][indicator][i:i+1].flatten()[:60], 
                                alpha=0.3, color='blue')
                plt.xlabel('時間 (秒)')
                plt.ylabel('信号値')
                plt.title(f'{indicator} - Fold {fold} 訓練データ波形サンプル')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 1, 2)
                plt.plot(test_result['waveform_targets'][0], 'b-', label='真値', alpha=0.7, linewidth=1)
                plt.plot(test_result['waveform_predictions'][0], color=task_color, linestyle='-', 
                         label=f'予測 ({test_task})', alpha=0.7, linewidth=1)
                plt.xlabel('時間 (秒)')
                plt.ylabel('信号値')
                plt.title(f'{indicator} - Fold {fold} テストデータ波形 ({test_task})')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'fold{fold}_{indicator}_waveforms.png', dpi=150, bbox_inches='tight')
            plt.close()

def plot_subject_summary_multi_indicator(fold_results, all_test_predictions, all_test_targets, 
                                        all_test_tasks, subject, subject_save_dir, config):
    """被験者の全体結果をプロット（複数指標・タスクごとに色分け）"""
    
    metrics = {}
    
    for indicator in config.target_indicators:
        if indicator not in all_test_predictions:
            continue
        
        # 全訓練データ統合
        all_train_predictions = []
        all_train_targets = []
        for r in fold_results:
            if indicator in r['train_predictions']:
                all_train_predictions.extend(r['train_predictions'][indicator])
                all_train_targets.extend(r['train_targets'][indicator])
        
        if all_train_predictions:
            all_train_predictions = np.array(all_train_predictions)
            all_train_targets = np.array(all_train_targets)
            all_train_mae = mean_absolute_error(all_train_targets, all_train_predictions)
            all_train_corr, _ = pearsonr(all_train_targets, all_train_predictions)
        else:
            all_train_mae = np.nan
            all_train_corr = np.nan
        
        # 全テストデータメトリクス
        all_test_preds = all_test_predictions[indicator]
        all_test_targs = all_test_targets[indicator]
        all_test_mae = mean_absolute_error(all_test_targs, all_test_preds)
        all_test_corr, _ = pearsonr(all_test_targs, all_test_preds)
        
        metrics[indicator] = {
            'train_mae': all_train_mae,
            'train_corr': all_train_corr,
            'test_mae': all_test_mae,
            'test_corr': all_test_corr
        }
        
        # 全訓練データ散布図
        if all_train_predictions is not None and len(all_train_predictions) > 0:
            plt.figure(figsize=(10, 8))
            plt.scatter(all_train_targets, all_train_predictions, alpha=0.5, s=10, color='gray')
            min_val = min(all_train_targets.min(), all_train_predictions.min())
            max_val = max(all_train_targets.max(), all_train_predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('真値')
            plt.ylabel('予測値')
            plt.title(f"{subject} - {indicator} 全訓練データ - MAE: {all_train_mae:.3f}, Corr: {all_train_corr:.3f}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(subject_save_dir / f'{indicator}_all_train_scatter.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 全テストデータ散布図（タスクごとに色分け）
        plt.figure(figsize=(12, 8))
        for task in config.tasks:
            mask = all_test_tasks == task
            if np.any(mask):
                plt.scatter(all_test_targs[mask], all_test_preds[mask], 
                           alpha=0.6, s=20, color=config.task_colors[task], label=task)
        
        min_val = min(all_test_targs.min(), all_test_preds.min())
        max_val = max(all_test_targs.max(), all_test_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"{subject} - {indicator} 全テストデータ - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}")
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(subject_save_dir / f'{indicator}_all_test_scatter_colored.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 全テストデータ連結波形（タスクごとに色分け）
        plt.figure(figsize=(20, 8))
        
        # 真値を薄い色でプロット
        plt.plot(all_test_targs, 'k-', label='真値', alpha=0.4, linewidth=1)
        
        # 予測値をタスクごとに色分けしてプロット
        for i, task in enumerate(config.tasks):
            start_idx = i * config.task_duration
            end_idx = (i + 1) * config.task_duration
            plt.plot(range(start_idx, end_idx), all_test_preds[start_idx:end_idx], 
                    color=config.task_colors[task], label=f'予測 ({task})', 
                    alpha=0.8, linewidth=1.5)
        
        # タスク境界に縦線
        for i in range(1, 6):
            plt.axvline(x=i*60, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('時間 (秒)')
        plt.ylabel('信号値')
        plt.title(f'{subject} - {indicator} 全テストデータ連結波形 - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}')
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(subject_save_dir / f'{indicator}_all_test_waveform_colored.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    return metrics

def plot_all_subjects_summary_unified_multi_indicator(all_subjects_results, config):
    """全被験者のサマリープロット（複数指標・1つのグラフに統合）"""
    save_dir = Path(config.save_path)
    
    for indicator in config.target_indicators:
        # カラーマップを準備（32人の被験者用）
        colors = plt.cm.hsv(np.linspace(0, 1, len(all_subjects_results)))
        
        train_data_exists = False
        test_data_exists = False
        
        # データの存在確認
        for result in all_subjects_results:
            if indicator in result['metrics']:
                if not np.isnan(result['metrics'][indicator]['train_mae']):
                    train_data_exists = True
                if not np.isnan(result['metrics'][indicator]['test_mae']):
                    test_data_exists = True
        
        # 訓練データ：全被験者を1つのグラフにプロット
        if train_data_exists:
            plt.figure(figsize=(14, 10))
            for i, result in enumerate(all_subjects_results):
                if indicator in result['fold_results'][0]['train_predictions']:
                    # 各被験者のデータを取得
                    all_train_predictions = []
                    all_train_targets = []
                    for r in result['fold_results']:
                        if indicator in r['train_predictions']:
                            all_train_predictions.extend(r['train_predictions'][indicator])
                            all_train_targets.extend(r['train_targets'][indicator])
                    
                    if all_train_predictions:
                        # 散布図をプロット
                        plt.scatter(all_train_targets, all_train_predictions, 
                                   alpha=0.3, s=5, color=colors[i], label=result['subject'])
            
            # 対角線
            plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], 'k--', lw=2, alpha=0.7)
            
            plt.xlabel('真値')
            plt.ylabel('予測値')
            
            # 平均メトリクス
            mae_list = []
            corr_list = []
            for result in all_subjects_results:
                if indicator in result['metrics'] and not np.isnan(result['metrics'][indicator]['train_mae']):
                    mae_list.append(result['metrics'][indicator]['train_mae'])
                    corr_list.append(result['metrics'][indicator]['train_corr'])
            
            if mae_list:
                avg_train_mae = np.mean(mae_list)
                avg_train_corr = np.mean(corr_list)
                
                plt.title(f'{indicator} - 全被験者 訓練データ - 平均MAE: {avg_train_mae:.3f}, 平均Corr: {avg_train_corr:.3f}')
            
            plt.grid(True, alpha=0.3)
            
            # 凡例を2列で右側に配置
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(save_dir / f'{indicator}_all_subjects_train_unified.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # テストデータ：全被験者を1つのグラフにプロット
        if test_data_exists:
            plt.figure(figsize=(14, 10))
            for i, result in enumerate(all_subjects_results):
                if indicator in result['all_test_predictions']:
                    # 各被験者のテストデータを取得
                    all_test_predictions = result['all_test_predictions'][indicator]
                    all_test_targets = result['all_test_targets'][indicator]
                    
                    # 散布図をプロット
                    plt.scatter(all_test_targets, all_test_predictions, 
                               alpha=0.4, s=8, color=colors[i], label=result['subject'])
            
            # 対角線
            plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], 'k--', lw=2, alpha=0.7)
            
            plt.xlabel('真値')
            plt.ylabel('予測値')
            
            # 平均メトリクス
            mae_list = []
            corr_list = []
            for result in all_subjects_results:
                if indicator in result['metrics']:
                    mae_list.append(result['metrics'][indicator]['test_mae'])
                    corr_list.append(result['metrics'][indicator]['test_corr'])
            
            if mae_list:
                avg_test_mae = np.mean(mae_list)
                avg_test_corr = np.mean(corr_list)
                
                plt.title(f'{indicator} - 全被験者 テストデータ - 平均MAE: {avg_test_mae:.3f}, 平均Corr: {avg_test_corr:.3f}')
            
            plt.grid(True, alpha=0.3)
            
            # 凡例を2列で右側に配置
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(save_dir / f'{indicator}_all_subjects_test_unified.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # 被験者ごとのパフォーマンス比較（棒グラフ）
        if test_data_exists:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            
            subjects = []
            test_maes = []
            test_corrs = []
            
            for result in all_subjects_results:
                if indicator in result['metrics']:
                    subjects.append(result['subject'])
                    test_maes.append(result['metrics'][indicator]['test_mae'])
                    test_corrs.append(result['metrics'][indicator]['test_corr'])
            
            if subjects:
                x = np.arange(len(subjects))
                
                # MAE比較
                bars1 = ax1.bar(x, test_maes, color=colors[:len(subjects)])
                avg_mae = np.mean(test_maes)
                ax1.axhline(y=avg_mae, color='r', linestyle='--', label=f'平均: {avg_mae:.3f}')
                ax1.set_ylabel('MAE')
                ax1.set_title(f'{indicator} - テストデータ MAE')
                ax1.set_ylim([0, max(test_maes) * 1.1])
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 相関係数比較
                bars2 = ax2.bar(x, test_corrs, color=colors[:len(subjects)])
                avg_corr = np.mean(test_corrs)
                ax2.axhline(y=avg_corr, color='r', linestyle='--', label=f'平均: {avg_corr:.3f}')
                ax2.set_ylabel('相関係数')
                ax2.set_xlabel('被験者')
                ax2.set_title(f'{indicator} - テストデータ相関係数')
                ax2.set_xticks(x)
                ax2.set_xticklabels(subjects, rotation=45, ha='right')
                ax2.set_ylim([0, 1])
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_dir / f'{indicator}_all_subjects_performance_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    # 乱数シード設定
    set_all_seeds(config.random_seed)
    
    # 開始時間
    total_start_time = time.time()
    
    print("\n" + "="*60)
    print("PhysNet2DCNN - マルチモード解析システム")
    print("="*60)
    print(f"解析タイプ: {config.analysis_type}")
    print(f"モデルアーキテクチャ: {config.model_architecture}")
    print(f"対象指標: {config.target_indicators}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    
    if config.use_lab:
        print(f"LABデータ: 使用（RGB+LAB = {config.num_channels}チャンネル）")
    else:
        print(f"LABデータ: 未使用")
    
    print(f"\n【高速化設定】")
    print(f"  AMP (自動混合精度): {'有効' if config.use_amp else '無効'}")
    print(f"  torch.compile: {'有効' if config.use_compile and torch.__version__.startswith('2.') else '無効'}")
    print(f"  DataLoader設定:")
    print(f"    Workers: {config.num_workers}")
    print(f"    Pin memory: {config.pin_memory}")
    print(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  バッチサイズ: {config.batch_size}")
    
    print(f"\nデータ拡張: {'有効' if config.use_augmentation else '無効'}")
    if config.use_augmentation:
        print(f"  拡張手法:")
        if config.crop_enabled:
            print(f"    - ランダムクロップ ({config.crop_size_ratio*100:.0f}%)")
        if config.rotation_enabled:
            print(f"    - 回転 (±{config.rotation_range}度)")
        if config.time_stretch_enabled:
            print(f"    - 時間軸ストレッチング ({config.time_stretch_range[0]:.1f}x-{config.time_stretch_range[1]:.1f}x)")
        if config.brightness_contrast_enabled:
            print(f"    - 明度・コントラスト調整 (±{config.brightness_range*100:.0f}%)")
    
    # GPU情報の表示
    if torch.cuda.is_available():
        print(f"\nGPU情報:")
        print(f"  デバイス名: {torch.cuda.get_device_name(0)}")
        print(f"  メモリ容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    print(f"\n保存先: {config.save_path}")
    print(f"被験者数: {len(config.subjects)}")
    print(f"乱数シード: {config.random_seed}")
    
    if config.analysis_type == "inter_subject":
        inter_subject_cross_validation(config)
    elif config.analysis_type == "intra_subject":
        # 個人内解析の実装は省略（必要に応じて追加）
        print("個人内解析は現在実装中です")
    else:
        print(f"不明な解析タイプ: {config.analysis_type}")
        return
    
    # 総処理時間
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("処理完了")
    print(f"結果保存先: {config.save_path}")
    print(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print("="*60)

if __name__ == "__main__":
    main()
