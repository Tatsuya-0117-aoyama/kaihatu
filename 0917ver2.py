import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
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
from itertools import combinations
import seaborn as sns
import time
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
# 設定クラス（完全版）
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
        
        # ================================
        # 交差検証モード設定
        # ================================
        # "within_subject": 個人内6分割交差検証
        # "all_subjects": 全被験者5分割交差検証
        self.cross_validation_mode = "all_subjects"
        
        # ================================
        # 推定モード設定
        # ================================
        # "single": 単一指標推定
        # "multi": 複数指標同時推定
        self.estimation_mode = "multi"
        
        # ================================
        # 推定する指標の選択
        # ================================
        # 利用可能な指標: ["CO", "SV", "HR_CO_SV", "Cwk", "Rp", "Zao", "I0", "LVET", "reDIA", "reSYS"]
        if self.estimation_mode == "single":
            self.target_signals = ["CO"]  # 単一指標の場合
        else:
            # 複数指標の場合（任意の組み合わせ可能）
            self.target_signals = ["CO", "SV", "HR_CO_SV"]  # 例：3指標
            # self.target_signals = ["CO", "SV", "HR_CO_SV", "Cwk", "Rp", "Zao", "I0", "LVET", "reDIA", "reSYS"]  # 全指標
        
        # 出力チャンネル数（推定する指標の数）
        self.output_channels = len(self.target_signals)
        
        # ================================
        # 全被験者交差検証設定
        # ================================
        self.n_folds_all_subjects = 5  # 全被験者交差検証のfold数
        
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
        self.use_lab = True  # LABデータを使用するか
        self.lab_filename = "_downsampled_1Hzver2.npy"  # LABデータのファイル名
        
        # ================================
        # 訓練・検証分割設定
        # ================================
        self.train_val_split_ratio = 0.9  # 訓練データの割合（90%）
        
        # 検証データ分割戦略
        self.validation_split_strategy = 'stratified'  # 'sequential' または 'stratified'
        
        # 層化サンプリング設定
        self.n_strata = 5  # 層の数
        self.stratification_method = 'quantile'  # 'equal_range' or 'quantile'
        
        # 被験者設定（bp001～bp032）
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定（6分割交差検証用）
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60  # 各タスクの長さ（秒）
        
        # タスクごとの色設定（交差検証プロット用）
        self.task_colors = {
            "t1-1": "#FF6B6B",  # 赤系
            "t2": "#4ECDC4",    # 青緑系
            "t1-2": "#45B7D1",  # 青系
            "t4": "#96CEB4",    # 緑系
            "t1-3": "#FECA57",  # 黄系
            "t5": "#DDA0DD"     # 紫系
        }
        
        # 5分割交差検証用の色設定
        self.fold_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
        
        # ================================
        # モデル設定
        # ================================
        # モデルタイプ選択（"3d" or "2d"）
        self.model_type = "2d"
        
        # 使用チャンネル設定
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
        
        # データ形状設定
        self.time_frames = 360  # 時間フレーム数
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
        # 学習設定
        # ================================
        if self.model_type == "3d":
            self.batch_size = 8
            self.epochs = 150
        else:  # 2d
            self.batch_size = 32 if self.use_amp else 16
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
        self.warmup_lr_factor = 0.1  # 初期学習率を1/10からスタート
        
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

# ================================
# データ拡張関数（完全版）
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
        """高速な時間軸ストレッチング"""
        x = torch.from_numpy(rgb_np).permute(3,0,1,2).unsqueeze(0).float()
        T = x.shape[2]
        T2 = max(1, int(T*factor))
        
        with torch.no_grad():
            x = F.interpolate(x, size=(T2, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
            x = F.interpolate(x, size=(T, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        
        out = x.squeeze(0).permute(1,2,3,0).cpu().numpy()
        return out
    
    def time_stretch(self, rgb_data, signal_data=None):
        """時間軸ストレッチング（高速版）"""
        if np.random.random() > self.config.aug_probability or not self.config.time_stretch_enabled:
            return rgb_data, signal_data
        
        if rgb_data.ndim != 4:
            return rgb_data, signal_data
        
        stretch_factor = np.random.uniform(*self.config.time_stretch_range)
        
        rgb_stretched = self.time_stretch_fast(rgb_data, stretch_factor)
        
        if signal_data is not None and signal_data.ndim >= 1:
            if signal_data.ndim == 1:
                # 単一指標
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
                
            elif signal_data.ndim == 2:  # 複数指標の場合
                t_original, n_signals = signal_data.shape
                t_stretched = int(t_original * stretch_factor)
                
                signal_resampled = np.zeros_like(signal_data)
                for i in range(n_signals):
                    f_signal = interp1d(np.arange(t_original), signal_data[:, i], 
                                      kind='linear', fill_value='extrapolate')
                    signal_stretched = f_signal(np.linspace(0, t_original-1, t_stretched))
                    
                    f_signal_back = interp1d(np.arange(len(signal_stretched)), signal_stretched,
                                            kind='linear', fill_value='extrapolate')
                    signal_resampled[:, i] = f_signal_back(np.linspace(0, len(signal_stretched)-1, t_original))
                    signal_resampled[:, i] = signal_resampled[:, i] * stretch_factor
                
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
# カスタム損失関数（完全版 - HuberLoss含む）
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
    """Huber損失と相関損失の組み合わせ（外れ値にロバスト）"""
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

class MultiTaskCombinedLoss(nn.Module):
    """複数タスク対応の複合損失関数"""
    def __init__(self, alpha=0.7, beta=0.3, n_tasks=1, loss_type='mse'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_tasks = n_tasks
        self.loss_type = loss_type
        
        if loss_type == 'huber':
            self.base_loss = nn.HuberLoss(delta=1.0)
        else:
            self.base_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        total_loss = 0
        mse_losses = []
        corr_losses = []
        
        if self.n_tasks == 1:
            # 単一タスクの場合
            base_loss = self.base_loss(pred, target)
            
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
            
            total_loss = self.alpha * base_loss + self.beta * corr_loss
            return total_loss, base_loss, corr_loss
        
        else:
            # 複数タスクの場合
            for i in range(self.n_tasks):
                pred_i = pred[:, :, i] if pred.dim() == 3 else pred[:, i].unsqueeze(1)
                target_i = target[:, :, i] if target.dim() == 3 else target[:, i].unsqueeze(1)
                
                base_loss = self.base_loss(pred_i, target_i)
                mse_losses.append(base_loss)
                
                if pred_i.dim() == 2 and pred_i.size(1) > 1:
                    pred_centered = pred_i - torch.mean(pred_i, dim=1, keepdim=True)
                    target_centered = target_i - torch.mean(target_i, dim=1, keepdim=True)
                    
                    numerator = torch.sum(pred_centered * target_centered, dim=1)
                    denominator = torch.sqrt(
                        torch.sum(pred_centered**2, dim=1) * 
                        torch.sum(target_centered**2, dim=1) + 1e-8
                    )
                    
                    correlation = numerator / denominator
                    corr_loss = torch.mean(1 - correlation)
                else:
                    pred_mean = pred_i - pred_i.mean()
                    target_mean = target_i - target_i.mean()
                    
                    numerator = torch.sum(pred_mean * target_mean)
                    denominator = torch.sqrt(
                        torch.sum(pred_mean ** 2) * 
                        torch.sum(target_mean ** 2) + 1e-8
                    )
                    correlation = numerator / denominator
                    corr_loss = 1 - correlation
                
                corr_losses.append(corr_loss)
                total_loss += self.alpha * base_loss + self.beta * corr_loss
            
            # 各タスクの損失の平均
            total_loss = total_loss / self.n_tasks
            avg_base_loss = sum(mse_losses) / self.n_tasks
            avg_corr_loss = sum(corr_losses) / self.n_tasks
            
            return total_loss, avg_base_loss, avg_corr_loss
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
    n_samples = len(task_signal)
    
    # 信号データが複数指標の場合は最初の指標で層化
    if task_signal.ndim == 1:
        signal_for_stratification = task_signal
    elif task_signal.ndim == 2:
        signal_for_stratification = task_signal[:, 0]
    else:  # 3次元の場合
        signal_for_stratification = task_signal[:, 0, 0]
    
    signal_min = signal_for_stratification.min()
    signal_max = signal_for_stratification.max()
    signal_mean = signal_for_stratification.mean()
    signal_std = signal_for_stratification.std()
    
    # 層の境界を決定
    if method == 'equal_range':
        bin_edges = np.linspace(signal_min, signal_max + 1e-10, n_strata + 1)
    elif method == 'quantile':
        quantiles = np.linspace(0, 1, n_strata + 1)
        bin_edges = np.quantile(signal_for_stratification, quantiles)
        bin_edges[-1] += 1e-10
    else:
        raise ValueError(f"Unknown stratification method: {method}")
    
    # 各サンプルを層に割り当て
    strata_assignment = np.digitize(signal_for_stratification, bin_edges) - 1
    
    train_indices = []
    val_indices = []
    
    strata_info = []
    
    # 各層から比例してサンプリング
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
    
    # デバッグ情報を表示
    if len(strata_info) > 0:
        print(f"      層化サンプリング詳細 (方法: {method}):")
        for info in strata_info:
            print(f"        層{info['stratum_id']}: "
                  f"信号値[{info['signal_range'][0]:.3f}, {info['signal_range'][1]:.3f}] "
                  f"計{info['n_total']}個 → 訓練{info['n_train']}個, 検証{info['n_val']}個")
    
    # 検証データの信号値分布を確認
    val_signals = signal_for_stratification[val_indices]
    train_signals = signal_for_stratification[train_indices]
    
    print(f"      信号値の分布確認:")
    print(f"        元データ: 平均={signal_mean:.3f}, 標準偏差={signal_std:.3f}")
    print(f"        訓練データ: 平均={train_signals.mean():.3f}, 標準偏差={train_signals.std():.3f}")
    print(f"        検証データ: 平均={val_signals.mean():.3f}, 標準偏差={val_signals.std():.3f}")
    
    return (task_rgb[train_indices], task_signal[train_indices],
            task_rgb[val_indices], task_signal[val_indices])

# ================================
# データセット（複数指標対応）
# ================================
class MultiSignalDataset(Dataset):
    """複数指標対応のデータセット"""
    def __init__(self, rgb_data, signal_data, model_type='3d', 
                 use_channel='RGB', config=None, is_training=True):
        self.model_type = model_type
        self.use_channel = use_channel
        self.is_training = is_training
        
        self.augmentation = DataAugmentation(config) if config else None
        
        self.rgb_data_raw = rgb_data
        self.signal_data_raw = signal_data
        
        rgb_data_selected = select_channels(rgb_data, use_channel)
        
        self.rgb_data = torch.FloatTensor(rgb_data_selected)
        
        # 信号データの処理（複数指標対応）
        if signal_data.ndim == 1:
            # 単一指標で時間次元なし
            signal_data = np.repeat(signal_data[:, np.newaxis], rgb_data.shape[1], axis=1)
            signal_data = signal_data[:, :, np.newaxis]  # (N, T, 1)
        elif signal_data.ndim == 2:
            # 判定が必要: (N, T) or (N, n_signals)
            if signal_data.shape[1] == rgb_data.shape[1]:
                # (N, T) - 単一指標で時間次元あり
                signal_data = signal_data[:, :, np.newaxis]  # (N, T, 1)
            else:
                # (N, n_signals) - 複数指標で時間次元なし
                n_samples, n_signals = signal_data.shape
                expanded_signal = np.zeros((n_samples, rgb_data.shape[1], n_signals))
                for i in range(n_signals):
                    expanded_signal[:, :, i] = np.repeat(signal_data[:, i:i+1], rgb_data.shape[1], axis=1)
                signal_data = expanded_signal
        elif signal_data.ndim == 3:
            # (N, T, n_signals) - そのまま使用
            pass
        
        self.signal_data = torch.FloatTensor(signal_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        # 元データを取得
        rgb = self.rgb_data_raw[idx]
        signal = self.signal_data_raw[idx]
        
        # データ拡張を適用（学習時のみ）
        if self.augmentation and self.is_training:
            rgb, signal = self.augmentation.apply_augmentation(rgb, signal, self.is_training)
        
        # チャンネル選択
        rgb = select_channels(rgb, self.use_channel)
        
        # ========================================
        # 元コードに合わせた修正部分
        # ========================================
        # Tensorに変換
        rgb_tensor = torch.FloatTensor(rgb)
        signal_tensor = torch.FloatTensor(signal if isinstance(signal, np.ndarray) else [signal])
        
        # 次元調整（元コードと同じロジック）
        if signal_tensor.dim() == 0:
            signal_tensor = signal_tensor.unsqueeze(0)
        
        # 単一値を時間次元に拡張
        if signal_tensor.size(0) == 1 and rgb_tensor.size(0) > 1:
            signal_tensor = signal_tensor.repeat(rgb_tensor.size(0))
        
        # 複数指標の場合の追加処理
        elif signal_tensor.dim() == 1:
            # 複数指標で時間次元がない場合
            if len(signal_tensor) != rgb_tensor.size(0):
                # 複数指標を時間次元に拡張
                signal_tensor = signal_tensor.unsqueeze(0).repeat(rgb_tensor.size(0), 1)
            else:
                # 時間次元のみの場合は最後に次元追加
                signal_tensor = signal_tensor.unsqueeze(1)
        
        # 3次元データ（時間×指標）の場合はそのまま
        elif signal_tensor.dim() == 2:
            # (T, n_signals) の形状を維持
            pass
        
        return rgb_tensor, signal_tensor

# ================================
# PhysNet2DCNN (3D版) - 複数出力対応
# ================================
class PhysNet2DCNN_3D_Multi(nn.Module):
    """3D畳み込みPhysNet（複数出力対応）"""
    def __init__(self, input_shape=None, output_channels=1):
        super(PhysNet2DCNN_3D_Multi, self).__init__()
        
        self.output_channels = output_channels
        
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
        
        # Final Conv - 複数出力対応
        self.conv_final = nn.Conv3d(64, self.output_channels, kernel_size=1)
        
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
        batch_size = x.size(0)
        time_frames = x.size(1)
        
        x = x.permute(0, 4, 1, 2, 3)
        
        # ConvBlocks
        x = self.elu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.elu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.dropout(self.pool1(x))
        
        x = self.elu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.elu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.dropout(self.pool2(x))
        
        x = self.elu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.elu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.dropout(self.pool3(x))
        
        x = self.elu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.elu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.dropout(self.pool4(x))
        
        x = self.elu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.elu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.upsample(x)
        
        x = self.elu6_1(self.bn6_1(self.conv6_1(x)))
        x = self.elu6_2(self.bn6_2(self.conv6_2(x)))
        
        x = self.spatial_pool(x)
        x = self.conv_final(x)
        
        # 出力形状: (B, output_channels, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, output_channels, T)
        x = x.permute(0, 2, 1)  # (B, T, output_channels)
        
        # 元の時間長に補間
        if x.size(1) != time_frames:
            x = x.permute(0, 2, 1)  # (B, output_channels, T)
            x = F.interpolate(x, size=time_frames, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # (B, T, output_channels)
        
        # 単一出力の場合は次元を削除
        if self.output_channels == 1:
            x = x.squeeze(-1)
        
        return x

# ================================
# PhysNet2DCNN (2D版) - 複数出力対応
# ================================
class PhysNet2DCNN_2D_Multi(nn.Module):
    """2D畳み込みPhysNet（複数出力対応）"""
    def __init__(self, input_shape=None, output_channels=1):
        super(PhysNet2DCNN_2D_Multi, self).__init__()
        
        self.output_channels = output_channels
        
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
        
        # ConvBlock 4: 64 filters（大きい入力の場合のみ）
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
        
        # Final layer - 複数出力対応
        self.fc = nn.Linear(32, self.output_channels)
        
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
        batch_size, time_frames = x.size(0), x.size(1)
        
        x = x.view(batch_size * time_frames, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 3, 1, 2)
        
        # ConvBlocks
        x = self.elu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.elu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.dropout(self.pool1(x))
        
        x = self.elu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.elu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.dropout(self.pool2(x))
        
        x = self.elu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.elu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.dropout(self.pool3(x))
        
        if not self.small_input:
            x = self.elu4_1(self.bn4_1(self.conv4_1(x)))
            x = self.elu4_2(self.bn4_2(self.conv4_2(x)))
            x = self.dropout(self.pool4(x))
        
        x = self.elu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.elu5_2(self.bn5_2(self.conv5_2(x)))
        
        x = self.spatial_pool(x)
        x = x.view(batch_size, time_frames, 64)
        
        x = x.permute(0, 2, 1)
        x = self.temporal_elu1(self.temporal_bn1(self.temporal_conv1(x)))
        x = self.temporal_elu2(self.temporal_bn2(self.temporal_conv2(x)))
        x = x.permute(0, 2, 1)
        
        x = self.fc(x)
        
        # 単一出力の場合は次元を削除
        if self.output_channels == 1:
            x = x.squeeze(-1)
        
        return x
# ================================
# モデル作成関数
# ================================
def create_model(config):
    """設定に基づいてモデルを作成"""
    if config.model_type == "3d":
        model = PhysNet2DCNN_3D_Multi(config.input_shape, config.output_channels)
        model_name = f"PhysNet2DCNN_3D ({config.output_channels}出力)"
    elif config.model_type == "2d":
        model = PhysNet2DCNN_2D_Multi(config.input_shape, config.output_channels)
        model_name = f"PhysNet2DCNN_2D ({config.output_channels}出力)"
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    # PyTorch 2.0+の場合、torch.compileを使用
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
        print(f"推定指標: {', '.join(config.target_signals)}")
        print(f"使用チャンネル: {config.use_channel}")
        if config.use_lab:
            print(f"LABデータ: 使用（RGB+LAB = {config.num_channels}チャンネル）")
        else:
            print(f"LABデータ: 未使用（RGB = {config.num_channels}チャンネル）")
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        if config.model_type == "3d":
            print("【注意】3Dモデルはメモリを多く使用します。バッチサイズの調整を推奨します。")
        else:
            print("【推奨】2Dモデルは計算効率が良く、大きなバッチサイズでも動作します。")
    
    return model

# ================================
# データ読み込み関数（複数指標対応）
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み（複数指標対応）"""
    
    # RGBデータの読み込み
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません: {rgb_path}")
        return None, None
    
    rgb_data = np.load(rgb_path)  # Shape: (360, 14, 16, 3)
    print(f"  RGBデータ読み込み成功: {rgb_data.shape}")
    
    # LABデータの読み込み（オプション）
    if config.use_lab:
        lab_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_downsampled_1Hzver2.npy")
        
        if os.path.exists(lab_path):
            lab_data = np.load(lab_path)
            print(f"  LABデータ読み込み成功: {lab_data.shape}")
            
            # RGBとLABデータの形状を確認
            if rgb_data.shape == lab_data.shape:
                # RGB+LABデータを結合（チャンネル次元で連結）
                combined_data = np.concatenate([rgb_data, lab_data], axis=-1)
                print(f"  RGB+LAB結合データ: {combined_data.shape}")
                
                # データの正規化（LABデータも0-1の範囲に）
                if lab_data.max() > 1.0:
                    combined_data[..., 3:] = combined_data[..., 3:] / 255.0
                
                rgb_data = combined_data
            else:
                print(f"警告: {subject}のRGBとLABデータの形状が一致しません")
                print(f"  RGB shape: {rgb_data.shape}, LAB shape: {lab_data.shape}")
                config.use_lab = False
                config.use_channel = 'RGB'
                config.num_channels = 3
        else:
            print(f"警告: {subject}のLABデータが見つかりません: {lab_path}")
            print(f"  LABデータなしで処理を続行します。")
            config.use_lab = False
            config.use_channel = 'RGB'
            config.num_channels = 3
    
    # データ形状の確認と調整
    if rgb_data.ndim == 5:  # (N, T, H, W, C)
        pass
    elif rgb_data.ndim == 4:  # (N, H, W, C) の場合、時間次元を追加
        rgb_data = np.expand_dims(rgb_data, axis=1)
        rgb_data = np.repeat(rgb_data, config.time_frames, axis=1)
    
    # 複数指標の信号データ読み込み
    all_signals_data = []
    
    for signal_name in config.target_signals:
        signal_data_list = []
        for task in config.tasks:
            signal_path = os.path.join(config.signal_base_path, subject, 
                                      signal_name, 
                                      f"{signal_name}_s2_{task}.npy")
            if not os.path.exists(signal_path):
                print(f"警告: {subject}の{task}の{signal_name}データが見つかりません")
                return None, None
            signal_data_list.append(np.load(signal_path))
        
        signal_data = np.concatenate(signal_data_list)
        all_signals_data.append(signal_data)
    
    # 複数指標を結合
    if len(all_signals_data) == 1:
        signal_data = all_signals_data[0]
    else:
        signal_data = np.stack(all_signals_data, axis=-1)  # (360, n_signals)
    
    # データの正規化（0-1の範囲に）
    if rgb_data[..., :3].max() > 1.0:
        rgb_data[..., :3] = rgb_data[..., :3] / 255.0
    
    return rgb_data, signal_data

def load_all_subjects_data(config):
    """全被験者のデータを読み込み"""
    all_rgb_data = []
    all_signal_data = []
    all_subject_task_info = []  # (subject_idx, task_idx)のタプルのリスト
    
    print("\n全被験者データ読み込み中...")
    for subj_idx, subject in enumerate(config.subjects):
        print(f"  {subject}を読み込み中... ({subj_idx+1}/{len(config.subjects)})")
        rgb_data, signal_data = load_data_single_subject(subject, config)
        
        if rgb_data is None or signal_data is None:
            print(f"  {subject}のデータ読み込み失敗。スキップします。")
            continue
        
        # 各タスクを分割
        for task_idx in range(len(config.tasks)):
            start_idx = task_idx * config.task_duration
            end_idx = (task_idx + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signal = signal_data[start_idx:end_idx]
            
            all_rgb_data.append(task_rgb)
            all_signal_data.append(task_signal)
            all_subject_task_info.append((subj_idx, task_idx))
    
    print(f"読み込み完了: {len(all_rgb_data)}タスク")
    
    return all_rgb_data, all_signal_data, all_subject_task_info

# ================================
# 全被験者5分割交差検証
# ================================
def create_stratified_folds(all_subject_task_info, n_folds=5, random_seed=42):
    """
    5分割交差検証のfoldを作成
    制約：各foldの訓練データに全32被験者から最低1タスク含める
    """
    np.random.seed(random_seed)
    
    # 被験者ごとにタスクをグループ化
    subject_tasks = {}
    for idx, (subj_idx, task_idx) in enumerate(all_subject_task_info):
        if subj_idx not in subject_tasks:
            subject_tasks[subj_idx] = []
        subject_tasks[subj_idx].append(idx)
    
    n_subjects = len(subject_tasks)
    n_total_tasks = len(all_subject_task_info)
    fold_size = n_total_tasks // n_folds  # 各foldのテストデータサイズ（約38）
    
    # 各foldのテストデータインデックスを保存
    test_folds = [[] for _ in range(n_folds)]
    
    # 各被験者のタスクを各foldに配分
    for subj_idx, task_indices in subject_tasks.items():
        np.random.shuffle(task_indices)
        n_tasks = len(task_indices)
        
        if n_tasks >= n_folds:
            # タスクが5個以上ある場合、各foldに均等に配分
            for fold_idx in range(n_folds):
                # 各foldに最低1つは配分
                start_idx = fold_idx * n_tasks // n_folds
                end_idx = (fold_idx + 1) * n_tasks // n_folds
                test_folds[fold_idx].extend(task_indices[start_idx:end_idx])
        else:
            # タスクが5個未満の場合、ランダムに配分
            fold_assignments = np.random.choice(n_folds, n_tasks, replace=False)
            for task_idx, fold_idx in zip(task_indices, fold_assignments):
                test_folds[fold_idx].append(task_idx)
    
    # 各foldのサイズを調整（約38タスクになるように）
    adjusted_folds = []
    for fold_idx in range(n_folds):
        current_fold = test_folds[fold_idx]
        
        if len(current_fold) > fold_size:
            # 多すぎる場合は削減
            np.random.shuffle(current_fold)
            current_fold = current_fold[:fold_size]
        elif len(current_fold) < fold_size:
            # 少ない場合は他のタスクから補充
            # ただし、訓練データに全被験者が含まれることを保証する必要がある
            all_indices = set(range(n_total_tasks))
            used_indices = set()
            for f in test_folds:
                used_indices.update(f)
            available_indices = list(all_indices - used_indices)
            
            if available_indices:
                np.random.shuffle(available_indices)
                n_needed = min(fold_size - len(current_fold), len(available_indices))
                current_fold.extend(available_indices[:n_needed])
        
        adjusted_folds.append(current_fold)
    
    # 検証：各foldの訓練データに全被験者が含まれることを確認
    print("\n5分割交差検証 - Fold構成確認:")
    for fold_idx, test_indices in enumerate(adjusted_folds):
        # このfoldのテストデータ
        test_subjects = set()
        for idx in test_indices:
            subj_idx, _ = all_subject_task_info[idx]
            test_subjects.add(subj_idx)
        
        # このfoldの訓練データ
        train_indices = []
        for i, fold in enumerate(adjusted_folds):
            if i != fold_idx:
                train_indices.extend(fold)
        
        train_subjects = set()
        for idx in range(n_total_tasks):
            if idx not in test_indices:
                subj_idx, _ = all_subject_task_info[idx]
                train_subjects.add(subj_idx)
        
        print(f"  Fold {fold_idx+1}:")
        print(f"    テストデータ: {len(test_indices)}タスク, {len(test_subjects)}人の被験者")
        print(f"    訓練データ: {n_total_tasks - len(test_indices)}タスク, {len(train_subjects)}人の被験者")
        
        # 訓練データに含まれない被験者をチェック
        missing_subjects = set(range(n_subjects)) - train_subjects
        if missing_subjects:
            print(f"    警告: 訓練データに含まれない被験者: {missing_subjects}")
            
            # 修正：訓練データに全被験者を含めるように調整
            for missing_subj in missing_subjects:
                # この被験者のタスクから1つを訓練データに移動
                subj_task_indices = subject_tasks[missing_subj]
                for task_idx in subj_task_indices:
                    if task_idx in test_indices:
                        # テストデータから削除
                        test_indices.remove(task_idx)
                        break
            
            adjusted_folds[fold_idx] = test_indices
            print(f"    修正後テストデータ: {len(test_indices)}タスク")
    
    return adjusted_folds

def all_subjects_cross_validation(config):
    """全被験者5分割交差検証"""
    all_rgb_data, all_signal_data, all_subject_task_info = load_all_subjects_data(config)
    
    if not all_rgb_data:
        print("データ読み込みエラー")
        return None
    
    # 層化5分割を作成
    folds = create_stratified_folds(all_subject_task_info, config.n_folds_all_subjects, config.random_seed)
    
    # 分割情報を詳細に保存
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'fold_split_info.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("全被験者5分割交差検証 - データ分割情報\n")
        f.write("="*60 + "\n\n")
        f.write(f"総タスク数: {len(all_subject_task_info)}\n")
        f.write(f"総被験者数: {len(config.subjects)}\n")
        f.write(f"Fold数: {config.n_folds_all_subjects}\n\n")
        
        for fold_idx, test_indices in enumerate(folds):
            f.write(f"{'='*40}\n")
            f.write(f"Fold {fold_idx+1}\n")
            f.write(f"{'='*40}\n")
            
            # テストデータの詳細
            f.write(f"テストデータ: {len(test_indices)}タスク\n")
            test_subject_tasks = {}
            for idx in test_indices:
                subj_idx, task_idx = all_subject_task_info[idx]
                subj_name = config.subjects[subj_idx]
                task_name = config.tasks[task_idx]
                
                if subj_name not in test_subject_tasks:
                    test_subject_tasks[subj_name] = []
                test_subject_tasks[subj_name].append(task_name)
            
            f.write(f"  含まれる被験者数: {len(test_subject_tasks)}\n")
            for subj, tasks in sorted(test_subject_tasks.items()):
                f.write(f"    {subj}: {tasks}\n")
            
            # 訓練データの詳細
            train_indices = [i for i in range(len(all_subject_task_info)) if i not in test_indices]
            f.write(f"\n訓練データ: {len(train_indices)}タスク\n")
            
            train_subject_tasks = {}
            for idx in train_indices:
                subj_idx, task_idx = all_subject_task_info[idx]
                subj_name = config.subjects[subj_idx]
                task_name = config.tasks[task_idx]
                
                if subj_name not in train_subject_tasks:
                    train_subject_tasks[subj_name] = []
                train_subject_tasks[subj_name].append(task_name)
            
            f.write(f"  含まれる被験者数: {len(train_subject_tasks)}\n")
            
            # 各被験者のタスク数を集計
            subject_task_counts = {}
            for subj in sorted(train_subject_tasks.keys()):
                subject_task_counts[subj] = len(train_subject_tasks[subj])
            
            # タスク数でソート
            sorted_subjects = sorted(subject_task_counts.items(), key=lambda x: x[1])
            
            f.write(f"  被験者別タスク数:\n")
            for subj, count in sorted_subjects:
                f.write(f"    {subj}: {count}タスク {train_subject_tasks[subj]}\n")
            
            # 統計情報
            f.write(f"\n  統計:\n")
            f.write(f"    最小タスク数/被験者: {min(subject_task_counts.values())}\n")
            f.write(f"    最大タスク数/被験者: {max(subject_task_counts.values())}\n")
            f.write(f"    平均タスク数/被験者: {np.mean(list(subject_task_counts.values())):.2f}\n")
            
            # 全被験者が含まれているか確認
            missing_subjects = set(config.subjects) - set(train_subject_tasks.keys())
            if missing_subjects:
                f.write(f"  警告: 訓練データに含まれない被験者: {missing_subjects}\n")
            else:
                f.write(f"  ✓ 全{len(config.subjects)}人の被験者が訓練データに含まれています\n")
            
            f.write("\n")
    
    # DataLoader用のワーカー初期化関数
    seed_worker = set_all_seeds(config.random_seed)
    
    fold_results = []
    all_test_predictions = {signal: [] for signal in config.target_signals}
    all_test_targets = {signal: [] for signal in config.target_signals}
    all_test_fold_indices = []
    
    cv_start_time = time.time()
    
    for fold_idx in range(config.n_folds_all_subjects):
        fold_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{config.n_folds_all_subjects}")
        print(f"{'='*60}")
        
        # フォルダ作成
        fold_dir = save_dir / f"fold{fold_idx+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # テストインデックス
        test_indices = folds[fold_idx]
        # 訓練インデックス
        train_val_indices = []
        for i in range(config.n_folds_all_subjects):
            if i != fold_idx:
                train_val_indices.extend(folds[i])
        
        # 訓練・検証分割
        np.random.shuffle(train_val_indices)
        val_size = int(len(train_val_indices) * (1 - config.train_val_split_ratio))
        train_indices = train_val_indices[val_size:]
        val_indices = train_val_indices[:val_size]
        
        print(f"データサイズ - 訓練: {len(train_indices)}, 検証: {len(val_indices)}, テスト: {len(test_indices)}")
        
        # データ準備
        train_rgb = np.array([all_rgb_data[i] for i in train_indices])
        train_signal = np.array([all_signal_data[i] for i in train_indices])
        val_rgb = np.array([all_rgb_data[i] for i in val_indices])
        val_signal = np.array([all_signal_data[i] for i in val_indices])
        test_rgb = np.array([all_rgb_data[i] for i in test_indices])
        test_signal = np.array([all_signal_data[i] for i in test_indices])
        
        # データセット作成
        train_dataset = MultiSignalDataset(train_rgb, train_signal, config.model_type, 
                                          config.use_channel, config, is_training=True)
        val_dataset = MultiSignalDataset(val_rgb, val_signal, config.model_type, 
                                        config.use_channel, config, is_training=False)
        test_dataset = MultiSignalDataset(test_rgb, test_signal, config.model_type, 
                                         config.use_channel, config, is_training=False)
        
        # データローダー作成
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
        model, train_metrics = train_model_multi(
            model, train_loader, val_loader, config, fold_idx, None, fold_dir
        )
        
        # 評価
        test_metrics = evaluate_model_multi(model, test_loader, config)
        
        fold_time = time.time() - fold_start_time
        
        # 結果表示
        print(f"\nFold {fold_idx+1} 結果:")
        for signal in config.target_signals:
            print(f"  {signal}:")
            print(f"    Train: MAE={train_metrics[signal]['mae']:.4f}, Corr={train_metrics[signal]['corr']:.4f}")
            print(f"    Test:  MAE={test_metrics[signal]['mae']:.4f}, Corr={test_metrics[signal]['corr']:.4f}")
        print(f"  処理時間: {fold_time:.1f}秒")
        
        # 結果保存
        fold_results.append({
            'fold': fold_idx + 1,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
        
        # CSV保存
        save_fold_results_csv(fold_idx, train_metrics, test_metrics, fold_dir, config)
        
        # テスト結果集約
        for signal in config.target_signals:
            all_test_predictions[signal].extend(test_metrics[signal]['predictions'])
            all_test_targets[signal].extend(test_metrics[signal]['targets'])
        all_test_fold_indices.extend([fold_idx] * len(test_indices))
        
        # Foldごとのプロット
        plot_fold_results_multi(fold_idx, train_metrics, test_metrics, fold_dir, config)
    
    cv_total_time = time.time() - cv_start_time
    print(f"\n交差検証総処理時間: {cv_total_time:.1f}秒 ({cv_total_time/60:.1f}分)")
    
    # 全体のプロット
    plot_all_cv_results_multi(fold_results, all_test_predictions, all_test_targets, 
                              all_test_fold_indices, save_dir, config)
    
    # 全体結果のCSV保存
    save_all_cv_results_csv(fold_results, save_dir, config)
    
    return fold_results
# ================================
# 学習関数（複数指標対応）
# ================================
def train_model_multi(model, train_loader, val_loader, config, fold=None, subject=None, save_dir=None):
    """モデルの学習（複数指標対応）"""
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    subject_str = f"{subject}" if subject is not None else ""
    
    if config.verbose:
        print(f"\n  学習開始 {subject_str} {fold_str}")
        print(f"    モデル: {config.model_type}")
        print(f"    エポック数: {config.epochs}")
        print(f"    バッチサイズ: {config.batch_size}")
        print(f"    AMP使用: {'有効' if config.use_amp else '無効'}")
        print(f"    Warmupエポック数: {config.warmup_epochs}")
        print(f"    初期学習率: {config.learning_rate}")
        if config.warmup_epochs > 0:
            print(f"    Warmup開始学習率: {config.learning_rate * config.warmup_lr_factor}")
    
    model = model.to(config.device)
    
    # 損失関数の選択
    if config.loss_type == "huber_combined":
        criterion = MultiTaskCombinedLoss(
            alpha=config.loss_alpha, 
            beta=config.loss_beta,
            n_tasks=config.output_channels,
            loss_type='huber'
        )
        if config.verbose:
            print(f"    損失関数: HuberCorrelationLoss (α={config.loss_alpha}, β={config.loss_beta})")
    else:
        criterion = MultiTaskCombinedLoss(
            alpha=config.loss_alpha, 
            beta=config.loss_beta,
            n_tasks=config.output_channels,
            loss_type='mse'
        )
        if config.verbose:
            print(f"    損失関数: CombinedLoss (α={config.loss_alpha}, β={config.loss_beta})")
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # AMP用のGradScaler初期化
    scaler = GradScaler() if config.use_amp else None
    if config.use_amp and config.verbose:
        print("    GradScaler初期化完了（AMP有効）")
    
    # Warmup用の初期学習率設定
    initial_lr = config.learning_rate
    if config.warmup_epochs > 0:
        warmup_lr = initial_lr * config.warmup_lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
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
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 各指標の記録
    train_metrics_best = {signal: {'predictions': [], 'targets': []} 
                         for signal in config.target_signals}
    
    # エポックループ
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
        train_preds_epoch = {signal: [] for signal in config.target_signals}
        train_targets_epoch = {signal: [] for signal in config.target_signals}
        
        for rgb, sig in train_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            
            optimizer.zero_grad()
            
            # AMP使用時はautocast内で順伝播
            if config.use_amp:
                with autocast():
                    pred = model(rgb)
                    loss, mse_loss, corr_loss = criterion(pred, sig)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(rgb)
                loss, mse_loss, corr_loss = criterion(pred, sig)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
                optimizer.step()
            
            if scheduler_per_batch and epoch >= config.warmup_epochs:
                scheduler.step()
            
            train_loss += loss.item()
            
            # 予測値を保存
            if config.output_channels == 1:
                pred_np = pred.mean(dim=1).detach().cpu().numpy() if pred.dim() == 2 else pred.detach().cpu().numpy()
                sig_np = sig.mean(dim=1).detach().cpu().numpy() if sig.dim() == 2 else sig.detach().cpu().numpy()
                train_preds_epoch[config.target_signals[0]].extend(pred_np)
                train_targets_epoch[config.target_signals[0]].extend(sig_np)
            else:
                for i, signal in enumerate(config.target_signals):
                    if pred.dim() == 3:
                        pred_np = pred[:, :, i].mean(dim=1).detach().cpu().numpy()
                        sig_np = sig[:, :, i].mean(dim=1).detach().cpu().numpy()
                    else:
                        pred_np = pred[:, i].detach().cpu().numpy()
                        sig_np = sig[:, i].detach().cpu().numpy()
                    train_preds_epoch[signal].extend(pred_np)
                    train_targets_epoch[signal].extend(sig_np)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for rgb, sig in val_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                
                if config.use_amp:
                    with autocast():
                        pred = model(rgb)
                        loss, mse_loss, corr_loss = criterion(pred, sig)
                else:
                    pred = model(rgb)
                    loss, mse_loss, corr_loss = criterion(pred, sig)
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # スケジューラー更新
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
            
            # 最良時の予測値を保存
            for signal in config.target_signals:
                train_metrics_best[signal]['predictions'] = np.array(train_preds_epoch[signal])
                train_metrics_best[signal]['targets'] = np.array(train_targets_epoch[signal])
            
            # モデル保存先の決定
            if save_dir is None:
                save_dir = Path(config.save_path)
                if config.cross_validation_mode == "within_subject" and subject is not None:
                    save_dir = save_dir / subject
            else:
                save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if fold is not None:
                model_name = f'best_model_fold{fold+1}.pth'
            else:
                model_name = f'best_model_{config.model_type}.pth'
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'model_type': config.model_type,
                'target_signals': config.target_signals
            }, save_dir / model_name)
        else:
            patience_counter += 1
        
        # エポック時間計算
        epoch_time = time.time() - epoch_start_time
        
        # ログ出力
        if config.verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch < config.warmup_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            if epoch < config.warmup_epochs:
                print(f"    [Warmup] Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e} Time: {epoch_time:.1f}s")
            else:
                print(f"    Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e} Time: {epoch_time:.1f}s")
            print(f"      Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 各指標の相関を表示
            for signal in config.target_signals:
                if len(train_preds_epoch[signal]) > 0:
                    corr = np.corrcoef(train_preds_epoch[signal], train_targets_epoch[signal])[0, 1]
                    print(f"      {signal} Train Corr: {corr:.4f}")
            
            # AMP使用時はスケール値も表示
            if config.use_amp and scaler is not None:
                print(f"      AMP Scale: {scaler.get_scale():.1f}")
        
        # Early Stopping
        if patience_counter >= config.patience:
            if config.verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # ベストモデル読み込み
    checkpoint = torch.load(save_dir / model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # メトリクス計算
    train_metrics = {}
    for signal in config.target_signals:
        preds = train_metrics_best[signal]['predictions']
        targets = train_metrics_best[signal]['targets']
        
        mae = mean_absolute_error(targets, preds)
        corr = np.corrcoef(targets, preds)[0, 1] if len(preds) > 1 else 0.0
        
        train_metrics[signal] = {
            'mae': mae,
            'corr': corr,
            'predictions': preds,
            'targets': targets
        }
    
    return model, train_metrics

# ================================
# 評価関数（複数指標対応）
# ================================
def evaluate_model_multi(model, test_loader, config):
    """モデルの評価（複数指標対応）"""
    model.eval()
    
    predictions = {signal: [] for signal in config.target_signals}
    targets = {signal: [] for signal in config.target_signals}
    waveform_predictions = {signal: [] for signal in config.target_signals}
    waveform_targets = {signal: [] for signal in config.target_signals}
    
    with torch.no_grad():
        for rgb, sig in test_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            
            if config.use_amp:
                with autocast():
                    pred = model(rgb)
            else:
                pred = model(rgb)
            
            # 予測値を保存
            if config.output_channels == 1:
                # 波形全体を保存
                if pred.dim() == 2:
                    waveform_predictions[config.target_signals[0]].append(pred.cpu().numpy())
                    waveform_targets[config.target_signals[0]].append(sig.cpu().numpy())
                    # 平均値も計算
                    pred_np = pred.mean(dim=1).cpu().numpy()
                    sig_np = sig.mean(dim=1).cpu().numpy()
                else:
                    pred_np = pred.cpu().numpy()
                    sig_np = sig.cpu().numpy()
                predictions[config.target_signals[0]].extend(pred_np)
                targets[config.target_signals[0]].extend(sig_np)
            else:
                for i, signal in enumerate(config.target_signals):
                    if pred.dim() == 3:
                        # 波形全体を保存
                        waveform_predictions[signal].append(pred[:, :, i].cpu().numpy())
                        waveform_targets[signal].append(sig[:, :, i].cpu().numpy())
                        # 平均値も計算
                        pred_np = pred[:, :, i].mean(dim=1).cpu().numpy()
                        sig_np = sig[:, :, i].mean(dim=1).cpu().numpy()
                    else:
                        pred_np = pred[:, i].cpu().numpy()
                        sig_np = sig[:, i].cpu().numpy()
                    predictions[signal].extend(pred_np)
                    targets[signal].extend(sig_np)
    
    # メトリクス計算
    test_metrics = {}
    for signal in config.target_signals:
        preds = np.array(predictions[signal])
        targs = np.array(targets[signal])
        
        mae = mean_absolute_error(targs, preds)
        rmse = np.sqrt(np.mean((targs - preds) ** 2))
        corr, p_value = pearsonr(targs, preds) if len(preds) > 1 else (0.0, 1.0)
        
        # R^2スコア
        ss_res = np.sum((targs - preds) ** 2)
        ss_tot = np.sum((targs - np.mean(targs)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        test_metrics[signal] = {
            'mae': mae,
            'rmse': rmse,
            'corr': corr,
            'r2': r2,
            'p_value': p_value,
            'predictions': preds,
            'targets': targs
        }
        
        # 波形データがある場合は追加
        if waveform_predictions[signal]:
            test_metrics[signal]['waveform_predictions'] = np.concatenate(waveform_predictions[signal], axis=0)
            test_metrics[signal]['waveform_targets'] = np.concatenate(waveform_targets[signal], axis=0)
    
    return test_metrics

# ================================
# 個人内6分割交差検証（複数指標対応）
# ================================
def within_subject_cross_validation(config):
    """個人内6分割交差検証（複数指標対応）"""
    all_subjects_results = []
    
    for subj_idx, subject in enumerate(config.subjects):
        subject_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"被験者 {subject} ({subj_idx+1}/{len(config.subjects)})")
        print(f"{'='*60}")
        
        subject_save_dir = Path(config.save_path) / subject
        subject_save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            rgb_data, signal_data = load_data_single_subject(subject, config)
            
            if rgb_data is None or signal_data is None:
                print(f"  {subject}のデータ読み込み失敗。スキップします。")
                continue
            
            print(f"  データ形状: RGB={rgb_data.shape}, Signal={signal_data.shape}")
            
            # 6分割交差検証実行
            fold_results = task_cross_validation_multi(
                rgb_data, signal_data, config, subject, subject_save_dir
            )
            
            # 被験者全体のサマリー計算
            subject_metrics = {}
            for signal in config.target_signals:
                all_train_predictions = np.concatenate([r['train_metrics'][signal]['predictions'] 
                                                       for r in fold_results])
                all_train_targets = np.concatenate([r['train_metrics'][signal]['targets'] 
                                                   for r in fold_results])
                all_test_predictions = np.concatenate([r['test_metrics'][signal]['predictions'] 
                                                      for r in fold_results])
                all_test_targets = np.concatenate([r['test_metrics'][signal]['targets'] 
                                                  for r in fold_results])
                
                subject_metrics[signal] = {
                    'train_mae': mean_absolute_error(all_train_targets, all_train_predictions),
                    'train_corr': pearsonr(all_train_targets, all_train_predictions)[0],
                    'test_mae': mean_absolute_error(all_test_targets, all_test_predictions),
                    'test_corr': pearsonr(all_test_targets, all_test_predictions)[0]
                }
            
            # 結果保存
            all_subjects_results.append({
                'subject': subject,
                'fold_results': fold_results,
                'subject_metrics': subject_metrics
            })
            
            subject_time = time.time() - subject_start_time
            
            print(f"\n  {subject} 完了:")
            for signal in config.target_signals:
                print(f"    {signal}:")
                print(f"      全体訓練: MAE={subject_metrics[signal]['train_mae']:.4f}, "
                      f"Corr={subject_metrics[signal]['train_corr']:.4f}")
                print(f"      全体テスト: MAE={subject_metrics[signal]['test_mae']:.4f}, "
                      f"Corr={subject_metrics[signal]['test_corr']:.4f}")
            print(f"    処理時間: {subject_time:.1f}秒 ({subject_time/60:.1f}分)")
            
            # CSV保存
            save_subject_results_csv(subject, subject_metrics, subject_save_dir, config)
            
        except Exception as e:
            print(f"  {subject}でエラー発生: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 全被験者の結果がある場合のみ統合プロットとCSV保存を実行
    if all_subjects_results:
        print(f"\n{'='*60}")
        print("全被験者サマリー作成中...")
        print(f"{'='*60}")
        
        # 全被験者のサマリープロット（統合版）
        plot_all_subjects_summary_unified_multi(all_subjects_results, config)
        
        # 全被験者結果のCSV保存
        save_all_subjects_summary_csv(all_subjects_results, config)
        
        # 統計サマリーの表示
        print(f"\n{'='*60}")
        print("全被験者統計サマリー")
        print(f"{'='*60}")
        
        for signal in config.target_signals:
            train_maes = [r['subject_metrics'][signal]['train_mae'] for r in all_subjects_results]
            train_corrs = [r['subject_metrics'][signal]['train_corr'] for r in all_subjects_results]
            test_maes = [r['subject_metrics'][signal]['test_mae'] for r in all_subjects_results]
            test_corrs = [r['subject_metrics'][signal]['test_corr'] for r in all_subjects_results]
            
            print(f"\n{signal}推定モデル:")
            print(f"  訓練データ:")
            print(f"    MAE: {np.mean(train_maes):.4f} ± {np.std(train_maes):.4f}")
            print(f"    相関: {np.mean(train_corrs):.4f} ± {np.std(train_corrs):.4f}")
            print(f"  テストデータ:")
            print(f"    MAE: {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
            print(f"    相関: {np.mean(test_corrs):.4f} ± {np.std(test_corrs):.4f}")
            print(f"  処理被験者数: {len(all_subjects_results)}/{len(config.subjects)}")
    else:
        print(f"\n{'='*60}")
        print("警告: 処理完了した被験者がありません")
        print(f"{'='*60}")
    return all_subjects_results

def task_cross_validation_multi(rgb_data, signal_data, config, subject, subject_save_dir):
    """タスクごとの6分割交差検証（複数指標対応）"""
    
    fold_results = []
    all_test_predictions = {signal: [] for signal in config.target_signals}
    all_test_targets = {signal: [] for signal in config.target_signals}
    all_test_task_indices = []
    all_test_tasks = []  # タスク名を記録
    
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    seed_worker = set_all_seeds(config.random_seed)
    
    cv_start_time = time.time()
    
    for fold, test_task in enumerate(config.tasks):
        fold_start_time = time.time()
        
        if config.verbose:
            print(f"\n  Fold {fold+1}/6 - テストタスク: {test_task}")
            print(f"    検証データ分割戦略: {config.validation_split_strategy}")
        
        # タスクごとにデータを分割
        train_rgb_list = []
        train_signal_list = []
        val_rgb_list = []
        val_signal_list = []
        test_rgb_list = []
        test_signal_list = []
        
        for i, task in enumerate(config.tasks):
            start_idx = i * config.task_duration
            end_idx = (i + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signal = signal_data[start_idx:end_idx]
            
            if task == test_task:
                # テストデータ
                test_rgb_list.append(task_rgb)
                test_signal_list.append(task_signal)
                all_test_task_indices.extend([i] * config.task_duration)
                all_test_tasks.extend([test_task] * config.task_duration)
            else:
                # 訓練・検証データの分割
                if config.validation_split_strategy == 'stratified':
                    train_rgb, train_signal, val_rgb, val_signal = stratified_sampling_split(
                        task_rgb, task_signal,
                        val_ratio=(1 - config.train_val_split_ratio),
                        n_strata=config.n_strata,
                        method=config.stratification_method
                    )
                    train_rgb_list.append(train_rgb)
                    train_signal_list.append(train_signal)
                    val_rgb_list.append(val_rgb)
                    val_signal_list.append(val_signal)
                else:  # 'sequential'
                    val_size = int(config.task_duration * (1 - config.train_val_split_ratio))
                    val_start_idx = config.task_duration - val_size
                    
                    train_rgb_list.append(task_rgb[:val_start_idx])
                    train_signal_list.append(task_signal[:val_start_idx])
                    val_rgb_list.append(task_rgb[val_start_idx:])
                    val_signal_list.append(task_signal[val_start_idx:])
        
        # データ結合
        train_rgb = np.concatenate(train_rgb_list)
        train_signal = np.concatenate(train_signal_list)
        val_rgb = np.concatenate(val_rgb_list)
        val_signal = np.concatenate(val_signal_list)
        test_rgb = np.concatenate(test_rgb_list)
        test_signal = np.concatenate(test_signal_list)
        
        if config.verbose:
            print(f"    データサイズ - 訓練: {len(train_rgb)}, 検証: {len(val_rgb)}, テスト: {len(test_rgb)}")
            print(f"    訓練:検証 = {len(train_rgb)}:{len(val_rgb)} ≈ 9:1")
        
        # データセット作成
        train_dataset = MultiSignalDataset(train_rgb, train_signal, config.model_type, 
                                          config.use_channel, config, is_training=True)
        val_dataset = MultiSignalDataset(val_rgb, val_signal, config.model_type, 
                                        config.use_channel, config, is_training=False)
        test_dataset = MultiSignalDataset(test_rgb, test_signal, config.model_type, 
                                         config.use_channel, config, is_training=False)
        
        # データローダー作成
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, worker_init_fn=seed_worker,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False
        )
        
        # モデル作成
        model = create_model(config)
        
        # Foldディレクトリ作成
        fold_dir = subject_save_dir / f"fold{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        
        # モデル学習
        model, train_metrics = train_model_multi(
            model, train_loader, val_loader, config, fold, subject, fold_dir
        )
        
        # 評価
        test_metrics = evaluate_model_multi(model, test_loader, config)
        
        fold_time = time.time() - fold_start_time
        
        if config.verbose:
            print(f"    Fold結果:")
            for signal in config.target_signals:
                print(f"      {signal}:")
                print(f"        Train: MAE={train_metrics[signal]['mae']:.4f}, Corr={train_metrics[signal]['corr']:.4f}")
                print(f"        Test:  MAE={test_metrics[signal]['mae']:.4f}, Corr={test_metrics[signal]['corr']:.4f}")
            print(f"    Fold処理時間: {fold_time:.1f}秒")
        
        # 結果保存
        fold_results.append({
            'fold': fold + 1,
            'test_task': test_task,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        })
        
        # テスト結果集約
        for signal in config.target_signals:
            all_test_predictions[signal].extend(test_metrics[signal]['predictions'])
            all_test_targets[signal].extend(test_metrics[signal]['targets'])
        
        # 各Foldのプロット
        plot_fold_results_multi(fold, train_metrics, test_metrics, fold_dir, config, test_task)
    
    cv_total_time = time.time() - cv_start_time
    if config.verbose:
        print(f"\n  交差検証総処理時間: {cv_total_time:.1f}秒 ({cv_total_time/60:.1f}分)")
    
    # テスト予測を元の順序に並び替え
    sorted_indices = np.argsort(all_test_task_indices)
    for signal in config.target_signals:
        all_test_predictions[signal] = np.array(all_test_predictions[signal])[sorted_indices]
        all_test_targets[signal] = np.array(all_test_targets[signal])[sorted_indices]
    all_test_tasks = np.array(all_test_tasks)[sorted_indices]
    
    # 被験者全体のプロット
    plot_subject_summary_multi(fold_results, all_test_predictions, all_test_targets,
                               all_test_tasks, subject, subject_save_dir, config)
    
    return fold_results

# ================================
# プロット関数（複数指標対応）
# ================================
def plot_fold_results_multi(fold_idx, train_metrics, test_metrics, save_dir, config, test_task=None):
    """Foldごとの結果をプロット（複数指標対応）"""
    
    for signal in config.target_signals:
        # 散布図（訓練データ）
        plt.figure(figsize=(10, 8))
        plt.scatter(train_metrics[signal]['targets'], train_metrics[signal]['predictions'], 
                   alpha=0.5, s=10, color='gray', label='訓練データ')
        min_val = min(train_metrics[signal]['targets'].min(), train_metrics[signal]['predictions'].min())
        max_val = max(train_metrics[signal]['targets'].max(), train_metrics[signal]['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"{signal}推定モデル - Fold {fold_idx+1} 訓練データ\n"
                 f"MAE: {train_metrics[signal]['mae']:.3f}, Corr: {train_metrics[signal]['corr']:.3f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_train_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 散布図（テストデータ）
        plt.figure(figsize=(10, 8))
        if config.cross_validation_mode == "all_subjects":
            color = config.fold_colors[fold_idx % len(config.fold_colors)]
        else:
            color = config.task_colors.get(test_task, 'blue') if test_task else 'blue'
        
        plt.scatter(test_metrics[signal]['targets'], test_metrics[signal]['predictions'], 
                   alpha=0.6, s=20, color=color, 
                   label=f'テストタスク: {test_task}' if test_task else f'Fold {fold_idx+1}')
        min_val = min(test_metrics[signal]['targets'].min(), test_metrics[signal]['predictions'].min())
        max_val = max(test_metrics[signal]['targets'].max(), test_metrics[signal]['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"{signal}推定モデル - Fold {fold_idx+1} テストデータ\n"
                 f"MAE: {test_metrics[signal]['mae']:.3f}, Corr: {test_metrics[signal]['corr']:.3f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_test_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 波形プロット（データがある場合）
        if 'waveform_predictions' in test_metrics[signal]:
            plt.figure(figsize=(16, 8))
            
            # 最初の3サンプルの波形を表示
            n_samples = min(3, len(test_metrics[signal]['waveform_predictions']))
            for i in range(n_samples):
                plt.subplot(n_samples, 1, i+1)
                plt.plot(test_metrics[signal]['waveform_targets'][i], 'b-', 
                        label='真値', alpha=0.7, linewidth=1)
                plt.plot(test_metrics[signal]['waveform_predictions'][i], 'r-', 
                        label='予測', alpha=0.7, linewidth=1)
                plt.xlabel('時間 (秒)')
                plt.ylabel('信号値')
                plt.title(f'{signal} - サンプル {i+1}')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / f'{signal}_waveforms.png', dpi=150, bbox_inches='tight')
            plt.close()

def plot_all_cv_results_multi(fold_results, all_test_predictions, all_test_targets, 
                              all_test_fold_indices, save_dir, config):
    """全交差検証結果をプロット（複数指標対応）"""
    
    # 各指標ごとにプロット
    for signal in config.target_signals:
        # 訓練データ統合プロット
        plt.figure(figsize=(12, 10))
        for fold_idx, result in enumerate(fold_results):
            train_preds = result['train_metrics'][signal]['predictions']
            train_targets = result['train_metrics'][signal]['targets']
            plt.scatter(train_targets, train_preds, alpha=0.4, s=10, 
                       color=config.fold_colors[fold_idx], label=f'Fold {fold_idx+1}')
        
        all_train_targets = np.concatenate([r['train_metrics'][signal]['targets'] for r in fold_results])
        all_train_preds = np.concatenate([r['train_metrics'][signal]['predictions'] for r in fold_results])
        min_val = all_train_targets.min()
        max_val = all_train_targets.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        overall_mae = mean_absolute_error(all_train_targets, all_train_preds)
        overall_corr = np.corrcoef(all_train_targets, all_train_preds)[0, 1]
        
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"{signal}推定モデル - 全訓練データ\n"
                 f"Overall MAE: {overall_mae:.3f}, Overall Corr: {overall_corr:.3f}")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_all_train_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # テストデータ統合プロット
        plt.figure(figsize=(12, 10))
        all_test_preds = np.array(all_test_predictions[signal])
        all_test_targs = np.array(all_test_targets[signal])
        fold_indices = np.array(all_test_fold_indices)
        
        for fold_idx in range(config.n_folds_all_subjects):
            mask = fold_indices == fold_idx
            if np.any(mask):
                plt.scatter(all_test_targs[mask], all_test_preds[mask], 
                           alpha=0.6, s=20, color=config.fold_colors[fold_idx], 
                           label=f'Fold {fold_idx+1}')
        
        min_val = all_test_targs.min()
        max_val = all_test_targs.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        overall_mae = mean_absolute_error(all_test_targs, all_test_preds)
        overall_corr = np.corrcoef(all_test_targs, all_test_preds)[0, 1]
        
        plt.xlabel('真値')
        plt.ylabel('予測値')
        plt.title(f"{signal}推定モデル - 全テストデータ\n"
                 f"Overall MAE: {overall_mae:.3f}, Overall Corr: {overall_corr:.3f}")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_all_test_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()

def plot_subject_summary_multi(fold_results, all_test_predictions, all_test_targets,
                              all_test_tasks, subject, subject_save_dir, config):
    """被験者の全体結果をプロット（複数指標対応）"""
    
    for signal in config.target_signals:
        # 全訓練データ統合
        all_train_predictions = np.concatenate([r['train_metrics'][signal]['predictions'] 
                                               for r in fold_results])
        all_train_targets = np.concatenate([r['train_metrics'][signal]['targets'] 
                                           for r in fold_results])
        
        # メトリクス計算
        all_train_mae = mean_absolute_error(all_train_targets, all_train_predictions)
        all_train_corr, _ = pearsonr(all_train_targets, all_train_predictions)
        
        all_test_preds = np.array(all_test_predictions[signal])
        all_test_targs = np.array(all_test_targets[signal])
        all_test_mae = mean_absolute_error(all_test_targs, all_test_preds)
        all_test_corr, _ = pearsonr(all_test_targs, all_test_preds)
        
        # プロット
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # 訓練データ散布図
        ax1.scatter(all_train_targets, all_train_predictions, alpha=0.5, s=10, color='gray')
        min_val = min(all_train_targets.min(), all_train_predictions.min())
        max_val = max(all_train_targets.max(), all_train_predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax1.set_xlabel('真値')
        ax1.set_ylabel('予測値')
        ax1.set_title(f"{signal}推定モデル - {subject} 全訓練データ\n"
                     f"MAE: {all_train_mae:.3f}, Corr: {all_train_corr:.3f}")
        ax1.grid(True, alpha=0.3)
        
        # テストデータ散布図（タスクごとに色分け）
        for task in config.tasks:
            mask = all_test_tasks == task
            if np.any(mask):
                ax2.scatter(all_test_targs[mask], all_test_preds[mask], 
                           alpha=0.6, s=20, color=config.task_colors[task], label=task)
        
        min_val = min(all_test_targs.min(), all_test_preds.min())
        max_val = max(all_test_targs.max(), all_test_preds.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax2.set_xlabel('真値')
        ax2.set_ylabel('予測値')
        ax2.set_title(f"{signal}推定モデル - {subject} 全テストデータ\n"
                     f"MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 訓練データ波形
        ax3.plot(all_train_targets[:300], 'b-', label='真値', alpha=0.7, linewidth=1)
        ax3.plot(all_train_predictions[:300], 'g-', label='予測', alpha=0.7, linewidth=1)
        ax3.set_xlabel('時間 (秒)')
        ax3.set_ylabel('信号値')
        ax3.set_title(f'{signal} 訓練データ波形（最初の300秒）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # テストデータ連結波形（タスクごとに色分け）
        ax4.plot(all_test_targs, 'k-', label='真値', alpha=0.4, linewidth=1)
        
        for i, task in enumerate(config.tasks):
            start_idx = i * config.task_duration
            end_idx = (i + 1) * config.task_duration
            ax4.plot(range(start_idx, end_idx), all_test_preds[start_idx:end_idx], 
                    color=config.task_colors[task], label=f'予測 ({task})', 
                    alpha=0.8, linewidth=1.5)
        
        # タスク境界に縦線
        for i in range(1, 6):
            ax4.axvline(x=i*60, color='gray', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('時間 (秒)')
        ax4.set_ylabel('信号値')
        ax4.set_title(f'{signal} 全テストデータ連結波形')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(subject_save_dir / f'{signal}_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

# ================================
# CSV保存関数
# ================================
def save_fold_results_csv(fold_idx, train_metrics, test_metrics, save_dir, config):
    """Foldごとの結果をCSVに保存"""
    results = []
    for signal in config.target_signals:
        results.append({
            'Fold': fold_idx + 1,
            'Signal': signal,
            'Train_MAE': train_metrics[signal]['mae'],
            'Train_Corr': train_metrics[signal]['corr'],
            'Test_MAE': test_metrics[signal]['mae'],
            'Test_Corr': test_metrics[signal]['corr'],
            'Test_RMSE': test_metrics[signal]['rmse'],
            'Test_R2': test_metrics[signal]['r2']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(save_dir / f'fold{fold_idx+1}_results.csv', index=False)

def save_all_cv_results_csv(fold_results, save_dir, config):
    """全交差検証結果をCSVに保存"""
    all_results = []
    
    for result in fold_results:
        fold = result['fold']
        for signal in config.target_signals:
            all_results.append({
                'Fold': fold,
                'Signal': signal,
                'Train_MAE': result['train_metrics'][signal]['mae'],
                'Train_Corr': result['train_metrics'][signal]['corr'],
                'Test_MAE': result['test_metrics'][signal]['mae'],
                'Test_Corr': result['test_metrics'][signal]['corr'],
                'Test_RMSE': result['test_metrics'][signal]['rmse'],
                'Test_R2': result['test_metrics'][signal]['r2']
            })
    
    df = pd.DataFrame(all_results)
    
    # 平均と標準偏差を計算
    summary = df.groupby('Signal').agg({
        'Train_MAE': ['mean', 'std'],
        'Train_Corr': ['mean', 'std'],
        'Test_MAE': ['mean', 'std'],
        'Test_Corr': ['mean', 'std'],
        'Test_RMSE': ['mean', 'std'],
        'Test_R2': ['mean', 'std']
    }).round(4)
    
    # 結果を保存
    df.to_csv(save_dir / 'all_folds_results.csv', index=False)
    summary.to_csv(save_dir / 'summary_statistics.csv')
    
    print("\n統計サマリー:")
    print(summary)

def save_subject_results_csv(subject, subject_metrics, save_dir, config):
    """被験者ごとの結果をCSVに保存"""
    results = []
    for signal in config.target_signals:
        results.append({
            'Subject': subject,
            'Signal': signal,
            'Train_MAE': subject_metrics[signal]['train_mae'],
            'Train_Corr': subject_metrics[signal]['train_corr'],
            'Test_MAE': subject_metrics[signal]['test_mae'],
            'Test_Corr': subject_metrics[signal]['test_corr']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(save_dir / 'subject_summary.csv', index=False)

def save_config_summary(config, total_time, save_dir):
    """設定情報を保存"""
    with open(save_dir / 'config_summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("実験設定\n")
        f.write("="*60 + "\n")
        f.write(f"実行日時: {config.timestamp}\n")
        f.write(f"交差検証モード: {config.cross_validation_mode}\n")
        f.write(f"推定モード: {config.estimation_mode}\n")
        f.write(f"推定指標: {', '.join(config.target_signals)}\n")
        f.write(f"モデルタイプ: {config.model_type}\n")
        f.write(f"使用チャンネル: {config.use_channel}\n")
        f.write(f"LABデータ使用: {config.use_lab}\n")
        f.write(f"訓練:検証比率: {config.train_val_split_ratio}:{1-config.train_val_split_ratio}\n")
        f.write(f"検証分割戦略: {config.validation_split_strategy}\n")
        f.write(f"データ拡張: {config.use_augmentation}\n")
        f.write(f"バッチサイズ: {config.batch_size}\n")
        f.write("\n高速化設定:\n")
        f.write(f"  AMP (自動混合精度): {config.use_amp}\n")
        f.write(f"  torch.compile: {config.use_compile}\n")
        f.write(f"  DataLoader workers: {config.num_workers}\n")
        f.write(f"  Pin memory: {config.pin_memory}\n")
        f.write(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}\n")
        f.write(f"損失関数: {config.loss_type}\n")
        f.write(f"Warmupエポック: {config.warmup_epochs}\n")
        f.write(f"学習率: {config.learning_rate}\n")
        f.write(f"エポック数: {config.epochs}\n")
        f.write(f"勾配クリッピング: {config.gradient_clip_val}\n")
        f.write(f"乱数シード: {config.random_seed}\n")
        f.write(f"\n総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)\n")
        
        # GPU情報も記録
        if torch.cuda.is_available():
            f.write(f"\nGPU情報:\n")
            f.write(f"  デバイス名: {torch.cuda.get_device_name(0)}\n")
            f.write(f"  メモリ容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
            f.write(f"  PyTorchバージョン: {torch.__version__}\n")
            f.write(f"  CUDAバージョン: {torch.version.cuda}\n")
# ================================
# 追加実装が必要な関数
# ================================

def plot_all_subjects_summary_unified_multi(all_subjects_results, config):
    """全被験者のサマリープロット（複数指標対応・統合版）"""
    save_dir = Path(config.save_path)
    
    for signal in config.target_signals:
        # カラーマップを準備（32人の被験者用）
        colors = plt.cm.hsv(np.linspace(0, 1, len(all_subjects_results)))
        
        # 訓練データ：全被験者を1つのグラフにプロット
        plt.figure(figsize=(14, 10))
        for i, result in enumerate(all_subjects_results):
            # 各被験者のデータを取得
            all_train_predictions = np.concatenate([r['train_metrics'][signal]['predictions'] 
                                                   for r in result['fold_results']])
            all_train_targets = np.concatenate([r['train_metrics'][signal]['targets'] 
                                               for r in result['fold_results']])
            
            # 散布図をプロット
            plt.scatter(all_train_targets, all_train_predictions, 
                       alpha=0.3, s=5, color=colors[i], label=result['subject'])
        
        # 対角線
        all_min = min([np.concatenate([r['train_metrics'][signal]['targets'] 
                                      for r in res['fold_results']]).min() 
                      for res in all_subjects_results])
        all_max = max([np.concatenate([r['train_metrics'][signal]['targets'] 
                                      for r in res['fold_results']]).max() 
                      for res in all_subjects_results])
        plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.7)
        
        plt.xlabel('真値')
        plt.ylabel('予測値')
        
        # 平均メトリクス
        avg_train_mae = np.mean([r['subject_metrics'][signal]['train_mae'] 
                                for r in all_subjects_results])
        avg_train_corr = np.mean([r['subject_metrics'][signal]['train_corr'] 
                                 for r in all_subjects_results])
        
        plt.title(f'{signal}推定モデル - 全被験者訓練データ\n'
                 f'平均MAE: {avg_train_mae:.3f}, 平均Corr: {avg_train_corr:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 凡例を2列で右側に配置
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_all_subjects_train_unified.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # テストデータ：全被験者を1つのグラフにプロット  
        plt.figure(figsize=(14, 10))
        for i, result in enumerate(all_subjects_results):
            # 各被験者のテストデータを取得
            all_test_predictions = np.concatenate([r['test_metrics'][signal]['predictions'] 
                                                  for r in result['fold_results']])
            all_test_targets = np.concatenate([r['test_metrics'][signal]['targets'] 
                                              for r in result['fold_results']])
            
            # 散布図をプロット
            plt.scatter(all_test_targets, all_test_predictions, 
                       alpha=0.4, s=8, color=colors[i], label=result['subject'])
        
        # 対角線
        all_min = min([np.concatenate([r['test_metrics'][signal]['targets'] 
                                     for r in res['fold_results']]).min() 
                     for res in all_subjects_results])
        all_max = max([np.concatenate([r['test_metrics'][signal]['targets'] 
                                     for r in res['fold_results']]).max() 
                     for res in all_subjects_results])
        plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.7)
        
        plt.xlabel('真値')
        plt.ylabel('予測値')
        
        # 平均メトリクス
        avg_test_mae = np.mean([r['subject_metrics'][signal]['test_mae'] 
                               for r in all_subjects_results])
        avg_test_corr = np.mean([r['subject_metrics'][signal]['test_corr'] 
                                for r in all_subjects_results])
        
        plt.title(f'{signal}推定モデル - 全被験者テストデータ\n'
                 f'平均MAE: {avg_test_mae:.3f}, 平均Corr: {avg_test_corr:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 凡例を2列で右側に配置
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_all_subjects_test_unified.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 被験者ごとのパフォーマンス比較（棒グラフ）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        subjects = [r['subject'] for r in all_subjects_results]
        train_corrs = [r['subject_metrics'][signal]['train_corr'] for r in all_subjects_results]
        test_corrs = [r['subject_metrics'][signal]['test_corr'] for r in all_subjects_results]
        
        x = np.arange(len(subjects))
        
        # 訓練相関
        bars1 = ax1.bar(x, train_corrs, color=colors)
        ax1.axhline(y=avg_train_corr, color='r', linestyle='--', 
                   label=f'平均: {avg_train_corr:.3f}')
        ax1.set_ylabel('相関係数')
        ax1.set_title(f'{signal} - 訓練データ相関')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # テスト相関
        bars2 = ax2.bar(x, test_corrs, color=colors)
        ax2.axhline(y=avg_test_corr, color='r', linestyle='--', 
                   label=f'平均: {avg_test_corr:.3f}')
        ax2.set_ylabel('相関係数')
        ax2.set_xlabel('被験者')
        ax2.set_title(f'{signal} - テストデータ相関')
        ax2.set_xticks(x)
        ax2.set_xticklabels(subjects, rotation=45, ha='right')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{signal}_all_subjects_performance_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

def save_all_subjects_summary_csv(all_subjects_results, config):
    """全被験者の結果をCSVに保存"""
    save_dir = Path(config.save_path)
    
    for signal in config.target_signals:
        all_results = []
        
        for result in all_subjects_results:
            all_results.append({
                'Subject': result['subject'],
                'Train_MAE': result['subject_metrics'][signal]['train_mae'],
                'Train_Corr': result['subject_metrics'][signal]['train_corr'],
                'Test_MAE': result['subject_metrics'][signal]['test_mae'],
                'Test_Corr': result['subject_metrics'][signal]['test_corr']
            })
        
        df = pd.DataFrame(all_results)
        
        # 平均と標準偏差を追加
        mean_row = pd.DataFrame({
            'Subject': ['Mean'],
            'Train_MAE': [df['Train_MAE'].mean()],
            'Train_Corr': [df['Train_Corr'].mean()],
            'Test_MAE': [df['Test_MAE'].mean()],
            'Test_Corr': [df['Test_Corr'].mean()]
        })
        
        std_row = pd.DataFrame({
            'Subject': ['Std'],
            'Train_MAE': [df['Train_MAE'].std()],
            'Train_Corr': [df['Train_Corr'].std()],
            'Test_MAE': [df['Test_MAE'].std()],
            'Test_Corr': [df['Test_Corr'].std()]
        })
        
        df = pd.concat([df, mean_row, std_row], ignore_index=True)
        df.to_csv(save_dir / f'{signal}_all_subjects_results.csv', index=False)
        
        print(f"\n{signal} - 全被験者平均結果:")
        print(f"  訓練: MAE={df.loc[df['Subject']=='Mean', 'Train_MAE'].values[0]:.4f}±"
              f"{df.loc[df['Subject']=='Std', 'Train_MAE'].values[0]:.4f}, "
              f"Corr={df.loc[df['Subject']=='Mean', 'Train_Corr'].values[0]:.4f}±"
              f"{df.loc[df['Subject']=='Std', 'Train_Corr'].values[0]:.4f}")
        print(f"  テスト: MAE={df.loc[df['Subject']=='Mean', 'Test_MAE'].values[0]:.4f}±"
              f"{df.loc[df['Subject']=='Std', 'Test_MAE'].values[0]:.4f}, "
              f"Corr={df.loc[df['Subject']=='Mean', 'Test_Corr'].values[0]:.4f}±"
              f"{df.loc[df['Subject']=='Std', 'Test_Corr'].values[0]:.4f}")

def plot_learning_curves(train_losses, val_losses, save_path, fold=None):
    """学習曲線をプロット"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curves{f" - Fold {fold+1}" if fold is not None else ""}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'learning_curves_fold{fold+1}.png' if fold is not None else 'learning_curves.png'
    plt.savefig(save_path / filename, dpi=150, bbox_inches='tight')
    plt.close()

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    set_all_seeds(config.random_seed)
    
    total_start_time = time.time()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - 交差検証解析")
    print("="*60)
    print(f"交差検証モード: {config.cross_validation_mode}")
    print(f"推定モード: {config.estimation_mode}")
    print(f"推定指標: {', '.join(config.target_signals)}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    if config.use_lab:
        print(f"LABデータ: 使用（RGB+LAB = {config.num_channels}チャンネル）")
    else:
        print(f"LABデータ: 未使用")
    
    print(f"\n【高速化設定】")
    print(f"  AMP (自動混合精度): {'有効 (float16)' if config.use_amp else '無効 (float32)'}")
    print(f"  torch.compile: {'有効' if config.use_compile and torch.__version__.startswith('2.') else '無効'}")
    print(f"  バッチサイズ: {config.batch_size}")
    
    print(f"\nデータ拡張: {'有効' if config.use_augmentation else '無効'}")
    print(f"デバイス: {config.device}")
    
    # GPU情報の表示
    if torch.cuda.is_available():
        print(f"\nGPU情報:")
        print(f"  デバイス名: {torch.cuda.get_device_name(0)}")
        print(f"  メモリ容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n保存先: {config.save_path}")
    print(f"乱数シード: {config.random_seed}")
    
    if config.cross_validation_mode == "all_subjects":
        # 全被験者5分割交差検証
        print(f"\n全被験者{config.n_folds_all_subjects}分割交差検証を実行します")
        print(f"被験者数: {len(config.subjects)}")
        fold_results = all_subjects_cross_validation(config)
    else:
        # 個人内6分割交差検証
        print(f"\n個人内6分割交差検証を実行します")
        print(f"被験者数: {len(config.subjects)}")
        all_subjects_results = within_subject_cross_validation(config)
    
    total_time = time.time() - total_start_time
    
    # 設定サマリー保存
    save_dir = Path(config.save_path)
    save_config_summary(config, total_time, save_dir)
    
    print(f"\n{'='*60}")
    print("処理完了")
    print(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
    print(f"結果保存先: {config.save_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
