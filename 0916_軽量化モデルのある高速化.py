import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    
    # CuDNNの決定的動作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # データローダーのワーカー初期化
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker

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
        
        # ================================
        # LAB変換データ使用設定
        # ================================
        self.use_lab = True  # LABデータを使用するか（True: RGB+LAB, False: RGBのみ）
        self.lab_filename = "_downsampled_1Hzver2.npy"  # LABデータのファイル名
        
        # ================================
        # 訓練・検証分割設定
        # ================================
        self.train_val_split_ratio = 0.9  # 訓練データの割合（90%）→1:9分割
        
        # 検証データ分割戦略
        self.validation_split_strategy = 'stratified'  # 'sequential' または 'stratified'
        
        # 層化サンプリング設定（stratified選択時のみ使用）
        self.n_strata = 5  # 層の数（信号値を5つの範囲に分割）
        self.stratification_method = 'quantile'  # 'equal_range' or 'quantile'
        
        # 血行動態信号タイプ設定
        self.signal_type = "CO"  # "CO", "HbO", "HbR", "HbT" など
        self.signal_prefix = "CO_s2"  # ファイル名のプレフィックス
        
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
        
        # ================================
        # モデル設定
        # ================================
        # モデルタイプ選択（"3d", "2d", "3d_light", "2d_light"）
        self.model_type = "2d_light"  # 軽量化版2Dモデルを使用
        
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
        
        # データ形状設定（CalibrationPhys準拠モデル用）
        self.time_frames = 360  # 時間フレーム数
        self.height = 36  # 画像の高さ
        self.width = 36   # 画像の幅
        self.input_shape = (self.time_frames, self.height, self.width, self.num_channels)
        
        # ================================
        # データ拡張設定
        # ================================
        self.use_augmentation = True  # データ拡張を使用するか
        
        # データ拡張のパラメータ
        if self.use_augmentation:
            # ランダムクロップ
            self.crop_enabled = True
            self.crop_size_ratio = 0.9  # 元のサイズの90%にクロップ
            
            # 回転
            self.rotation_enabled = True
            self.rotation_range = 5  # ±5度の範囲で回転
            
            # 時間軸ストレッチング
            self.time_stretch_enabled = True
            self.time_stretch_range = (0.9, 1.1)  # 90%～110%の速度変化
            
            # 明度・コントラスト調整
            self.brightness_contrast_enabled = True
            self.brightness_range = 0.2  # ±20%の明度変化
            self.contrast_range = 0.2    # ±20%のコントラスト変化
            
            # 各拡張の適用確率
            self.aug_probability = 0.5  # 各拡張を50%の確率で適用
        else:
            self.crop_enabled = False
            self.rotation_enabled = False
            self.time_stretch_enabled = False
            self.brightness_contrast_enabled = False
        
        # ================================
        # 学習設定（モデルタイプに応じて自動調整）
        # ================================
        if self.model_type == "3d":
            # 標準3Dモデル
            self.batch_size = 8
            self.epochs = 150
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 30
            self.gradient_clip_val = 0.5
        elif self.model_type == "3d_light":
            # 軽量化3Dモデル
            self.batch_size = 16  # 軽量化により大きめのバッチサイズ可能
            self.epochs = 200  # 高速化により多めのエポック
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 40
            self.gradient_clip_val = 0.5
        elif self.model_type == "2d":
            # 標準2Dモデル
            self.batch_size = 16
            self.epochs = 150
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 30
            self.gradient_clip_val = 0.5
        elif self.model_type == "2d_light":
            # 軽量化2Dモデル
            self.batch_size = 32  # 軽量化により大きめのバッチサイズ可能
            self.epochs = 200  # 高速化により多めのエポック
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 40
            self.gradient_clip_val = 0.5
        else:
            # デフォルト設定
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
        self.patience_improvement_threshold = 0.995  # 0.5%以上の改善を要求
        self.min_delta = 0.0001  # 最小改善量
        
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
        """
        ランダムクロップ
        data: (T, H, W, C) or (H, W, C)
        """
        if np.random.random() > self.config.aug_probability or not self.config.crop_enabled:
            return data
        
        if data.ndim == 4:  # (T, H, W, C)
            t, h, w, c = data.shape
            new_h = int(h * self.config.crop_size_ratio)
            new_w = int(w * self.config.crop_size_ratio)
            
            # ランダムな開始位置
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
            
            # クロップ
            cropped = data[:, top:top+new_h, left:left+new_w, :]
            
            # 元のサイズにリサイズ
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
        """
        ランダム回転
        data: (T, H, W, C) or (H, W, C)
        """
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
        """
        高速な時間軸ストレッチング（PyTorch F.interpolate使用）
        rgb_np: (T,H,W,C) in [0,1]
        factor: ストレッチファクター
        """
        # NumPy配列をTorchテンソルに変換
        x = torch.from_numpy(rgb_np).permute(3,0,1,2).unsqueeze(0).float()  # (1,C,T,H,W)
        T = x.shape[2]
        T2 = max(1, int(T*factor))
        
        with torch.no_grad():
            # ストレッチ
            x = F.interpolate(x, size=(T2, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
            # 元の長さに戻す
            x = F.interpolate(x, size=(T, x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        
        # NumPy配列に戻す
        out = x.squeeze(0).permute(1,2,3,0).cpu().numpy()  # (T,H,W,C)
        return out
    
    def time_stretch(self, rgb_data, signal_data=None):
        """
        時間軸ストレッチング（高速版）
        rgb_data: (T, H, W, C)
        signal_data: (T,) or None
        """
        if np.random.random() > self.config.aug_probability or not self.config.time_stretch_enabled:
            return rgb_data, signal_data
        
        if rgb_data.ndim != 4:  # 時間次元がない場合はスキップ
            return rgb_data, signal_data
        
        stretch_factor = np.random.uniform(*self.config.time_stretch_range)
        
        # RGBデータの高速ストレッチング
        rgb_stretched = self.time_stretch_fast(rgb_data, stretch_factor)
        
        # 信号データのストレッチング（提供されている場合）
        if signal_data is not None and signal_data.ndim == 1:
            # 信号も同様にストレッチング
            t_original = len(signal_data)
            t_stretched = int(t_original * stretch_factor)
            
            f_signal = interp1d(np.arange(t_original), signal_data, 
                              kind='linear', fill_value='extrapolate')
            signal_stretched = f_signal(np.linspace(0, t_original-1, t_stretched))
            
            # 元の長さに戻す
            f_signal_back = interp1d(np.arange(len(signal_stretched)), signal_stretched,
                                    kind='linear', fill_value='extrapolate')
            signal_resampled = f_signal_back(np.linspace(0, len(signal_stretched)-1, t_original))
            
            # 周波数を調整（ストレッチファクターに応じて）
            signal_resampled = signal_resampled * stretch_factor
            
            return rgb_stretched, signal_resampled
        
        return rgb_stretched, signal_data
    
    def brightness_contrast_adjust(self, data):
        """
        明度・コントラスト調整
        data: (T, H, W, C) or (H, W, C)
        """
        if np.random.random() > self.config.aug_probability or not self.config.brightness_contrast_enabled:
            return data
        
        # 明度とコントラストの変化量
        brightness_delta = np.random.uniform(-self.config.brightness_range, 
                                            self.config.brightness_range)
        contrast_factor = np.random.uniform(1 - self.config.contrast_range, 
                                           1 + self.config.contrast_range)
        
        # データの正規化（0-1の範囲と仮定）
        data_adjusted = data.copy()
        
        # コントラスト調整
        mean = np.mean(data_adjusted, axis=tuple(range(data_adjusted.ndim-1)), keepdims=True)
        data_adjusted = (data_adjusted - mean) * contrast_factor + mean
        
        # 明度調整
        data_adjusted = data_adjusted + brightness_delta
        
        # クリッピング
        data_adjusted = np.clip(data_adjusted, 0, 1)
        
        return data_adjusted
    
    def apply_augmentation(self, rgb_data, signal_data=None, is_training=True):
        """
        すべてのデータ拡張を適用
        rgb_data: (T, H, W, C) or (H, W, C)
        signal_data: (T,) or scalar or None
        is_training: 学習時のみTrueでデータ拡張を適用
        """
        if not is_training or not self.config.use_augmentation:
            return rgb_data, signal_data
        
        # 各拡張を順次適用
        rgb_data = self.random_crop(rgb_data)
        rgb_data = self.random_rotation(rgb_data)
        
        # 時間軸ストレッチング（時間次元がある場合のみ）
        if rgb_data.ndim == 4 and self.config.time_stretch_enabled:
            rgb_data, signal_data = self.time_stretch(rgb_data, signal_data)
        
        rgb_data = self.brightness_contrast_adjust(rgb_data)
        
        return rgb_data, signal_data

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
        # 波形全体のMSE損失（3D/2Dモデル対応）
        if pred.dim() == 2 and pred.size(1) > 1:  # (B, T)形状の場合
            mse_loss = self.mse(pred, target)
            
            # 時系列相関損失
            batch_corr_loss = 0
            for i in range(pred.size(0)):
                pred_i = pred[i] - pred[i].mean()
                target_i = target[i] - target[i].mean()
                
                numerator = torch.sum(pred_i * target_i)
                denominator = torch.sqrt(torch.sum(pred_i ** 2) * torch.sum(target_i ** 2) + 1e-8)
                correlation = numerator / denominator
                batch_corr_loss += (1 - correlation)
            
            corr_loss = batch_corr_loss / pred.size(0)
        else:
            # スカラー値の場合の処理
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
        if pred.dim() == 2 and pred.size(1) > 1:  # (B, T)形状の場合
            batch_corr_loss = 0
            for i in range(pred.size(0)):
                pred_i = pred[i] - pred[i].mean()
                target_i = target[i] - target[i].mean()
                
                numerator = torch.sum(pred_i * target_i)
                denominator = torch.sqrt(torch.sum(pred_i ** 2) * torch.sum(target_i ** 2) + 1e-8)
                correlation = numerator / denominator
                batch_corr_loss += (1 - correlation)
            
            corr_loss = batch_corr_loss / pred.size(0)
        else:
            pred_mean = pred - pred.mean()
            target_mean = target - target.mean()
            
            numerator = torch.sum(pred_mean * target_mean)
            denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
            correlation = numerator / denominator
            corr_loss = 1 - correlation
        
        total_loss = self.alpha * huber_loss + self.beta * corr_loss
        
        return total_loss, huber_loss, corr_loss

# ================================
# チャンネル選択ユーティリティ
# ================================
def select_channels(data, use_channel):
    """
    データから指定されたチャンネルを選択
    data: (..., 3) or (..., 6) 形状のRGBまたはRGB+LABデータ
    use_channel: 'R', 'G', 'B', 'RGB', 'RG', 'GB', 'RB', 'LAB', 'RGB+LAB'
    """
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
    """
    信号値の分布を保持したまま訓練・検証データを分割
    
    Parameters:
    -----------
    task_rgb : array
        RGBデータ (60, H, W, C) or (60, 14, 16, C)
    task_signal : array  
        信号データ (60,) - CO/HbO等の信号値
    val_ratio : float
        検証データの割合（0.1 = 10%）
    n_strata : int
        層の数
    method : str
        'equal_range': 信号値の範囲を等間隔に分割
        'quantile': 各層のサンプル数が均等になるように分割
    
    Returns:
    --------
    train_rgb, train_signal, val_rgb, val_signal
    """
    
    n_samples = len(task_signal)
    
    # 信号値の統計情報
    signal_min = task_signal.min()
    signal_max = task_signal.max()
    signal_mean = task_signal.mean()
    signal_std = task_signal.std()
    
    # 層の境界を決定
    if method == 'equal_range':
        # 信号値の範囲を等間隔に分割
        bin_edges = np.linspace(signal_min, signal_max + 1e-10, n_strata + 1)
    elif method == 'quantile':
        # 各層のサンプル数が均等になるように分割（分位数ベース）
        quantiles = np.linspace(0, 1, n_strata + 1)
        bin_edges = np.quantile(task_signal, quantiles)
        bin_edges[-1] += 1e-10  # 最大値を含むように調整
    else:
        raise ValueError(f"Unknown stratification method: {method}")
    
    # 各サンプルを層に割り当て
    strata_assignment = np.digitize(task_signal, bin_edges) - 1
    
    train_indices = []
    val_indices = []
    
    # デバッグ情報
    strata_info = []
    
    # 各層から比例してサンプリング
    for stratum_id in range(n_strata):
        # この層に属するサンプルのインデックス
        stratum_mask = (strata_assignment == stratum_id)
        stratum_indices = np.where(stratum_mask)[0]
        
        if len(stratum_indices) == 0:
            continue
        
        # この層の信号値の範囲
        stratum_signals = task_signal[stratum_indices]
        stratum_min = stratum_signals.min()
        stratum_max = stratum_signals.max()
        
        # この層から取る検証データ数
        n_val_from_stratum = max(1, int(len(stratum_indices) * val_ratio))
        n_train_from_stratum = len(stratum_indices) - n_val_from_stratum
        
        # ランダムにシャッフルして分割
        np.random.shuffle(stratum_indices)
        val_from_stratum = stratum_indices[:n_val_from_stratum]
        train_from_stratum = stratum_indices[n_val_from_stratum:]
        
        val_indices.extend(val_from_stratum)
        train_indices.extend(train_from_stratum)
        
        # 情報を記録
        strata_info.append({
            'stratum_id': stratum_id + 1,
            'signal_range': (stratum_min, stratum_max),
            'n_total': len(stratum_indices),
            'n_train': n_train_from_stratum,
            'n_val': n_val_from_stratum
        })
    
    # インデックスをソート（時系列順を保持するため）
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    
    # 分割結果の統計情報を表示
    if len(strata_info) > 0:
        print(f"      層化サンプリング詳細 (方法: {method}):")
        for info in strata_info:
            print(f"        層{info['stratum_id']}: "
                  f"信号値[{info['signal_range'][0]:.3f}, {info['signal_range'][1]:.3f}] "
                  f"計{info['n_total']}個 → 訓練{info['n_train']}個, 検証{info['n_val']}個")
    
    # 検証データの信号値分布を確認
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
        config: Config オブジェクト（データ拡張用）
        is_training: 学習時True、検証/テスト時False
        """
        self.model_type = model_type
        self.use_channel = use_channel
        self.is_training = is_training
        
        # データ拡張の初期化
        self.augmentation = DataAugmentation(config) if config else None
        
        # データを保存（拡張前）
        self.rgb_data_raw = rgb_data
        self.signal_data_raw = signal_data
        
        # チャンネル選択を適用
        rgb_data_selected = select_channels(rgb_data, use_channel)
        
        # CalibrationPhys準拠モデル用
        self.rgb_data = torch.FloatTensor(rgb_data_selected)
        
        # signal_dataが1次元の場合、時間次元に拡張
        if signal_data.ndim == 1:
            signal_data = np.repeat(signal_data[:, np.newaxis], rgb_data.shape[1], axis=1)
        
        self.signal_data = torch.FloatTensor(signal_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        # 元データを取得
        rgb = self.rgb_data_raw[idx]
        signal = self.signal_data_raw[idx] if self.signal_data_raw.ndim > 1 else self.signal_data_raw[idx:idx+1].squeeze()
        
        # データ拡張を適用（学習時のみ）
        if self.augmentation and self.is_training:
            rgb, signal = self.augmentation.apply_augmentation(rgb, signal, self.is_training)
        
        # チャンネル選択
        rgb = select_channels(rgb, self.use_channel)
        
        # Tensorに変換
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
    """
    CalibrationPhys論文準拠のPhysNet2DCNN（3D畳み込み版）
    様々な入力サイズに対応（14x16, 36x36など）
    入力: (batch_size, time_frames, height, width, channels)
    出力: (batch_size, time_frames) の脈波/呼吸波形
    """
    def __init__(self, input_shape=None):
        super(PhysNet2DCNN_3D, self).__init__()
        
        # 入力チャンネル数を動的に設定
        if input_shape is not None:
            in_channels = input_shape[-1]
            height = input_shape[1] if len(input_shape) >= 3 else 36
            width = input_shape[2] if len(input_shape) >= 3 else 36
        else:
            in_channels = 3
            height = 36
            width = 36
        
        # 入力サイズに基づいてプーリングサイズを調整
        self.adaptive_pooling = height < 36 or width < 36
        
        # ConvBlock 1: 32 filters
        self.conv1_1 = nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1_1 = nn.BatchNorm3d(32, momentum=0.01, eps=1e-5)
        self.elu1_1 = nn.ELU(inplace=True)
        
        self.conv1_2 = nn.Conv3d(32, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1_2 = nn.BatchNorm3d(32, momentum=0.01, eps=1e-5)
        self.elu1_2 = nn.ELU(inplace=True)
        
        # 小さい入力に対応したプーリング
        if height <= 16 or width <= 16:
            self.pool1 = nn.Identity()  # プーリングをスキップ
        else:
            self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # ConvBlock 2: 64 filters
        self.conv2_1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu2_1 = nn.ELU(inplace=True)
        
        self.conv2_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu2_2 = nn.ELU(inplace=True)
        
        # 条件付きプーリング
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
        
        # Upsample
        self.upsample = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)
        
        # ConvBlock 6: 64 filters
        self.conv6_1 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn6_1 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu6_1 = nn.ELU(inplace=True)
        
        self.conv6_2 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn6_2 = nn.BatchNorm3d(64, momentum=0.01, eps=1e-5)
        self.elu6_2 = nn.ELU(inplace=True)
        
        # Adaptive Spatial Global Average Pooling（どんなサイズでも1x1に）
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Final Conv
        self.conv_final = nn.Conv3d(64, 1, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # 重み初期化
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
        """
        入力: x shape (B, T, H, W, C)
        出力: shape (B, T) 脈波/呼吸波形
        """
        batch_size = x.size(0)
        time_frames = x.size(1)
        
        # PyTorchのConv3dは (B, C, D, H, W)を期待
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        # ConvBlock 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # ConvBlock 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # ConvBlock 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # ConvBlock 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.elu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.elu4_2(x)
        x = self.pool4(x)
        x = self.dropout(x)
        
        # ConvBlock 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.elu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.elu5_2(x)
        x = self.upsample(x)
        
        # ConvBlock 6
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.elu6_1(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.elu6_2(x)
        
        # Spatial Global Average Pooling（どんなサイズでも対応）
        x = self.spatial_pool(x)
        
        # Final Conv
        x = self.conv_final(x)
        
        # 出力を整形
        x = x.squeeze(1).squeeze(-1).squeeze(-1)
        
        # 元の時間長に補間
        if x.size(-1) != time_frames:
            x = F.interpolate(x.unsqueeze(1), size=time_frames, mode='linear', align_corners=False)
            x = x.squeeze(1)
        
        return x

# ================================
# CalibrationPhys準拠 PhysNet2DCNN (2D版)
# ================================
class PhysNet2DCNN_2D(nn.Module):
    """
    2D畳み込みを使用したPhysNet2DCNN（効率的な実装）
    様々な入力サイズに対応（14x16, 36x36など）
    入力: (batch_size, time_frames, height, width, channels)
    """
    def __init__(self, input_shape=None):
        super(PhysNet2DCNN_2D, self).__init__()
        
        # 入力チャンネル数とサイズを動的に設定
        if input_shape is not None:
            in_channels = input_shape[-1]
            height = input_shape[1] if len(input_shape) >= 3 else 36
            width = input_shape[2] if len(input_shape) >= 3 else 36
        else:
            in_channels = 3
            height = 36
            width = 36
        
        # 小さい入力サイズの検出
        self.small_input = height <= 16 or width <= 16
        
        # ConvBlock 1: 32 filters
        if self.small_input:
            # 小さい入力用：カーネルサイズを小さく
            self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
            self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        
        self.bn1_1 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-5)
        self.elu1_1 = nn.ELU(inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32, momentum=0.01, eps=1e-5)
        self.elu1_2 = nn.ELU(inplace=True)
        
        # 条件付きプーリング
        if self.small_input:
            self.pool1 = nn.Identity()  # プーリングをスキップ
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 2: 64 filters
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu2_1 = nn.ELU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64, momentum=0.01, eps=1e-5)
        self.elu2_2 = nn.ELU(inplace=True)
        
        # 条件付きプーリング
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
        
        # 条件付きプーリング
        if self.small_input:
            self.pool3 = nn.Identity()  # スキップ
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
        
        # Adaptive Spatial Global Average Pooling（どんなサイズでも1x1に）
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
        
        # 重み初期化
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
        """
        入力: x shape (B, T, H, W, C)
        出力: shape (B, T) 脈波/呼吸波形
        """
        batch_size, time_frames = x.size(0), x.size(1)
        
        # 時間次元をバッチに結合
        x = x.view(batch_size * time_frames, x.size(2), x.size(3), x.size(4))
        # (B*T, H, W, C) -> (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # ConvBlock 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # ConvBlock 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # ConvBlock 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # ConvBlock 4（大きい入力の場合のみ）
        if not self.small_input:
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = self.elu4_1(x)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x = self.elu4_2(x)
            x = self.pool4(x)
            x = self.dropout(x)
        
        # ConvBlock 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.elu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.elu5_2(x)
        
        # Adaptive Spatial pooling（どんなサイズでも対応）
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
        
        x = x.permute(0, 2, 1)
        
        # 各時間ステップで予測
        x = self.fc(x)
        x = x.squeeze(-1)
        
        return x

# ================================
# 軽量化版 CalibrationPhys準拠 PhysNet2DCNN (3D版)
# ================================
class LightPhysNet2DCNN_3D(nn.Module):
    """
    軽量化版3D PhysNet2DCNN
    - ブロック数を6→4に削減
    - チャンネル数を削減（32/64→16/32）
    - Depthwise Separable Convolutionを使用
    """
    def __init__(self, input_shape=None):
        super(LightPhysNet2DCNN_3D, self).__init__()
        
        # 入力チャンネル数とサイズを動的に設定
        if input_shape is not None:
            in_channels = input_shape[-1]
            height = input_shape[1] if len(input_shape) >= 3 else 36
            width = input_shape[2] if len(input_shape) >= 3 else 36
        else:
            in_channels = 3
            height = 36
            width = 36
        
        # 小さい入力サイズの検出
        self.small_input = height <= 16 or width <= 16
        
        # チャンネル数を削減
        base_channels = 16  # 32から16に削減
        
        # ConvBlock 1: 16 filters (32→16に削減)
        self.conv1_1 = nn.Conv3d(in_channels, base_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1_1 = nn.BatchNorm3d(base_channels)
        self.elu1_1 = nn.ELU(inplace=True)
        
        # Depthwise Separable Convolution
        self.conv1_2_dw = nn.Conv3d(base_channels, base_channels, kernel_size=(1, 3, 3), 
                                     padding=(0, 1, 1), groups=base_channels)
        self.conv1_2_pw = nn.Conv3d(base_channels, base_channels, kernel_size=1)
        self.bn1_2 = nn.BatchNorm3d(base_channels)
        self.elu1_2 = nn.ELU(inplace=True)
        
        if self.small_input:
            self.pool1 = nn.Identity()
        else:
            self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # ConvBlock 2: 32 filters (64→32に削減)
        self.conv2_1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_1 = nn.BatchNorm3d(base_channels*2)
        self.elu2_1 = nn.ELU(inplace=True)
        
        # Depthwise Separable
        self.conv2_2_dw = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=(3, 3, 3), 
                                     padding=(1, 1, 1), groups=base_channels*2)
        self.conv2_2_pw = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=1)
        self.bn2_2 = nn.BatchNorm3d(base_channels*2)
        self.elu2_2 = nn.ELU(inplace=True)
        
        self.pool2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # ConvBlock 3: 32 filters (ブロック数削減のため4,5を省略)
        self.conv3_1 = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.bn3_1 = nn.BatchNorm3d(base_channels*2)
        self.elu3_1 = nn.ELU(inplace=True)
        
        self.conv3_2_dw = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=(3, 1, 1), 
                                     padding=(1, 0, 0), groups=base_channels*2)
        self.conv3_2_pw = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=1)
        self.bn3_2 = nn.BatchNorm3d(base_channels*2)
        self.elu3_2 = nn.ELU(inplace=True)
        
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # Upsample（元の時間長に戻す）
        self.upsample = nn.Upsample(scale_factor=(4, 1, 1), mode='trilinear', align_corners=False)
        
        # ConvBlock 4: Final processing
        self.conv4_1 = nn.Conv3d(base_channels*2, base_channels*2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn4_1 = nn.BatchNorm3d(base_channels*2)
        self.elu4_1 = nn.ELU(inplace=True)
        
        # Adaptive Spatial Global Average Pooling
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Final Conv
        self.conv_final = nn.Conv3d(base_channels*2, 1, kernel_size=1)
        
        # Dropout（軽減）
        self.dropout = nn.Dropout(0.1)
        
        # 重み初期化
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
        """
        入力: x shape (B, T, H, W, C)
        出力: shape (B, T) 脈波/呼吸波形
        """
        batch_size = x.size(0)
        time_frames = x.size(1)
        
        # PyTorchのConv3dは (B, C, D, H, W)を期待
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        # ConvBlock 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        
        # Depthwise Separable
        x = self.conv1_2_dw(x)
        x = self.conv1_2_pw(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # ConvBlock 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        
        # Depthwise Separable
        x = self.conv2_2_dw(x)
        x = self.conv2_2_pw(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # ConvBlock 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        
        # Depthwise Separable
        x = self.conv3_2_dw(x)
        x = self.conv3_2_pw(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        x = self.pool3(x)
        
        # Upsample
        x = self.upsample(x)
        
        # ConvBlock 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.elu4_1(x)
        
        # Spatial Global Average Pooling
        x = self.spatial_pool(x)
        
        # Final Conv
        x = self.conv_final(x)
        
        # 出力を整形
        x = x.squeeze(1).squeeze(-1).squeeze(-1)
        
        # 元の時間長に補間
        if x.size(-1) != time_frames:
            x = F.interpolate(x.unsqueeze(1), size=time_frames, mode='linear', align_corners=False)
            x = x.squeeze(1)
        
        return x

# ================================
# 軽量化版 CalibrationPhys準拠 PhysNet2DCNN (2D版)
# ================================
class LightPhysNet2DCNN_2D(nn.Module):
    """
    軽量化版2D PhysNet2DCNN
    - ブロック数を5→3に削減
    - チャンネル数を削減
    - Depthwise Separable Convolutionを使用
    - Temporal処理を簡略化
    """
    def __init__(self, input_shape=None):
        super(LightPhysNet2DCNN_2D, self).__init__()
        
        # 入力チャンネル数とサイズを動的に設定
        if input_shape is not None:
            in_channels = input_shape[-1]
            height = input_shape[1] if len(input_shape) >= 3 else 36
            width = input_shape[2] if len(input_shape) >= 3 else 36
        else:
            in_channels = 3
            height = 36
            width = 36
        
        # 小さい入力サイズの検出
        self.small_input = height <= 16 or width <= 16
        
        # チャンネル数を削減
        base_channels = 16  # 32から16に削減
        
        # ConvBlock 1: 16 filters
        if self.small_input:
            kernel_size = 3
        else:
            kernel_size = 5
        
        self.conv1_1 = nn.Conv2d(in_channels, base_channels, kernel_size=kernel_size, 
                                 padding=kernel_size//2)
        self.bn1_1 = nn.BatchNorm2d(base_channels)
        self.elu1_1 = nn.ELU(inplace=True)
        
        # Depthwise Separable Convolution
        self.conv1_2_dw = nn.Conv2d(base_channels, base_channels, kernel_size=3, 
                                     padding=1, groups=base_channels)
        self.conv1_2_pw = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.bn1_2 = nn.BatchNorm2d(base_channels)
        self.elu1_2 = nn.ELU(inplace=True)
        
        if self.small_input:
            self.pool1 = nn.Identity()
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 2: 32 filters
        self.conv2_1 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(base_channels*2)
        self.elu2_1 = nn.ELU(inplace=True)
        
        # Depthwise Separable
        self.conv2_2_dw = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, 
                                     padding=1, groups=base_channels*2)
        self.conv2_2_pw = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=1)
        self.bn2_2 = nn.BatchNorm2d(base_channels*2)
        self.elu2_2 = nn.ELU(inplace=True)
        
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ConvBlock 3: 32 filters (最終ブロック)
        self.conv3_1 = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(base_channels*2)
        self.elu3_1 = nn.ELU(inplace=True)
        
        # Adaptive Spatial Global Average Pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal processing（簡略化）
        self.temporal_conv = nn.Conv1d(base_channels*2, base_channels, kernel_size=3, padding=1)
        self.temporal_bn = nn.BatchNorm1d(base_channels)
        self.temporal_elu = nn.ELU(inplace=True)
        
        # Final layer
        self.fc = nn.Linear(base_channels, 1)
        
        # Dropout（軽減）
        self.dropout = nn.Dropout(0.1)
        
        # 重み初期化
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
        """
        入力: x shape (B, T, H, W, C)
        出力: shape (B, T) 脈波/呼吸波形
        """
        batch_size, time_frames = x.size(0), x.size(1)
        
        # 時間次元をバッチに結合
        x = x.view(batch_size * time_frames, x.size(2), x.size(3), x.size(4))
        # (B*T, H, W, C) -> (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # ConvBlock 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        
        # Depthwise Separable
        x = self.conv1_2_dw(x)
        x = self.conv1_2_pw(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # ConvBlock 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        
        # Depthwise Separable
        x = self.conv2_2_dw(x)
        x = self.conv2_2_pw(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # ConvBlock 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        
        # Adaptive Spatial pooling
        x = self.spatial_pool(x)
        x = x.view(batch_size, time_frames, -1)
        
        # 時間次元の処理（簡略化）
        x = x.permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.temporal_elu(x)
        
        x = x.permute(0, 2, 1)
        
        # 各時間ステップで予測
        x = self.fc(x)
        x = x.squeeze(-1)
        
        return x

# ================================
# モデル作成関数
# ================================
def create_model(config):
    """設定に基づいてモデルを作成"""
    if config.model_type == "3d":
        model = PhysNet2DCNN_3D(config.input_shape)
        model_name = "PhysNet2DCNN_3D (CalibrationPhys準拠)"
    elif config.model_type == "3d_light":
        model = LightPhysNet2DCNN_3D(config.input_shape)
        model_name = "LightPhysNet2DCNN_3D (軽量化版)"
    elif config.model_type == "2d":
        model = PhysNet2DCNN_2D(config.input_shape)
        model_name = "PhysNet2DCNN_2D (CalibrationPhys準拠 効率版)"
    elif config.model_type == "2d_light":
        model = LightPhysNet2DCNN_2D(config.input_shape)
        model_name = "LightPhysNet2DCNN_2D (軽量化版)"
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    
    if config.verbose:
        print(f"\n選択モデル: {model_name}")
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
        print(f"データ拡張: {'有効' if config.use_augmentation else '無効'}")
        if config.use_augmentation:
            print(f"  - ランダムクロップ: {'有効' if config.crop_enabled else '無効'}")
            print(f"  - 回転: {'有効' if config.rotation_enabled else '無効'}")
            print(f"  - 時間軸ストレッチング: {'有効 (高速版)' if config.time_stretch_enabled else '無効'}")
            print(f"  - 明度・コントラスト調整: {'有効' if config.brightness_contrast_enabled else '無効'}")
        print(f"Warmup: {config.warmup_epochs}エポック (初期学習率×{config.warmup_lr_factor})")
        print(f"勾配クリッピング: {config.gradient_clip_val}")
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
        
        if "light" in config.model_type:
            print("【軽量化版】高速処理のため、精度と速度のバランスを最適化")
        
        if config.model_type == "3d":
            print("【注意】3Dモデルはメモリを多く使用します。バッチサイズの調整を推奨します。")
        elif config.model_type == "3d_light":
            print("【推奨】軽量化3Dモデルは処理速度が約2倍向上します")
        elif config.model_type == "2d":
            print("【推奨】2Dモデルは計算効率が良く、大きなバッチサイズでも動作します。")
        elif config.model_type == "2d_light":
            print("【推奨】軽量化2Dモデルは処理速度が約3倍向上します")
    
    return model

def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み（LABデータ対応）"""
    
    # すべてのモデルで同じRGBデータファイルを使用
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません: {rgb_path}")
        return None, None
    
    rgb_data = np.load(rgb_path)  # Shape: (360, 14, 16, 3)
    print(f"  RGBデータ読み込み成功: {rgb_data.shape}")
    
    # LABデータの読み込み（オプション）
    if config.use_lab:
        # LABデータは同じディレクトリの _downsampled_1Hzver2.npy
        lab_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_downsampled_1Hzver2.npy")
        
        if not os.path.exists(lab_path):
            print(f"警告: {subject}のLABデータが見つかりません: {lab_path}")
            print(f"  LABデータなしで処理を続行します。")
            # LABデータなしで続行（RGBのみ使用）
            config.use_lab = False
            config.use_channel = 'RGB'
            config.num_channels = 3
        else:
            lab_data = np.load(lab_path)
            print(f"  LABデータ読み込み成功: {lab_data.shape}")
            
            # RGBとLABデータの形状を確認
            if rgb_data.shape != lab_data.shape:
                print(f"警告: {subject}のRGBとLABデータの形状が一致しません")
                print(f"  RGB shape: {rgb_data.shape}, LAB shape: {lab_data.shape}")
                # 形状が異なる場合もRGBのみで続行
                config.use_lab = False
                config.use_channel = 'RGB'
                config.num_channels = 3
            else:
                # RGB+LABデータを結合（チャンネル次元で連結）
                combined_data = np.concatenate([rgb_data, lab_data], axis=-1)
                print(f"  RGB+LAB結合データ: {combined_data.shape}")
                
                # データの正規化（LABデータも0-1の範囲に）
                if lab_data.max() > 1.0:
                    combined_data[..., 3:] = combined_data[..., 3:] / 255.0
                
                rgb_data = combined_data
    
    # 3D/2Dモデル用にデータ形状を調整
    # データ形状の確認と調整
    if rgb_data.ndim == 5:  # (N, T, H, W, C)
        pass
    elif rgb_data.ndim == 4:  # (N, H, W, C) の場合、時間次元を追加
        rgb_data = np.expand_dims(rgb_data, axis=1)
        rgb_data = np.repeat(rgb_data, config.time_frames, axis=1)
    
    # 信号データの読み込み
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
    
    # データの正規化（0-1の範囲に）
    if rgb_data[..., :3].max() > 1.0:  # RGBチャンネルのみチェック
        rgb_data[..., :3] = rgb_data[..., :3] / 255.0
    
    return rgb_data, signal_data

# ================================
# 学習関数（Warmup追加）
# ================================
def train_model(model, train_loader, val_loader, config, fold=None, subject=None):
    """モデルの学習（Warmup対応）"""
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    subject_str = f"{subject}" if subject is not None else ""
    
    if config.verbose:
        print(f"\n  学習開始 {subject_str} {fold_str}")
        print(f"    モデル: {config.model_type}")
        print(f"    エポック数: {config.epochs}")
        print(f"    バッチサイズ: {config.batch_size}")
        print(f"    Warmupエポック数: {config.warmup_epochs}")
        print(f"    初期学習率: {config.learning_rate}")
        print(f"    Warmup開始学習率: {config.learning_rate * config.warmup_lr_factor}")
    
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
        criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
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
    train_correlations = []
    val_correlations = []
    best_val_loss = float('inf')
    best_val_corr = -1
    patience_counter = 0
    
    # 学習時の予測値と真値を保存
    train_preds_best = None
    train_targets_best = None
    
    for epoch in range(config.epochs):
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
            pred = model(rgb)
            
            # 損失計算
            if hasattr(criterion, 'alpha'):
                loss, mse_loss, corr_loss = criterion(pred, sig)
            else:
                loss = criterion(pred, sig)
            
            loss.backward()
            
            # 勾配クリッピング（設定値使用）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
            optimizer.step()
            
            if scheduler_per_batch and epoch >= config.warmup_epochs:
                scheduler.step()
            
            train_loss += loss.item()
            
            # 予測値と真値を保存
            if pred.dim() == 2:
                # 3D/2Dモデルの場合、時系列の平均値を保存
                train_preds_all.extend(pred.mean(dim=1).detach().cpu().numpy())
                train_targets_all.extend(sig.mean(dim=1).detach().cpu().numpy())
            else:
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
                
                # 損失計算
                if hasattr(criterion, 'alpha'):
                    loss, mse_loss, corr_loss = criterion(pred, sig)
                else:
                    loss = criterion(pred, sig)
                
                val_loss += loss.item()
                
                # 予測値と真値を保存
                if pred.dim() == 2:
                    val_preds.extend(pred.mean(dim=1).cpu().numpy())
                    val_targets.extend(sig.mean(dim=1).cpu().numpy())
                else:
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
        
        # スケジューラー更新（Warmup終了後）
        if not scheduler_per_batch and epoch >= config.warmup_epochs:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # モデル保存（改善判定を厳密化）
        improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss > 0 else 1
        if improvement > config.min_delta or val_corr > best_val_corr * config.patience_improvement_threshold:
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
        if config.verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch < config.warmup_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            if epoch < config.warmup_epochs:
                print(f"    [Warmup] Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e}")
            else:
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
    waveform_predictions = []
    waveform_targets = []
    
    with torch.no_grad():
        for rgb, sig in test_loader:
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            pred = model(rgb)
            
            # 波形全体を保存（3D/2Dモデル用）
            if pred.dim() == 2:
                waveform_predictions.append(pred.cpu().numpy())
                waveform_targets.append(sig.cpu().numpy())
                # 平均値も計算
                predictions.extend(pred.mean(dim=1).cpu().numpy())
                targets.extend(sig.mean(dim=1).cpu().numpy())
            else:
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
    
    result = {
        'mae': mae, 'rmse': rmse, 'corr': corr,
        'r2': r2, 'p_value': p_value,
        'predictions': predictions, 'targets': targets
    }
    
    # 波形データがある場合は追加
    if waveform_predictions:
        result['waveform_predictions'] = np.concatenate(waveform_predictions, axis=0)
        result['waveform_targets'] = np.concatenate(waveform_targets, axis=0)
    
    return result

# ================================
# 6分割交差検証（改良版：分割戦略選択対応）
# ================================
def task_cross_validation(rgb_data, signal_data, config, subject, subject_save_dir):
    """タスクごとの6分割交差検証（分割戦略選択対応版）"""
    
    fold_results = []
    all_test_predictions = []
    all_test_targets = []
    all_test_task_indices = []
    all_test_tasks = []  # タスク名を記録
    
    # 乱数シードを設定（再現性のため）
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    # DataLoader用のワーカー初期化関数を取得
    seed_worker = set_all_seeds(config.random_seed)
    
    for fold, test_task in enumerate(config.tasks):
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
                all_test_tasks.extend([test_task] * config.task_duration)  # タスク名を記録
            else:
                # 分割戦略に応じて訓練・検証データを分離
                if config.validation_split_strategy == 'stratified':
                    # 層化サンプリング
                    train_rgb, train_signal, val_rgb, val_signal = stratified_sampling_split(
                        task_rgb, 
                        task_signal,
                        val_ratio=(1 - config.train_val_split_ratio),
                        n_strata=config.n_strata,
                        method=config.stratification_method
                    )
                    train_rgb_list.append(train_rgb)
                    train_signal_list.append(train_signal)
                    val_rgb_list.append(val_rgb)
                    val_signal_list.append(val_signal)
                else:  # 'sequential' (default)
                    # 既存の方法（各タスクの最後10%）
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
            
            # 分割後の信号値分布を確認
            print(f"    全体の信号値分布:")
            print(f"      訓練: 平均={train_signal.mean():.3f}, 標準偏差={train_signal.std():.3f}, "
                  f"範囲=[{train_signal.min():.3f}, {train_signal.max():.3f}]")
            print(f"      検証: 平均={val_signal.mean():.3f}, 標準偏差={val_signal.std():.3f}, "
                  f"範囲=[{val_signal.min():.3f}, {val_signal.max():.3f}]")
            print(f"      テスト: 平均={test_signal.mean():.3f}, 標準偏差={test_signal.std():.3f}, "
                  f"範囲=[{test_signal.min():.3f}, {test_signal.max():.3f}]")
        
        # データローダー作成（データ拡張対応、決定的動作のため設定）
        train_dataset = CODataset(train_rgb, train_signal, config.model_type, 
                                 config.use_channel, config, is_training=True)
        val_dataset = CODataset(val_rgb, val_signal, config.model_type, 
                               config.use_channel, config, is_training=False)
        test_dataset = CODataset(test_rgb, test_signal, config.model_type, 
                                config.use_channel, config, is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                 num_workers=0, worker_init_fn=seed_worker, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                               num_workers=0, worker_init_fn=seed_worker, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=0, worker_init_fn=seed_worker, pin_memory=False)
        
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
        
        # 各Foldのプロット（色分け対応）
        plot_fold_results_colored(fold_results[-1], subject_save_dir, config)
    
    # テスト予測を元の順序に並び替え
    sorted_indices = np.argsort(all_test_task_indices)
    all_test_predictions = np.array(all_test_predictions)[sorted_indices]
    all_test_targets = np.array(all_test_targets)[sorted_indices]
    all_test_tasks = np.array(all_test_tasks)[sorted_indices]
    
    return fold_results, all_test_predictions, all_test_targets, all_test_tasks

# ================================
# プロット関数（色分け対応）
# ================================
def plot_fold_results_colored(result, save_dir, config):
    """各Foldの結果をプロット（色分け対応）"""
    fold = result['fold']
    test_task = result['test_task']
    task_color = config.task_colors[test_task]
    
    # 訓練データ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['train_targets'], result['train_predictions'], 
                alpha=0.5, s=10, color='gray', label='訓練データ')
    min_val = min(result['train_targets'].min(), result['train_predictions'].min())
    max_val = max(result['train_targets'].max(), result['train_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"Fold {fold} 訓練データ - MAE: {result['train_mae']:.3f}, Corr: {result['train_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # テストデータ散布図（色分け）
    plt.figure(figsize=(10, 8))
    plt.scatter(result['test_targets'], result['test_predictions'], 
                alpha=0.6, s=20, color=task_color, label=f'テストタスク: {test_task}')
    min_val = min(result['test_targets'].min(), result['test_predictions'].min())
    max_val = max(result['test_targets'].max(), result['test_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"Fold {fold} テストデータ ({test_task}) - MAE: {result['test_mae']:.3f}, Corr: {result['test_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 波形比較（色分け）
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
    plt.plot(result['test_predictions'], color=task_color, linestyle='-', 
             label=f'予測 ({test_task})', alpha=0.7, linewidth=1)
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'Fold {fold} テストデータ波形 ({test_task})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_waveforms.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_subject_summary_colored(fold_results, all_test_predictions, all_test_targets, 
                                all_test_tasks, subject, subject_save_dir, config):
    """被験者の全体結果をプロット（タスクごとに色分け）"""
    
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
    plt.scatter(all_train_targets, all_train_predictions, alpha=0.5, s=10, color='gray')
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
    
    # 全テストデータ散布図（タスクごとに色分け）
    plt.figure(figsize=(12, 8))
    for task in config.tasks:
        mask = all_test_tasks == task
        if np.any(mask):
            plt.scatter(all_test_targets[mask], all_test_predictions[mask], 
                       alpha=0.6, s=20, color=config.task_colors[task], label=task)
    
    min_val = min(all_test_targets.min(), all_test_predictions.min())
    max_val = max(all_test_targets.max(), all_test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"{subject} 全テストデータ - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_scatter_colored.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 全テストデータ連結波形（タスクごとに色分け）
    plt.figure(figsize=(20, 8))
    
    # 真値を薄い色でプロット
    plt.plot(all_test_targets, 'k-', label='真値', alpha=0.4, linewidth=1)
    
    # 予測値をタスクごとに色分けしてプロット
    for i, task in enumerate(config.tasks):
        start_idx = i * config.task_duration
        end_idx = (i + 1) * config.task_duration
        plt.plot(range(start_idx, end_idx), all_test_predictions[start_idx:end_idx], 
                color=config.task_colors[task], label=f'予測 ({task})', 
                alpha=0.8, linewidth=1.5)
    
    # タスク境界に縦線
    for i in range(1, 6):
        plt.axvline(x=i*60, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'{subject} 全テストデータ連結波形 - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_waveform_colored.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return all_train_mae, all_train_corr, all_test_mae, all_test_corr

def plot_all_subjects_summary_unified(all_subjects_results, config):
    """全被験者のサマリープロット（1つのグラフに統合）"""
    save_dir = Path(config.save_path)
    
    # カラーマップを準備（32人の被験者用）
    colors = plt.cm.hsv(np.linspace(0, 1, len(all_subjects_results)))
    
    # 訓練データ：全被験者を1つのグラフにプロット
    plt.figure(figsize=(14, 10))
    for i, result in enumerate(all_subjects_results):
        # 各被験者のデータを取得
        all_train_predictions = np.concatenate([r['train_predictions'] for r in result['fold_results']])
        all_train_targets = np.concatenate([r['train_targets'] for r in result['fold_results']])
        
        # 散布図をプロット
        plt.scatter(all_train_targets, all_train_predictions, 
                   alpha=0.3, s=5, color=colors[i], label=result['subject'])
    
    # 対角線
    all_min = min([np.concatenate([r['train_targets'] for r in res['fold_results']]).min() 
                  for res in all_subjects_results])
    all_max = max([np.concatenate([r['train_targets'] for r in res['fold_results']]).max() 
                  for res in all_subjects_results])
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.7)
    
    plt.xlabel('真値')
    plt.ylabel('予測値')
    
    # 平均メトリクス
    avg_train_mae = np.mean([r['train_mae'] for r in all_subjects_results])
    avg_train_corr = np.mean([r['train_corr'] for r in all_subjects_results])
    
    plt.title(f'全被験者 訓練データ - 平均MAE: {avg_train_mae:.3f}, 平均Corr: {avg_train_corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 凡例を2列で右側に配置
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_train_unified.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # テストデータ：全被験者を1つのグラフにプロット
    plt.figure(figsize=(14, 10))
    for i, result in enumerate(all_subjects_results):
        # 各被験者のテストデータを取得
        all_test_predictions = result['all_test_predictions']
        all_test_targets = result['all_test_targets']
        
        # 散布図をプロット
        plt.scatter(all_test_targets, all_test_predictions, 
                   alpha=0.4, s=8, color=colors[i], label=result['subject'])
    
    # 対角線
    all_min = min([res['all_test_targets'].min() for res in all_subjects_results])
    all_max = max([res['all_test_targets'].max() for res in all_subjects_results])
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, alpha=0.7)
    
    plt.xlabel('真値')
    plt.ylabel('予測値')
    
    # 平均メトリクス
    avg_test_mae = np.mean([r['test_mae'] for r in all_subjects_results])
    avg_test_corr = np.mean([r['test_corr'] for r in all_subjects_results])
    
    plt.title(f'全被験者 テストデータ - 平均MAE: {avg_test_mae:.3f}, 平均Corr: {avg_test_corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    # 凡例を2列で右側に配置
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_test_unified.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 被験者ごとのパフォーマンス比較（棒グラフ）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    subjects = [r['subject'] for r in all_subjects_results]
    train_corrs = [r['train_corr'] for r in all_subjects_results]
    test_corrs = [r['test_corr'] for r in all_subjects_results]
    
    x = np.arange(len(subjects))
    
    # 訓練相関
    bars1 = ax1.bar(x, train_corrs, color=colors)
    ax1.axhline(y=avg_train_corr, color='r', linestyle='--', label=f'平均: {avg_train_corr:.3f}')
    ax1.set_ylabel('相関係数')
    ax1.set_title('訓練データ相関')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # テスト相関
    bars2 = ax2.bar(x, test_corrs, color=colors)
    ax2.axhline(y=avg_test_corr, color='r', linestyle='--', label=f'平均: {avg_test_corr:.3f}')
    ax2.set_ylabel('相関係数')
    ax2.set_xlabel('被験者')
    ax2.set_title('テストデータ相関')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subjects, rotation=45, ha='right')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    # 完全な再現性のための乱数シード設定
    set_all_seeds(config.random_seed)
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - 個人内解析（6分割交差検証）")
    print("="*60)
    print(f"血行動態信号: {config.signal_type}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    if config.use_lab:
        print(f"LABデータ: 使用（RGB+LAB = {config.num_channels}チャンネル）")
    else:
        print(f"LABデータ: 未使用")
    print(f"訓練:検証データ比率: {config.train_val_split_ratio*100:.0f}:{(1-config.train_val_split_ratio)*100:.0f}")
    print(f"検証データ分割戦略: {config.validation_split_strategy}")
    if config.validation_split_strategy == 'stratified':
        print(f"  層化サンプリング設定:")
        print(f"    層の数: {config.n_strata}")
        print(f"    分割方法: {config.stratification_method}")
    print(f"データ拡張: {'有効' if config.use_augmentation else '無効'}")
    if config.use_augmentation:
        print(f"  拡張手法:")
        if config.crop_enabled:
            print(f"    - ランダムクロップ ({config.crop_size_ratio*100:.0f}%)")
        if config.rotation_enabled:
            print(f"    - 回転 (±{config.rotation_range}度)")
        if config.time_stretch_enabled:
            print(f"    - 時間軸ストレッチング ({config.time_stretch_range[0]:.1f}x-{config.time_stretch_range[1]:.1f}x) [高速版]")
        if config.brightness_contrast_enabled:
            print(f"    - 明度・コントラスト調整 (±{config.brightness_range*100:.0f}%)")
    print(f"Warmup: {config.warmup_epochs}エポック (初期学習率×{config.warmup_lr_factor})")
    print(f"損失関数: {config.loss_type}")
    print(f"スケジューラー: {config.scheduler_type}")
    print(f"勾配クリッピング: {config.gradient_clip_val}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    print(f"被験者数: {len(config.subjects)}")
    print(f"乱数シード: {config.random_seed} (完全再現性モード)")
    
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
            if config.use_lab and rgb_data.shape[-1] == 6:
                print(f"  チャンネル構成: RGB(3) + LAB(3) = {rgb_data.shape[-1]}チャンネル")
            
            # 6分割交差検証実行（色分け対応、分割戦略選択対応）
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
            
            print(f"\n  {subject} 完了:")
            print(f"    全体訓練: MAE={train_mae:.4f}, Corr={train_corr:.4f}")
            print(f"    全体テスト: MAE={test_mae:.4f}, Corr={test_corr:.4f}")
            
            # 結果をCSVファイルに保存
            results_df = pd.DataFrame({
                'Subject': [subject],
                'Train_MAE': [train_mae],
                'Train_Corr': [train_corr],
                'Test_MAE': [test_mae],
                'Test_Corr': [test_corr]
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
        
        # 設定情報も保存
        with open(save_dir / 'config_summary.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("実験設定\n")
            f.write("="*60 + "\n")
            f.write(f"実行日時: {config.timestamp}\n")
            f.write(f"モデルタイプ: {config.model_type}\n")
            f.write(f"信号タイプ: {config.signal_type}\n")
            f.write(f"使用チャンネル: {config.use_channel}\n")
            f.write(f"LABデータ使用: {config.use_lab}\n")
            f.write(f"訓練:検証比率: {config.train_val_split_ratio}:{1-config.train_val_split_ratio}\n")
            f.write(f"検証分割戦略: {config.validation_split_strategy}\n")
            f.write(f"データ拡張: {config.use_augmentation}\n")
            f.write(f"  時間軸ストレッチング: 高速版 (F.interpolate使用)\n")
            f.write(f"Warmupエポック: {config.warmup_epochs}\n")
            f.write(f"学習率: {config.learning_rate}\n")
            f.write(f"バッチサイズ: {config.batch_size}\n")
            f.write(f"エポック数: {config.epochs}\n")
            f.write(f"損失関数: {config.loss_type}\n")
            f.write(f"勾配クリッピング: {config.gradient_clip_val}\n")
            f.write(f"乱数シード: {config.random_seed}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("実験結果\n")
            f.write("="*60 + "\n")
            f.write(f"処理被験者数: {len(all_subjects_results)}/{len(config.subjects)}\n")
            f.write(f"平均訓練MAE: {avg_train_mae:.4f}±{std_train_mae:.4f}\n")
            f.write(f"平均訓練相関: {avg_train_corr:.4f}±{std_train_corr:.4f}\n")
            f.write(f"平均テストMAE: {avg_test_mae:.4f}±{std_test_mae:.4f}\n")
            f.write(f"平均テスト相関: {avg_test_corr:.4f}±{std_test_corr:.4f}\n")
    
    print(f"\n{'='*60}")
    print("処理完了")
    print(f"結果保存先: {config.save_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # メイン実行
    main()
