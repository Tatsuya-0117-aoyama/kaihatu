import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy import ndimage as ndi  # ★ 画像回転などで使用
import random
import math
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
        
        # ★血行動態信号タイプ設定
        self.signal_type = "CO"       # "CO", "HbO", "HbR", "HbT" など
        self.signal_prefix = "CO_s2"  # ファイル名のプレフィックス
        
        # 被験者設定（bp001～bp032）
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定（6分割交差検証用：各60秒）
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60   # 各タスク 60 秒（1Hz前提）
        
        # モデルタイプ選択
        # "standard": 既存1D（5ブロック）
        # "deep":     既存1D（8ブロック）
        # "physnet_2d": 2D版（C×M×T→1×1×T出力）
        # "physnet_3d": 3D版（C×H×W×T→1×1×T出力）
        self.model_type = "physnet_2d"  # ★推奨
        
        # 入力形状（36x36想定）とチャンネル
        self.use_channel = 'RGB'  # 'R','G','B','RGB'
        self.input_shape = (36, 36, (3 if self.use_channel == 'RGB' else 1))  # 既存1Dモデルで参照
        
        # PhysNet系のウィンドウ設定
        self.temporal_window = 60   # 60秒窓
        self.temporal_stride = 30   # 30秒ステップ（重複窓）
        
        # ROI整形（2D版で使用）
        # "flatten"     : H×WをそのままM=H*Wにフラット化
        # "block16x14"  : 36×36を≈16×14に平均プーリング（M≈224）
        self.roi_mode = "flatten"
        
        # ======== ★ データ拡張（オーグメンテーション）設定 ========
        self.aug_enable = True    # 拡張ON/OFF
        # それぞれの適用確率
        self.aug_prob_rotate = 0.5
        self.aug_prob_crop   = 0.5
        self.aug_prob_time   = 0.5
        self.aug_prob_bc     = 0.5  # 明度/コントラスト
        # パラメータ強度
        self.rotate_max_deg = 10.0      # ±deg
        self.crop_scale_min = 0.85      # 辺をこの比率までクロップ（例: 36→約31ピクセル）
        self.time_stretch_min = 0.90    # 0.90～1.10倍で伸縮
        self.time_stretch_max = 1.10
        self.brightness_delta = 0.08    # [0,1]に正規化したときの±シフト幅
        self.contrast_delta   = 0.10    # 1±delta の倍率

        # ======== 学習設定 ========
        if self.model_type == "standard":
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        elif self.model_type == "deep":
            self.batch_size = 32
            self.epochs = 200
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 40
        elif self.model_type in ["physnet_2d", "physnet_3d"]:
            self.batch_size = 16
            self.epochs = 120
            self.learning_rate = 1e-3
            self.weight_decay = 1e-4
            self.patience = 20
        else:
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7  # MSE/Huberの重み
        self.loss_beta = 0.3   # 相関損失の重み
        
        # 学習率スケジューラー設定
        self.scheduler_type = "cosine"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 30
        self.scheduler_T_mult = 2
        
        # 表示設定
        self.verbose = True
        self.random_seed = 42

# ================================
# カスタム損失関数
# ================================
class CombinedLoss(nn.Module):
    """MSE損失と相関損失を組み合わせた複合損失"""
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
    """Huber損失と相関損失（外れ値にロバスト）"""
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
# 既存データセット（1Dモデル用）
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
            selected_data = rgb_data  # RGB
        
        self.rgb_data = torch.FloatTensor(selected_data).permute(0, 3, 1, 2)  # (N,C,H,W)
        self.signal_data = torch.FloatTensor(signal_data)                     # (N,)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.signal_data[idx]

# ================================
# クリップデータセット（PhysNet系 + オーグメンテーション）
# ================================
class ClipDataset(Dataset):
    """
    rgb_clip: (T, H, W, C)
    signal:   (T,)  or (T,1)
    mode: '2d' -> (C, M, Tw) を返す
          '3d' -> (C, H, W, Tw) を返す
    さらに start_idx（窓の開始時刻）も返す（テスト時の重畳平均に使用）
    """
    def __init__(self, rgb_clip, signal, config, mode='2d', is_train=True):
        assert mode in ['2d', '3d']
        self.mode = mode
        self.cfg = config
        self.is_train = is_train

        # 対象チャンネル抽出
        if config.use_channel == 'R':
            rgb_clip = rgb_clip[..., 0:1]
        elif config.use_channel == 'G':
            rgb_clip = rgb_clip[..., 1:2]
        elif config.use_channel == 'B':
            rgb_clip = rgb_clip[..., 2:3]
        else:  # 'RGB'
            pass

        self.rgb = rgb_clip.astype(np.float32)           # (T,H,W,C’)
        self.sig = signal.reshape(-1).astype(np.float32) # (T,)
        self.T, self.H, self.W, self.C = self.rgb.shape

        Tw = config.temporal_window
        St = config.temporal_stride
        self.windows = []
        for s in range(0, self.T - Tw + 1, St):
            e = s + Tw
            self.windows.append((s, e))

    def __len__(self):
        return len(self.windows)

    # ---------- ここからオーグメンテーション群 ----------
    def _random_rotate(self, frame):
        # frame: (H,W,C)
        deg = random.uniform(-self.cfg.rotate_max_deg, self.cfg.rotate_max_deg)
        # 各チャンネル同じ回転
        out = np.empty_like(frame)
        for ch in range(frame.shape[2]):
            out[..., ch] = ndi.rotate(frame[..., ch], deg, reshape=False, order=1, mode='reflect')
        return out

    def _random_crop_and_resize(self, clip):
        # clip: (Tw,H,W,C)
        H, W = clip.shape[1], clip.shape[2]
        scale = random.uniform(self.cfg.crop_scale_min, 1.0)
        ch = max(1, int(round(H * scale)))
        cw = max(1, int(round(W * scale)))
        if ch == H and cw == W:
            return clip  # no-op

        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        cropped = clip[:, top:top+ch, left:left+cw, :]  # (Tw,ch,cw,C)

        # リサイズ back to (H,W) using torch (bilinear)
        # (Tw,H,W,C) → (Tw,C,H,W)
        tens = torch.from_numpy(cropped.transpose(0,3,1,2))  # (Tw,C,ch,cw)
        tens = torch.nn.functional.interpolate(
            tens, size=(H, W), mode='bilinear', align_corners=False
        )
        out = tens.numpy().transpose(0,2,3,1)  # (Tw,H,W,C)
        return out.astype(np.float32)

    def _random_brightness_contrast(self, clip):
        # clip: (Tw,H,W,C), 値スケール不明のため、min-max正規化で調整→元スケールへ戻す
        vmin = np.min(clip)
        vmax = np.max(clip)
        rng = max(vmax - vmin, 1e-6)
        norm = (clip - vmin) / rng

        # コントラスト（1±delta）、明度（±delta）
        c = 1.0 + random.uniform(-self.cfg.contrast_delta, self.cfg.contrast_delta)
        b = random.uniform(-self.cfg.brightness_delta, self.cfg.brightness_delta)
        out = norm * c + b
        out = np.clip(out, 0.0, 1.0)

        # 元スケールに戻す
        out = out * rng + vmin
        return out.astype(np.float32)

    def _random_time_stretch(self, clip, target):  # (Tw,H,W,C), (Tw,)
        Tw = clip.shape[0]
        fac = random.uniform(self.cfg.time_stretch_min, self.cfg.time_stretch_max)
        t = np.arange(Tw, dtype=np.float32)
        # 目標グリッド（0..Tw-1）に対し、ソース座標を 1/fac 倍で逆写像
        src = t / fac
        src = np.clip(src, 0, Tw - 1)

        # クリップを (Tw, P) にして各列で線形補間
        P = clip.shape[1] * clip.shape[2] * clip.shape[3]
        flat = clip.reshape(Tw, P)
        stretched = np.empty_like(flat)
        for p in range(P):
            stretched[:, p] = np.interp(t, src, flat[:, p])
        clip_out = stretched.reshape(Tw, clip.shape[1], clip.shape[2], clip.shape[3])

        # ターゲットも同様に伸縮
        target_out = np.interp(t, src, target)

        return clip_out.astype(np.float32), target_out.astype(np.float32)

    def _apply_augmentation(self, clip, target):
        # clip: (Tw,H,W,C), target: (Tw,)
        if not self.cfg.aug_enable or not self.is_train:
            return clip, target

        # 1) 回転（各フレーム、同角度）
        if random.random() < self.cfg.aug_prob_rotate:
            # 時系列の各フレームに同じ回転を適用
            deg = random.uniform(-self.cfg.rotate_max_deg, self.cfg.rotate_max_deg)
            out = np.empty_like(clip)
            for t in range(clip.shape[0]):
                for ch in range(clip.shape[3]):
                    out[t, ..., ch] = ndi.rotate(clip[t, ..., ch], deg, reshape=False, order=1, mode='reflect')
            clip = out

        # 2) ランダムクロップ（→元サイズにリサイズ）
        if random.random() < self.cfg.aug_prob_crop:
            clip = self._random_crop_and_resize(clip)

        # 3) 明度/コントラスト
        if random.random() < self.cfg.aug_prob_bc:
            clip = self._random_brightness_contrast(clip)

        # 4) 時間ストレッチ（Twは維持）
        if random.random() < self.cfg.aug_prob_time:
            clip, target = self._random_time_stretch(clip, target)

        return clip, target
    # ---------- オーグメンテーション終わり ----------

    def _to_C_M_T(self, clip):  # (Tw,H,W,C’) -> (C’, M, Tw)
        Tw, H, W, C = clip.shape
        x = clip.transpose(3, 1, 2, 0)  # (C,H,W,Tw)
        x = x.reshape(C, H * W, Tw)     # (C,M,Tw)
        return x

    def __getitem__(self, idx):
        s, e = self.windows[idx]
        clip = self.rgb[s:e]         # (Tw,H,W,C’)
        target = self.sig[s:e]       # (Tw,)

        # オーグメンテーション（学習時のみ）
        clip, target = self._apply_augmentation(clip, target)

        if self.mode == '2d':
            if self.cfg.roi_mode == 'block16x14':
                ph = math.floor(self.H / 14)
                pw = math.floor(self.W / 16)
                hh = ph * 14
                ww = pw * 16
                c = clip[:, :hh, :ww, :].reshape(
                    clip.shape[0], 14, ph, 16, pw, clip.shape[3]
                ).mean(axis=(2, 4))   # (Tw,14,16,C’)
                x = c.transpose(3, 1, 2, 0).reshape(c.shape[3], 14 * 16, c.shape[0])  # (C,M,Tw)
            else:
                x = self._to_C_M_T(clip)  # (C,M,Tw)
        else:  # '3d'
            x = clip.transpose(3, 1, 2, 0)  # (C,H,W,Tw)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(target).float()  # (Tw,)
        start_idx = torch.tensor(s, dtype=torch.long)
        return x, y, start_idx

# ================================
# 既存1Dモデル（5/8ブロック）
# ================================
class PhysNet2DCNN(nn.Module):
    """1D畳み込み版（既存）"""
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
            nn.ELU()
        )
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (B,C,H,W) → (B,C,H*W)
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        x = self.conv_block1(x); x = self.avgpool1(x); x = self.dropout(x)
        x = self.conv_block2(x); x = self.avgpool2(x); x = self.dropout(x)
        x = self.conv_block3(x); x = self.avgpool3(x); x = self.dropout(x)
        x = self.conv_block4(x); x = self.avgpool4(x); x = self.dropout(x)
        x = self.conv_block5(x)
        x = self.global_avgpool(x)
        x = self.conv_final(x)   # (B,1,1)
        x = x.squeeze()
        if B == 1: x = x.unsqueeze(0)
        return x

class DeepPhysNet2DCNN(nn.Module):
    """1D畳み込み版（8ブロック）"""
    def __init__(self, input_shape, depth_level="deep"):
        super(DeepPhysNet2DCNN, self).__init__()
        in_channels = input_shape[2]
        assert depth_level == "deep"
        self.num_blocks = 8
        self.channels = [32, 64, 64, 128, 128, 128, 256, 256]
        self.use_residual = True
        
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        prev_channels = in_channels
        for i in range(self.num_blocks):
            out_channels = self.channels[i]
            if i == 0:
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels), nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels), nn.ELU()
                )
            elif i < 4:
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels), nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels), nn.ELU()
                )
            else:
                mid = out_channels
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, mid, kernel_size=1),
                    nn.BatchNorm1d(mid), nn.ELU(),
                    nn.Conv1d(mid, mid, kernel_size=3, padding=1),
                    nn.BatchNorm1d(mid), nn.ELU(),
                    nn.Conv1d(mid, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels), nn.ELU()
                )
            self.conv_blocks.append(conv_block)
            if i < 2:
                self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            elif i < 4:
                self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2) if i % 2 == 0 else None)
            else:
                self.pools.append(None)
            if prev_channels != out_channels:
                self.residual_convs.append(nn.Conv1d(prev_channels, out_channels, kernel_size=1))
            else:
                self.residual_convs.append(None)
            prev_channels = out_channels
        
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = 128
        self.dropout_rates = [0.3, 0.2]
        self.fc = nn.Sequential(
            nn.Linear(prev_channels, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(),
            nn.Dropout(self.dropout_rates[0]),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ELU(),
            nn.Dropout(self.dropout_rates[1]),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.block_dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1)
        for i in range(self.num_blocks):
            identity = x
            x = self.conv_blocks[i](x)
            if self.residual_convs[i] is not None:
                identity = self.residual_convs[i](identity)
            if self.pools[i] is not None:
                x = self.pools[i](x)
                identity = self.pools[i](identity)
            x = x + identity * 0.3
            if i >= 4:
                x = self.block_dropout(x)
        x = self.global_avgpool(x).squeeze(-1)
        x = self.fc(x).squeeze(-1)
        if B == 1: x = x.unsqueeze(0)
        return x

# ================================
# PhysNet2DCNN_2D（C×M×T→1×1×T）
# ================================
class PhysNet2DCNN_2D(nn.Module):
    """
    入力: (B, C, M, T)   出力: (B, 1, T)
    空間方向を主に縮小し、時間を一旦1/4にダウンサンプル後、x2×x2で復元。
    最終は空間GAP→Conv1d(1×1)で (B,1,T)。
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(32), nn.ELU()
        )
        self.p1 = nn.AvgPool2d(kernel_size=(2,1), stride=(2,1))  # M 1/2

        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64), nn.ELU()
        )
        self.p2 = nn.AvgPool2d(kernel_size=(4,2), stride=(4,2))  # M 1/4, T 1/2

        self.c3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64), nn.ELU()
        )
        self.p3 = nn.AvgPool2d(kernel_size=(2,1), stride=(2,1))  # M 1/2（累積 M/16）

        self.ct = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(64), nn.ELU()
        )

        self.up1 = nn.Upsample(scale_factor=(1,2), mode='nearest')  # T ×2
        self.up2 = nn.Upsample(scale_factor=(1,2), mode='nearest')  # T ×2（合計×4）

        self.fout = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,C,M,T)
        x = self.c1(x); x = self.p1(x)
        x = self.c2(x); x = self.p2(x)
        x = self.c3(x); x = self.p3(x)
        x = self.ct(x)
        x = self.up1(x); x = self.up2(x)  # (B,64,M',T)
        x = x.mean(dim=2)                 # 空間GAP → (B,64,T)
        x = self.fout(x)                  # (B,1,T)
        return x

# ================================
# PhysNet2DCNN_3D（C×H×W×T→1×1×T）
# ================================
class PhysNet2DCNN_3D(nn.Module):
    """
    入力: (B, C, H, W, T)   出力: (B, 1, T)
    3D畳み込みで空間中心にプーリング→空間GAP→1x1Conv(時間)。
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(1,5,5), padding=(0,2,2)),
            nn.BatchNorm3d(32), nn.ELU(),
            nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64), nn.ELU(),
            nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(64), nn.ELU(),
            nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        )
        self.fout = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,C,H,W,T) → (B,C,T,H,W)
        x = x.permute(0, 1, 4, 2, 3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.mean(dim=[3, 4])        # 空間GAP → (B,64,T)
        x = self.fout(x)              # (B,1,T)
        return x

# ================================
# モデル作成関数
# ================================
def create_model(config, sample_shape=None):
    """
    sample_shape:
      - physnet_2d: (C, M, Tw)
      - physnet_3d: (C, H, W, Tw)
    """
    if config.model_type == "standard":
        model = PhysNet2DCNN(config.input_shape)
        model_name = "PhysNet2DCNN (Standard - 5 blocks)"
    elif config.model_type == "deep":
        model = DeepPhysNet2DCNN(config.input_shape, depth_level="deep")
        model_name = "DeepPhysNet2DCNN (Deep - 8 blocks)"
    elif config.model_type == "physnet_2d":
        in_c = sample_shape[0] if sample_shape is not None else (3 if config.use_channel == 'RGB' else 1)
        model = PhysNet2DCNN_2D(in_channels=in_c)
        model_name = "PhysNet2DCNN_2D"
    elif config.model_type == "physnet_3d":
        in_c = sample_shape[0] if sample_shape is not None else (3 if config.use_channel == 'RGB' else 1)
        model = PhysNet2DCNN_3D(in_channels=in_c)
        model_name = "PhysNet2DCNN_3D"
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    if config.verbose:
        print(f"\n選択モデル: {model_name}")
        print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    return model

# ================================
# データ読み込み
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み"""
    rgb_path = os.path.join(config.rgb_base_path, subject, f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません")
        return None, None
    rgb_data = np.load(rgb_path)  # 期待形状: (360,36,36,3)

    signal_data_list = []
    for task in config.tasks:
        signal_path = os.path.join(config.signal_base_path, subject, config.signal_type, f"{config.signal_prefix}_{task}.npy")
        if not os.path.exists(signal_path):
            print(f"警告: {subject}の{task}の{config.signal_type}データが見つかりません")
            return None, None
        signal_data_list.append(np.load(signal_path))
    signal_data = np.concatenate(signal_data_list)  # 期待形状: (360,)
    return rgb_data, signal_data

# ================================
# 学習関数
# ================================
def train_model(model, train_loader, val_loader, config, fold=None, subject=None):
    """モデルの学習（PhysNet系の(B,1,T)出力にも対応）"""
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    subject_str = f"{subject}" if subject is not None else ""
    if config.verbose:
        print(f"\n  学習開始 {subject_str} {fold_str}")
        print(f"    モデル: {config.model_type}")
        print(f"    エポック数: {config.epochs}")
        print(f"    バッチサイズ: {config.batch_size}")

    model = model.to(config.device)

    # 損失関数
    if config.loss_type == "combined":
        if config.verbose: print(f"    損失関数: CombinedLoss (α={config.loss_alpha}, β={config.loss_beta})")
        criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    elif config.loss_type == "huber_combined":
        if config.verbose: print(f"    損失関数: HuberCorrelationLoss")
        criterion = HuberCorrelationLoss(alpha=config.loss_alpha, beta=config.loss_beta)
    else:
        if config.verbose: print("    損失関数: MSE")
        mse = nn.MSELoss()
        def criterion(pred, target):
            l = mse(pred, target)
            z = torch.tensor(0.0, device=pred.device)
            return l, l, z

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # スケジューラー
    if config.scheduler_type == "cosine":
        if config.verbose: print(f"    スケジューラー: CosineAnnealingWarmRestarts")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=1e-6
        )
        scheduler_per_batch = False
    elif config.scheduler_type == "onecycle":
        if config.verbose: print("    スケジューラー: OneCycleLR")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate * 10,
            epochs=config.epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3, anneal_strategy='cos'
        )
        scheduler_per_batch = True
    else:
        if config.verbose: print("    スケジューラー: ReduceLROnPlateau")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)
        scheduler_per_batch = False

    train_losses, val_losses = [], []
    train_correlations, val_correlations = [], []
    best_val_loss = float('inf')
    best_val_corr = -1
    patience_counter = 0
    train_preds_best = None
    train_targets_best = None

    save_dir = Path(config.save_path)
    if subject is not None:
        save_dir = save_dir / subject
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = f'best_model_fold{fold+1}.pth' if fold is not None else f'best_model_{config.model_type}.pth'

    for epoch in range(config.epochs):
        # === Train ===
        model.train()
        train_loss = 0.0
        train_preds_all = []
        train_targets_all = []

        for batch in train_loader:
            if len(batch) == 3:
                rgb, sig, _ = batch
            else:
                rgb, sig = batch
            rgb, sig = rgb.to(config.device), sig.to(config.device)
            optimizer.zero_grad()
            pred = model(rgb)

            # (B,1,T) → (B,T)
            if pred.dim() == 3: pred_use = pred.squeeze(1)
            else:               pred_use = pred
            target_use = sig

            loss, mse_loss, corr_loss = criterion(pred_use, target_use)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler_per_batch: scheduler.step()

            train_loss += loss.item()
            train_preds_all.append(pred_use.detach().cpu().numpy())
            train_targets_all.append(target_use.detach().cpu().numpy())

        # === Val ===
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    rgb, sig, _ = batch
                else:
                    rgb, sig = batch
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                pred = model(rgb)
                pred_use = pred.squeeze(1) if pred.dim() == 3 else pred
                target_use = sig
                l, _, _ = criterion(pred_use, target_use)
                val_loss += l.item()
                val_preds.append(pred_use.cpu().numpy())
                val_targets.append(target_use.cpu().numpy())

        # === Metrics（フラット化） ===
        def _flatten(lst): return np.concatenate([a.reshape(-1) for a in lst], axis=0)
        train_preds_all = _flatten(train_preds_all)
        train_targets_all = _flatten(train_targets_all)
        val_preds = _flatten(val_preds)
        val_targets = _flatten(val_targets)

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        train_corr = np.corrcoef(train_preds_all, train_targets_all)[0, 1] if len(train_preds_all)>1 else 0.0
        val_corr   = np.corrcoef(val_preds, val_targets)[0, 1] if len(val_preds)>1 else 0.0
        val_mae    = mean_absolute_error(val_targets, val_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_correlations.append(train_corr)
        val_correlations.append(val_corr)

        if not scheduler_per_batch:
            if config.scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ベスト更新
        if val_loss < best_val_loss or (val_loss < best_val_loss * 1.1 and val_corr > best_val_corr):
            best_val_loss = val_loss
            best_val_corr = val_corr
            patience_counter = 0
            train_preds_best = train_preds_all.copy()
            train_targets_best = train_targets_all.copy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_corr': best_val_corr,
                'model_type': config.model_type
            }, save_dir / model_name)
        else:
            patience_counter += 1

        # ログ
        if config.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1:3d}/{config.epochs}] LR: {current_lr:.2e}")
            print(f"      Train Loss: {train_loss:.4f}, Corr: {train_corr:.4f}")
            print(f"      Val   Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")

        if patience_counter >= config.patience:
            if config.verbose: print(f"    Early stopping at epoch {epoch+1}")
            break

    # ベストモデル復元
    checkpoint = torch.load(save_dir / model_name, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, train_preds_best, train_targets_best

# ================================
# 評価関数
# ================================
def evaluate_model(model, test_loader, config):
    """テスト時：ClipDataset(PhysNet系)は窓出力を重畳平均して元の長さTに復元"""
    model.eval()

    # 既定（1Dモデルなど通常パス）
    if not isinstance(test_loader.dataset, ClipDataset):
        predictions, targets = [], []
        with torch.no_grad():
            for rgb, sig in test_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                pred = model(rgb)
                pred_use = pred.squeeze(1) if pred.dim() == 3 else pred
                predictions.append(pred_use.cpu().numpy())
                targets.append(sig.cpu().numpy())
        predictions = np.concatenate([p.reshape(-1) for p in predictions], axis=0)
        targets     = np.concatenate([t.reshape(-1) for t in targets], axis=0)

    else:
        ds = test_loader.dataset
        T_total = ds.T
        Tw = config.temporal_window
        sum_pred = np.zeros(T_total, dtype=np.float64)
        cnt_pred = np.zeros(T_total, dtype=np.float64)
        gt = ds.sig.copy()

        with torch.no_grad():
            for x, y, starts in test_loader:
                x = x.to(config.device)
                pred = model(x)
                pred_use = pred.squeeze(1) if pred.dim() == 3 else pred  # (B,Tw)
                pred_np = pred_use.cpu().numpy()
                starts_np = starts.numpy()
                B = pred_np.shape[0]
                for b in range(B):
                    s = int(starts_np[b])
                    e = s + pred_np.shape[1]
                    sum_pred[s:e] += pred_np[b]
                    cnt_pred[s:e] += 1.0

        # 重畳平均
        cnt_pred[cnt_pred == 0] = 1.0
        predictions = (sum_pred / cnt_pred).astype(np.float32)
        targets = gt.astype(np.float32)

    mae  = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    if np.std(targets) == 0 or np.std(predictions) == 0:
        corr, p_value = 0.0, 1.0
    else:
        corr, p_value = pearsonr(targets, predictions)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2) + 1e-8
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

    for fold, test_task in enumerate(config.tasks):
        if config.verbose:
            print(f"\n  Fold {fold+1}/6 - テストタスク: {test_task}")

        # タスクごとに分割
        train_rgb_list, train_signal_list = [], []
        test_rgb_list, test_signal_list = [], []
        for i, task in enumerate(config.tasks):
            s = i * config.task_duration
            e = (i + 1) * config.task_duration
            task_rgb = rgb_data[s:e]
            task_signal = signal_data[s:e]
            if task == test_task:
                test_rgb_list.append(task_rgb)
                test_signal_list.append(task_signal)
            else:
                train_rgb_list.append(task_rgb)
                train_signal_list.append(task_signal)

        train_rgb = np.concatenate(train_rgb_list) if len(train_rgb_list)>0 else np.empty((0,)+rgb_data.shape[1:])
        train_signal = np.concatenate(train_signal_list) if len(train_signal_list)>0 else np.empty((0,))
        test_rgb = np.concatenate(test_rgb_list)
        test_signal = np.concatenate(test_signal_list)

        # 訓練/検証（8:2）
        split_idx = int(len(train_rgb) * 0.8) if len(train_rgb) > 0 else 0
        val_rgb = train_rgb[split_idx:] if split_idx < len(train_rgb) else train_rgb
        val_signal = train_signal[split_idx:] if split_idx < len(train_signal) else train_signal
        train_rgb = train_rgb[:split_idx]
        train_signal = train_signal[:split_idx]

        # ===== データローダ作成 =====
        if config.model_type in ["physnet_2d", "physnet_3d"]:
            mode = '2d' if config.model_type == "physnet_2d" else '3d'
            train_dataset = ClipDataset(train_rgb, train_signal, config, mode=mode, is_train=True)
            val_dataset   = ClipDataset(val_rgb,   val_signal,   config, mode=mode, is_train=False)
            test_dataset  = ClipDataset(test_rgb,  test_signal,  config, mode=mode, is_train=False)

            # サンプル形状をモデルに渡す
            if len(train_dataset) > 0:
                sample_x, _, _ = train_dataset[0]
            else:
                sample_x, _, _ = test_dataset[0]
            sample_shape = sample_x.shape  # 2d:(C,M,Tw) / 3d:(C,H,W,Tw)

            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

            model = create_model(config, sample_shape)
        else:
            train_dataset = CODataset(train_rgb, train_signal, config.use_channel)
            val_dataset   = CODataset(val_rgb,   val_signal,   config.use_channel)
            test_dataset  = CODataset(test_rgb,  test_signal,  config.use_channel)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)
            model = create_model(config)

        # 学習
        model, train_preds, train_targets = train_model(model, train_loader, val_loader, config, fold, subject)

        # 評価（PhysNet系は時間復元済み）
        test_results = evaluate_model(model, test_loader, config)

        if config.verbose:
            tr_corr = np.corrcoef(train_targets, train_preds)[0, 1] if len(train_preds)>1 else 0.0
            print(f"    Train: MAE={mean_absolute_error(train_targets, train_preds):.4f}, Corr={tr_corr:.4f}")
            print(f"    Test:  MAE={test_results['mae']:.4f}, Corr={test_results['corr']:.4f}")

        # 結果保存
        fold_results.append({
            'fold': fold + 1,
            'test_task': test_task,
            'train_predictions': np.array(train_preds),
            'train_targets': np.array(train_targets),
            'test_predictions': test_results['predictions'],
            'test_targets': test_results['targets'],
            'train_mae': mean_absolute_error(train_targets, train_preds),
            'train_corr': (np.corrcoef(train_targets, train_preds)[0, 1] if len(train_preds)>1 else 0.0),
            'test_mae': test_results['mae'],
            'test_corr': test_results['corr']
        })

        # 全体テストの集約（60点×1タスク）
        all_test_predictions.extend(test_results['predictions'])
        all_test_targets.extend(test_results['targets'])

        # 各Foldのプロット
        plot_fold_results(fold_results[-1], subject_save_dir)

    # リスト→配列
    all_test_predictions = np.array(all_test_predictions)
    all_test_targets = np.array(all_test_targets)
    return fold_results, all_test_predictions, all_test_targets

# ================================
# プロット関数
# ================================
def plot_fold_results(result, save_dir):
    """各Foldの結果をプロット"""
    fold = result['fold']
    # 訓練散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['train_targets'], result['train_predictions'], alpha=0.5, s=10)
    min_val = min(result['train_targets'].min(), result['train_predictions'].min())
    max_val = max(result['train_targets'].max(), result['train_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値'); plt.ylabel('予測値')
    plt.title(f"Fold {fold} 訓練データ - MAE: {result['train_mae']:.3f}, Corr: {result['train_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # テスト散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['test_targets'], result['test_predictions'], alpha=0.5, s=10)
    min_val = min(result['test_targets'].min(), result['test_predictions'].min())
    max_val = max(result['test_targets'].max(), result['test_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値'); plt.ylabel('予測値')
    plt.title(f"Fold {fold} テストデータ ({result['test_task']}) - MAE: {result['test_mae']:.3f}, Corr: {result['test_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 波形（訓練は窓フラット、テストは60点復元）
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(result['train_targets'], 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(result['train_predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
    plt.xlabel('サンプルIndex'); plt.ylabel('信号値')
    plt.title(f'Fold {fold} 訓練データ波形（フラット表示）')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(result['test_targets'], 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(result['test_predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
    plt.xlabel('時間 (秒)'); plt.ylabel('信号値')
    plt.title(f'Fold {fold} テストデータ波形 ({result["test_task"]})')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_waveforms.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_subject_summary(fold_results, all_test_predictions, all_test_targets, subject, subject_save_dir):
    """被験者の全体結果をプロット"""
    # 訓練データ統合（窓フラット）
    all_train_predictions = np.concatenate([r['train_predictions'] for r in fold_results])
    all_train_targets = np.concatenate([r['train_targets'] for r in fold_results])
    all_train_mae = mean_absolute_error(all_train_targets, all_train_predictions)
    try:
        all_train_corr, _ = pearsonr(all_train_targets, all_train_predictions)
    except Exception:
        all_train_corr = 0.0

    # 全テストメトリクス（60*6=360点）
    all_test_mae = mean_absolute_error(all_test_targets, all_test_predictions)
    try:
        all_test_corr, _ = pearsonr(all_test_targets, all_test_predictions)
    except Exception:
        all_test_corr = 0.0

    # 訓練散布
    plt.figure(figsize=(10, 8))
    plt.scatter(all_train_targets, all_train_predictions, alpha=0.5, s=10)
    min_val = min(all_train_targets.min(), all_train_predictions.min())
    max_val = max(all_train_targets.max(), all_train_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値'); plt.ylabel('予測値')
    plt.title(f"{subject} 全訓練データ - MAE: {all_train_mae:.3f}, Corr: {all_train_corr:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # テスト散布
    plt.figure(figsize=(10, 8))
    plt.scatter(all_test_targets, all_test_predictions, alpha=0.5, s=10)
    min_val = min(all_test_targets.min(), all_test_predictions.min())
    max_val = max(all_test_targets.max(), all_test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値'); plt.ylabel('予測値')
    plt.title(f"{subject} 全テストデータ - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 全テスト連結波形（360点）
    plt.figure(figsize=(20, 6))
    plt.plot(all_test_targets, 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(all_test_predictions, 'g-', label='予測', alpha=0.7, linewidth=1)
    for i in range(1, 6):
        plt.axvline(x=i*60, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('時間 (秒)'); plt.ylabel('信号値')
    plt.title(f'{subject} 全テストデータ連結波形 - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_waveform.png', dpi=150, bbox_inches='tight')
    plt.close()

    return all_train_mae, all_train_corr, all_test_mae, all_test_corr

def plot_all_subjects_summary(all_subjects_results, config):
    """全被験者のサマリープロット"""
    save_dir = Path(config.save_path)
    subjects, train_maes, train_corrs, test_maes, test_corrs = [], [], [], [], []
    for result in all_subjects_results:
        subjects.append(result['subject'])
        train_maes.append(result['train_mae'])
        train_corrs.append(result['train_corr'])
        test_maes.append(result['test_mae'])
        test_corrs.append(result['test_corr'])

    # 訓練サマリー
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.ravel()
    for i, result in enumerate(all_subjects_results[:32]):
        ax = axes[i]
        ax.text(0.5, 0.5, f"{result['subject']}\nMAE: {result['train_mae']:.3f}\nCorr: {result['train_corr']:.3f}",
                ha='center', va='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#e8f4f8' if result['train_corr'] > 0.7 else '#f8e8e8')
    fig.suptitle(f'全被験者 訓練データ結果 - 平均MAE: {np.mean(train_maes):.3f}, 平均Corr: {np.mean(train_corrs):.3f}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_train_summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    # テストサマリー
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.ravel()
    for i, result in enumerate(all_subjects_results[:32]):
        ax = axes[i]
        ax.text(0.5, 0.5, f"{result['subject']}\nMAE: {result['test_mae']:.3f}\nCorr: {result['test_corr']:.3f}",
                ha='center', va='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
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
    print(" PhysNet2DCNN - 個人内解析（6分割交差検証） with Augmentation")
    print("="*60)
    print(f"血行動態信号: {config.signal_type}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
    print(f"損失関数: {config.loss_type}")
    print(f"スケジューラー: {config.scheduler_type}")
    print(f"オーグメンテーション: {config.aug_enable}")
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
            # データ読み込み（(360,36,36,3), (360,)）
            rgb_data, signal_data = load_data_single_subject(subject, config)
            if rgb_data is None or signal_data is None:
                print(f"  {subject}のデータ読み込み失敗。スキップします。")
                continue
            
            print(f"  データ形状: RGB={rgb_data.shape}, Signal={signal_data.shape}")
            
            # 6分割交差検証
            fold_results, all_test_predictions, all_test_targets = task_cross_validation(
                rgb_data, signal_data, config, subject, subject_save_dir
            )
            
            # 被験者サマリー
            train_mae, train_corr, test_mae, test_corr = plot_subject_summary(
                fold_results, all_test_predictions, all_test_targets, subject, subject_save_dir
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
    
    # 全被験者サマリープロット
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
