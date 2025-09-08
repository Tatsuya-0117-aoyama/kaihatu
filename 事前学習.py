"""
PhysNet2DCNN 事前学習プログラム
被験者の個人差（性別、年齢、肌の色、撮影条件）にロバストな特徴抽出器を構築

使用方法:
1. このファイルを単独で実行して事前学習を実施
2. 生成された pretrained_encoder.pth を元のCO推定コードで使用
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import random
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# フォント設定
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# グローバル設定
# ================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================
# 設定クラス
# ================================
class PretrainConfig:
    """事前学習の設定"""
    
    def __init__(self):
        # データパス
        self.rgb_base_path = r"C:\Users\EyeBelow"
        self.save_base_path = r"D:\EPSCAN\pretrain"
        
        # 被験者設定
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        
        # モデル設定
        self.encoder_type = "deep"  # "standard", "deep", "very_deep"
        self.projection_dim = 128    # 投影空間の次元数
        self.feature_dim = 256       # 特徴ベクトルの次元数
        
        # 事前学習手法
        self.methods = {
            "simclr": True,           # SimCLR (Contrastive Learning)
            "byol": True,             # BYOL (Bootstrap Your Own Latent)
            "temporal": True,         # Temporal Consistency
            "rotation": True,         # Rotation Prediction
        }
        
        # 学習設定
        self.batch_size = 64
        self.epochs = 150
        self.warmup_epochs = 10
        self.base_lr = 0.001
        self.weight_decay = 1e-4
        
        # Contrastive Learning設定
        self.temperature = 0.07
        self.augmentation_prob = 0.8
        
        # データ拡張強度
        self.aug_strength = {
            "brightness": 0.4,
            "contrast": 0.4,
            "saturation": 0.2,
            "noise": 0.1,
            "spatial": 0.2,
        }
        
        # システム設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4
        
        # 保存設定
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = Path(self.save_base_path) / self.timestamp
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # ログ設定
        self.log_interval = 10
        self.save_interval = 20

# ================================
# データ拡張クラス
# ================================
class AdvancedAugmentation:
    """個人差に対応するための高度なデータ拡張"""
    
    def __init__(self, config):
        self.config = config
        self.aug_strength = config.aug_strength
    
    def __call__(self, x, augment_type="all"):
        """
        x: (C, H, W) tensor
        augment_type: "all", "color", "spatial", "noise"
        """
        augmented = x.clone()
        
        if augment_type in ["all", "color"]:
            augmented = self._color_augmentation(augmented)
        
        if augment_type in ["all", "spatial"]:
            augmented = self._spatial_augmentation(augmented)
        
        if augment_type in ["all", "noise"]:
            augmented = self._noise_augmentation(augmented)
        
        return torch.clamp(augmented, 0, 1)
    
    def _color_augmentation(self, x):
        """色調補正（肌の色、照明条件の違いに対応）"""
        
        # 明度調整
        if random.random() < self.config.augmentation_prob:
            brightness = 1 + random.uniform(-self.aug_strength["brightness"], 
                                           self.aug_strength["brightness"])
            x = x * brightness
        
        # コントラスト調整
        if random.random() < self.config.augmentation_prob:
            contrast = 1 + random.uniform(-self.aug_strength["contrast"], 
                                         self.aug_strength["contrast"])
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = (x - mean) * contrast + mean
        
        # 彩度調整（RGB画像の場合）
        if x.shape[0] == 3 and random.random() < self.config.augmentation_prob:
            saturation = 1 + random.uniform(-self.aug_strength["saturation"], 
                                           self.aug_strength["saturation"])
            gray = x.mean(dim=0, keepdim=True)
            x = x * saturation + gray * (1 - saturation)
        
        # 色相シフト（肌の色調変化）
        if x.shape[0] == 3 and random.random() < 0.3:
            shift = random.uniform(-0.05, 0.05)
            x = x + shift
        
        return x
    
    def _spatial_augmentation(self, x):
        """空間変換（顔の位置、画角の違いに対応）"""
        
        h, w = x.shape[1], x.shape[2]
        
        # ランダムクロップ＆リサイズ
        if random.random() < self.config.augmentation_prob:
            crop_ratio = 1 - random.uniform(0, self.aug_strength["spatial"])
            crop_h = int(h * crop_ratio)
            crop_w = int(w * crop_ratio)
            
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            
            x_cropped = x[:, top:top+crop_h, left:left+crop_w]
            x = F.interpolate(
                x_cropped.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # 水平反転
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        
        # 微小な回転（±5度程度）
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            x = self._rotate_tensor(x, angle)
        
        return x
    
    def _noise_augmentation(self, x):
        """ノイズ追加（撮影条件の違いに対応）"""
        
        # ガウシアンノイズ
        if random.random() < self.config.augmentation_prob:
            noise_std = random.uniform(0, self.aug_strength["noise"])
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        
        # スペックルノイズ（局所的なノイズ）
        if random.random() < 0.2:
            speckle = torch.randn_like(x) * 0.05
            x = x + x * speckle
        
        return x
    
    def _rotate_tensor(self, x, angle):
        """テンソルの回転（簡易実装）"""
        # 簡単のため、小さい角度では無視
        if abs(angle) < 2:
            return x
        
        # より複雑な回転が必要な場合はtorchvision.transforms.functional.rotateを使用
        return x

# ================================
# SimCLR用データセット
# ================================
class SimCLRDataset(Dataset):
    """SimCLR用のペアデータセット"""
    
    def __init__(self, rgb_data, config):
        self.rgb_data = torch.FloatTensor(rgb_data).permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W)
        self.augmentation = AdvancedAugmentation(config)
        self.config = config
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        anchor = self.rgb_data[idx]
        
        # 2つの異なる拡張版を生成
        aug1 = self.augmentation(anchor, "all")
        aug2 = self.augmentation(anchor, "all")
        
        return aug1, aug2, idx

# ================================
# Temporal Contrastive用データセット
# ================================
class TemporalDataset(Dataset):
    """時系列の連続性を利用したデータセット"""
    
    def __init__(self, rgb_data, config, window_size=5):
        self.rgb_data = torch.FloatTensor(rgb_data).permute(0, 3, 1, 2)
        self.augmentation = AdvancedAugmentation(config)
        self.window_size = window_size
        self.config = config
    
    def __len__(self):
        return len(self.rgb_data) - self.window_size
    
    def __getitem__(self, idx):
        # アンカーフレーム
        anchor = self.rgb_data[idx]
        
        # 時間的に近いフレームを正例として選択
        pos_offset = random.randint(1, self.window_size)
        positive = self.rgb_data[idx + pos_offset]
        
        # 時間的に遠いフレームを負例として選択
        neg_candidates = list(range(0, max(0, idx - self.window_size))) + \
                        list(range(min(len(self.rgb_data), idx + self.window_size + 1), 
                                 len(self.rgb_data)))
        
        if neg_candidates:
            neg_idx = random.choice(neg_candidates)
            negative = self.rgb_data[neg_idx]
        else:
            # 負例が取れない場合は別の被験者のデータを使用（仮）
            negative = self.rgb_data[random.randint(0, len(self.rgb_data) - 1)]
        
        # 軽い拡張を適用
        anchor = self.augmentation(anchor, "color")
        positive = self.augmentation(positive, "color")
        negative = self.augmentation(negative, "color")
        
        return anchor, positive, negative

# ================================
# エンコーダーモデル
# ================================
class UniversalEncoder(nn.Module):
    """汎用的な特徴抽出エンコーダー"""
    
    def __init__(self, config):
        super(UniversalEncoder, self).__init__()
        
        self.config = config
        
        # エンコーダーブロック構築
        if config.encoder_type == "standard":
            self.encoder = self._build_standard_encoder()
        elif config.encoder_type == "deep":
            self.encoder = self._build_deep_encoder()
        elif config.encoder_type == "very_deep":
            self.encoder = self._build_very_deep_encoder()
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection Head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.BatchNorm1d(config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim, config.projection_dim)
        )
        
        # Prediction Head (for BYOL)
        self.predictor = nn.Sequential(
            nn.Linear(config.projection_dim, config.projection_dim // 2),
            nn.BatchNorm1d(config.projection_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.projection_dim // 2, config.projection_dim)
        )
    
    def _build_standard_encoder(self):
        """標準的なエンコーダー（5ブロック）"""
        return nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Conv1d(32, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            # Block 4
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),
                nn.AvgPool1d(2)
            ),
            # Block 5
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
                # 最初のブロック
                block = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU(),
                    nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU()
                )
            else:
                # 通常のブロック
                block = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU(),
                    nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ELU()
                )
            
            # Poolingは必要に応じて追加
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
    
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # (B, C, H, W) -> (B, C, H*W)
        x = x.view(batch_size, x.size(1), -1)
        
        # エンコーダーブロックを通過
        for block in self.encoder:
            x = block(x)
        
        # Global pooling
        features = self.global_pool(x).squeeze(-1)
        
        if return_features:
            return features
        
        # Projection
        z = self.projection(features)
        
        return z, features
    
    def predict(self, z):
        """BYOL用の予測"""
        return self.predictor(z)

# ================================
# 損失関数
# ================================
class MultiTaskLoss(nn.Module):
    """複数の自己教師あり学習タスクの損失を組み合わせ"""
    
    def __init__(self, config):
        super(MultiTaskLoss, self).__init__()
        self.config = config
        self.temperature = config.temperature
    
    def simclr_loss(self, z1, z2):
        """SimCLR損失"""
        batch_size = z1.size(0)
        
        # L2正規化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 全ペアの類似度計算
        representations = torch.cat([z1, z2], dim=0)
        similarity = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        # 自分自身を除外するマスク
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity = similarity.masked_fill(mask, -1e9)
        
        # 正例ペアの位置
        pos_mask = torch.zeros_like(similarity).bool()
        pos_mask[torch.arange(batch_size), batch_size + torch.arange(batch_size)] = True
        pos_mask[batch_size + torch.arange(batch_size), torch.arange(batch_size)] = True
        
        # Temperature scaling
        similarity = similarity / self.temperature
        
        # Cross entropy loss
        exp_sim = torch.exp(similarity)
        
        # 各サンプルごとの損失
        loss = 0
        for i in range(2 * batch_size):
            pos_sim = exp_sim[i, pos_mask[i]].sum()
            neg_sim = exp_sim[i, ~pos_mask[i] & ~mask[i]].sum()
            loss -= torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        
        return loss / (2 * batch_size)
    
    def byol_loss(self, pred1, z2, pred2, z1):
        """BYOL損失（予測と目標の間のコサイン類似度）"""
        pred1 = F.normalize(pred1, dim=1)
        pred2 = F.normalize(pred2, dim=1)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        loss1 = 2 - 2 * (pred1 * z2.detach()).sum(dim=1).mean()
        loss2 = 2 - 2 * (pred2 * z1.detach()).sum(dim=1).mean()
        
        return (loss1 + loss2) / 2
    
    def temporal_loss(self, anchor, positive, negative):
        """時系列コントラスト損失"""
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)
        
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        
        # Margin loss
        margin = 0.5
        loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()
        
        return loss

# ================================
# 学習用トレーナー
# ================================
class PretrainTrainer:
    """事前学習の管理クラス"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # モデル初期化
        self.encoder = UniversalEncoder(config).to(self.device)
        self.target_encoder = UniversalEncoder(config).to(self.device)
        
        # Target encoderは勾配計算しない
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # 損失関数
        self.criterion = MultiTaskLoss(config)
        
        # オプティマイザー
        self.optimizer = optim.AdamW(
            self.encoder.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )
        
        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=30, T_mult=2
        )
        
        # EMA更新用
        self.ema_tau = 0.996
        
        # ログ
        self.train_losses = []
        self.best_loss = float('inf')
    
    def update_target_encoder(self):
        """Target encoderのEMA更新"""
        for param, target_param in zip(self.encoder.parameters(), 
                                      self.target_encoder.parameters()):
            target_param.data = self.ema_tau * target_param.data + \
                               (1 - self.ema_tau) * param.data
    
    def train_epoch(self, simclr_loader, temporal_loader=None):
        """1エポックの学習"""
        self.encoder.train()
        epoch_losses = {"total": 0, "simclr": 0, "byol": 0, "temporal": 0}
        
        # SimCLR/BYOL学習
        for batch_idx, (x1, x2, _) in enumerate(simclr_loader):
            x1, x2 = x1.to(self.device), x2.to(self.device)
            
            # Forward pass
            z1, _ = self.encoder(x1)
            z2, _ = self.encoder(x2)
            
            # Target encoder (BYOL)
            with torch.no_grad():
                target_z1, _ = self.target_encoder(x1)
                target_z2, _ = self.target_encoder(x2)
            
            # Predictions for BYOL
            pred1 = self.encoder.predict(z1)
            pred2 = self.encoder.predict(z2)
            
            # 損失計算
            loss = 0
            
            if self.config.methods["simclr"]:
                simclr_loss = self.criterion.simclr_loss(z1, z2)
                loss += simclr_loss
                epoch_losses["simclr"] += simclr_loss.item()
            
            if self.config.methods["byol"]:
                byol_loss = self.criterion.byol_loss(pred1, target_z2, pred2, target_z1)
                loss += byol_loss * 0.5
                epoch_losses["byol"] += byol_loss.item()
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Target encoder更新
            self.update_target_encoder()
            
            epoch_losses["total"] += loss.item()
        
        # Temporal Contrastive学習（オプション）
        if temporal_loader and self.config.methods["temporal"]:
            for anchor, positive, negative in temporal_loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                z_anchor, _ = self.encoder(anchor)
                z_pos, _ = self.encoder(positive)
                z_neg, _ = self.encoder(negative)
                
                temporal_loss = self.criterion.temporal_loss(z_anchor, z_pos, z_neg)
                
                self.optimizer.zero_grad()
                temporal_loss.backward()
                self.optimizer.step()
                
                epoch_losses["temporal"] += temporal_loss.item()
        
        # 平均損失計算
        n_batches = len(simclr_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, loss):
        """チェックポイント保存"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'target_encoder_state_dict': self.target_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config.__dict__
        }
        
        # ベストモデル
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(checkpoint, self.config.save_path / 'best_pretrained_encoder.pth')
        
        # 定期保存
        if (epoch + 1) % self.config.save_interval == 0:
            torch.save(checkpoint, self.config.save_path / f'checkpoint_epoch_{epoch+1}.pth')
    
    def train(self, simclr_loader, temporal_loader=None, epochs=None):
        """完全な学習ループ"""
        epochs = epochs or self.config.epochs
        
        print("\n" + "="*60)
        print(" 事前学習開始")
        print("="*60)
        print(f"エンコーダータイプ: {self.config.encoder_type}")
        print(f"デバイス: {self.device}")
        print(f"エポック数: {epochs}")
        print(f"バッチサイズ: {self.config.batch_size}")
        
        for epoch in range(epochs):
            # 学習
            losses = self.train_epoch(simclr_loader, temporal_loader)
            
            # スケジューラー更新
            self.scheduler.step()
            
            # ログ保存
            self.train_losses.append(losses["total"])
            
            # チェックポイント保存
            self.save_checkpoint(epoch, losses["total"])
            
            # 表示
            if (epoch + 1) % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1:3d}/{epochs}] LR: {lr:.2e}")
                print(f"  Loss - Total: {losses['total']:.4f}, "
                      f"SimCLR: {losses['simclr']:.4f}, "
                      f"BYOL: {losses['byol']:.4f}, "
                      f"Temporal: {losses['temporal']:.4f}")
        
        print("\n事前学習完了！")
        return self.encoder

# ================================
# データ読み込み関数
# ================================
def load_all_rgb_data(config):
    """全被験者のRGBデータを読み込み"""
    print("\nデータ読み込み中...")
    
    all_data = []
    loaded_subjects = []
    
    for subject in tqdm(config.subjects, desc="被験者データ読み込み"):
        rgb_path = Path(config.rgb_base_path) / subject / f"{subject}_downsampled_1Hz.npy"
        
        if rgb_path.exists():
            data = np.load(rgb_path)
            all_data.append(data)
            loaded_subjects.append(subject)
    
    if len(all_data) == 0:
        raise ValueError("データが見つかりません")
    
    all_data = np.concatenate(all_data, axis=0)
    
    print(f"✓ 読み込み完了")
    print(f"  被験者数: {len(loaded_subjects)}")
    print(f"  総データ数: {len(all_data)}")
    print(f"  データ形状: {all_data.shape}")
    
    return all_data, loaded_subjects

# ================================
# 可視化関数
# ================================
def visualize_training(config):
    """学習結果の可視化"""
    save_path = config.save_path
    
    # 損失曲線を読み込み（保存されている場合）
    loss_file = save_path / "training_losses.npy"
    if loss_file.exists():
        losses = np.load(loss_file)
        
        plt.figure(figsize=(12, 5))
        
        # 損失推移
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('事前学習の損失推移')
        plt.grid(True, alpha=0.3)
        
        # 損失の移動平均
        plt.subplot(1, 2, 2)
        window = 10
        if len(losses) > window:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(moving_avg)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (移動平均)')
            plt.title(f'損失の移動平均（window={window}）')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 学習曲線を保存: {save_path / 'training_curves.png'}")

# ================================
# メイン実行関数
# ================================
def main():
    """メイン実行"""
    
    # 設定
    config = PretrainConfig()
    
    # 設定を保存
    with open(config.save_path / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    
    # データ読み込み
    rgb_data, subjects = load_all_rgb_data(config)
    
    # データ分割（訓練用）
    train_size = int(len(rgb_data) * 0.9)
    train_data = rgb_data[:train_size]
    
    print(f"\n訓練データ: {len(train_data)}サンプル")
    
    # データセット作成
    simclr_dataset = SimCLRDataset(train_data, config)
    simclr_loader = DataLoader(
        simclr_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Temporal dataset（オプション）
    temporal_loader = None
    if config.methods["temporal"]:
        temporal_dataset = TemporalDataset(train_data, config)
        temporal_loader = DataLoader(
            temporal_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    # トレーナー作成と学習
    trainer = PretrainTrainer(config)
    encoder = trainer.train(simclr_loader, temporal_loader)
    
    # 損失を保存
    np.save(config.save_path / "training_losses.npy", trainer.train_losses)
    
    # 可視化
    visualize_training(config)
    
    # 最終モデル保存（使いやすい形式で）
    final_checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'config': {
            'encoder_type': config.encoder_type,
            'feature_dim': config.feature_dim,
            'projection_dim': config.projection_dim,
        }
    }
    torch.save(final_checkpoint, config.save_path / 'pretrained_encoder_final.pth')
    
    print("\n" + "="*60)
    print(" 事前学習完了")
    print("="*60)
    print(f"保存先: {config.save_path}")
    print(f"最良モデル: best_pretrained_encoder.pth")
    print(f"最終モデル: pretrained_encoder_final.pth")
    print("\n使用方法:")
    print("1. 生成された pretrained_encoder_final.pth を元のCO推定コードで読み込む")
    print("2. エンコーダー部分の重みを転移学習に使用")
    print("3. CO推定タスクでファインチューニング")

if __name__ == "__main__":
    main()
