import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# フォント設定（メイリオ）
plt.rcParams['font.sans-serif'] = ['Meiryo', 'Yu Gothic', 'Hiragino Sans', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
mpl.rcParams['font.size'] = 10

# ================================
# 設定クラス（すべての設定をここで管理）
# ================================
class Config:
    def __init__(self):
        # パス設定
        self.rgb_base_path = r"C:\Users\EyeBelow"
        self.signal_base_path = r"C:\Users\Data_signals_bp"
        self.save_path = r"D:\EPSCAN\001"
        
        # データ設定
        self.subject = "bp001"
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60  # 各タスクの秒数
        
        # =============================================
        # モデル選択（ここを変更してモデルを切り替え）
        # =============================================
        # "CNN-LSTM", "3D-CNN", "Vision-Transformer" から選択
        self.model_type = "CNN-LSTM"  
        
        # =============================================
        # 使用チャンネル選択（ここを変更してチャンネルを切り替え）
        # =============================================
        # 'R': 赤成分のみ, 'G': 緑成分のみ, 'B': 青成分のみ, 'RGB': 全チャンネル
        self.use_channel = 'B'  # B成分のみで学習  
        
        # モデル設定
        self.input_shape = (14, 16, 3)  # H, W, C
        
        # 学習設定
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # データ分割比率（各タスク内で）
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        
        # Early Stopping
        self.patience = 20
        
        # 表示設定
        self.verbose = True
        self.save_every = 10  # 何エポックごとにチェックポイントを保存

# ================================
# データセット
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, co_data):
        """
        rgb_data: (N, H, W, C) numpy array
        co_data: (N,) numpy array
        """
        self.rgb_data = torch.FloatTensor(rgb_data).permute(0, 3, 1, 2)  # (N, C, H, W)
        self.co_data = torch.FloatTensor(co_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.co_data[idx]

# ================================
# モデルアーキテクチャ
# ================================

# 1. CNN-LSTM モデル
class CNN_LSTM(nn.Module):
    def __init__(self, input_shape):
        super(CNN_LSTM, self).__init__()
        
        # CNN部分（特徴抽出）
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # LSTM部分
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.2)
        
        # 出力層
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN特徴抽出
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, -1)
        
        # LSTMのための形状変換
        lstm_in = cnn_out.unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        
        # 最終出力
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()

# 2. 3D-CNN モデル
class CNN3D(nn.Module):
    def __init__(self, input_shape):
        super(CNN3D, self).__init__()
        
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 1)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 3))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 3D-CNNのための形状変換 (B, C, H, W) -> (B, 1, C, H, W)
        x = x.unsqueeze(1)
        
        # 3D畳み込み
        conv_out = self.conv3d_layers(x)
        conv_out = conv_out.view(batch_size, -1)
        
        # 全結合層
        out = self.fc_layers(conv_out)
        return out.squeeze()

# 3. Vision Transformer モデル
class VisionTransformer(nn.Module):
    def __init__(self, input_shape, patch_size=4, embed_dim=256, num_heads=8, num_layers=6):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.patch_dim = 3 * patch_size * patch_size
        
        # パッチ埋め込み
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        
        # 位置エンコーディング
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 出力層
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # パッチ化 (B, C, H, W) -> (B, num_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 3, -1, self.patch_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, self.num_patches, -1)
        
        # パッチ埋め込み
        x = self.patch_embed(x)
        
        # CLSトークンと位置エンコーディング
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        
        # CLSトークンから出力
        out = self.fc(x[:, 0])
        return out.squeeze()

# ================================
# データ読み込み関数
# ================================
def load_data(config):
    """データの読み込み"""
    print("="*60)
    print("データ読み込み中...")
    print("="*60)
    
    # RGB画像データの読み込み
    rgb_path = os.path.join(config.rgb_base_path, config.subject, 
                            f"{config.subject}_downsampled_1Hz.npy")
    rgb_data = np.load(rgb_path)  # (360, 14, 16, 3)
    print(f"✓ RGB画像データ: {rgb_data.shape}")
    
    # COデータの読み込みと結合
    co_data_list = []
    for task in config.tasks:
        co_path = os.path.join(config.signal_base_path, config.subject, 
                              "CO", f"CO_s2_{task}.npy")
        co_task_data = np.load(co_path)  # (60,)
        co_data_list.append(co_task_data)
    
    co_data = np.concatenate(co_data_list)  # (360,)
    print(f"✓ COデータ: {co_data.shape}")
    print(f"  - COの範囲: [{co_data.min():.2f}, {co_data.max():.2f}]")
    print(f"  - COの平均: {co_data.mean():.2f} ± {co_data.std():.2f}")
    
    return rgb_data, co_data

# ================================
# データ分割関数
# ================================
def split_data_by_task(rgb_data, co_data, config):
    """タスクごとにデータを分割"""
    print("\nデータ分割中...")
    
    train_rgb, train_co = [], []
    val_rgb, val_co = [], []
    test_rgb, test_co = [], []
    
    for i, task in enumerate(config.tasks):
        start_idx = i * config.task_duration
        end_idx = (i + 1) * config.task_duration
        
        task_rgb = rgb_data[start_idx:end_idx]
        task_co = co_data[start_idx:end_idx]
        
        # 各タスクを7:1:2に分割
        train_end = int(config.task_duration * config.train_ratio)
        val_end = train_end + int(config.task_duration * config.val_ratio)
        
        train_rgb.append(task_rgb[:train_end])
        train_co.append(task_co[:train_end])
        
        val_rgb.append(task_rgb[train_end:val_end])
        val_co.append(task_co[train_end:val_end])
        
        test_rgb.append(task_rgb[val_end:])
        test_co.append(task_co[val_end:])
        
        if config.verbose:
            print(f"  Task {task}: Train {train_end}, Val {val_end-train_end}, Test {config.task_duration-val_end}")
    
    # 結合
    train_rgb = np.concatenate(train_rgb)
    train_co = np.concatenate(train_co)
    val_rgb = np.concatenate(val_rgb)
    val_co = np.concatenate(val_co)
    test_rgb = np.concatenate(test_rgb)
    test_co = np.concatenate(test_co)
    
    print(f"\n分割結果:")
    print(f"  訓練データ: {len(train_rgb)}サンプル")
    print(f"  検証データ: {len(val_rgb)}サンプル")
    print(f"  テストデータ: {len(test_rgb)}サンプル")
    
    return (train_rgb, train_co), (val_rgb, val_co), (test_rgb, test_co)

# ================================
# モデル作成関数
# ================================
def create_model(config):
    """指定されたタイプのモデルを作成"""
    print(f"\nモデル作成: {config.model_type}")
    
    if config.model_type == "CNN-LSTM":
        model = CNN_LSTM(config.input_shape)
    elif config.model_type == "3D-CNN":
        model = CNN3D(config.input_shape)
    elif config.model_type == "Vision-Transformer":
        model = VisionTransformer(config.input_shape)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    # パラメータ数の表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  総パラメータ数: {total_params:,}")
    print(f"  学習可能パラメータ数: {trainable_params:,}")
    
    return model

# ================================
# 学習関数
# ================================
def train_model(model, train_loader, val_loader, config):
    """モデルの学習"""
    print("\n" + "="*60)
    print("モデル学習開始")
    print("="*60)
    
    model = model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     patience=10, 
                                                     factor=0.5,
                                                     verbose=config.verbose)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # 学習モード
        model.train()
        train_loss = 0
        train_mae = 0
        
        for batch_idx, (rgb, co) in enumerate(train_loader):
            rgb, co = rgb.to(config.device), co.to(config.device)
            
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, co)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += mean_absolute_error(pred.detach().cpu().numpy(), 
                                            co.detach().cpu().numpy())
        
        # 検証モード
        model.eval()
        val_loss = 0
        val_mae = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for rgb, co in val_loader:
                rgb, co = rgb.to(config.device), co.to(config.device)
                pred = model(rgb)
                loss = criterion(pred, co)
                val_loss += loss.item()
                
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(co.cpu().numpy())
        
        # 平均損失の計算
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_mae /= len(train_loader)
        val_mae = mean_absolute_error(val_preds, val_targets)
        val_corr, _ = pearsonr(val_preds, val_targets)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # スケジューラー更新
        scheduler.step(val_loss)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # ベストモデルの保存
            save_dir = Path(config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 
                      save_dir / f'best_model_{config.model_type}.pth')
        else:
            patience_counter += 1
        
        # ログ出力
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f} | "
                  f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")
        
        # チェックポイント保存
        if (epoch + 1) % config.save_every == 0:
            save_dir = Path(config.save_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_{config.model_type}_epoch{epoch+1}.pth')
        
        # Early Stopping判定
        if patience_counter >= config.patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # ベストモデルの読み込み
    model.load_state_dict(torch.load(save_dir / f'best_model_{config.model_type}.pth'))
    
    return model, train_losses, val_losses

# ================================
# 評価関数
# ================================
def evaluate_model(model, test_loader, config):
    """モデルの評価"""
    print("\n" + "="*60)
    print("テストデータでの評価")
    print("="*60)
    
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
    
    # 評価指標の計算
    mae = mean_absolute_error(targets, predictions)
    mse = np.mean((targets - predictions) ** 2)
    rmse = np.sqrt(mse)
    corr, p_value = pearsonr(targets, predictions)
    
    # R²スコア
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  相関係数: {corr:.4f} (p値: {p_value:.2e})")
    print(f"  R²スコア: {r2:.4f}")
    
    return mae, corr, predictions, targets, {'mse': mse, 'rmse': rmse, 'r2': r2}

# ================================
# プロット関数
# ================================
def plot_results(predictions, targets, train_losses, val_losses, mae, corr, config):
    """結果のプロット"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 学習曲線
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('学習曲線')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 予測値 vs 真値（散布図）
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想線')
    ax2.set_xlabel('真値 (CO)')
    ax2.set_ylabel('予測値 (CO)')
    ax2.set_title(f'予測結果\nMAE: {mae:.3f}, Corr: {corr:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差プロット
    ax3 = plt.subplot(2, 3, 3)
    residuals = targets - predictions
    ax3.scatter(predictions, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('予測値')
    ax3.set_ylabel('残差 (真値 - 予測値)')
    ax3.set_title(f'残差プロット\n平均: {residuals.mean():.3f}, 標準偏差: {residuals.std():.3f}')
    ax3.grid(True, alpha=0.3)
    
    # 4. 時系列プロット（最初の120サンプル）
    ax4 = plt.subplot(2, 3, 4)
    sample_range = min(120, len(targets))
    x_axis = np.arange(sample_range)
    ax4.plot(x_axis, targets[:sample_range], 'b-', label='真値', alpha=0.7)
    ax4.plot(x_axis, predictions[:sample_range], 'r-', label='予測値', alpha=0.7)
    ax4.set_xlabel('時間 (秒)')
    ax4.set_ylabel('CO値')
    ax4.set_title('時系列比較（最初の120秒）')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 誤差のヒストグラム
    ax5 = plt.subplot(2, 3, 5)
    errors = np.abs(targets - predictions)
    ax5.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax5.axvline(x=mae, color='r', linestyle='--', lw=2, label=f'MAE: {mae:.3f}')
    ax5.set_xlabel('絶対誤差')
    ax5.set_ylabel('頻度')
    ax5.set_title('絶対誤差の分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. タスクごとの性能
    ax6 = plt.subplot(2, 3, 6)
    task_maes = []
    task_corrs = []
    for i, task in enumerate(config.tasks):
        start_idx = i * int(len(targets) / 6)
        end_idx = (i + 1) * int(len(targets) / 6) if i < 5 else len(targets)
        task_mae = mean_absolute_error(targets[start_idx:end_idx], 
                                       predictions[start_idx:end_idx])
        task_corr, _ = pearsonr(targets[start_idx:end_idx], 
                               predictions[start_idx:end_idx])
        task_maes.append(task_mae)
        task_corrs.append(task_corr)
    
    x_pos = np.arange(len(config.tasks))
    width = 0.35
    ax6.bar(x_pos - width/2, task_maes, width, label='MAE', alpha=0.7)
    ax6_twin = ax6.twinx()
    ax6_twin.bar(x_pos + width/2, task_corrs, width, label='Correlation', 
                color='orange', alpha=0.7)
    
    ax6.set_xlabel('タスク')
    ax6.set_ylabel('MAE', color='blue')
    ax6_twin.set_ylabel('相関係数', color='orange')
    ax6.set_title('タスクごとの性能')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(config.tasks)
    ax6.tick_params(axis='y', labelcolor='blue')
    ax6_twin.tick_params(axis='y', labelcolor='orange')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'CO推定結果 - {config.model_type} ({config.subject})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'results_{config.model_type}_{config.subject}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nグラフを保存しました: {save_dir / f'results_{config.model_type}_{config.subject}.png'}")

# ================================
# メイン実行関数
# ================================
def main():
    # 設定の初期化
    config = Config()
    
    print("\n" + "="*60)
    print(f" CO推定モデル - 個人内学習 ({config.subject})")
    print("="*60)
    print(f"モデル: {config.model_type}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    
    try:
        # 1. データ読み込み
        rgb_data, co_data = load_data(config)
        
        # 2. データ分割
        train_data, val_data, test_data = split_data_by_task(rgb_data, co_data, config)
        train_rgb, train_co = train_data
        val_rgb, val_co = val_data
        test_rgb, test_co = test_data
        
        # 3. データセットとDataLoaderの作成
        train_dataset = CODataset(train_rgb, train_co)
        val_dataset = CODataset(val_rgb, val_co)
        test_dataset = CODataset(test_rgb, test_co)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                              shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                               shuffle=False, num_workers=0)
        
        # 4. モデル作成
        model = create_model(config)
        
        # 5. モデル学習
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, config)
        
        # 6. テストデータで評価
        mae, corr, predictions, targets, metrics = evaluate_model(model, test_loader, config)
        
        # 7. 結果の可視化
        plot_results(predictions, targets, train_losses, val_losses, mae, corr, config)
        
        # 8. 結果の保存
        save_dir = Path(config.save_path)
        results = {
            'predictions': predictions,
            'targets': targets,
            'mae': mae,
            'correlation': corr,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': {
                'model_type': config.model_type,
                'subject': config.subject,
                'batch_size': config.batch_size,
                'epochs': len(train_losses),
                'learning_rate': config.learning_rate
            }
        }
        
        np.save(save_dir / f'results_{config.model_type}_{config.subject}.npy', 
                results, allow_pickle=True)
        
        # 9. 最終サマリーの表示
        print("\n" + "="*60)
        print(" 学習完了 - 最終結果")
        print("="*60)
        print(f"モデル: {config.model_type}")
        print(f"被験者: {config.subject}")
        print(f"テストデータ性能:")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - 相関係数: {corr:.4f}")
        print(f"  - RMSE: {metrics['rmse']:.4f}")
        print(f"  - R²スコア: {metrics['r2']:.4f}")
        print(f"\n保存先: {save_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
