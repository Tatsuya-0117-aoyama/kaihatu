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
        
        # =============================================
        # 解析タイプ選択（個人内 or 個人間）
        # =============================================
        self.analysis_type = "individual"  # "individual"（個人内） or "cross"（個人間）
        
        # データ設定
        if self.analysis_type == "individual":
            self.subjects = ["bp001"]  # 個人内解析の場合は1人のみ
            self.n_folds = 1  # 交差検証なし
        else:  # cross
            # 個人間解析の場合は全被験者
            self.subjects = [f"bp{i:03d}" for i in range(1, 33)]  # bp001からbp032
            self.n_folds = 8  # 8分割交差検証
        
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
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)  # H, W, C
        
        # 学習設定
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # データ分割比率（個人内解析の場合は各タスク内で、個人間の場合は訓練セット内で）
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
    def __init__(self, rgb_data, co_data, use_channel='B'):
        """
        rgb_data: (N, H, W, C) numpy array
        co_data: (N,) numpy array
        use_channel: 'R', 'G', 'B', 'RGB'から選択
        """
        # チャンネル選択
        if use_channel == 'R':
            selected_data = rgb_data[:, :, :, 0:1]  # R成分
        elif use_channel == 'G':
            selected_data = rgb_data[:, :, :, 1:2]  # G成分
        elif use_channel == 'B':
            selected_data = rgb_data[:, :, :, 2:3]  # B成分
        else:  # 'RGB'
            selected_data = rgb_data
        
        # (N, H, W, C) -> (N, C, H, W)
        self.rgb_data = torch.FloatTensor(selected_data).permute(0, 3, 1, 2)
        self.co_data = torch.FloatTensor(co_data)
        self.use_channel = use_channel
    
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
        
        # 入力チャンネル数を取得
        in_channels = input_shape[2]
        
        # CNN部分（特徴抽出）
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
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
        
        # 入力チャンネル数を取得（RGBの場合は3、単一チャンネルの場合は1）
        # 3D-CNNの場合、チャンネル次元を時間次元として扱うため、入力は1チャンネルとして扱う
        
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
            nn.AdaptiveAvgPool3d((1, 1, input_shape[2]))  # チャンネル数に応じて調整
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * input_shape[2], 64),
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
        self.in_channels = input_shape[2]  # 入力チャンネル数
        self.patch_dim = self.in_channels * patch_size * patch_size
        
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
        x = x.contiguous().view(batch_size, self.in_channels, -1, self.patch_dim // self.in_channels)
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
def load_data_single_subject(subject, config):
    """単一被験者のデータ読み込み"""
    # RGB画像データの読み込み
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません")
        return None, None
        
    rgb_data = np.load(rgb_path)  # (360, 14, 16, 3)
    
    # COデータの読み込みと結合
    co_data_list = []
    for task in config.tasks:
        co_path = os.path.join(config.signal_base_path, subject, 
                              "CO", f"CO_s2_{task}.npy")
        if not os.path.exists(co_path):
            print(f"警告: {subject}の{task}のCOデータが見つかりません")
            return None, None
        co_task_data = np.load(co_path)  # (60,)
        co_data_list.append(co_task_data)
    
    co_data = np.concatenate(co_data_list)  # (360,)
    return rgb_data, co_data

def load_all_data(config):
    """全データの読み込み（個人内または個人間）"""
    print("="*60)
    print("データ読み込み中...")
    print("="*60)
    
    all_rgb_data = []
    all_co_data = []
    all_subject_ids = []  # 被験者IDを記録（個人間解析用）
    
    for subject in config.subjects:
        rgb_data, co_data = load_data_single_subject(subject, config)
        if rgb_data is not None and co_data is not None:
            all_rgb_data.append(rgb_data)
            all_co_data.append(co_data)
            # 被験者IDを360回繰り返して記録
            all_subject_ids.extend([subject] * len(rgb_data))
            print(f"✓ {subject}のデータ読み込み完了")
    
    if len(all_rgb_data) == 0:
        raise ValueError("データが読み込めませんでした")
    
    # データを結合
    all_rgb_data = np.concatenate(all_rgb_data, axis=0)
    all_co_data = np.concatenate(all_co_data, axis=0)
    
    print(f"\n読み込み完了:")
    print(f"  被験者数: {len(config.subjects)}")
    print(f"  RGB画像データ: {all_rgb_data.shape}")
    print(f"  COデータ: {all_co_data.shape}")
    print(f"  使用チャンネル: {config.use_channel}成分")
    print(f"  COの範囲: [{all_co_data.min():.2f}, {all_co_data.max():.2f}]")
    print(f"  COの平均: {all_co_data.mean():.2f} ± {all_co_data.std():.2f}")
    
    return all_rgb_data, all_co_data, all_subject_ids

# ================================
# データ分割関数
# ================================
def split_data_individual(rgb_data, co_data, config):
    """個人内解析用のデータ分割（タスクごと）"""
    print("\nデータ分割中（個人内解析）...")
    
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
    print(f"  入力形状: {config.input_shape} (H×W×C)")
    print(f"  使用チャンネル: {config.use_channel}成分")
    
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
def train_model(model, train_loader, val_loader, config, fold=None):
    """モデルの学習"""
    fold_str = f"Fold {fold+1}" if fold is not None else "全データ"
    print(f"\n学習開始 - {fold_str}")
    
    model = model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     patience=10, 
                                                     factor=0.5,
                                                     verbose=False)
    
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
            if fold is not None:
                model_name = f'best_model_{config.model_type}_{config.use_channel}_fold{fold+1}.pth'
            else:
                model_name = f'best_model_{config.model_type}_{config.use_channel}.pth'
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_counter += 1
        
        # ログ出力
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        
        # Early Stopping判定
        if patience_counter >= config.patience:
            print(f"  Early stopping triggered at epoch {epoch+1}")
            break
    
    # ベストモデルの読み込み
    model.load_state_dict(torch.load(save_dir / model_name))
    
    return model, train_losses, val_losses

# ================================
# 評価関数
# ================================
def evaluate_model(model, test_loader, config):
    """モデルの評価"""
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
    
    return mae, corr, predictions, targets, {'mse': mse, 'rmse': rmse, 'r2': r2, 'p_value': p_value}

# ================================
# 交差検証関数（個人間解析用）
# ================================
def cross_validation(rgb_data, co_data, subject_ids, config):
    """個人間解析の交差検証"""
    print("\n" + "="*60)
    print(f"{config.n_folds}分割交差検証を開始（個人間解析）")
    print("="*60)
    
    # 被験者ごとにインデックスをグループ化
    unique_subjects = sorted(list(set(subject_ids)))
    subject_indices = {subj: [] for subj in unique_subjects}
    for i, subj in enumerate(subject_ids):
        subject_indices[subj].append(i)
    
    # KFoldで被験者を分割
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
    
    results = []
    for fold, (train_subj_idx, test_subj_idx) in enumerate(kf.split(unique_subjects)):
        print(f"\n--- Fold {fold+1}/{config.n_folds} ---")
        
        # 被験者レベルで訓練とテストに分割
        train_subjects = [unique_subjects[i] for i in train_subj_idx]
        test_subjects = [unique_subjects[i] for i in test_subj_idx]
        
        print(f"  テスト被験者: {test_subjects}")
        
        # インデックスを取得
        train_indices = []
        for subj in train_subjects:
            train_indices.extend(subject_indices[subj])
        test_indices = []
        for subj in test_subjects:
            test_indices.extend(subject_indices[subj])
        
        # データ分割
        train_val_rgb = rgb_data[train_indices]
        train_val_co = co_data[train_indices]
        test_rgb = rgb_data[test_indices]
        test_co = co_data[test_indices]
        
        # 訓練・検証データをさらに分割（8:2）
        split_idx = int(len(train_val_rgb) * 0.8)
        train_rgb = train_val_rgb[:split_idx]
        train_co = train_val_co[:split_idx]
        val_rgb = train_val_rgb[split_idx:]
        val_co = train_val_co[split_idx:]
        
        print(f"  訓練: {len(train_rgb)}サンプル, 検証: {len(val_rgb)}サンプル, テスト: {len(test_rgb)}サンプル")
        
        # データセットとローダー作成
        train_dataset = CODataset(train_rgb, train_co, config.use_channel)
        val_dataset = CODataset(val_rgb, val_co, config.use_channel)
        test_dataset = CODataset(test_rgb, test_co, config.use_channel)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # モデルの作成と学習
        model = create_model(config)
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, config, fold)
        
        # 評価
        mae, corr, predictions, targets, metrics = evaluate_model(model, test_loader, config)
        print(f"  結果 - MAE: {mae:.4f}, 相関係数: {corr:.4f}")
        
        results.append({
            'fold': fold,
            'mae': mae,
            'corr': corr,
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_subjects': test_subjects
        })
    
    return results

# ================================
# プロット関数（個人内解析用）
# ================================
def plot_results_individual(predictions, targets, train_losses, val_losses, mae, corr, config):
    """個人内解析の結果プロット"""
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
    
    subjects_str = config.subjects[0] if len(config.subjects) == 1 else f"{len(config.subjects)}名"
    plt.suptitle(f'CO推定結果 - {config.model_type} ({subjects_str}) - {config.use_channel}成分使用', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    analysis_type = "individual" if config.analysis_type == "individual" else "cross"
    plt.savefig(save_dir / f'results_{config.model_type}_{analysis_type}_{config.use_channel}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nグラフを保存しました")

# ================================
# プロット関数（個人間解析用）
# ================================
def plot_results_cross(results, config):
    """個人間解析の結果プロット"""
    n_plots = min(config.n_folds, 8)
    fig = plt.figure(figsize=(20, 12))
    
    # 各foldの結果をプロット（最大8個）
    for i in range(n_plots):
        ax = plt.subplot(3, 3, i+1)
        fold_result = results[i]
        
        ax.scatter(fold_result['targets'], fold_result['predictions'], alpha=0.5, s=10)
        min_val = min(fold_result['targets'].min(), fold_result['predictions'].min())
        max_val = max(fold_result['targets'].max(), fold_result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('真値 (CO)')
        ax.set_ylabel('予測値 (CO)')
        ax.set_title(f'Fold {i+1}\nMAE: {fold_result["mae"]:.3f}, Corr: {fold_result["corr"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    # 全体の結果（最後のサブプロット）
    ax = plt.subplot(3, 3, 9)
    all_targets = np.concatenate([r['targets'] for r in results])
    all_predictions = np.concatenate([r['predictions'] for r in results])
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_corr, _ = pearsonr(all_targets, all_predictions)
    
    ax.scatter(all_targets, all_predictions, alpha=0.5, s=10)
    min_val = min(all_targets.min(), all_predictions.min())
    max_val = max(all_targets.max(), all_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('真値 (CO)')
    ax.set_ylabel('予測値 (CO)')
    ax.set_title(f'全体結果\nMAE: {overall_mae:.3f}, Corr: {overall_corr:.3f}')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'CO推定結果 - {config.model_type} (個人間解析, {config.n_folds}分割交差検証) - {config.use_channel}成分使用', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'results_{config.model_type}_cross_{config.use_channel}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return overall_mae, overall_corr

# ================================
# メイン実行関数
# ================================
def main():
    # 設定の初期化
    config = Config()
    
    print("\n" + "="*60)
    print(f" CO推定モデル")
    print("="*60)
    print(f"解析タイプ: {'個人内解析' if config.analysis_type == 'individual' else '個人間解析'}")
    print(f"モデル: {config.model_type}")
    print(f"使用チャンネル: {config.use_channel}成分")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    
    try:
        # データ読み込み
        rgb_data, co_data, subject_ids = load_all_data(config)
        
        if config.analysis_type == "individual":
            # ========== 個人内解析 ==========
            # データ分割
            train_data, val_data, test_data = split_data_individual(rgb_data, co_data, config)
            train_rgb, train_co = train_data
            val_rgb, val_co = val_data
            test_rgb, test_co = test_data
            
            # データセットとDataLoaderの作成
            train_dataset = CODataset(train_rgb, train_co, config.use_channel)
            val_dataset = CODataset(val_rgb, val_co, config.use_channel)
            test_dataset = CODataset(test_rgb, test_co, config.use_channel)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                                  shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                                   shuffle=False, num_workers=0)
            
            # モデル作成と学習
            model = create_model(config)
            model, train_losses, val_losses = train_model(model, train_loader, val_loader, config)
            
            # テストデータで評価
            print("\n" + "="*60)
            print("テストデータでの評価")
            print("="*60)
            mae, corr, predictions, targets, metrics = evaluate_model(model, test_loader, config)
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  相関係数: {corr:.4f} (p値: {metrics['p_value']:.2e})")
            print(f"  R²スコア: {metrics['r2']:.4f}")
            
            # 結果の可視化
            plot_results_individual(predictions, targets, train_losses, val_losses, mae, corr, config)
            
            # 結果の保存
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
                    'analysis_type': config.analysis_type,
                    'model_type': config.model_type,
                    'subjects': config.subjects,
                    'use_channel': config.use_channel,
                    'batch_size': config.batch_size,
                    'epochs': len(train_losses),
                    'learning_rate': config.learning_rate
                }
            }
            np.save(save_dir / f'results_{config.model_type}_individual_{config.use_channel}.npy', 
                    results, allow_pickle=True)
            
        else:
            # ========== 個人間解析 ==========
            results = cross_validation(rgb_data, co_data, subject_ids, config)
            
            # 結果の可視化
            overall_mae, overall_corr = plot_results_cross(results, config)
            
            # 結果の保存
            save_dir = Path(config.save_path)
            np.save(save_dir / f'results_{config.model_type}_cross_{config.use_channel}.npy', 
                    results, allow_pickle=True)
            
            # 各Foldの結果サマリー
            print("\n" + "="*60)
            print(" 交差検証結果サマリー")
            print("="*60)
            for r in results:
                print(f"Fold {r['fold']+1}: MAE={r['mae']:.4f}, Corr={r['corr']:.4f}")
            print(f"\n全体: MAE={overall_mae:.4f}, Corr={overall_corr:.4f}")
        
        # 最終サマリーの表示
        print("\n" + "="*60)
        print(" 学習完了 - 最終結果")
        print("="*60)
        print(f"解析タイプ: {'個人内解析' if config.analysis_type == 'individual' else '個人間解析'}")
        print(f"モデル: {config.model_type}")
        print(f"被験者: {config.subjects[0] if len(config.subjects) == 1 else f'{len(config.subjects)}名'}")
        print(f"使用チャンネル: {config.use_channel}成分")
        
        if config.analysis_type == "individual":
            print(f"テストデータ性能:")
            print(f"  - MAE: {mae:.4f}")
            print(f"  - 相関係数: {corr:.4f}")
            print(f"  - RMSE: {metrics['rmse']:.4f}")
            print(f"  - R²スコア: {metrics['r2']:.4f}")
        else:
            print(f"交差検証結果:")
            print(f"  - 全体MAE: {overall_mae:.4f}")
            print(f"  - 全体相関係数: {overall_corr:.4f}")
        
        print(f"\n保存先: {save_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
