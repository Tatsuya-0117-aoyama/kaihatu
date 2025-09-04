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
            self.subjects = ["bp001"]  # 個人内解析
            self.n_folds = 1  # 交差検証なし
        else:
            self.subjects = [f"bp{i:03d}" for i in range(1, 33)]  # bp001～bp032
            self.n_folds = 8  # 8分割交差検証
        
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60  # 各タスクの秒数
        
        # 使用チャンネル
        self.use_channel = 'B'  # 'R', 'G', 'B', 'RGB'
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)
        
        # 学習設定
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # データ分割
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        self.random_split = True  # True: ランダム分割, False: 順番分割
        self.random_seed = 42
        
        # Early Stopping
        self.patience = 20
        
        # 表示設定
        self.verbose = True

# ================================
# データセット
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, co_data, use_channel='B'):
        """
        rgb_data: (N, H, W, C) numpy array
        co_data: (N,) numpy array
        """
        # チャンネル選択
        if use_channel == 'R':
            selected_data = rgb_data[:, :, :, 0:1]
        elif use_channel == 'G':
            selected_data = rgb_data[:, :, :, 1:2]
        elif use_channel == 'B':
            selected_data = rgb_data[:, :, :, 2:3]
        else:  # 'RGB'
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
    """
    PhysNetベースの2DCNNモデル
    14×16の画像を224次元ベクトルとして処理
    """
    def __init__(self, input_shape):
        super(PhysNet2DCNN, self).__init__()
        
        in_channels = input_shape[2]
        
        # ConvBlock 1 (kernel_size=5)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU()
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # ConvBlock 2 (kernel_size=3)
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
        
        # 1x1 Convolution
        self.conv_final = nn.Conv1d(64, 1, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        
        # (B, C, H, W) -> (B, C, H*W=224)
        x = x.view(batch_size, in_channels, -1)
        
        # ConvBlocks
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
        
        # Global Average Pooling
        x = self.global_avgpool(x)
        
        # Final 1x1 Convolution
        x = self.conv_final(x)
        
        x = x.squeeze()
        if batch_size == 1:
            x = x.unsqueeze(0)
        
        return x

# ================================
# データ読み込み
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータ読み込み"""
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
    """全データの読み込み"""
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
    print(f"  RGB画像データ: {all_rgb_data.shape}")
    print(f"  COデータ: {all_co_data.shape}")
    print(f"  使用チャンネル: {config.use_channel}成分")
    print(f"  COの範囲: [{all_co_data.min():.2f}, {all_co_data.max():.2f}]")
    print(f"  COの平均: {all_co_data.mean():.2f} ± {all_co_data.std():.2f}")
    
    return all_rgb_data, all_co_data, all_subject_ids

# ================================
# データ分割
# ================================
def split_data_individual(rgb_data, co_data, config):
    """個人内解析用のデータ分割"""
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
        test_size = config.task_duration - train_size - val_size
        
        if config.random_split:
            # ランダム分割
            task_indices = np.arange(config.task_duration)
            np.random.shuffle(task_indices)
            
            train_indices = task_indices[:train_size]
            val_indices = task_indices[train_size:train_size + val_size]
            test_indices = task_indices[train_size + val_size:]
            
            train_rgb.append(task_rgb[train_indices])
            train_co.append(task_co[train_indices])
            val_rgb.append(task_rgb[val_indices])
            val_co.append(task_co[val_indices])
            test_rgb.append(task_rgb[test_indices])
            test_co.append(task_co[test_indices])
        else:
            # 順番分割
            train_rgb.append(task_rgb[:train_size])
            train_co.append(task_co[:train_size])
            val_rgb.append(task_rgb[train_size:train_size + val_size])
            val_co.append(task_co[train_size:train_size + val_size])
            test_rgb.append(task_rgb[train_size + val_size:])
            test_co.append(task_co[train_size + val_size:])
        
        if config.verbose:
            split_type = "ランダム" if config.random_split else "順番"
            print(f"  Task {task}: Train {train_size}, Val {val_size}, Test {test_size} ({split_type})")
    
    train_rgb = np.concatenate(train_rgb)
    train_co = np.concatenate(train_co)
    val_rgb = np.concatenate(val_rgb)
    val_co = np.concatenate(val_co)
    test_rgb = np.concatenate(test_rgb)
    test_co = np.concatenate(test_co)
    
    print(f"\n分割結果:")
    print(f"  訓練: {len(train_rgb)}サンプル")
    print(f"  検証: {len(val_rgb)}サンプル")
    print(f"  テスト: {len(test_rgb)}サンプル")
    
    return (train_rgb, train_co), (val_rgb, val_co), (test_rgb, test_co)

# ================================
# モデル作成
# ================================
def create_model(config):
    """モデルの作成"""
    print("\nPhysNet2DCNNモデル作成中...")
    print(f"  入力形状: {config.input_shape} (14×16×{config.input_shape[2]})")
    print(f"  使用チャンネル: {config.use_channel}成分")
    
    model = PhysNet2DCNN(config.input_shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  総パラメータ数: {total_params:,}")
    print(f"  学習可能パラメータ数: {trainable_params:,}")
    
    return model

# ================================
# 学習
# ================================
def train_model(model, train_loader, val_loader, config, fold=None):
    """モデルの学習"""
    fold_str = f"Fold {fold+1}" if fold is not None else ""
    print(f"\n学習開始 {fold_str}")
    
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
        # 学習
        model.train()
        train_loss = 0
        for rgb, co in train_loader:
            rgb, co = rgb.to(config.device), co.to(config.device)
            
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, co)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 検証
        model.eval()
        val_loss = 0
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
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae = mean_absolute_error(val_preds, val_targets)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_dir = Path(config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            if fold is not None:
                model_name = f'best_model_fold{fold+1}.pth'
            else:
                model_name = 'best_model.pth'
            torch.save(model.state_dict(), save_dir / model_name)
        else:
            patience_counter += 1
        
        # ログ出力
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{config.epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        
        # Early Stopping
        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # ベストモデル読み込み
    model.load_state_dict(torch.load(save_dir / model_name))
    
    return model, train_losses, val_losses

# ================================
# 評価
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
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(np.mean((targets - predictions) ** 2))
    corr, p_value = pearsonr(targets, predictions)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
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
# 交差検証（個人間解析用）
# ================================
def cross_validation(rgb_data, co_data, subject_ids, config):
    """個人間解析の交差検証"""
    print("\n" + "="*60)
    print(f"{config.n_folds}分割交差検証開始")
    print("="*60)
    
    unique_subjects = sorted(list(set(subject_ids)))
    subject_indices = {subj: [] for subj in unique_subjects}
    for i, subj in enumerate(subject_ids):
        subject_indices[subj].append(i)
    
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    
    results = []
    for fold, (train_subj_idx, test_subj_idx) in enumerate(kf.split(unique_subjects)):
        print(f"\n--- Fold {fold+1}/{config.n_folds} ---")
        
        train_subjects = [unique_subjects[i] for i in train_subj_idx]
        test_subjects = [unique_subjects[i] for i in test_subj_idx]
        
        print(f"  テスト被験者: {test_subjects}")
        
        # インデックス取得
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
        
        # 訓練・検証分割
        split_idx = int(len(train_val_rgb) * 0.8)
        train_rgb = train_val_rgb[:split_idx]
        train_co = train_val_co[:split_idx]
        val_rgb = train_val_rgb[split_idx:]
        val_co = train_val_co[split_idx:]
        
        # データローダー作成
        train_dataset = CODataset(train_rgb, train_co, config.use_channel)
        val_dataset = CODataset(val_rgb, val_co, config.use_channel)
        test_dataset = CODataset(test_rgb, test_co, config.use_channel)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # モデル作成と学習
        model = create_model(config)
        model, train_losses, val_losses = train_model(model, train_loader, val_loader, config, fold)
        
        # 評価
        eval_results = evaluate_model(model, test_loader, config)
        print(f"  結果 - MAE: {eval_results['mae']:.4f}, Corr: {eval_results['corr']:.4f}")
        
        results.append({
            'fold': fold,
            'mae': eval_results['mae'],
            'corr': eval_results['corr'],
            'predictions': eval_results['predictions'],
            'targets': eval_results['targets'],
            'test_subjects': test_subjects
        })
    
    return results

# ================================
# プロット（個人内）
# ================================
def plot_results_individual(eval_results, train_losses, val_losses, config):
    """個人内解析の結果プロット"""
    fig = plt.figure(figsize=(15, 10))
    
    predictions = eval_results['predictions']
    targets = eval_results['targets']
    
    # 1. 学習曲線
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('学習曲線')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 予測値 vs 真値
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想線')
    ax2.set_xlabel('真値 (CO)')
    ax2.set_ylabel('予測値 (CO)')
    ax2.set_title(f"予測結果\nMAE: {eval_results['mae']:.3f}, Corr: {eval_results['corr']:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差プロット
    ax3 = plt.subplot(2, 3, 3)
    residuals = targets - predictions
    ax3.scatter(predictions, residuals, alpha=0.5, s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('予測値')
    ax3.set_ylabel('残差')
    ax3.set_title(f'残差プロット\n平均: {residuals.mean():.3f}, 標準偏差: {residuals.std():.3f}')
    ax3.grid(True, alpha=0.3)
    
    # 4. 時系列プロット
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
    
    # 5. 誤差ヒストグラム
    ax5 = plt.subplot(2, 3, 5)
    errors = np.abs(targets - predictions)
    ax5.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax5.axvline(x=eval_results['mae'], color='r', linestyle='--', lw=2, 
                label=f"MAE: {eval_results['mae']:.3f}")
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
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'PhysNet2DCNN - CO推定結果 ({config.subjects[0]}) - {config.use_channel}成分', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'results_individual_{config.use_channel}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

# ================================
# プロット（個人間）
# ================================
def plot_results_cross(results, config):
    """個人間解析の結果プロット"""
    n_plots = min(config.n_folds, 8)
    fig = plt.figure(figsize=(20, 12))
    
    # 各fold結果
    for i in range(n_plots):
        ax = plt.subplot(3, 3, i+1)
        fold_result = results[i]
        
        ax.scatter(fold_result['targets'], fold_result['predictions'], alpha=0.5, s=10)
        min_val = min(fold_result['targets'].min(), fold_result['predictions'].min())
        max_val = max(fold_result['targets'].max(), fold_result['predictions'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel('真値 (CO)')
        ax.set_ylabel('予測値 (CO)')
        ax.set_title(f"Fold {i+1}\nMAE: {fold_result['mae']:.3f}, Corr: {fold_result['corr']:.3f}")
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
    
    ax.set_xlabel('真値 (CO)')
    ax.set_ylabel('予測値 (CO)')
    ax.set_title(f'全体結果\nMAE: {overall_mae:.3f}, Corr: {overall_corr:.3f}')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'PhysNet2DCNN - CO推定結果 ({config.n_folds}分割交差検証) - {config.use_channel}成分', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    save_dir = Path(config.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'results_cross_{config.use_channel}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    return overall_mae, overall_corr

# ================================
# メイン実行
# ================================
def main():
    config = Config()
    
    print("\n" + "="*60)
    print(" PhysNet2DCNN - CO推定モデル")
    print("="*60)
    print(f"解析タイプ: {'個人内' if config.analysis_type == 'individual' else '個人間'}")
    print(f"使用チャンネル: {config.use_channel}成分")
    print(f"データ分割: {'ランダム' if config.random_split else '順番'}")
    print(f"デバイス: {config.device}")
    print(f"保存先: {config.save_path}")
    
    try:
        # データ読み込み
        rgb_data, co_data, subject_ids = load_all_data(config)
        
        if config.analysis_type == "individual":
            # 個人内解析
            train_data, val_data, test_data = split_data_individual(rgb_data, co_data, config)
            train_rgb, train_co = train_data
            val_rgb, val_co = val_data
            test_rgb, test_co = test_data
            
            # データローダー作成
            train_dataset = CODataset(train_rgb, train_co, config.use_channel)
            val_dataset = CODataset(val_rgb, val_co, config.use_channel)
            test_dataset = CODataset(test_rgb, test_co, config.use_channel)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            
            # モデル作成と学習
            model = create_model(config)
            model, train_losses, val_losses = train_model(model, train_loader, val_loader, config)
            
            # 評価
            print("\nテストデータで評価中...")
            eval_results = evaluate_model(model, test_loader, config)
            
            print("\n結果:")
            print(f"  MAE: {eval_results['mae']:.4f}")
            print(f"  RMSE: {eval_results['rmse']:.4f}")
            print(f"  相関係数: {eval_results['corr']:.4f} (p={eval_results['p_value']:.2e})")
            print(f"  R²スコア: {eval_results['r2']:.4f}")
            
            # プロット
            plot_results_individual(eval_results, train_losses, val_losses, config)
            
            # 結果保存
            save_dir = Path(config.save_path)
            np.save(save_dir / f'results_individual_{config.use_channel}.npy', 
                    {'eval_results': eval_results, 
                     'train_losses': train_losses,
                     'val_losses': val_losses}, 
                    allow_pickle=True)
            
        else:
            # 個人間解析
            results = cross_validation(rgb_data, co_data, subject_ids, config)
            
            # プロット
            overall_mae, overall_corr = plot_results_cross(results, config)
            
            # 結果保存
            save_dir = Path(config.save_path)
            np.save(save_dir / f'results_cross_{config.use_channel}.npy', 
                    results, allow_pickle=True)
            
            # サマリー表示
            print("\n" + "="*60)
            print(" 交差検証結果")
            print("="*60)
            for r in results:
                print(f"Fold {r['fold']+1}: MAE={r['mae']:.4f}, Corr={r['corr']:.4f}")
            print(f"\n全体: MAE={overall_mae:.4f}, Corr={overall_corr:.4f}")
        
        print("\n完了しました。")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
