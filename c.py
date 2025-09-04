import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pickle

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CODataset(Dataset):
    """CO推定用のデータセット"""
    def __init__(self, rgb_data_list, co_data_list, window_size=10, stride=1):
        """
        Args:
            rgb_data_list: 複数被験者のRGB画像データのリスト
            co_data_list: 複数被験者のCOデータのリスト
            window_size: スライディングウィンドウのサイズ
            stride: ウィンドウのストライド
        """
        self.window_size = window_size
        self.stride = stride
        
        # 全被験者のデータを結合
        all_rgb_data = []
        all_co_data = []
        
        for rgb_data, co_data in zip(rgb_data_list, co_data_list):
            # 14×16を224に変換（フラット化）
            N = rgb_data.shape[0]
            rgb_flat = rgb_data.reshape(N, -1, 3)  # (N, 224, 3)
            rgb_flat = rgb_flat.transpose(0, 2, 1)  # (N, 3, 224)
            all_rgb_data.append(rgb_flat)
            all_co_data.append(co_data)
        
        self.rgb_data = np.concatenate(all_rgb_data, axis=0)
        self.co_data = np.concatenate(all_co_data, axis=0)
        
        # スライディングウィンドウのインデックスを作成
        self.indices = []
        for i in range(0, len(self.rgb_data) - window_size + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        # RGB入力: (3, 224, window_size)
        rgb_window = self.rgb_data[start_idx:end_idx].transpose(1, 2, 0)  # (3, 224, window_size)
        # CO出力: (window_size,)
        co_window = self.co_data[start_idx:end_idx]
        
        return torch.FloatTensor(rgb_window), torch.FloatTensor(co_window)

class PhysNet2DCNN(nn.Module):
    """PhysNetベースの2D CNNモデル"""
    def __init__(self, input_channels=3, window_size=10):
        super(PhysNet2DCNN, self).__init__()
        
        # エンコーダー部分
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(5, 3), padding=(2, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # グローバルプーリング
        self.global_pool = nn.AdaptiveAvgPool2d((1, window_size))
        
        # デコーダー部分（1×1×window_sizeのCO出力）
        self.conv_out = nn.Conv2d(128, 1, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (batch, 3, 224, window_size)
        
        # エンコーダー
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # グローバルプーリング
        x = self.global_pool(x)
        
        # 出力層
        x = self.conv_out(x)  # (batch, 1, 1, window_size)
        x = x.squeeze(1).squeeze(1)  # (batch, window_size)
        
        return x

def load_subject_data(subject, base_rgb_path='C:/Users/EyeBelow', 
                     base_co_path='C:/Users/Data signals_bp'):
    """1人の被験者のデータを読み込み"""
    
    # タスクリストと順番
    task_order = ['t1-1', 't2', 't1-2', 't4', 't1-3', 't5']
    
    try:
        # RGBデータの読み込み
        rgb_path = Path(base_rgb_path) / subject / f'{subject}_downsampled_1Hz.npy'
        rgb_data = np.load(rgb_path)  # (360, 14, 16, 3)
        
        # COデータの読み込みと結合
        co_data_list = []
        for task in task_order:
            co_path = Path(base_co_path) / subject / 'CO' / f'CO_s2_{task}.npy'
            co_task_data = np.load(co_path)  # (60,)
            co_data_list.append(co_task_data)
        
        co_data = np.concatenate(co_data_list)  # (360,)
        
        return rgb_data, co_data
    
    except Exception as e:
        print(f"Error loading data for {subject}: {e}")
        return None, None

def load_all_subjects_data(base_rgb_path='C:/Users/EyeBelow', 
                          base_co_path='C:/Users/Data signals_bp'):
    """全被験者のデータを読み込み"""
    
    subjects = [f'bp{i:03d}' for i in range(1, 33)]  # bp001からbp032
    
    all_rgb_data = []
    all_co_data = []
    valid_subjects = []
    
    print("Loading data for all subjects...")
    for subject in subjects:
        rgb_data, co_data = load_subject_data(subject, base_rgb_path, base_co_path)
        
        if rgb_data is not None and co_data is not None:
            all_rgb_data.append(rgb_data)
            all_co_data.append(co_data)
            valid_subjects.append(subject)
            print(f"  Loaded {subject}: RGB {rgb_data.shape}, CO {co_data.shape}")
    
    print(f"\nSuccessfully loaded {len(valid_subjects)} subjects")
    
    return all_rgb_data, all_co_data, valid_subjects

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """モデルの学習"""
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for rgb_batch, co_batch in train_loader:
            rgb_batch = rgb_batch.to(device)
            co_batch = co_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb_batch)
            loss = criterion(outputs, co_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * rgb_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rgb_batch, co_batch in val_loader:
                rgb_batch = rgb_batch.to(device)
                co_batch = co_batch.to(device)
                
                outputs = model(rgb_batch)
                loss = criterion(outputs, co_batch)
                val_loss += loss.item() * rgb_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Best model保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'    Epoch [{epoch+1}/{num_epochs}], Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}')
    
    # Best modelを復元
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, best_val_loss

def evaluate_model(model, test_loader):
    """モデルの評価"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for rgb_batch, co_batch in test_loader:
            rgb_batch = rgb_batch.to(device)
            outputs = model(rgb_batch).cpu().numpy()
            
            all_predictions.extend(outputs.flatten())
            all_targets.extend(co_batch.numpy().flatten())
    
    mae = mean_absolute_error(all_targets, all_predictions)
    return mae, np.array(all_predictions), np.array(all_targets)

def cross_validation_8fold(all_rgb_data, all_co_data, valid_subjects, 
                          window_size=10, stride=5, batch_size=32, 
                          num_epochs=100, learning_rate=0.001):
    """8分割交差検証"""
    
    n_subjects = len(valid_subjects)
    n_folds = 8
    subjects_per_fold = n_subjects // n_folds
    
    cv_results = {
        'fold_mae': [],
        'fold_predictions': [],
        'fold_targets': [],
        'fold_train_subjects': [],
        'fold_test_subjects': [],
        'best_models': []
    }
    
    print(f"\nStarting 8-fold cross validation with {n_subjects} subjects")
    print(f"Each fold will have {subjects_per_fold} test subjects\n")
    
    # 8分割交差検証
    for fold in range(n_folds):
        print(f"{'='*60}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        # テストセットとトレーニングセットの分割
        test_start_idx = fold * subjects_per_fold
        test_end_idx = min(test_start_idx + subjects_per_fold, n_subjects)
        
        test_indices = list(range(test_start_idx, test_end_idx))
        train_val_indices = [i for i in range(n_subjects) if i not in test_indices]
        
        # 訓練データから検証データを分離（訓練データの20%を検証用に）
        n_val = max(1, len(train_val_indices) // 5)
        val_indices = train_val_indices[-n_val:]
        train_indices = train_val_indices[:-n_val]
        
        # 被験者名の記録
        test_subjects = [valid_subjects[i] for i in test_indices]
        train_subjects = [valid_subjects[i] for i in train_indices]
        val_subjects = [valid_subjects[i] for i in val_indices]
        
        print(f"Train subjects ({len(train_subjects)}): {train_subjects[:3]}...{train_subjects[-3:]}")
        print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        
        # データの準備
        train_rgb = [all_rgb_data[i] for i in train_indices]
        train_co = [all_co_data[i] for i in train_indices]
        
        val_rgb = [all_rgb_data[i] for i in val_indices]
        val_co = [all_co_data[i] for i in val_indices]
        
        test_rgb = [all_rgb_data[i] for i in test_indices]
        test_co = [all_co_data[i] for i in test_indices]
        
        # データセットの作成
        train_dataset = CODataset(train_rgb, train_co, window_size, stride)
        val_dataset = CODataset(val_rgb, val_co, window_size, stride)
        test_dataset = CODataset(test_rgb, test_co, window_size, stride)
        
        # データローダーの作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
        
        # モデルの作成と学習
        model = PhysNet2DCNN(input_channels=3, window_size=window_size).to(device)
        
        print(f"\nTraining fold {fold+1}...")
        train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, num_epochs, learning_rate
        )
        
        # 評価
        test_mae, predictions, targets = evaluate_model(model, test_loader)
        print(f"Fold {fold+1} Test MAE: {test_mae:.4f}")
        
        # 結果の保存
        cv_results['fold_mae'].append(test_mae)
        cv_results['fold_predictions'].append(predictions)
        cv_results['fold_targets'].append(targets)
        cv_results['fold_train_subjects'].append(train_subjects)
        cv_results['fold_test_subjects'].append(test_subjects)
        cv_results['best_models'].append(model.state_dict().copy())
        
        # モデルを保存
        torch.save(model.state_dict(), f'physnet_co_model_fold{fold+1}.pth')
    
    # 全体の結果を計算
    avg_mae = np.mean(cv_results['fold_mae'])
    std_mae = np.std(cv_results['fold_mae'])
    
    print(f"\n{'='*60}")
    print("Cross Validation Results:")
    print(f"{'='*60}")
    for fold in range(n_folds):
        print(f"Fold {fold+1}: MAE = {cv_results['fold_mae'][fold]:.4f}")
    print(f"{'='*60}")
    print(f"Average MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    
    return cv_results

def plot_cv_results(cv_results):
    """交差検証結果の可視化"""
    n_folds = len(cv_results['fold_mae'])
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for fold in range(n_folds):
        ax = axes[fold]
        predictions = cv_results['fold_predictions'][fold]
        targets = cv_results['fold_targets'][fold]
        mae = cv_results['fold_mae'][fold]
        
        # 散布図
        ax.scatter(targets, predictions, alpha=0.3, s=1)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        ax.set_xlabel('True CO')
        ax.set_ylabel('Predicted CO')
        ax.set_title(f'Fold {fold+1} (MAE: {mae:.4f})')
        ax.grid(True, alpha=0.3)
        
        # R²スコアの計算と表示
        from sklearn.metrics import r2_score
        r2 = r2_score(targets, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'8-Fold Cross Validation Results\nAverage MAE: {np.mean(cv_results["fold_mae"]):.4f} ± {np.std(cv_results["fold_mae"]):.4f}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cv_results_8fold.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # MAEのバープロット
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fold_indices = np.arange(1, n_folds + 1)
    mae_values = cv_results['fold_mae']
    
    bars = ax.bar(fold_indices, mae_values, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(mae_values), color='red', linestyle='--', label=f'Mean MAE: {np.mean(mae_values):.4f}')
    ax.axhline(y=np.mean(mae_values) + np.std(mae_values), color='orange', linestyle=':', alpha=0.5)
    ax.axhline(y=np.mean(mae_values) - np.std(mae_values), color='orange', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Fold Number')
    ax.set_ylabel('MAE')
    ax.set_title('MAE per Fold in 8-Fold Cross Validation')
    ax.set_xticks(fold_indices)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 各バーの上にMAE値を表示
    for bar, mae in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('mae_per_fold.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # ハイパーパラメータ
    WINDOW_SIZE = 10
    STRIDE = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # 全被験者のデータを読み込み
    all_rgb_data, all_co_data, valid_subjects = load_all_subjects_data()
    
    if len(valid_subjects) < 8:
        print(f"Warning: Only {len(valid_subjects)} subjects available. Need at least 8 for 8-fold CV.")
        return None
    
    # 8分割交差検証の実行
    cv_results = cross_validation_8fold(
        all_rgb_data, all_co_data, valid_subjects,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # 結果の可視化
    plot_cv_results(cv_results)

    # 保存パスの設定
    save_dir = r"D:\EPSCAN\001"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 結果の保存
    with open('cv_results_8fold.pkl', 'wb') as f:
        pickle.dump(cv_results, f)
    print("\nCross validation results saved to 'cv_results_8fold.pkl'")
    
    # 被験者ごとのMAEを計算
    print("\n" + "="*60)
    print("Per-Subject Test Performance:")
    print("="*60)
    
    subject_mae = {}
    for fold_idx, test_subjects in enumerate(cv_results['fold_test_subjects']):
        mae = cv_results['fold_mae'][fold_idx]
        for subject in test_subjects:
            subject_mae[subject] = mae
    
    # ソートして表示
    for subject in sorted(subject_mae.keys()):
        print(f"{subject}: MAE = {subject_mae[subject]:.4f}")
    
    return cv_results

if __name__ == "__main__":
    cv_results = main()
