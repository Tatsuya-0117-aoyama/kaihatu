import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CODataset(Dataset):
    """CO推定用のデータセット"""
    def __init__(self, rgb_data, co_data, window_size=10, stride=1):
        """
        Args:
            rgb_data: RGB画像データ (N, 14, 16, 3)
            co_data: COデータ (N,)
            window_size: スライディングウィンドウのサイズ
            stride: ウィンドウのストライド
        """
        self.window_size = window_size
        self.stride = stride
        
        # 14×16を224に変換（フラット化）
        N = rgb_data.shape[0]
        rgb_flat = rgb_data.reshape(N, -1, 3)  # (N, 224, 3)
        self.rgb_data = rgb_flat.transpose(0, 2, 1)  # (N, 3, 224)
        self.co_data = co_data
        
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

def load_data(subject='bp001', base_rgb_path='C:/Users/EyeBelow', 
              base_co_path='C:/Users/Data signals_bp'):
    """データの読み込みと分割"""
    
    # タスクリストと順番
    task_order = ['t1-1', 't2', 't1-2', 't4', 't1-3', 't5']
    
    # RGBデータの読み込み
    rgb_path = Path(base_rgb_path) / subject / f'{subject}_downsampled_1Hz.npy'
    rgb_data = np.load(rgb_path)  # (360, 14, 16, 3)
    print(f"RGB data shape: {rgb_data.shape}")
    
    # COデータの読み込みと結合
    co_data_list = []
    for task in task_order:
        co_path = Path(base_co_path) / subject / 'CO' / f'CO_s2_{task}.npy'
        co_task_data = np.load(co_path)  # (60,)
        co_data_list.append(co_task_data)
    
    co_data = np.concatenate(co_data_list)  # (360,)
    print(f"CO data shape: {co_data.shape}")
    
    # 各タスクごとにtrain/val/testに分割
    train_rgb, train_co = [], []
    val_rgb, val_co = [], []
    test_rgb, test_co = [], []
    
    for i, task in enumerate(task_order):
        start_idx = i * 60
        end_idx = start_idx + 60
        
        task_rgb = rgb_data[start_idx:end_idx]
        task_co = co_data[start_idx:end_idx]
        
        # 6:2:2で分割
        train_end = 36
        val_end = 48
        
        train_rgb.append(task_rgb[:train_end])
        train_co.append(task_co[:train_end])
        
        val_rgb.append(task_rgb[train_end:val_end])
        val_co.append(task_co[train_end:val_end])
        
        test_rgb.append(task_rgb[val_end:])
        test_co.append(task_co[val_end:])
    
    # 結合
    train_rgb = np.concatenate(train_rgb, axis=0)  # (216, 14, 16, 3)
    train_co = np.concatenate(train_co, axis=0)     # (216,)
    val_rgb = np.concatenate(val_rgb, axis=0)       # (72, 14, 16, 3)
    val_co = np.concatenate(val_co, axis=0)         # (72,)
    test_rgb = np.concatenate(test_rgb, axis=0)     # (72, 14, 16, 3)
    test_co = np.concatenate(test_co, axis=0)       # (72,)
    
    print(f"Train: RGB {train_rgb.shape}, CO {train_co.shape}")
    print(f"Val: RGB {val_rgb.shape}, CO {val_co.shape}")
    print(f"Test: RGB {test_rgb.shape}, CO {test_co.shape}")
    
    return (train_rgb, train_co), (val_rgb, val_co), (test_rgb, test_co)

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """モデルの学習"""
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
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
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}')
    
    return train_losses, val_losses

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

def main():
    # ハイパーパラメータ
    WINDOW_SIZE = 10
    STRIDE = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # データの読み込み
    print("Loading data...")
    train_data, val_data, test_data = load_data()
    
    # データセットの作成
    train_dataset = CODataset(train_data[0], train_data[1], WINDOW_SIZE, STRIDE)
    val_dataset = CODataset(val_data[0], val_data[1], WINDOW_SIZE, STRIDE)
    test_dataset = CODataset(test_data[0], test_data[1], WINDOW_SIZE, STRIDE)
    
    # データローダーの作成
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # モデルの作成
    print("\nCreating model...")
    model = PhysNet2DCNN(input_channels=3, window_size=WINDOW_SIZE).to(device)
    
    # 学習
    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    
    # 評価
    print("\nEvaluating model...")
    test_mae, predictions, targets = evaluate_model(model, test_loader)
    print(f"Test MAE: {test_mae:.4f}")
    
    # 結果のプロット
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 学習曲線
    axes[0].plot(train_losses, label='Train MAE')
    axes[0].plot(val_losses, label='Val MAE')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True)
    
    # 予測vs実測（散布図）
    axes[1].scatter(targets, predictions, alpha=0.5)
    axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    axes[1].set_xlabel('True CO')
    axes[1].set_ylabel('Predicted CO')
    axes[1].set_title(f'Predictions vs True (MAE: {test_mae:.4f})')
    axes[1].grid(True)
    
    # 時系列比較（一部）
    sample_len = min(100, len(targets))
    axes[2].plot(targets[:sample_len], label='True CO', alpha=0.7)
    axes[2].plot(predictions[:sample_len], label='Predicted CO', alpha=0.7)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('CO')
    axes[2].set_title('Time Series Comparison (First 100 samples)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('co_estimation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # モデルの保存
    torch.save(model.state_dict(), 'physnet_co_model_bp001.pth')
    print("\nModel saved as 'physnet_co_model_bp001.pth'")
    
    return model, test_mae

if __name__ == "__main__":
    model, test_mae = main()
