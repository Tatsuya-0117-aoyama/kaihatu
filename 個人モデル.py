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
        
        # ★血行動態信号タイプ設定（ユーザー変更可能）
        self.signal_type = "CO"  # "CO", "HbO", "HbR", "HbT" など
        self.signal_prefix = "CO_s2"  # ファイル名のプレフィックス
        
        # 被験者設定（bp001～bp032）
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定（6分割交差検証用）
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60
        
        # モデルタイプ選択
        self.model_type = "deep"  # "standard", "deep", "very_deep", "resnet"
        
        # 使用チャンネル設定
        self.use_channel = 'B'  # 'R', 'G', 'B', 'RGB'
        self.input_shape = (14, 16, 1 if self.use_channel != 'RGB' else 3)
        
        # 学習設定
        self.batch_size = 32
        self.epochs = 200
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.patience = 40
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7  # MSE/Huber損失の重み
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
    """MSE損失と相関損失を組み合わせた複合損失関数"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE損失
        mse_loss = self.mse(pred, target)
        
        # 相関損失
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
        pred_mean = pred - pred.mean()
        target_mean = target - target.mean()
        
        numerator = torch.sum(pred_mean * target_mean)
        denominator = torch.sqrt(torch.sum(pred_mean ** 2) * torch.sum(target_mean ** 2) + 1e-8)
        correlation = numerator / denominator
        corr_loss = 1 - correlation
        
        total_loss = self.alpha * huber_loss + self.beta * corr_loss
        
        return total_loss, huber_loss, corr_loss

# ================================
# データセット
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
            selected_data = rgb_data
        
        self.rgb_data = torch.FloatTensor(selected_data).permute(0, 3, 1, 2)
        self.signal_data = torch.FloatTensor(signal_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.signal_data[idx]

# ================================
# DeepPhysNet2DCNNモデル
# ================================
class DeepPhysNet2DCNN(nn.Module):
    """深層PhysNet2DCNN（8ブロック構成）"""
    def __init__(self, input_shape):
        super(DeepPhysNet2DCNN, self).__init__()
        
        in_channels = input_shape[2]
        
        self.num_blocks = 8
        self.channels = [32, 64, 64, 128, 128, 128, 256, 256]
        
        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        
        prev_channels = in_channels
        
        for i in range(self.num_blocks):
            out_channels = self.channels[i]
            
            if i == 0:
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=7, padding=3),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU()
                )
            elif i < 4:
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU(),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU()
                )
            else:
                mid_channels = out_channels
                conv_block = nn.Sequential(
                    nn.Conv1d(prev_channels, mid_channels, kernel_size=1),
                    nn.BatchNorm1d(mid_channels),
                    nn.ELU(),
                    nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(mid_channels),
                    nn.ELU(),
                    nn.Conv1d(mid_channels, out_channels, kernel_size=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ELU()
                )
            
            self.conv_blocks.append(conv_block)
            
            if i < 2:
                self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
            elif i < 4:
                if i % 2 == 0:
                    self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
                else:
                    self.pools.append(None)
            else:
                self.pools.append(None)
            
            if prev_channels != out_channels:
                self.residual_convs.append(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=1)
                )
            else:
                self.residual_convs.append(None)
            
            prev_channels = out_channels
        
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        
        hidden_dim = 128
        self.fc = nn.Sequential(
            nn.Linear(prev_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.block_dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        in_channels = x.size(1)
        
        x = x.view(batch_size, in_channels, -1)
        
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
        
        x = self.global_avgpool(x)
        x = x.squeeze(-1)
        
        x = self.fc(x)
        x = x.squeeze(-1)
        
        if batch_size == 1:
            x = x.unsqueeze(0)
        
        return x

# ================================
# データ読み込み
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み"""
    rgb_path = os.path.join(config.rgb_base_path, subject, 
                            f"{subject}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_path):
        print(f"警告: {subject}のRGBデータが見つかりません")
        return None, None
    
    rgb_data = np.load(rgb_path)
    
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
    return rgb_data, signal_data

# ================================
# 6分割交差検証
# ================================
def task_cross_validation(rgb_data, signal_data, config, subject, subject_save_dir):
    """タスクごとの6分割交差検証"""
    
    fold_results = []
    all_test_predictions = []
    all_test_targets = []
    all_test_task_indices = []
    
    for fold, test_task in enumerate(config.tasks):
        print(f"\n  Fold {fold+1}/6 - テストタスク: {test_task}")
        
        # タスクごとにデータを分割
        train_rgb_list = []
        train_signal_list = []
        test_rgb_list = []
        test_signal_list = []
        
        for i, task in enumerate(config.tasks):
            start_idx = i * config.task_duration
            end_idx = (i + 1) * config.task_duration
            
            task_rgb = rgb_data[start_idx:end_idx]
            task_signal = signal_data[start_idx:end_idx]
            
            if task == test_task:
                test_rgb_list.append(task_rgb)
                test_signal_list.append(task_signal)
                all_test_task_indices.extend([i] * config.task_duration)
            else:
                train_rgb_list.append(task_rgb)
                train_signal_list.append(task_signal)
        
        # データ結合
        train_rgb = np.concatenate(train_rgb_list)
        train_signal = np.concatenate(train_signal_list)
        test_rgb = np.concatenate(test_rgb_list)
        test_signal = np.concatenate(test_signal_list)
        
        # 訓練・検証分割（8:2）
        split_idx = int(len(train_rgb) * 0.8)
        val_rgb = train_rgb[split_idx:]
        val_signal = train_signal[split_idx:]
        train_rgb = train_rgb[:split_idx]
        train_signal = train_signal[:split_idx]
        
        # データローダー作成
        train_dataset = CODataset(train_rgb, train_signal, config.use_channel)
        val_dataset = CODataset(val_rgb, val_signal, config.use_channel)
        test_dataset = CODataset(test_rgb, test_signal, config.use_channel)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # モデル作成と学習
        model = DeepPhysNet2DCNN(config.input_shape)
        model = model.to(config.device)
        
        # 損失関数とオプティマイザ
        criterion = CombinedLoss(alpha=config.loss_alpha, beta=config.loss_beta)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                             weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult, eta_min=1e-6
        )
        
        # 学習
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # 訓練
            model.train()
            train_loss = 0
            for rgb, sig in train_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                
                optimizer.zero_grad()
                pred = model(rgb)
                loss, _, _ = criterion(pred, sig)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 検証
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for rgb, sig in val_loader:
                    rgb, sig = rgb.to(config.device), sig.to(config.device)
                    pred = model(rgb)
                    loss, _, _ = criterion(pred, sig)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step()
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # モデル保存
                torch.save(model.state_dict(), 
                          subject_save_dir / f'best_model_fold{fold+1}.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= config.patience:
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # ベストモデル読み込み
        model.load_state_dict(torch.load(subject_save_dir / f'best_model_fold{fold+1}.pth'))
        
        # 評価
        model.eval()
        train_predictions = []
        train_targets = []
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            # 訓練データの予測
            for rgb, sig in train_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                pred = model(rgb)
                train_predictions.extend(pred.cpu().numpy())
                train_targets.extend(sig.cpu().numpy())
            
            # テストデータの予測
            for rgb, sig in test_loader:
                rgb, sig = rgb.to(config.device), sig.to(config.device)
                pred = model(rgb)
                test_predictions.extend(pred.cpu().numpy())
                test_targets.extend(sig.cpu().numpy())
        
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # メトリクス計算
        train_mae = mean_absolute_error(train_targets, train_predictions)
        train_corr, _ = pearsonr(train_targets, train_predictions)
        test_mae = mean_absolute_error(test_targets, test_predictions)
        test_corr, _ = pearsonr(test_targets, test_predictions)
        
        print(f"    Train: MAE={train_mae:.4f}, Corr={train_corr:.4f}")
        print(f"    Test:  MAE={test_mae:.4f}, Corr={test_corr:.4f}")
        
        # 結果保存
        fold_results.append({
            'fold': fold + 1,
            'test_task': test_task,
            'train_predictions': train_predictions,
            'train_targets': train_targets,
            'test_predictions': test_predictions,
            'test_targets': test_targets,
            'train_mae': train_mae,
            'train_corr': train_corr,
            'test_mae': test_mae,
            'test_corr': test_corr
        })
        
        # 全体のテストデータ集約
        all_test_predictions.extend(test_predictions)
        all_test_targets.extend(test_targets)
        
        # 各Foldのプロット
        plot_fold_results(fold_results[-1], subject_save_dir)
    
    # テスト予測を元の順序に並び替え
    sorted_indices = np.argsort(all_test_task_indices)
    all_test_predictions = np.array(all_test_predictions)[sorted_indices]
    all_test_targets = np.array(all_test_targets)[sorted_indices]
    
    return fold_results, all_test_predictions, all_test_targets

# ================================
# プロット関数
# ================================
def plot_fold_results(result, save_dir):
    """各Foldの結果をプロット"""
    fold = result['fold']
    
    # 訓練データ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['train_targets'], result['train_predictions'], alpha=0.5, s=10)
    min_val = min(result['train_targets'].min(), result['train_predictions'].min())
    max_val = max(result['train_targets'].max(), result['train_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"Fold {fold} 訓練データ - MAE: {result['train_mae']:.3f}, Corr: {result['train_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_train_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # テストデータ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(result['test_targets'], result['test_predictions'], alpha=0.5, s=10)
    min_val = min(result['test_targets'].min(), result['test_predictions'].min())
    max_val = max(result['test_targets'].max(), result['test_predictions'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"Fold {fold} テストデータ ({result['test_task']}) - MAE: {result['test_mae']:.3f}, Corr: {result['test_corr']:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 波形比較
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
    plt.plot(result['test_predictions'], 'g-', label='予測', alpha=0.7, linewidth=1)
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'Fold {fold} テストデータ波形 ({result["test_task"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'fold{fold}_waveforms.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_subject_summary(fold_results, all_test_predictions, all_test_targets, 
                         subject, subject_save_dir):
    """被験者の全体結果をプロット"""
    
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
    plt.scatter(all_train_targets, all_train_predictions, alpha=0.5, s=10)
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
    
    # 全テストデータ散布図
    plt.figure(figsize=(10, 8))
    plt.scatter(all_test_targets, all_test_predictions, alpha=0.5, s=10)
    min_val = min(all_test_targets.min(), all_test_predictions.min())
    max_val = max(all_test_targets.max(), all_test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真値')
    plt.ylabel('予測値')
    plt.title(f"{subject} 全テストデータ - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 全テストデータ連結波形
    plt.figure(figsize=(20, 6))
    plt.plot(all_test_targets, 'b-', label='真値', alpha=0.7, linewidth=1)
    plt.plot(all_test_predictions, 'g-', label='予測', alpha=0.7, linewidth=1)
    
    # タスク境界に縦線
    for i in range(1, 6):
        plt.axvline(x=i*60, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('時間 (秒)')
    plt.ylabel('信号値')
    plt.title(f'{subject} 全テストデータ連結波形 - MAE: {all_test_mae:.3f}, Corr: {all_test_corr:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(subject_save_dir / 'all_test_waveform.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return all_train_mae, all_train_corr, all_test_mae, all_test_corr

def plot_all_subjects_summary(all_subjects_results, config):
    """全被験者のサマリープロット"""
    save_dir = Path(config.save_path)
    
    # 被験者ごとのメトリクスを抽出
    subjects = []
    train_maes = []
    train_corrs = []
    test_maes = []
    test_corrs = []
    
    for result in all_subjects_results:
        subjects.append(result['subject'])
        train_maes.append(result['train_mae'])
        train_corrs.append(result['train_corr'])
        test_maes.append(result['test_mae'])
        test_corrs.append(result['test_corr'])
    
    # 全被験者の訓練データサマリー
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.ravel()
    
    for i, result in enumerate(all_subjects_results[:32]):
        ax = axes[i]
        ax.text(0.5, 0.5, f"{result['subject']}\nMAE: {result['train_mae']:.3f}\nCorr: {result['train_corr']:.3f}", 
                ha='center', va='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#e8f4f8' if result['train_corr'] > 0.7 else '#f8e8e8')
    
    fig.suptitle(f'全被験者 訓練データ結果 - 平均MAE: {np.mean(train_maes):.3f}, 平均Corr: {np.mean(train_corrs):.3f}', 
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_subjects_train_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 全被験者のテストデータサマリー
    fig, axes = plt.subplots(4, 8, figsize=(24, 12))
    axes = axes.ravel()
    
    for i, result in enumerate(all_subjects_results[:32]):
        ax = axes[i]
        ax.text(0.5, 0.5, f"{result['subject']}\nMAE: {result['test_mae']:.3f}\nCorr: {result['test_corr']:.3f}", 
                ha='center', va='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
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
    print(" PhysNet2DCNN - 個人内解析（6分割交差検証）")
    print("="*60)
    print(f"血行動態信号: {config.signal_type}")
    print(f"モデルタイプ: {config.model_type}")
    print(f"チャンネル: {config.use_channel}")
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
            # データ読み込み
            rgb_data, signal_data = load_data_single_subject(subject, config)
            
            if rgb_data is None or signal_data is None:
                print(f"  {subject}のデータ読み込み失敗。スキップします。")
                continue
            
            print(f"  データ形状: RGB={rgb_data.shape}, Signal={signal_data.shape}")
            
            # 6分割交差検証実行
            fold_results, all_test_predictions, all_test_targets = task_cross_validation(
                rgb_data, signal_data, config, subject, subject_save_dir
            )
            
            # 被験者全体のサマリープロット
            train_mae, train_corr, test_mae, test_corr = plot_subject_summary(
                fold_results, all_test_predictions, all_test_targets, 
                subject, subject_save_dir
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
    
    # 全被験者のサマリープロット
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
