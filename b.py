import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def extract_rgb_features(rgb_data):
    """RGB信号から特徴量を抽出（平均輝度の時系列）"""
    # RGBチャネルの平均を計算 (frames, h, w, 3) -> (frames,)
    mean_intensity = np.mean(rgb_data, axis=(1, 2, 3))
    
    # ピークを検出
    peaks, _ = find_peaks(mean_intensity, distance=20)
    
    return mean_intensity, peaks

def extract_hemodynamic_features(hemo_data, param_name):
    """血行動態データから特徴量を抽出"""
    # パラメータによって特徴抽出方法を変える
    if param_name in ['CO', 'SV', 'HR_CO_SV']:
        # これらは重要な血行動態指標
        return hemo_data
    else:
        return hemo_data

def find_optimal_lag(rgb_features, hemo_features, max_lag=1000):
    """相互相関により最適な時間遅れを推定"""
    # 正規化
    rgb_norm = (rgb_features - np.mean(rgb_features)) / (np.std(rgb_features) + 1e-8)
    hemo_norm = (hemo_features - np.mean(hemo_features)) / (np.std(hemo_features) + 1e-8)
    
    # ヘモダイナミクスデータを30Hzにアップサンプリング（線形補間）
    hemo_upsampled = np.interp(
        np.linspace(0, len(hemo_norm)-1, len(hemo_norm)*30),
        np.arange(len(hemo_norm)),
        hemo_norm
    )
    
    # RGB信号の長さに合わせる
    if len(hemo_upsampled) < len(rgb_norm):
        hemo_upsampled = np.pad(hemo_upsampled, (0, len(rgb_norm) - len(hemo_upsampled)), 'edge')
    else:
        hemo_upsampled = hemo_upsampled[:len(rgb_norm)]
    
    # 相互相関を計算
    correlation = correlate(rgb_norm, hemo_upsampled, mode='same')
    
    # 最大相関の位置を見つける
    lag = np.argmax(np.abs(correlation)) - len(rgb_norm) // 2
    
    # 妥当な範囲に制限
    lag = np.clip(lag, -max_lag, max_lag)
    
    return lag, np.max(np.abs(correlation))

def process_subject(subject_id, rgb_base_path, hemo_base_path, graph_output_dir):
    """各被験者のデータを処理"""
    print(f"Processing {subject_id}...")
    
    # パス設定
    rgb_path = os.path.join(rgb_base_path, subject_id, f"{subject_id}_downsampled_1Hz.npy")
    hemo_subject_path = os.path.join(hemo_base_path, subject_id)
    output_path = os.path.join(rgb_base_path, subject_id, f"{subject_id}_downsampled_1Hzver2.npy")
    
    # 出力ディレクトリの作成
    os.makedirs(graph_output_dir, exist_ok=True)
    
    try:
        # RGB信号データを読み込み
        rgb_data = np.load(rgb_path)
        print(f"  Original RGB shape: {rgb_data.shape}")
        
        # RGB信号の特徴量を抽出
        rgb_features, rgb_peaks = extract_rgb_features(rgb_data)
        
        # 血行動態データとの相関を計算
        best_lag = 0
        best_correlation = 0
        best_param = None
        
        # 主要な血行動態パラメータを確認
        key_params = ['CO', 'SV', 'HR_CO_SV', 'CI', 'SVI']
        
        for param in key_params:
            param_path = os.path.join(hemo_subject_path, param)
            if not os.path.exists(param_path):
                continue
            
            # 各タスクのデータを結合
            all_hemo_data = []
            task_order = ['t1-1', 't2', 't1-2', 't4', 't1-3', 't5']
            
            for task in task_order:
                task_file = os.path.join(param_path, f"{param}_s2_{task}.npy")
                if os.path.exists(task_file):
                    try:
                        task_data = np.load(task_file)
                        all_hemo_data.append(task_data)
                    except:
                        continue
            
            if len(all_hemo_data) == 6:  # 全タスクのデータがある場合
                hemo_data = np.concatenate(all_hemo_data)
                hemo_features = extract_hemodynamic_features(hemo_data, param)
                
                # 時間遅れを推定
                lag, correlation = find_optimal_lag(rgb_features, hemo_features)
                
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_lag = lag
                    best_param = param
        
        print(f"  Best correlation with {best_param}: {best_correlation:.3f}, Lag: {best_lag} frames")
        
        # RGB信号のトリミング
        # 時間遅れに基づいて開始位置を調整
        start_idx = max(0, best_lag)
        start_idx = min(start_idx, rgb_data.shape[0] - 10800)
        
        # (10800, 14, 16, 3)にトリミング
        rgb_trimmed = rgb_data[start_idx:start_idx+10800]
        
        if rgb_trimmed.shape[0] < 10800:
            # パディングが必要な場合
            pad_size = 10800 - rgb_trimmed.shape[0]
            rgb_trimmed = np.pad(rgb_trimmed, 
                                ((0, pad_size), (0, 0), (0, 0), (0, 0)), 
                                mode='edge')
        
        print(f"  Trimmed RGB shape: {rgb_trimmed.shape}")
        
        # ダウンサンプリング (30点の平均で1Hz、360点に)
        downsampled_data = []
        for i in range(360):
            start = i * 30
            end = min(start + 30, rgb_trimmed.shape[0])
            if start < rgb_trimmed.shape[0]:
                segment = rgb_trimmed[start:end]
                if len(segment) > 0:
                    downsampled_data.append(np.mean(segment, axis=0))
                else:
                    # 最後のセグメントが短い場合は最後の値を使用
                    downsampled_data.append(rgb_trimmed[-1])
        
        downsampled_data = np.array(downsampled_data)
        print(f"  Downsampled RGB shape: {downsampled_data.shape}")
        
        # データを保存
        np.save(output_path, downsampled_data)
        print(f"  Saved to: {output_path}")
        
        # グラフを作成・保存
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 各チャネルの平均輝度をプロット
        time_axis = np.arange(360)
        
        for ch, color, label in zip(range(3), ['red', 'green', 'blue'], ['Red', 'Green', 'Blue']):
            channel_mean = np.mean(downsampled_data[:, :, :, ch], axis=(1, 2))
            axes[ch].plot(time_axis, channel_mean, color=color, linewidth=1)
            axes[ch].set_ylabel(f'{label} Intensity')
            axes[ch].grid(True, alpha=0.3)
            
            # タスク区切りを追加
            task_boundaries = [60, 120, 180, 240, 300]
            task_labels = ['Rest1', 'Hold', 'Rest2', 'Walk', 'Rest3', 'Leg']
            
            for i, boundary in enumerate(task_boundaries):
                axes[ch].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            
            if ch == 0:
                # 最初のサブプロットにタスクラベルを追加
                for i in range(6):
                    start = i * 60
                    axes[ch].text(start + 30, axes[ch].get_ylim()[1] * 0.95, 
                                task_labels[i], ha='center', fontsize=8)
        
        axes[2].set_xlabel('Time (seconds)')
        axes[0].set_title(f'{subject_id} - RGB Signal (1Hz, 360 samples)')
        
        plt.tight_layout()
        
        # グラフを保存
        graph_path = os.path.join(graph_output_dir, f"{subject_id}_rgb_signal.png")
        plt.savefig(graph_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Graph saved to: {graph_path}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {subject_id}: {str(e)}")
        return False

def main():
    """メイン処理"""
    print("Starting RGB signal synchronization and downsampling...")
    print("=" * 60)
    
    # 処理対象の被験者リスト
    subjects = [f"bp{str(i).zfill(3)}" for i in range(1, 33)]
    
    success_count = 0
    failed_subjects = []
    
    # ========== パス設定 ==========
    # RGB信号データのベースパス
    RGB_BASE_PATH = "C:\\Users\\EyeBelow"
    
    # 血行動態データのベースパス  
    HEMO_BASE_PATH = "C:\\Users\\Data signals_bp"
    
    # グラフ出力先ディレクトリ
    GRAPH_OUTPUT_DIR = "C:\\ダウンサンプリング後のRGB信号ver2"
    # ========== パス設定終了 ==========
    
    for subject in subjects:
        if process_subject(subject, RGB_BASE_PATH, HEMO_BASE_PATH, GRAPH_OUTPUT_DIR):
            success_count += 1
        else:
            failed_subjects.append(subject)
        print("-" * 40)
    
    # 結果サマリー
    print("=" * 60)
    print(f"Processing completed!")
    print(f"Successful: {success_count}/{len(subjects)}")
    
    if failed_subjects:
        print(f"Failed subjects: {', '.join(failed_subjects)}")
    
    print("\nAll downsampled data saved with suffix '_downsampled_1Hzver2.npy'")
    print(f"All graphs saved to '{GRAPH_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
