import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, correlate
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def extract_rgb_features(rgb_data):
    """RGB信号から特徴量（平均値の時系列）を抽出"""
    # 各フレームのRGB平均値を計算
    rgb_mean = np.mean(rgb_data, axis=(1, 2, 3))  # (11402,)
    return rgb_mean

def extract_hemodynamic_features(signals_path, subject, param='CO'):
    """血行動態パラメータを時系列として結合"""
    tasks = ['t1-1', 't2', 't1-2', 't4', 't1-3', 't5']
    hemodynamic_data = []
    
    param_path = os.path.join(signals_path, subject, param)
    
    for task in tasks:
        file_name = f"{param}_62_{task}.npy"
        file_path = os.path.join(param_path, file_name)
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            hemodynamic_data.append(data)
        else:
            # ファイルが存在しない場合はゼロで埋める
            hemodynamic_data.append(np.zeros(60))
    
    # 全タスクを結合（360秒分）
    return np.concatenate(hemodynamic_data)

def find_optimal_lag(rgb_features, hemodynamic_features, max_lag=1000):
    """相関が最大となる時間遅れを推定"""
    # RGB信号をリサンプリング（11402点から360点へ）
    # 約31.67点ごとに1点サンプリング
    sampling_rate = len(rgb_features) / 360
    rgb_resampled = rgb_features[::int(sampling_rate)][:360]
    
    # 相互相関を計算
    correlation = correlate(rgb_resampled, hemodynamic_features, mode='full')
    
    # 最大相関のラグを見つける
    lags = np.arange(-len(hemodynamic_features) + 1, len(rgb_resampled))
    max_corr_idx = np.argmax(np.abs(correlation))
    optimal_lag = lags[max_corr_idx]
    
    # 元のサンプリングレートに変換
    optimal_lag_frames = int(optimal_lag * sampling_rate)
    
    return optimal_lag_frames

def process_subject(subject_id, eye_below_path, signals_path, output_path, graph_path):
    """1人の被験者データを処理"""
    print(f"\n処理中: {subject_id}")
    
    # RGB信号を読み込み
    rgb_file = os.path.join(eye_below_path, subject_id, f"{subject_id}_downsampled_1Hz.npy")
    if not os.path.exists(rgb_file):
        print(f"  RGBファイルが見つかりません: {rgb_file}")
        return
    
    rgb_data = np.load(rgb_file)
    print(f"  元のRGBデータ形状: {rgb_data.shape}")
    
    # RGB特徴量を抽出
    rgb_features = extract_rgb_features(rgb_data)
    
    # 複数の血行動態パラメータで時間遅れを推定
    params_to_check = ['CO', 'HR_CO_SV_T', 'SV', 'reMAP']
    lags = []
    
    for param in params_to_check:
        try:
            hemodynamic_features = extract_hemodynamic_features(signals_path, subject_id, param)
            lag = find_optimal_lag(rgb_features, hemodynamic_features)
            lags.append(lag)
            print(f"  {param}での時間遅れ: {lag}フレーム")
        except Exception as e:
            print(f"  {param}の処理でエラー: {e}")
            continue
    
    # 平均的な時間遅れを使用（外れ値を除外）
    if lags:
        optimal_lag = int(np.median(lags))
    else:
        optimal_lag = 0
    
    print(f"  推定された時間遅れ（中央値）: {optimal_lag}フレーム")
    
    # データを10800フレームに切り出し
    # 前後から均等に削除
    total_to_remove = rgb_data.shape[0] - 10800
    
    if total_to_remove > 0:
        # 時間遅れを考慮して開始位置を調整
        start_idx = max(0, optimal_lag)
        end_idx = start_idx + 10800
        
        # 範囲を超える場合は調整
        if end_idx > rgb_data.shape[0]:
            end_idx = rgb_data.shape[0]
            start_idx = end_idx - 10800
        
        rgb_trimmed = rgb_data[start_idx:end_idx]
    else:
        rgb_trimmed = rgb_data
    
    print(f"  トリミング後の形状: {rgb_trimmed.shape}")
    
    # 30点平均で1Hzにダウンサンプリング（360点にする）
    downsampled_data = []
    for i in range(0, 10800, 30):
        window = rgb_trimmed[i:i+30]
        if len(window) > 0:
            downsampled_data.append(np.mean(window, axis=0))
    
    downsampled_data = np.array(downsampled_data)
    print(f"  ダウンサンプリング後の形状: {downsampled_data.shape}")
    
    # npyファイルとして保存
    output_file = os.path.join(eye_below_path, subject_id, f"{subject_id}_downsampled_1Hzver2.npy")
    np.save(output_file, downsampled_data)
    print(f"  保存完了: {output_file}")
    
    # グラフを作成して保存
    create_and_save_graph(downsampled_data, subject_id, graph_path)

def create_and_save_graph(data, subject_id, graph_path):
    """ダウンサンプリング後のRGB信号のグラフを作成"""
    os.makedirs(graph_path, exist_ok=True)
    
    # 各チャンネルの平均値を計算
    r_mean = np.mean(data[:, :, :, 0], axis=(1, 2))
    g_mean = np.mean(data[:, :, :, 1], axis=(1, 2))
    b_mean = np.mean(data[:, :, :, 2], axis=(1, 2))
    
    # 時間軸（秒）
    time = np.arange(len(data))
    
    # グラフを作成
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # タスクの境界を追加
    task_boundaries = [0, 60, 120, 180, 240, 300, 360]
    task_names = ['t1-1\n(安静)', 't2\n(息止め)', 't1-2\n(安静)', 't4\n(足踏み)', 't1-3\n(安静)', 't5\n(足に力)']
    
    for ax, data_channel, color, title in zip([ax1, ax2, ax3], 
                                               [r_mean, g_mean, b_mean],
                                               ['red', 'green', 'blue'],
                                               ['Red Channel', 'Green Channel', 'Blue Channel']):
        ax.plot(time, data_channel, color=color, linewidth=0.8)
        ax.set_ylabel('Mean Intensity')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # タスク境界を追加
        for i, boundary in enumerate(task_boundaries[:-1]):
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            if i < len(task_names):
                ax.text(boundary + 30, ax.get_ylim()[1] * 0.95, task_names[i], 
                       ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.7))
    
    ax3.set_xlabel('Time (seconds)')
    
    plt.suptitle(f'RGB Signal Analysis - {subject_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # グラフを保存
    output_file = os.path.join(graph_path, f"{subject_id}_rgb_signal.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  グラフ保存完了: {output_file}")

def main():
    """メイン処理"""
    # パスの設定
    eye_below_path = r"C:\Users\EyeBelow"
    signals_path = r"C:\Users\Data\signals_bp"
    output_path = eye_below_path  # 同じフォルダーに保存
    graph_path = r"C:\ダウンサンプリング後のRGB信号ver2"
    
    # グラフ保存フォルダーを作成
    os.makedirs(graph_path, exist_ok=True)
    
    # 被験者リスト（bp001からbp032まで）
    subjects = [f"bp{i:03d}" for i in range(1, 33)]
    
    print("=" * 60)
    print("RGB信号と血行動態データの同期・ダウンサンプリング処理")
    print("=" * 60)
    
    # 各被験者を処理
    for subject_id in subjects:
        try:
            process_subject(subject_id, eye_below_path, signals_path, output_path, graph_path)
        except Exception as e:
            print(f"\nエラー: {subject_id}の処理中にエラーが発生しました")
            print(f"  詳細: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("全ての処理が完了しました")
    print("=" * 60)

if __name__ == "__main__":
    main()
