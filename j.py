import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os
from pathlib import Path

class SpO2_FFT_Processor:
    def __init__(self, base_path="C:/Users/Data signals_bp"):
        self.base_path = Path(base_path)
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        self.tasks = ['t1-1', 't2', 't1-2', 't4', 't1-3', 't5']
        
    def process_spo2_fft(self, spo2_data, sampling_rate):
        """FFTでHRとCOを計算"""
        # 前処理
        spo2_data = spo2_data - np.mean(spo2_data)
        
        # FFT
        fft_vals = np.fft.rfft(spo2_data)
        power_spectrum = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(spo2_data), d=1/sampling_rate)
        
        # HR推定 (0.5-3 Hz = 30-180 bpm)
        hr_range = (freqs >= 0.5) & (freqs <= 3.0)
        if np.any(hr_range):
            hr_freqs = freqs[hr_range]
            hr_power = power_spectrum[hr_range]
            dominant_freq = hr_freqs[np.argmax(hr_power)]
            heart_rate = dominant_freq * 60  # bpm
        else:
            heart_rate = 60  # デフォルト値
            
        # SV指標（簡易推定）
        peaks, _ = find_peaks(spo2_data, distance=int(sampling_rate*0.5))
        if len(peaks) > 1:
            sv_index = np.std(spo2_data) * len(peaks)
        else:
            sv_index = np.std(spo2_data)
            
        # CO指標
        co_index = (heart_rate / 60) * sv_index
        
        # 60秒分のデータとして保存用配列を作成
        hr_array = np.full(60, heart_rate)  # 60秒分のHR値
        co_array = np.full(60, co_index)    # 60秒分のCO値
        
        return hr_array, co_array
    
    def process_subject(self, subject):
        """1人の被験者の全タスクを処理"""
        subject_path = self.base_path / subject
        spo2_path = subject_path / "Spo2_Wave"
        
        if not spo2_path.exists():
            print(f"SpO2_Waveフォルダーが見つかりません: {subject}")
            return None
        
        # HR_FFTとCO_FFTフォルダー作成
        hr_fft_path = subject_path / "HR_FFT"
        co_fft_path = subject_path / "CO_FFT"
        hr_fft_path.mkdir(exist_ok=True)
        co_fft_path.mkdir(exist_ok=True)
        
        all_hr = []
        all_co = []
        
        for task in self.tasks:
            spo2_file = spo2_path / f"Spo2_Wave_s2_{task}.npy"
            
            if not spo2_file.exists():
                print(f"ファイルが見つかりません: {spo2_file}")
                # デフォルト値で埋める
                hr_data = np.full(60, 60)
                co_data = np.full(60, 1.0)
            else:
                # SpO2データ読み込み
                spo2_data = np.load(spo2_file)
                
                # サンプリングレート推定（60秒のデータ）
                sampling_rate = len(spo2_data) / 60
                
                # FFT処理
                hr_data, co_data = self.process_spo2_fft(spo2_data, sampling_rate)
            
            # npyファイルとして保存
            np.save(hr_fft_path / f"HR_FFT_s2_{task}.npy", hr_data)
            np.save(co_fft_path / f"CO_FFT_s2_{task}.npy", co_data)
            
            all_hr.append(hr_data)
            all_co.append(co_data)
        
        return np.concatenate(all_hr), np.concatenate(all_co)
    
    def plot_all_tasks(self, subject):
        """全タスクを1つのグラフに表示"""
        subject_path = self.base_path / subject
        
        # 保存されたデータを読み込み
        all_hr = []
        all_co = []
        
        hr_fft_path = subject_path / "HR_FFT"
        co_fft_path = subject_path / "CO_FFT"
        
        for task in self.tasks:
            hr_file = hr_fft_path / f"HR_FFT_s2_{task}.npy"
            co_file = co_fft_path / f"CO_FFT_s2_{task}.npy"
            
            if hr_file.exists() and co_file.exists():
                all_hr.append(np.load(hr_file))
                all_co.append(np.load(co_file))
            else:
                all_hr.append(np.full(60, np.nan))
                all_co.append(np.full(60, np.nan))
        
        # 連結
        hr_concat = np.concatenate(all_hr)
        co_concat = np.concatenate(all_co)
        
        # 時間軸（秒）
        time = np.arange(len(hr_concat))
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # HR
        ax1.plot(time, hr_concat, 'b-', linewidth=1)
        ax1.set_ylabel('Heart Rate (bpm)', fontsize=12)
        ax1.set_title(f'{subject} - HR_FFT', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 360)
        
        # CO
        ax2.plot(time, co_concat, 'g-', linewidth=1)
        ax2.set_ylabel('CO Index', fontsize=12)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_title(f'{subject} - CO_FFT', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 360)
        
        # タスクの境目に赤線
        task_boundaries = [60, 120, 180, 240, 300]  # t1-1(60s), t2(60s), t1-2(60s), t4(60s), t1-3(60s), t5(60s)
        for boundary in task_boundaries:
            ax1.axvline(x=boundary, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
            ax2.axvline(x=boundary, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # タスク名を追加
        task_positions = [30, 90, 150, 210, 270, 330]
        task_labels = ['t1-1', 't2', 't1-2', 't4', 't1-3', 't5']
        for pos, label in zip(task_positions, task_labels):
            ax1.text(pos, ax1.get_ylim()[1] * 0.95, label, ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # グラフを保存
        output_path = subject_path / f"{subject}_HR_CO_FFT.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.show()
        
        print(f"グラフを保存しました: {output_path}")
    
    def process_all_subjects(self):
        """全被験者を処理"""
        for subject in self.subjects:
            print(f"処理中: {subject}")
            
            # FFT処理と保存
            result = self.process_subject(subject)
            
            if result is not None:
                # グラフ作成
                self.plot_all_tasks(subject)
                print(f"{subject} 完了")
            else:
                print(f"{subject} スキップ")

# 実行
def main():
    processor = SpO2_FFT_Processor(base_path="C:/Users/Data signals_bp")
    
    # 全被験者処理
    processor.process_all_subjects()

if __name__ == "__main__":
    main()
