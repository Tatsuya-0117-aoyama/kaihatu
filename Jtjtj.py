# ================================
# 設定クラス（修正版）
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
        
        # 血行動態信号タイプ設定
        self.signal_type = "CO"  # "CO", "HbO", "HbR", "HbT" など
        self.signal_prefix = "CO_s2"  # ファイル名のプレフィックス
        
        # 被験者設定（bp001～bp032）
        self.subjects = [f"bp{i:03d}" for i in range(1, 33)]
        
        # タスク設定（6分割交差検証用）
        self.tasks = ["t1-1", "t2", "t1-2", "t4", "t1-3", "t5"]
        self.task_duration = 60  # 各タスクの長さ（秒）
        
        # モデルタイプ選択
        self.model_type = "3d"  # デフォルトは3Dモデル
        
        # 使用チャンネル設定（全モデル共通）
        self.use_channel = 'RGB'  
        
        # チャンネル数の自動計算
        channel_map = {
            'R': 1, 'G': 1, 'B': 1,
            'RG': 2, 'GB': 2, 'RB': 2,
            'RGB': 3
        }
        self.num_channels = channel_map.get(self.use_channel, 3)
        
        # データ形状設定（実際のデータ形状に合わせて修正）
        if self.model_type in ["3d", "2d"]:
            # CalibrationPhys準拠モデル用（実際のデータ形状）
            self.time_frames = 360  # 時間フレーム数
            self.height = 14  # 画像の高さ（修正）
            self.width = 16   # 画像の幅（修正）
            self.input_shape = (self.time_frames, self.height, self.width, self.num_channels)
        else:
            # 従来モデル用
            self.input_shape = (14, 16, self.num_channels)
        
        # ================================
        # データ拡張設定
        # ================================
        self.use_augmentation = True  # データ拡張を使用するか
        
        # データ拡張のパラメータ
        if self.use_augmentation:
            # ランダムクロップ
            self.crop_enabled = True
            self.crop_size_ratio = 0.9  # 元のサイズの90%にクロップ
            
            # 回転
            self.rotation_enabled = True
            self.rotation_range = 5  # ±5度の範囲で回転
            
            # 時間軸ストレッチング
            self.time_stretch_enabled = True
            self.time_stretch_range = (0.9, 1.1)  # 90%～110%の速度変化
            
            # 明度・コントラスト調整
            self.brightness_contrast_enabled = True
            self.brightness_range = 0.2  # ±20%の明度変化
            self.contrast_range = 0.2    # ±20%のコントラスト変化
            
            # 各拡張の適用確率
            self.aug_probability = 0.5  # 各拡張を50%の確率で適用
        else:
            self.crop_enabled = False
            self.rotation_enabled = False
            self.time_stretch_enabled = False
            self.brightness_contrast_enabled = False
        
        # 学習設定（モデルタイプに応じて自動調整）
        if self.model_type == "standard":
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        elif self.model_type == "deep":
            self.batch_size = 32
            self.epochs = 200
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 40
        elif self.model_type == "3d":
            # 3Dモデルはメモリを多く使うため小さめのバッチサイズ
            self.batch_size = 8
            self.epochs = 150
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 30
        elif self.model_type == "2d":
            # 2Dモデルは効率的なので大きめのバッチサイズ可能
            self.batch_size = 16
            self.epochs = 150
            self.learning_rate = 0.001
            self.weight_decay = 1e-4
            self.patience = 30
        else:
            # デフォルト設定
            self.batch_size = 16
            self.epochs = 100
            self.learning_rate = 0.001
            self.weight_decay = 1e-5
            self.patience = 20
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 損失関数設定
        self.loss_type = "combined"  # "mse", "combined", "huber_combined"
        self.loss_alpha = 0.7  # MSE/Huber損失の重み
        self.loss_beta = 0.3   # 相関損失の重み
        
        # 学習率スケジューラー設定
        self.scheduler_type = "cosine"  # "cosine", "onecycle", "plateau"
        self.scheduler_T0 = 30 if self.model_type != "standard" else 20
        self.scheduler_T_mult = 2 if self.model_type != "standard" else 1
        
        # 表示設定
        self.verbose = True
        self.random_seed = 42

# ================================
# CalibrationPhys準拠 PhysNet2DCNN (3D版) - 修正版
# ================================
class PhysNet2DCNN_3D(nn.Module):
    """
    CalibrationPhys論文準拠のPhysNet2DCNN（3D畳み込み版）
    入力: (batch_size, time_frames, height, width, channels)
    出力: (batch_size, time_frames) の脈波/呼吸波形
    
    任意の入力サイズ（14x16など）に対応
    """
    def __init__(self, input_shape=None):
        super(PhysNet2DCNN_3D, self).__init__()
        
        # 入力チャンネル数を動的に設定
        if input_shape is not None:
            self.time_frames = input_shape[0]
            self.height = input_shape[1] 
            self.width = input_shape[2]
            in_channels = input_shape[-1]
        else:
            self.time_frames = 360
            self.height = 14  # 修正
            self.width = 16   # 修正
            in_channels = 3
        
        # ConvBlock 1: 32 filters
        self.conv1_1 = nn.Conv3d(in_channels, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1_1 = nn.BatchNorm3d(32)
        self.elu1_1 = nn.ELU(inplace=True)
        
        self.conv1_2 = nn.Conv3d(32, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.bn1_2 = nn.BatchNorm3d(32)
        self.elu1_2 = nn.ELU(inplace=True)
        
        # プーリングサイズを動的に計算（小さな画像に対応）
        pool1_h = min(2, self.height // 2) if self.height > 2 else 1
        pool1_w = min(2, self.width // 2) if self.width > 2 else 1
        self.pool1 = nn.AvgPool3d(kernel_size=(1, pool1_h, pool1_w), stride=(1, pool1_h, pool1_w))
        
        # 計算後のサイズ
        self.h_after_pool1 = self.height // pool1_h
        self.w_after_pool1 = self.width // pool1_w
        
        # ConvBlock 2: 64 filters
        self.conv2_1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_1 = nn.BatchNorm3d(64)
        self.elu2_1 = nn.ELU(inplace=True)
        
        self.conv2_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2_2 = nn.BatchNorm3d(64)
        self.elu2_2 = nn.ELU(inplace=True)
        
        # プーリングサイズを動的に計算
        pool2_h = min(2, self.h_after_pool1 // 2) if self.h_after_pool1 > 2 else 1
        pool2_w = min(2, self.w_after_pool1 // 2) if self.w_after_pool1 > 2 else 1
        self.pool2 = nn.AvgPool3d(kernel_size=(2, pool2_h, pool2_w), stride=(2, pool2_h, pool2_w))
        
        # 計算後のサイズ
        self.h_after_pool2 = self.h_after_pool1 // pool2_h
        self.w_after_pool2 = self.w_after_pool1 // pool2_w
        
        # ConvBlock 3: 64 filters
        self.conv3_1 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3_1 = nn.BatchNorm3d(64)
        self.elu3_1 = nn.ELU(inplace=True)
        
        self.conv3_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3_2 = nn.BatchNorm3d(64)
        self.elu3_2 = nn.ELU(inplace=True)
        
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # ConvBlock 4: 64 filters
        self.conv4_1 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4_1 = nn.BatchNorm3d(64)
        self.elu4_1 = nn.ELU(inplace=True)
        
        self.conv4_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4_2 = nn.BatchNorm3d(64)
        self.elu4_2 = nn.ELU(inplace=True)
        
        self.pool4 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        # ConvBlock 5: 64 filters with upsampling
        self.conv5_1 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5_1 = nn.BatchNorm3d(64)
        self.elu5_1 = nn.ELU(inplace=True)
        
        self.conv5_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5_2 = nn.BatchNorm3d(64)
        self.elu5_2 = nn.ELU(inplace=True)
        
        # Upsample
        self.upsample = nn.Upsample(scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)
        
        # ConvBlock 6: 64 filters
        self.conv6_1 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn6_1 = nn.BatchNorm3d(64)
        self.elu6_1 = nn.ELU(inplace=True)
        
        self.conv6_2 = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn6_2 = nn.BatchNorm3d(64)
        self.elu6_2 = nn.ELU(inplace=True)
        
        # Spatial Global Average Pooling
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # Final Conv
        self.conv_final = nn.Conv3d(64, 1, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        入力: x shape (B, T, H, W, C)
        出力: shape (B, T_out) 脈波/呼吸波形
        """
        batch_size = x.size(0)
        time_frames = x.size(1)
        
        # PyTorchのConv3dは (B, C, D, H, W)を期待
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        # ConvBlock 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # ConvBlock 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # ConvBlock 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # ConvBlock 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.elu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.elu4_2(x)
        x = self.pool4(x)
        x = self.dropout(x)
        
        # ConvBlock 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.elu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.elu5_2(x)
        x = self.upsample(x)
        x = self.dropout(x)
        
        # ConvBlock 6
        x = self.conv6_1(x)
        x = self.bn6_1(x)
        x = self.elu6_1(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.elu6_2(x)
        
        # Spatial Global Average Pooling
        x = self.spatial_pool(x)
        
        # Final Conv
        x = self.conv_final(x)
        
        # 出力を整形
        x = x.squeeze(1).squeeze(-1).squeeze(-1)  # (B, T_reduced)
        
        # 元の時間長に補間（必要に応じて）
        if x.size(1) != time_frames:
            x = F.interpolate(x.unsqueeze(1), size=time_frames, mode='linear', align_corners=False)
            x = x.squeeze(1)
        
        return x

# ================================
# CalibrationPhys準拠 PhysNet2DCNN (2D版 - 効率的) - 修正版
# ================================
class PhysNet2DCNN_2D(nn.Module):
    """
    2D畳み込みを使用したPhysNet2DCNN（効率的な実装）
    入力: (batch_size, time_frames, height, width, channels)
    
    任意の入力サイズ（14x16など）に対応
    """
    def __init__(self, input_shape=None):
        super(PhysNet2DCNN_2D, self).__init__()
        
        # 入力チャンネル数と画像サイズを動的に設定
        if input_shape is not None:
            self.time_frames = input_shape[0]
            self.height = input_shape[1]
            self.width = input_shape[2]
            in_channels = input_shape[-1]
        else:
            self.time_frames = 360
            self.height = 14  # 修正
            self.width = 16   # 修正
            in_channels = 3
        
        # ConvBlock 1: 32 filters
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.elu1_1 = nn.ELU(inplace=True)
        
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.elu1_2 = nn.ELU(inplace=True)
        
        # プーリングサイズを動的に計算
        pool1_h = min(2, self.height // 2) if self.height > 2 else 1
        pool1_w = min(2, self.width // 2) if self.width > 2 else 1
        self.pool1 = nn.AvgPool2d(kernel_size=(pool1_h, pool1_w), stride=(pool1_h, pool1_w))
        
        # 計算後のサイズ
        self.h_after_pool1 = self.height // pool1_h
        self.w_after_pool1 = self.width // pool1_w
        
        # ConvBlock 2: 64 filters
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.elu2_1 = nn.ELU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.elu2_2 = nn.ELU(inplace=True)
        
        # プーリングサイズを動的に計算
        pool2_h = min(2, self.h_after_pool1 // 2) if self.h_after_pool1 > 2 else 1
        pool2_w = min(2, self.w_after_pool1 // 2) if self.w_after_pool1 > 2 else 1
        self.pool2 = nn.AvgPool2d(kernel_size=(pool2_h, pool2_w), stride=(pool2_h, pool2_w))
        
        # 計算後のサイズ
        self.h_after_pool2 = self.h_after_pool1 // pool2_h
        self.w_after_pool2 = self.w_after_pool1 // pool2_w
        
        # ConvBlock 3: 64 filters
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm3d(64)
        self.elu3_1 = nn.ELU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.elu3_2 = nn.ELU(inplace=True)
        
        # 小さな画像サイズの場合はプーリングを調整
        if self.h_after_pool2 >= 3 and self.w_after_pool2 >= 3:
            pool3_h = min(2, self.h_after_pool2 // 2)
            pool3_w = min(2, self.w_after_pool2 // 2)
            self.pool3 = nn.AvgPool2d(kernel_size=(pool3_h, pool3_w), stride=(pool3_h, pool3_w))
            self.h_after_pool3 = self.h_after_pool2 // pool3_h
            self.w_after_pool3 = self.w_after_pool2 // pool3_w
        else:
            self.pool3 = None  # プーリングをスキップ
            self.h_after_pool3 = self.h_after_pool2
            self.w_after_pool3 = self.w_after_pool2
        
        # ConvBlock 4: 64 filters
        self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.elu4_1 = nn.ELU(inplace=True)
        
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(64)
        self.elu4_2 = nn.ELU(inplace=True)
        
        # さらに小さくなる場合はプーリングをスキップ
        if self.h_after_pool3 >= 3 and self.w_after_pool3 >= 3:
            pool4_h = min(2, self.h_after_pool3 // 2)
            pool4_w = min(2, self.w_after_pool3 // 2)
            self.pool4 = nn.AvgPool2d(kernel_size=(pool4_h, pool4_w), stride=(pool4_h, pool4_w))
        else:
            self.pool4 = None
        
        # ConvBlock 5: 64 filters
        self.conv5_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.elu5_1 = nn.ELU(inplace=True)
        
        self.conv5_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(64)
        self.elu5_2 = nn.ELU(inplace=True)
        
        # Spatial Global Average Pooling（どんなサイズでも1x1に）
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal processing
        self.temporal_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.temporal_bn1 = nn.BatchNorm1d(64)
        self.temporal_elu1 = nn.ELU(inplace=True)
        
        self.temporal_conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.temporal_bn2 = nn.BatchNorm1d(32)
        self.temporal_elu2 = nn.ELU(inplace=True)
        
        # Final layer
        self.fc = nn.Linear(32, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        入力: x shape (B, T, H, W, C)
        出力: shape (B, T) 脈波/呼吸波形
        """
        batch_size, time_frames = x.size(0), x.size(1)
        
        # 時間次元をバッチに結合
        x = x.view(batch_size * time_frames, x.size(2), x.size(3), x.size(4))
        # (B*T, H, W, C) -> (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # ConvBlock 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.elu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.elu1_2(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # ConvBlock 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.elu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.elu2_2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # ConvBlock 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.elu3_1(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.elu3_2(x)
        if self.pool3 is not None:
            x = self.pool3(x)
        x = self.dropout(x)
        
        # ConvBlock 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.elu4_1(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.elu4_2(x)
        if self.pool4 is not None:
            x = self.pool4(x)
        x = self.dropout(x)
        
        # ConvBlock 5
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.elu5_1(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.elu5_2(x)
        
        # Spatial pooling（どんなサイズでも1x1に収束）
        x = self.spatial_pool(x)
        
        # 時間次元を復元
        x = x.view(batch_size, time_frames, 64)
        
        # 時間次元の処理
        x = x.permute(0, 2, 1)  # (B, 64, T)
        x = self.temporal_conv1(x)
        x = self.temporal_bn1(x)
        x = self.temporal_elu1(x)
        
        x = self.temporal_conv2(x)
        x = self.temporal_bn2(x)
        x = self.temporal_elu2(x)
        
        x = x.permute(0, 2, 1)  # (B, T, 32)
        
        # 各時間ステップで予測
        x = self.fc(x)
        x = x.squeeze(-1)
        
        return x

# ================================
# データ読み込み（修正版）
# ================================
def load_data_single_subject(subject, config):
    """単一被験者のデータを読み込み - (360, 14, 16, 3)形状に対応"""
    
    if config.model_type in ['3d', '2d']:
        # CalibrationPhys準拠モデル用（360フレームデータ）
        rgb_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_video_data.npy")  # (360, 14, 16, 3)形状のデータ
        if not os.path.exists(rgb_path):
            print(f"警告: {subject}のRGBデータが見つかりません")
            return None, None
        
        rgb_data = np.load(rgb_path)  # Expected shape: (360, 14, 16, 3)
        
        # データ形状の確認と調整
        if rgb_data.ndim == 4:  # (T, H, W, C)
            # 各タスクごとに分割された形状に変換
            # 360フレーム = 6タスク × 60フレーム/タスク
            rgb_data = rgb_data.reshape(6, 60, config.height, config.width, config.num_channels)
            rgb_data = rgb_data.reshape(-1, config.height, config.width, config.num_channels)  # (360, H, W, C)
        elif rgb_data.ndim == 5:  # (N, T, H, W, C) - 既に適切な形状
            pass
        elif rgb_data.ndim == 3:  # (T, H, W) - チャンネル次元がない場合
            rgb_data = np.expand_dims(rgb_data, axis=-1)
            rgb_data = np.repeat(rgb_data, config.num_channels, axis=-1)
        else:
            raise ValueError(f"予期しないRGBデータ形状: {rgb_data.shape}")
            
    else:
        # 従来モデル用
        rgb_path = os.path.join(config.rgb_base_path, subject, 
                                f"{subject}_downsampled_1Hz.npy")
        if not os.path.exists(rgb_path):
            print(f"警告: {subject}のRGBデータが見つかりません")
            return None, None
        
        rgb_data = np.load(rgb_path)
    
    # 信号データの読み込み
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
    
    # データの正規化（0-1の範囲に）
    if rgb_data.max() > 1.0:
        rgb_data = rgb_data / 255.0
    
    # 形状チェック
    if config.verbose:
        print(f"  データ読み込み完了: RGB={rgb_data.shape}, Signal={signal_data.shape}")
        print(f"  想定入力形状: {config.input_shape}")
    
    return rgb_data, signal_data

# ================================
# データセット（修正版）
# ================================
class CODataset(Dataset):
    def __init__(self, rgb_data, signal_data, model_type='standard', 
                 use_channel='RGB', config=None, is_training=True):
        """
        rgb_data: 
            - standard/deep: (N, H, W, C) 形状
            - 3d/2d: (N, T, H, W, C) または (N, H, W, C) 形状
        signal_data: 
            - standard/deep: (N,) 形状
            - 3d/2d: (N, T) または (N,) 形状
        config: Config オブジェクト（データ拡張用）
        is_training: 学習時True、検証/テスト時False
        """
        self.model_type = model_type
        self.use_channel = use_channel
        self.is_training = is_training
        self.config = config
        
        # データ拡張の初期化
        self.augmentation = DataAugmentation(config) if config else None
        
        # データを保存（拡張前）
        self.rgb_data_raw = rgb_data
        self.signal_data_raw = signal_data
        
        if model_type in ['3d', '2d']:
            # 3D/2Dモデル用の処理
            if rgb_data.ndim == 3:  # (N, H, W, C) -> 時間次元を追加
                # 1フレームのデータを時間方向に複製（必要に応じて）
                # ここでは各データポイントが1つの時間ステップを表すと仮定
                rgb_data = rgb_data[:, np.newaxis, :, :, :]  # (N, 1, H, W, C)
                # 必要に応じて時間方向に拡張
                if config and hasattr(config, 'time_frames'):
                    rgb_data = np.repeat(rgb_data, min(config.time_frames, 1), axis=1)
            
            # チャンネル選択を適用
            rgb_data_selected = select_channels(rgb_data, use_channel)
            self.rgb_data = torch.FloatTensor(rgb_data_selected)
            
            # signal_dataが1次元の場合、時間次元に拡張
            if signal_data.ndim == 1:
                if rgb_data.shape[1] == 1:
                    # 各データポイントがスカラー値の場合
                    signal_data = signal_data[:, np.newaxis]  # (N, 1)
                else:
                    # 時間次元に拡張
                    signal_data = np.repeat(signal_data[:, np.newaxis], rgb_data.shape[1], axis=1)
            
            self.signal_data = torch.FloatTensor(signal_data)
        else:
            # 従来モデル用
            # チャンネル選択を適用
            rgb_data_selected = select_channels(rgb_data, use_channel)
            self.rgb_data = torch.FloatTensor(rgb_data_selected).permute(0, 3, 1, 2)
            self.signal_data = torch.FloatTensor(signal_data)
    
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        # 元データを取得
        rgb = self.rgb_data_raw[idx]
        
        # 信号データの処理
        if self.signal_data_raw.ndim > 1:
            signal = self.signal_data_raw[idx]
        else:
            signal = self.signal_data_raw[idx:idx+1].squeeze()
        
        # 3D/2Dモデルの場合の形状調整
        if self.model_type in ['3d', '2d']:
            if rgb.ndim == 3:  # (H, W, C) -> (1, H, W, C)
                rgb = rgb[np.newaxis, :, :, :]
            
            # データ拡張を適用（学習時のみ）
            if self.augmentation and self.is_training:
                rgb, signal = self.augmentation.apply_augmentation(rgb, signal, self.is_training)
            
            # チャンネル選択
            rgb = select_channels(rgb, self.use_channel)
            
            # Tensorに変換
            rgb_tensor = torch.FloatTensor(rgb)
            
            # 信号データの処理
            if isinstance(signal, (int, float)):
                signal_tensor = torch.FloatTensor([signal])
            elif isinstance(signal, np.ndarray):
                if signal.ndim == 0:
                    signal_tensor = torch.FloatTensor([signal.item()])
                else:
                    signal_tensor = torch.FloatTensor(signal)
            else:
                signal_tensor = torch.FloatTensor([signal])
            
            # 時間次元の調整
            if signal_tensor.dim() == 1 and signal_tensor.size(0) == 1 and rgb_tensor.size(0) > 1:
                signal_tensor = signal_tensor.repeat(rgb_tensor.size(0))
            elif signal_tensor.dim() == 0:
                signal_tensor = signal_tensor.unsqueeze(0)
                if rgb_tensor.size(0) > 1:
                    signal_tensor = signal_tensor.repeat(rgb_tensor.size(0))
        else:
            # 従来モデル用
            # データ拡張を適用（学習時のみ）
            if self.augmentation and self.is_training:
                rgb, signal = self.augmentation.apply_augmentation(rgb, signal, self.is_training)
            
            # チャンネル選択
            rgb = select_channels(rgb, self.use_channel)
            
            rgb_tensor = torch.FloatTensor(rgb).permute(2, 0, 1)
            signal_tensor = torch.FloatTensor([signal] if not isinstance(signal, np.ndarray) else signal)
        
        return rgb_tensor, signal_tensor

# ================================
# データ拡張クラス（修正版）
# ================================
class DataAugmentation:
    """データ拡張を管理するクラス - 小さな画像サイズに対応"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)
    
    def random_crop(self, data):
        """
        ランダムクロップ - 小さな画像サイズに対応
        data: (T, H, W, C) or (H, W, C)
        """
        if np.random.random() > self.config.aug_probability or not self.config.crop_enabled:
            return data
        
        if data.ndim == 4:  # (T, H, W, C)
            t, h, w, c = data.shape
            
            # 小さな画像の場合、クロップサイズを調整
            min_crop_size = max(2, min(h, w) // 2)  # 最小2ピクセル
            crop_ratio = max(0.8, min_crop_size / min(h, w))  # 最小80%
            
            new_h = max(min_crop_size, int(h * crop_ratio))
            new_w = max(min_crop_size, int(w * crop_ratio))
            
            # ランダムな開始位置
            top = np.random.randint(0, max(1, h - new_h + 1))
            left = np.random.randint(0, max(1, w - new_w + 1))
            
            # クロップ
            cropped = data[:, top:top+new_h, left:left+new_w, :]
            
            # 元のサイズにリサイズ
            resized = np.zeros_like(data)
            for i in range(t):
                for j in range(c):
                    resized[i, :, :, j] = cv2.resize(cropped[i, :, :, j], (w, h))
            
            return resized
        
        elif data.ndim == 3:  # (H, W, C)
            h, w, c = data.shape
            
            # 小さな画像の場合、クロップサイズを調整
            min_crop_size = max(2, min(h, w) // 2)
            crop_ratio = max(0.8, min_crop_size / min(h, w))
            
            new_h = max(min_crop_size, int(h * crop_ratio))
            new_w = max(min_crop_size, int(w * crop_ratio))
            
            top = np.random.randint(0, max(1, h - new_h + 1))
            left = np.random.randint(0, max(1, w - new_w + 1))
            
            cropped = data[top:top+new_h, left:left+new_w, :]
            resized = cv2.resize(cropped, (w, h))
            
            return resized
        
        return data
    
    def random_rotation(self, data):
        """
        ランダム回転 - 小さな画像サイズに対応
        data: (T, H, W, C) or (H, W, C)
        """
        if np.random.random() > self.config.aug_probability or not self.config.rotation_enabled:
            return data
        
        # 小さな画像の場合、回転角度を制限
        max_angle = min(self.config.rotation_range, 10)  # 最大10度
        angle = np.random.uniform(-max_angle, max_angle)
        
        if data.ndim == 4:  # (T, H, W, C)
            rotated = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[3]):
                    rotated[i, :, :, j] = scipy_rotate(data[i, :, :, j], angle, 
                                                       reshape=False, mode='reflect')
            return rotated
        
        elif data.ndim == 3:  # (H, W, C)
            rotated = np.zeros_like(data)
            for j in range(data.shape[2]):
                rotated[:, :, j] = scipy_rotate(data[:, :, j], angle, 
                                               reshape=False, mode='reflect')
            return rotated
        
        return data
    
    def time_stretch(self, rgb_data, signal_data=None):
        """
        時間軸ストレッチング
        rgb_data: (T, H, W, C)
        signal_data: (T,) or None
        """
        if np.random.random() > self.config.aug_probability or not self.config.time_stretch_enabled:
            return rgb_data, signal_data
        
        if rgb_data.ndim != 4:  # 時間次元がない場合はスキップ
            return rgb_data, signal_data
        
        stretch_factor = np.random.uniform(*self.config.time_stretch_range)
        t_original = rgb_data.shape[0]
        t_stretched = int(t_original * stretch_factor)
        
        # RGBデータのストレッチング
        rgb_stretched = np.zeros((t_stretched, *rgb_data.shape[1:]))
        
        for h in range(rgb_data.shape[1]):
            for w in range(rgb_data.shape[2]):
                for c in range(rgb_data.shape[3]):
                    # 補間用の関数を作成
                    f = interp1d(np.arange(t_original), rgb_data[:, h, w, c], 
                               kind='linear', fill_value='extrapolate')
                    # 新しい時間軸で補間
                    new_t = np.linspace(0, t_original-1, t_stretched)
                    rgb_stretched[:, h, w, c] = f(new_t)
        
        # 元の長さにリサンプリング
        rgb_resampled = np.zeros_like(rgb_data)
        for h in range(rgb_data.shape[1]):
            for w in range(rgb_data.shape[2]):
                for c in range(rgb_data.shape[3]):
                    f = interp1d(np.arange(t_stretched), rgb_stretched[:, h, w, c], 
                               kind='linear', fill_value='extrapolate')
                    rgb_resampled[:, h, w, c] = f(np.linspace(0, t_stretched-1, t_original))
        
        # 信号データのストレッチング（提供されている場合）
        if signal_data is not None and signal_data.ndim == 1:
            # 信号も同様にストレッチング
            f_signal = interp1d(np.arange(len(signal_data)), signal_data, 
                              kind='linear', fill_value='extrapolate')
            signal_stretched = f_signal(np.linspace(0, len(signal_data)-1, 
                                       int(len(signal_data) * stretch_factor)))
            # 元の長さに戻す
            f_signal_back = interp1d(np.arange(len(signal_stretched)), signal_stretched,
                                    kind='linear', fill_value='extrapolate')
            signal_resampled = f_signal_back(np.linspace(0, len(signal_stretched)-1, 
                                            len(signal_data)))
            
            # 周波数を調整（ストレッチファクターに応じて）
            signal_resampled = signal_resampled * stretch_factor
            
            return rgb_resampled, signal_resampled
        
        return rgb_resampled, signal_data
    
    def brightness_contrast_adjust(self, data):
        """
        明度・コントラスト調整
        data: (T, H, W, C) or (H, W, C)
        """
        if np.random.random() > self.config.aug_probability or not self.config.brightness_contrast_enabled:
            return data
        
        # 明度とコントラストの変化量
        brightness_delta = np.random.uniform(-self.config.brightness_range, 
                                            self.config.brightness_range)
        contrast_factor = np.random.uniform(1 - self.config.contrast_range, 
                                           1 + self.config.contrast_range)
        
        # データの正規化（0-1の範囲と仮定）
        data_adjusted = data.copy()
        
        # コントラスト調整
        mean = np.mean(data_adjusted, axis=tuple(range(data_adjusted.ndim-1)), keepdims=True)
        data_adjusted = (data_adjusted - mean) * contrast_factor + mean
        
        # 明度調整
        data_adjusted = data_adjusted + brightness_delta
        
        # クリッピング
        data_adjusted = np.clip(data_adjusted, 0, 1)
        
        return data_adjusted
    
    def apply_augmentation(self, rgb_data, signal_data=None, is_training=True):
        """
        すべてのデータ拡張を適用
        rgb_data: (T, H, W, C) or (H, W, C)
        signal_data: (T,) or scalar or None
        is_training: 学習時のみTrueでデータ拡張を適用
        """
        if not is_training or not self.config.use_augmentation:
            return rgb_data, signal_data
        
        # 各拡張を順次適用
        rgb_data = self.random_crop(rgb_data)
        rgb_data = self.random_rotation(rgb_data)
        
        # 時間軸ストレッチング（時間次元がある場合のみ）
        if rgb_data.ndim == 4 and self.config.time_stretch_enabled:
            rgb_data, signal_data = self.time_stretch(rgb_data, signal_data)
        
        rgb_data = self.brightness_contrast_adjust(rgb_data)
        
        return rgb_data, signal_data

# ================================
# 6分割交差検証（修正版）
# ================================
def task_cross_validation(rgb_data, signal_data, config, subject, subject_save_dir):
    """タスクごとの6分割交差検証 - (360, 14, 16, 3)形状対応"""
    
    fold_results = []
    all_test_predictions = []
    all_test_targets = []
    all_test_task_indices = []
    
    for fold, test_task in enumerate(config.tasks):
        if config.verbose:
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
        
        # データローダー作成（データ拡張対応）
        train_dataset = CODataset(train_rgb, train_signal, config.model_type, 
                                 config.use_channel, config, is_training=True)
        val_dataset = CODataset(val_rgb, val_signal, config.model_type, 
                               config.use_channel, config, is_training=False)
        test_dataset = CODataset(test_rgb, test_signal, config.model_type, 
                                config.use_channel, config, is_training=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # モデル作成
        model = create_model(config)
        
        # モデル学習
        model, train_preds, train_targets = train_model(
            model, train_loader, val_loader, config, fold, subject
        )
        
        # 評価
        test_results = evaluate_model(model, test_loader, config)
        
        if config.verbose:
            print(f"    Train: MAE={mean_absolute_error(train_targets, train_preds):.4f}, Corr={np.corrcoef(train_targets, train_preds)[0, 1]:.4f}")
            print(f"    Test:  MAE={test_results['mae']:.4f}, Corr={test_results['corr']:.4f}")
        
        # 結果保存
        fold_results.append({
            'fold': fold + 1,
            'test_task': test_task,
            'train_predictions': train_preds,
            'train_targets': train_targets,
            'test_predictions': test_results['predictions'],
            'test_targets': test_results['targets'],
            'train_mae': mean_absolute_error(train_targets, train_preds),
            'train_corr': np.corrcoef(train_targets, train_preds)[0, 1],
            'test_mae': test_results['mae'],
            'test_corr': test_results['corr']
        })
        
        # 全体のテストデータ集約
        all_test_predictions.extend(test_results['predictions'])
        all_test_targets.extend(test_results['targets'])
        
        # 各Foldのプロット
        plot_fold_results(fold_results[-1], subject_save_dir)
    
    # テスト予測を元の順序に並び替え
    sorted_indices = np.argsort(all_test_task_indices)
    all_test_predictions = np.array(all_test_predictions)[sorted_indices]
    all_test_targets = np.array(all_test_targets)[sorted_indices]
    
    return fold_results, all_test_predictions, all_test_targets

# ================================
# テスト関数（新規追加）
# ================================
def test_model_with_sample_data(config):
    """サンプルデータでモデルの動作確認"""
    print(f"\n{'='*60}")
    print("モデル動作テスト")
    print(f"{'='*60}")
    
    # サンプルデータ作成 (360, 14, 16, 3)
    sample_rgb = np.random.rand(360, 14, 16, 3).astype(np.float32)
    sample_signal = np.random.rand(360).astype(np.float32)
    
    print(f"サンプルデータ形状: RGB={sample_rgb.shape}, Signal={sample_signal.shape}")
    
    # データセット作成
    dataset = CODataset(sample_rgb, sample_signal, config.model_type, 
                       config.use_channel, config, is_training=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # モデル作成
    model = create_model(config)
    model.eval()
    
    # テスト実行
    with torch.no_grad():
        for i, (rgb, sig) in enumerate(dataloader):
            print(f"バッチ {i+1}: RGB input shape = {rgb.shape}, Signal shape = {sig.shape}")
            
            try:
                output = model(rgb)
                print(f"           Output shape = {output.shape}")
                print(f"           Output range = [{output.min().item():.4f}, {output.max().item():.4f}]")
                
                if i == 0:  # 最初のバッチのみ詳細表示
                    print(f"           Model type: {config.model_type}")
                    print(f"           Channel: {config.use_channel}")
                    print(f"           Parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                break  # 1バッチのみテスト
                
            except Exception as e:
                print(f"           エラー: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    print("モデル動作テスト完了 - 正常に動作しています")
    return True
