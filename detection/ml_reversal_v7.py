"""
ML Reversal V7 Detection - PyTorch LSTM
機器學習反轉檢測器 V7

使用 PyTorch LSTM 處理時序數據，自動學習反轉模式

改進：
- LSTM 自動學習時序模式
- 輸入：過去 20 根 K 線的序列
- 不需要手工計算變化率
- 使用原始技術指標

模型架構：
- LSTM (2 層，hidden_size=64)
- Dropout (0.3)
- 全連接層
- 二分類輸出

檢測過濾條件（與 V6 相同）：
通用條件：
- |sar_distance_pct| > 1.3%
- [t-1] -> [t] 價格變化 < 0.5%

Peak（做空）：
- peak 在 SAR 上方
- peak 在 bb_middle 上方
- entry 價格 < peak 價格

Valley（做多）：
- valley 在 SAR 下方
- valley 在 bb_middle 下方
- entry 價格 > valley 價格
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from .base import DetectionMethod
from utils.extremes import find_local_extremes


class ReversalLSTM(nn.Module):
    """反轉預測 LSTM 模型（與訓練時相同）"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.5):
        super(ReversalLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 層（單層不需要 dropout 參數）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全連接層
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Dropout
        x = self.dropout(last_output)

        # 全連接層
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)

        return output


class MLReversalV7Detector(DetectionMethod):
    """
    機器學習反轉檢測器 V7 - PyTorch LSTM

    使用 LSTM 模型預測極值點後的反轉入場點
    """

    def __init__(
        self,
        model_path: str = "models/reversal/reversal_model_v7_lstm.pth",
        prob_threshold: float = 0.5,
    ):
        """
        初始化檢測器

        Args:
            model_path: 模型檔案路徑
            prob_threshold: 預測概率閾值（0-1）
        """
        super().__init__("ML Reversal V7 Detection (LSTM)")
        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self.model = None
        self.feature_cols = None
        self.mean = None
        self.std = None
        self.seq_len = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"找不到模型檔案: {self.model_path}\n"
                f"請先執行訓練: python model_training/reversal/train_ml_reversal_v7.py"
            )

        # 設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")

        # 載入模型（PyTorch 2.6+ 需要 weights_only=False）
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        input_size = checkpoint["input_size"]
        self.feature_cols = checkpoint["feature_cols"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.seq_len = checkpoint["seq_len"]

        # 創建模型並載入權重
        self.model = ReversalLSTM(input_size=input_size)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ ML Reversal V7 (LSTM) 模型載入成功")
        print(f"   模型路徑: {self.model_path}")
        print(f"   特徵數: {len(self.feature_cols)}")
        print(f"   序列長度: {self.seq_len}")
        print(f"   概率閾值: {self.prob_threshold}")
        print(f"   設備: {self.device}")

    def _extract_sequence(self, df: pd.DataFrame, t_idx: int) -> np.ndarray:
        """
        提取 [t-seq_len+1, ..., t] 的序列

        Args:
            df: 包含所有指標的 DataFrame
            t_idx: [t] 極值點的索引

        Returns:
            序列數組 (seq_len, num_features) 或 None
        """
        # 確保有足夠的歷史數據
        if t_idx < self.seq_len:
            return None

        start_idx = t_idx - self.seq_len + 1
        end_idx = t_idx + 1

        try:
            sequence = df.iloc[start_idx:end_idx][self.feature_cols].values

            # 檢查是否有 NaN
            if np.isnan(sequence).any():
                return None

            # 標準化（去掉 mean/std 的多餘維度）
            mean = self.mean.squeeze()  # (1, 1, num_features) -> (num_features,)
            std = self.std.squeeze()  # (1, 1, num_features) -> (num_features,)
            sequence_norm = (sequence - mean) / std

            return sequence_norm

        except (KeyError, IndexError):
            return None

    def detect(self, df: pd.DataFrame) -> np.ndarray:
        """
        檢測反轉點

        檢測策略：
        1. 使用 find_local_extremes 找出極值點 [t]
        2. 對每個極值點，使用 [t+1] 作為入場點
        3. 使用 LSTM 模型預測是否為好的入場點

        Args:
            df: 包含 OHLCV 數據和技術指標的 DataFrame

        Returns:
            整數數組：
            - 0: 無信號
            - 1: 做空信號（peak 後的入場點）
            - 2: 做多信號（valley 後的入場點）
        """
        n = len(df)
        if n < self.seq_len + 20:
            return np.zeros(n, dtype=int)

        signals = np.zeros(n, dtype=int)

        # 1. 找出極值點
        prices = df["close"].values
        extremes, peaks, valleys = find_local_extremes(
            prices, window=20, prominence_factor=0.3
        )

        if len(peaks) == 0 and len(valleys) == 0:
            return signals

        # 統計
        candidates_count = 0
        peak_predictions = 0
        valley_predictions = 0
        peak_probas = []
        valley_probas = []

        # 2. 處理 peak（做空信號）
        for t_idx in sorted(peaks):
            entry_idx = t_idx + 1

            if entry_idx >= n:
                continue

            t_price = df.iloc[t_idx]["close"]
            entry_price = df.iloc[entry_idx]["close"]
            bb_middle = df.iloc[t_idx]["bb_middle"]
            sar_distance_pct = df.iloc[t_idx]["sar_distance_pct"]
            sar_value = df.iloc[t_idx]["sar"]

            # 計算價格變化
            if t_idx >= 1:
                t_minus_1_price = df.iloc[t_idx - 1]["close"]
                price_change_t = abs((t_price - t_minus_1_price) / t_minus_1_price)
            else:
                price_change_t = 0.0

            # 過濾條件（與 V6 相同）
            if abs(sar_distance_pct) <= 1.3:
                continue
            if t_price <= sar_value:
                continue
            if t_price <= bb_middle:
                continue
            if price_change_t >= 0.005:
                continue
            if entry_price >= t_price:
                continue

            candidates_count += 1

            # 提取序列
            sequence = self._extract_sequence(df, t_idx)
            if sequence is None:
                continue

            # 模型預測
            try:
                with torch.no_grad():
                    sequence_tensor = (
                        torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    )
                    output = self.model(sequence_tensor)
                    proba = output.item()

                peak_probas.append(proba)

                if proba >= self.prob_threshold:
                    signals[entry_idx] = 1  # 做空信號
                    peak_predictions += 1

            except Exception:
                continue

        # 3. 處理 valley（做多信號）
        for t_idx in sorted(valleys):
            entry_idx = t_idx + 1

            if entry_idx >= n:
                continue

            t_price = df.iloc[t_idx]["close"]
            entry_price = df.iloc[entry_idx]["close"]
            bb_middle = df.iloc[t_idx]["bb_middle"]
            sar_distance_pct = df.iloc[t_idx]["sar_distance_pct"]
            sar_value = df.iloc[t_idx]["sar"]

            if t_idx >= 1:
                t_minus_1_price = df.iloc[t_idx - 1]["close"]
                price_change_t = abs((t_price - t_minus_1_price) / t_minus_1_price)
            else:
                price_change_t = 0.0

            # 過濾條件
            if abs(sar_distance_pct) <= 1.3:
                continue
            if t_price >= sar_value:
                continue
            if t_price >= bb_middle:
                continue
            if price_change_t >= 0.005:
                continue
            if entry_price <= t_price:
                continue

            candidates_count += 1

            # 提取序列
            sequence = self._extract_sequence(df, t_idx)
            if sequence is None:
                continue

            # 模型預測
            try:
                with torch.no_grad():
                    sequence_tensor = (
                        torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    )
                    output = self.model(sequence_tensor)
                    proba = output.item()

                valley_probas.append(proba)

                if proba >= self.prob_threshold:
                    signals[entry_idx] = 2  # 做多信號
                    valley_predictions += 1

            except Exception:
                continue

        # Debug 信息
        if candidates_count > 0:
            print(
                f"   [ML V7 LSTM] 極值點: "
                f"{len(peaks)} 個 peak + {len(valleys)} 個 valley"
            )
            print(f"   [ML V7 LSTM] 候選入場點: {candidates_count} 個")
            print(
                f"   [ML V7 LSTM] 預測信號: "
                f"{peak_predictions} 個做空 + {valley_predictions} 個做多"
            )

            if len(peak_probas) > 0:
                print(
                    f"   [ML V7 LSTM] Peak 概率: "
                    f"min={min(peak_probas):.3f}, "
                    f"max={max(peak_probas):.3f}, "
                    f"avg={np.mean(peak_probas):.3f}, "
                    f"median={np.median(peak_probas):.3f}"
                )
                peak_pct = np.percentile(peak_probas, [25, 50, 75])
                print(f"   [ML V7 LSTM] Peak 概率分佈: {peak_pct}")
            if len(valley_probas) > 0:
                print(
                    f"   [ML V7 LSTM] Valley 概率: "
                    f"min={min(valley_probas):.3f}, "
                    f"max={max(valley_probas):.3f}, "
                    f"avg={np.mean(valley_probas):.3f}, "
                    f"median={np.median(valley_probas):.3f}"
                )
                valley_pct = np.percentile(valley_probas, [25, 50, 75])
                print(f"   [ML V7 LSTM] Valley 概率分佈: {valley_pct}")

        return signals

    def get_color(self):
        """返回檢測方法的顏色（用於可視化）"""
        return "darkblue"
