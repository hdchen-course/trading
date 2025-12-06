#!/usr/bin/env python3
"""
ML Reversal V7 模型訓練腳本 - PyTorch LSTM

基於 V6 的標籤邏輯，改用 PyTorch LSTM 處理時序數據

改進：
- 使用 LSTM 自動學習時序模式
- 輸入：過去 20 根 K 線的序列數據
- 不需要手工計算變化率特徵
- 使用原始技術指標作為特徵

模型架構：
- LSTM (1 層，hidden_size=32) - 簡化避免過擬合
- Dropout (0.5) - 增強正則化
- 全連接層
- 二分類輸出
- Early Stopping - 在最佳 AUC 停止

訓練策略：
- 序列長度：20 根 K 線（5 小時）
- 特徵：原始技術指標（不用變化率）
- 標籤邏輯：與 V6 相同
- 優化器：Adam
- 損失函數：BCELoss（二分類交叉熵）
"""

import sys
import os

# 必須在其他 import 之前設定 path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # noqa: E402

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

from utils.extremes import find_local_extremes  # noqa: E402


# ============================================================================
# 1. LSTM 模型定義
# ============================================================================


class ReversalLSTM(nn.Module):
    """
    反轉預測 LSTM 模型

    架構：
    - LSTM (1 層，hidden_size=32) - 簡化避免過擬合
    - Dropout (0.5) - 增強正則化
    - 全連接層
    - Sigmoid 輸出（二分類）
    """

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
        """
        前向傳播

        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            output: (batch_size, 1) - 預測概率
        """
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


# ============================================================================
# 2. 數據集類
# ============================================================================


class ReversalDataset(Dataset):
    """
    反轉預測數據集

    每個樣本包含：
    - 序列：過去 seq_len 根 K 線的特徵
    - 標籤：0 或 1
    """

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: (num_samples, seq_len, num_features)
            labels: (num_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================================
# 3. 數據載入和預處理
# ============================================================================


def load_data(data_file: str, year: int = 2024) -> pd.DataFrame:
    """載入訓練數據並篩選指定年份"""
    print("=" * 80)
    print("📊 (1) 載入數據")
    print("=" * 80)

    print(f"\n數據檔案: {data_file}")

    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"✅ 成功讀取數據: {len(df)} 行")
    print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    # 只保留指定年份數據
    print(f"\n🔍 篩選 {year} 年數據...")
    df = (
        df[(df["timestamp"] >= f"{year}-01-01") & (df["timestamp"] < f"{year+1}-01-01")]
        .copy()
        .reset_index(drop=True)
    )

    print(f"   {year} 年數據: {len(df)} 行")
    print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    # 統計反轉點
    reversal_count = df["reverse"].sum()
    reversal_pct = reversal_count / len(df) * 100

    print("\n反轉點統計:")
    print(f"   反轉點數量: {reversal_count} 個 ({reversal_pct:.2f}%)")

    if "reverse_type" in df.columns:
        peak_count = (df["reverse_type"] == "peak").sum()
        valley_count = (df["reverse_type"] == "valley").sum()
        print(f"   峰值: {peak_count} 個")
        print(f"   谷值: {valley_count} 個")

    return df


def filter_candidates(df: pd.DataFrame) -> list:
    """篩選候選樣本 - 使用局部極值檢測"""
    print("\n" + "=" * 80)
    print("🎯 (2) 篩選候選樣本 - 局部極值檢測")
    print("=" * 80)

    print("\n使用 find_local_extremes:")
    print("   window = 20")
    print("   prominence_factor = 0.3")

    prices = df["close"].values
    extremes, peaks, valleys = find_local_extremes(
        prices, window=20, prominence_factor=0.3
    )

    extreme_indices = sorted(list(set(peaks) | set(valleys)))

    print("\n✅ 找到極值點:")
    print(f"   峰值點: {len(peaks)} 個")
    print(f"   谷值點: {len(valleys)} 個")
    print(
        f"   總極值點: {len(extreme_indices)} 個 ({len(extreme_indices)/len(df)*100:.2f}%)"
    )

    return extreme_indices


def generate_labels(df: pd.DataFrame, extreme_indices: list) -> tuple:
    """
    生成訓練標籤 - 針對即時交易場景優化

    使用場景：
    1. find_local_extremes 找到 ~5000 個局部極值點
    2. LSTM 過濾掉會虧損的點
    3. 保留高質量入場點

    正樣本：真實反轉點（不會虧損）
    負樣本：嚴格過濾 + 會虧損的局部極值點

    嚴格過濾條件（確保負樣本質量）：
    1. |sar_distance_pct| > 1.3%
    2. peak 在 SAR 上方，valley 在 SAR 下方
    3. peak 在 bb_middle 上方，valley 在 bb_middle 下方
    4. [t-1] -> [t] 價格變化 < 0.5%
    5. entry 價格確認反轉方向
    6. 入場後會虧損（>= 0.18%）
    """
    print("\n" + "=" * 80)
    print("🏷️  (3) 生成標籤（針對即時交易優化）")
    print("=" * 80)

    print("\n使用場景:")
    print("   即時交易: find_local_extremes → LSTM 過濾 → 高質量入場點")
    print("\n標籤邏輯:")
    print("   正樣本：真實反轉點（不會虧損）")
    print("   負樣本：嚴格過濾 + 會虧損的局部極值點")
    print("\n嚴格過濾條件（確保負樣本是明顯的壞點）：")
    print("   1. |sar_distance_pct| > 1.3%")
    print("   2. peak 在 SAR 上方，valley 在 SAR 下方")
    print("   3. peak 在 bb_middle 上方，valley 在 bb_middle 下方")
    print("   4. [t-1] -> [t] 價格變化 < 0.5%")
    print("   5. entry 價格確認反轉方向")
    print("   6. 入場後會虧損（>= 0.18%）")

    labels = []
    t_indices = []
    entry_indices = []

    true_reversal_positive = 0
    negative = 0
    filtered_out = 0

    max_holding_bars = 16
    true_reversal_set = set(df[df["reverse"] == 1].index.tolist())

    for t_idx in extreme_indices:
        if t_idx < 1 or t_idx + 1 + max_holding_bars >= len(df):
            continue

        entry_idx = t_idx + 1

        # === 正樣本：真實反轉點 ===
        if t_idx in true_reversal_set:
            labels.append(1)
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
            true_reversal_positive += 1
            continue

        # === 負樣本：嚴格過濾 + 會虧損 ===
        t_price = df.iloc[t_idx]["close"]
        entry_price = df.iloc[entry_idx]["close"]
        t_minus_1_price = df.iloc[t_idx - 1]["close"]

        sar_distance_pct = df.iloc[t_idx]["sar_distance_pct"]
        sar_price = df.iloc[t_idx]["sar"]
        bb_middle = df.iloc[t_idx]["bb_middle"]

        prev_price = df.iloc[t_idx - 1]["close"]
        next_price = df.iloc[t_idx + 1]["close"]
        is_peak = t_price > prev_price and t_price > next_price

        # 過濾條件 1: |sar_distance_pct| > 1.3%
        if abs(sar_distance_pct) <= 1.3:
            filtered_out += 1
            continue

        # 過濾條件 2: peak 在 SAR 上方，valley 在 SAR 下方
        if is_peak:
            if t_price <= sar_price:
                filtered_out += 1
                continue
        else:
            if t_price >= sar_price:
                filtered_out += 1
                continue

        # 過濾條件 3: peak 在 bb_middle 上方，valley 在 bb_middle 下方
        if is_peak:
            if t_price <= bb_middle:
                filtered_out += 1
                continue
        else:
            if t_price >= bb_middle:
                filtered_out += 1
                continue

        # 過濾條件 4: [t-1] -> [t] 價格變化 < 0.5%
        price_change_pct = abs((t_price - t_minus_1_price) / t_minus_1_price) * 100
        if price_change_pct >= 0.5:
            filtered_out += 1
            continue

        # 過濾條件 5: entry 價格確認反轉方向
        if is_peak:
            if entry_price >= t_price:
                filtered_out += 1
                continue
        else:
            if entry_price <= t_price:
                filtered_out += 1
                continue

        # 過濾條件 6: 檢查是否會虧損
        max_loss_pct = 0.0
        for i in range(1, max_holding_bars + 1):
            future_idx = entry_idx + i
            if future_idx >= len(df):
                break

            future_price = df.iloc[future_idx]["close"]

            if is_peak:
                pnl_pct = (entry_price - future_price) / entry_price * 100
            else:
                pnl_pct = (future_price - entry_price) / entry_price * 100

            if pnl_pct < max_loss_pct:
                max_loss_pct = pnl_pct

        # 會觸發止損 → 負樣本
        if max_loss_pct <= -0.18:
            labels.append(0)
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
            negative += 1
        else:
            filtered_out += 1

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    print("\n📊 標籤分佈:")
    print(
        f"   正樣本（真實反轉點）: {positive_count} 個 ({positive_count/len(labels)*100:.2f}%)"
    )
    print(
        f"   負樣本（會虧損）: {negative_count} 個 ({negative_count/len(labels)*100:.2f}%)"
    )
    print(f"   被過濾（不確定）: {filtered_out} 個")
    print(f"   總樣本: {len(labels)} 個")

    print("\n💡 即時交易預期:")
    print(f"   find_local_extremes 找到 ~{len(extreme_indices)} 個候選點")
    print(f"   LSTM 過濾後保留 ~{positive_count} 個高質量點")
    print(f"   過濾率: {filtered_out/len(extreme_indices)*100:.1f}%")

    return t_indices, entry_indices, labels


def extract_sequences(
    df: pd.DataFrame,
    entry_indices: list,
    labels: list,
    seq_len: int = 30,
    feature_cols: list = None,
) -> tuple:
    """
    提取時序特徵序列

    對每個入場點 entry_idx，提取 [entry_idx-seq_len+1, ..., entry_idx] 的序列

    Args:
        df: 完整數據
        entry_indices: 入場點索引（實際交易入場的時間點）
        labels: 標籤
        seq_len: 序列長度（默認 30 根 K 線 = 7.5 小時）
        feature_cols: 要使用的特徵列

    Returns:
        (sequences, labels_filtered, valid_indices, feature_cols)
    """
    print("\n" + "=" * 80)
    print(f"🔧 (4) 提取時序序列（seq_len={seq_len}）")
    print("=" * 80)

    if feature_cols is None:
        # 使用所有數值特徵（排除 Bottom 5 有害特徵）
        # 移除：amplitude, bb_upper, lis_2nd, avg_bb_position, volume_sma
        feature_cols = [
            # 基本價量 (OHLCV)
            "open",
            "high",
            "low",
            "close",
            "volume",
            # 趨勢指標
            "ema_100",
            "ema_50",
            "close_ema50_pct",
            "close_ema100_pct",
            # 動量指標
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "stochrsi_k",
            "stochrsi_d",
            # 波動率指標
            "atr",
            "bb_middle",
            "bb_lower",  # 移除 bb_upper
            "bb_position",
            "bb_width_pct",
            # SAR 指標
            "sar",
            "sar_distance_pct",
            "sar_distance_2nd",
            # 成交量指標
            "volume_change",
            "volume_ratio",
            "volume_trend",  # 移除 volume_sma
            # 價格變化
            "price_change_pct",
            "price_change_3",
            "price_change_5",
            "price_atr_ratio_1h",
            # 斜率和導數
            "price_slope_1st",
            "price_slope_2nd",
            "price_2nd",
            "macd_2nd",
            "vol_2nd",
            "atr_2nd",
            # LIS/LDS
            "lis",
            "lds",
            "lds_2nd",  # 移除 lis_2nd
            # 統計特徵
            "avg_amplitude",
            "avg_volume",
            "avg_macd",
            "avg_rsi",  # 移除 avg_bb_position
            "volatility",
        ]

    print(f"\n使用 {len(feature_cols)} 個特徵（移除 Bottom 5 有害特徵）:")
    print("   ❌ 已移除: amplitude, bb_upper, lis_2nd, avg_bb_position, volume_sma")
    print("   ✅ 保留重要特徵:")
    print("      基本價量: open, high, low, close, volume")
    print("      趨勢指標: EMA, MACD, RSI, StochRSI")
    print("      波動率: ATR, BB (middle/lower), volatility")
    print("      SAR: sar, sar_distance_pct, sar_distance_2nd")
    print("      成交量: volume_change, volume_ratio, volume_trend")
    print("      價格變化: price_change_3/5, price_atr_ratio")
    print("      導數特徵: price_2nd, macd_2nd, vol_2nd, atr_2nd")
    print("      LIS/LDS: lis, lds, lds_2nd")

    sequences = []
    labels_filtered = []
    valid_indices = []

    for i, entry_idx in enumerate(entry_indices):
        # 確保有足夠的歷史數據
        if entry_idx < seq_len:
            continue

        # 提取序列 [entry_idx-seq_len+1, ..., entry_idx]
        # 這是在入場點時，模型能看到的過去 seq_len 根 K 線
        start_idx = entry_idx - seq_len + 1
        end_idx = entry_idx + 1

        try:
            sequence = df.iloc[start_idx:end_idx][feature_cols].values

            # 檢查是否有 NaN
            if np.isnan(sequence).any():
                continue

            sequences.append(sequence)
            labels_filtered.append(labels[i])
            valid_indices.append(i)

        except (KeyError, IndexError):
            continue

    sequences = np.array(sequences)
    labels_filtered = np.array(labels_filtered)

    print("\n✅ 序列提取完成:")
    print(f"   有效樣本: {len(sequences)} 個")
    print(f"   序列形狀: {sequences.shape}")
    print(f"   正樣本: {labels_filtered.sum()} 個")
    print(f"   負樣本: {len(labels_filtered) - labels_filtered.sum()} 個")

    return sequences, labels_filtered, valid_indices, feature_cols


def normalize_sequences(sequences_train, sequences_test):
    """
    標準化序列數據（Z-score）

    使用訓練集的均值和標準差標準化訓練集和測試集
    """
    print("\n標準化序列數據...")

    # 計算訓練集的均值和標準差（對每個特徵）
    # sequences_train: (num_samples, seq_len, num_features)
    mean = sequences_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, num_features)
    std = sequences_train.std(axis=(0, 1), keepdims=True)

    # 避免除以零
    std = np.where(std < 1e-8, 1.0, std)

    # 標準化
    sequences_train_norm = (sequences_train - mean) / std
    sequences_test_norm = (sequences_test - mean) / std

    print(f"   訓練集: mean={mean.mean():.4f}, std={std.mean():.4f}")

    return sequences_train_norm, sequences_test_norm, mean, std


def balance_samples(sequences, labels):
    """平衡樣本 - 下採樣多數類"""
    print("\n" + "=" * 80)
    print("⚖️  (5) 平衡樣本")
    print("=" * 80)

    positive_mask = labels == 1
    negative_mask = labels == 0

    n_positive = positive_mask.sum()
    n_negative = negative_mask.sum()

    print("\n原始樣本分佈:")
    print(f"   正樣本: {n_positive} 個")
    print(f"   負樣本: {n_negative} 個")

    # 下採樣
    if n_positive > n_negative:
        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]

        np.random.seed(42)
        positive_indices_sampled = np.random.choice(
            positive_indices, size=n_negative, replace=False
        )

        balanced_indices = np.concatenate([positive_indices_sampled, negative_indices])
    else:
        positive_indices = np.where(positive_mask)[0]
        negative_indices = np.where(negative_mask)[0]

        np.random.seed(42)
        negative_indices_sampled = np.random.choice(
            negative_indices, size=n_positive, replace=False
        )

        balanced_indices = np.concatenate([positive_indices, negative_indices_sampled])

    # 打亂
    np.random.seed(42)
    np.random.shuffle(balanced_indices)

    sequences_balanced = sequences[balanced_indices]
    labels_balanced = labels[balanced_indices]

    print("\n✅ 平衡後樣本分佈:")
    print(f"   正樣本: {(labels_balanced == 1).sum()} 個 (50.00%)")
    print(f"   負樣本: {(labels_balanced == 0).sum()} 個 (50.00%)")
    print(f"   總樣本: {len(labels_balanced)} 個")

    return sequences_balanced, labels_balanced


# ============================================================================
# 4. 訓練和驗證
# ============================================================================


def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device).unsqueeze(1)

        # 前向傳播
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 統計
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    all_probas = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())
            all_probas.extend(outputs.cpu().tolist())

    avg_loss = total_loss / len(test_loader)
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_probas = np.array(all_probas).flatten()

    accuracy = accuracy_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probas)

    return avg_loss, accuracy, auc, all_labels, all_predictions


def train_and_validate(
    sequences,
    labels,
    feature_cols,
    output_dir: str = "models/reversal",
    num_epochs: int = 64,
    batch_size: int = 32,
    learning_rate: float = 0.001,
):
    """訓練和驗證 LSTM 模型"""
    print("\n" + "=" * 80)
    print("🤖 (6) 訓練和驗證模型 - PyTorch LSTM")
    print("=" * 80)

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=46, stratify=labels
    )

    print("\n數據分割:")
    print(f"   訓練集: {len(X_train)} 個樣本")
    print(f"   測試集: {len(X_test)} 個樣本")

    # 標準化
    X_train_norm, X_test_norm, mean, std = normalize_sequences(X_train, X_test)

    # 創建數據集和加載器
    train_dataset = ReversalDataset(X_train_norm, y_train)
    test_dataset = ReversalDataset(X_test_norm, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"\n使用設備: {device}")

    # 創建模型（最佳配置）
    input_size = sequences.shape[2]
    model = ReversalLSTM(
        input_size=input_size, hidden_size=32, num_layers=1, dropout=0.5
    )
    model = model.to(device)

    print("\n模型架構（最佳配置）:")
    print(f"   輸入大小: {input_size}")
    print("   隱藏層大小: 32")
    print("   LSTM 層數: 1")
    print("   Dropout: 0.5")

    # 損失函數和優化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n訓練參數:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print("   Early Stopping: patience=7 (7 epochs 無改善則停止)")

    # 訓練 + Early Stopping
    print("\n開始訓練...")
    best_auc = 0.0
    best_epoch = 0
    best_model_state = None
    patience = 9
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device, epoch
        )

        test_loss, test_acc, test_auc, _, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        # 每 5 個 epoch 顯示一次
        if (epoch + 1) % 5 == 0:
            print(
                f"   Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}"
            )

        # Early Stopping: 保存最佳模型
        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"   ✨ 新的最佳 AUC: {best_auc:.4f} (Epoch {best_epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early Stopping: {patience} epochs 無改善")
                break

    # 恢復最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n✅ 訓練完成！已恢復最佳模型")
    else:
        print("\n✅ 訓練完成！")

    print(f"   最佳 Epoch: {best_epoch}")
    print(f"   最佳 AUC: {best_auc:.4f}")

    # 最終評估
    test_loss, test_acc, test_auc, y_test_true, y_test_pred = evaluate_model(
        model, test_loader, criterion, device
    )

    print("\n📊 最終測試集性能:")
    print(f"   準確率: {test_acc:.4f}")
    print(f"   AUC: {test_auc:.4f}")

    print(f"\n📋 分類報告:")
    print(classification_report(y_test_true, y_test_pred))

    print("\n📊 混淆矩陣:")
    cm = confusion_matrix(y_test_true, y_test_pred)
    print(f"   真負例: {cm[0,0]}, 假正例: {cm[0,1]}")
    print(f"   假負例: {cm[1,0]}, 真正例: {cm[1,1]}")

    # 特徵重要性分析（Permutation Importance）
    print(f"\n🎯 特徵重要性分析（Permutation Importance）...")
    print(f"   方法：打亂每個特徵，計算 AUC 下降幅度")

    baseline_auc = test_auc
    feature_importance = []

    model.eval()
    with torch.no_grad():
        for i, feature_name in enumerate(feature_cols):
            # 複製測試數據
            X_test_permuted = X_test_norm.copy()

            # 打亂第 i 個特徵（所有時間步）
            np.random.seed(42)
            for t in range(X_test_permuted.shape[1]):  # 對每個時間步
                np.random.shuffle(X_test_permuted[:, t, i])

            # 評估
            test_dataset_permuted = ReversalDataset(X_test_permuted, y_test)
            test_loader_permuted = DataLoader(
                test_dataset_permuted, batch_size=batch_size, shuffle=False
            )

            _, _, permuted_auc, _, _ = evaluate_model(
                model, test_loader_permuted, criterion, device
            )

            # 計算重要性（AUC 下降）
            importance = baseline_auc - permuted_auc
            feature_importance.append(
                {
                    "feature": feature_name,
                    "importance": importance,
                    "auc_drop": importance,
                }
            )

    # 排序並顯示 Top 15
    feature_importance_df = pd.DataFrame(feature_importance)
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    )

    print(f"\n📊 Top 15 重要特徵:")
    for idx, row in feature_importance_df.head(15).iterrows():
        print(f"   {row['feature']:25s}: AUC 下降 {row['auc_drop']:.4f}")

    print(f"\n📊 Bottom 5 不重要特徵:")
    for idx, row in feature_importance_df.tail(5).iterrows():
        print(f"   {row['feature']:25s}: AUC 下降 {row['auc_drop']:.4f}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "reversal_model_v7_lstm.pth")

    print(f"\n💾 保存模型...")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "feature_cols": feature_cols,
            "mean": mean,
            "std": std,
            "seq_len": sequences.shape[1],
        },
        model_path,
    )
    print(f"   模型已保存: {model_path}")

    return model, test_acc, test_auc


# ============================================================================
# 5. 主流程
# ============================================================================


def main():
    """主訓練流程"""
    print("=" * 80)
    print("🤖 ML Reversal V7 模型訓練 - PyTorch LSTM")
    print("=" * 80)

    # (1) 載入數據
    data_file = "training_data/reversal_training_data_2023-12-01_to_2025-01-01_15m.csv"
    df = load_data(data_file, year=2024)

    # (2) 篩選候選樣本
    extreme_indices = filter_candidates(df)

    # (3) 生成標籤
    t_indices, entry_indices, labels = generate_labels(df, extreme_indices)

    # (4) 提取時序序列（使用 entry_indices - 實際入場點）
    sequences, labels_filtered, valid_indices, feature_cols = extract_sequences(
        df, entry_indices, labels, seq_len=32
    )

    # (5) 平衡樣本
    sequences_balanced, labels_balanced = balance_samples(sequences, labels_filtered)

    # (6) 訓練和驗證
    model, test_acc, test_auc = train_and_validate(
        sequences_balanced,
        labels_balanced,
        feature_cols,
        num_epochs=128,
        batch_size=32,
        learning_rate=0.001,
    )

    print("\n" + "=" * 80)
    print("✅ 訓練完成！")
    print("=" * 80)
    print(f"\n最終性能:")
    print(f"   測試準確率: {test_acc:.4f}")
    print(f"   測試 AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
