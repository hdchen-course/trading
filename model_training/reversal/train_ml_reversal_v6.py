#!/usr/bin/env python3
"""
ML Reversal V6 模型訓練腳本

基於 V3 的架構，使用新的 SAR 距離特徵

改進：
- 添加 sar_distance_pct 和 sar_distance_2nd 特徵
- 使用 2024 年完整數據訓練
- XGBoost 模型
"""

import sys
import os

# 必須在其他 import 之前修改 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
import joblib  # noqa: E402

from utils.extremes import find_local_extremes  # noqa: E402


def load_data(data_file: str, year: int = 2024) -> pd.DataFrame:
    """
    載入訓練數據並篩選指定年份

    Args:
        data_file: 數據文件路徑
        year: 要篩選的年份

    Returns:
        篩選後的 DataFrame
    """
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

    # 檢查必要欄位
    required_columns = ["timestamp", "close", "volume", "reverse"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"缺少必要欄位: {missing_columns}")

    # 檢查 SAR 距離特徵
    sar_features = []
    if "sar_distance_pct" in df.columns:
        sar_features.append("sar_distance_pct")
    if "sar_distance_2nd" in df.columns:
        sar_features.append("sar_distance_2nd")

    if sar_features:
        print(f"\n✅ 發現 SAR 距離特徵: {', '.join(sar_features)}")
    else:
        print(f"\n⚠️  未發現 SAR 距離特徵")

    # 統計反轉點
    reversal_count = df["reverse"].sum()
    reversal_pct = reversal_count / len(df) * 100

    print(f"\n反轉點統計:")
    print(f"   反轉點數量: {reversal_count} 個 ({reversal_pct:.2f}%)")

    # 統計峰值和谷值
    if "reverse_type" in df.columns:
        peak_count = (df["reverse_type"] == "peak").sum()
        valley_count = (df["reverse_type"] == "valley").sum()
        print(f"   峰值: {peak_count} 個")
        print(f"   谷值: {valley_count} 個")

    print(f"   總欄位數: {len(df.columns)}")

    return df


def filter_candidates(df: pd.DataFrame) -> list:
    """
    篩選候選樣本 - 使用局部極值檢測

    使用 find_local_extremes 找出潛在的反轉點
    參數：window=20, prominence_factor=0.3

    Args:
        df: 原始數據

    Returns:
        候選反轉點索引列表
    """
    print("\n" + "=" * 80)
    print("🎯 (2) 篩選候選樣本 - 局部極值檢測")
    print("=" * 80)

    print(f"\n使用 find_local_extremes:")
    print(f"   window = 20")
    print(f"   prominence_factor = 0.3")

    # 使用 find_local_extremes 找出極值點
    prices = df["close"].values
    extremes, peaks, valleys = find_local_extremes(
        prices, window=20, prominence_factor=0.3
    )

    # 極值點索引（peaks 和 valleys 的聯集）
    extreme_indices = sorted(list(set(peaks) | set(valleys)))

    print(f"\n✅ 找到極值點:")
    print(f"   峰值點: {len(peaks)} 個")
    print(f"   谷值點: {len(valleys)} 個")
    print(
        f"   總極值點: {len(extreme_indices)} 個 ({len(extreme_indices)/len(df)*100:.2f}%)"
    )

    return extreme_indices


def generate_labels(df: pd.DataFrame, extreme_indices: list) -> tuple:
    """
    生成訓練標籤 - 基於真實反轉點和止損/止盈邏輯

    邏輯：
    1. 候選點 [t] 由 find_local_extremes 找出
    2. 入場點為 [t+1]
    3. 如果候選點是真實反轉點 => 正樣本
    4. 如果不是真實反轉點，檢查入場後 4 小時內：
       - 沒有虧損 0.18% 且曾經浮盈 1% => 正樣本
       - 其餘 => 負樣本

    Args:
        df: 完整數據
        extreme_indices: 極值點索引列表

    Returns:
        (t_indices, entry_indices, labels) 元組
    """
    print("\n" + "=" * 80)
    print("🏷️  (3) 生成標籤")
    print("=" * 80)

    print(f"\n標籤邏輯:")
    print(f"   候選點: [t] (由 find_local_extremes 找出)")
    print(f"   入場點: [t+1]")
    print(f"   正樣本條件: 候選點是真實反轉點")
    print(f"   負樣本條件: 虧損超過 0.35% 且 |sar_distance_pct| < 1.22%")

    labels = []
    t_indices = []  # 候選點索引
    entry_indices = []  # 入場點索引

    # 統計
    true_reversal_positive = 0  # 真實反轉點的正樣本
    negative = 0  # 負樣本

    max_holding_bars = 16  # 4 小時 = 16 根 15 分鐘 K 線

    # 獲取真實反轉點集合
    true_reversal_set = set(df[df["reverse"] == 1].index.tolist())

    for t_idx in extreme_indices:
        # 確保有足夠的未來數據
        if t_idx + 1 + max_holding_bars >= len(df):
            continue

        entry_idx = t_idx + 1

        # 條件 1: 檢查是否為真實反轉點
        if t_idx in true_reversal_set:
            label = 1  # 正樣本
            labels.append(label)
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
            true_reversal_positive += 1
            continue

        # 條件 2: 檢查是否為負樣本
        entry_price = df.iloc[entry_idx]["close"]
        t_price = df.iloc[t_idx]["close"]

        # 判斷交易方向（峰值做空，谷值做多）
        if t_idx > 0 and t_idx < len(df) - 1:
            prev_price = df.iloc[t_idx - 1]["close"]
            next_price = df.iloc[t_idx + 1]["close"]
            is_peak = t_price > prev_price and t_price > next_price
        else:
            continue

        # 模擬交易
        max_loss_pct = 0.0  # 追蹤最大虧損

        for i in range(1, max_holding_bars + 1):
            future_idx = entry_idx + i
            if future_idx >= len(df):
                break

            future_price = df.iloc[future_idx]["close"]

            if is_peak:  # 做空
                pnl_pct = (entry_price - future_price) / entry_price * 100
            else:  # 做多
                pnl_pct = (future_price - entry_price) / entry_price * 100

            # 追蹤最大虧損
            if pnl_pct < max_loss_pct:
                max_loss_pct = pnl_pct

        # 生成負樣本標籤
        # 負樣本：虧損超過 0.35% 且 |sar_distance_pct| < 1.22%
        if max_loss_pct <= -0.35:
            # 檢查 SAR 距離條件
            sar_dist = abs(df.iloc[t_idx]["sar_distance_pct"])
            if sar_dist < 1.22:
                label = 0  # 負樣本：虧損超過 0.35% 且 SAR 距離小
                labels.append(label)
                t_indices.append(t_idx)
                entry_indices.append(entry_idx)
                negative += 1

    # 統計標籤分佈
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    print(f"\n📊 標籤分佈:")
    print(f"   正樣本總數: {positive_count} 個 ({positive_count/len(labels)*100:.2f}%)")
    print(f"     - 真實反轉點: {true_reversal_positive} 個")
    print(f"   負樣本總數: {negative_count} 個 ({negative_count/len(labels)*100:.2f}%)")
    print(f"   總樣本數: {len(labels)} 個")

    # 顯示樣本示例
    print(f"\n📋 樣本示例（前 5 個正樣本 + 前 5 個負樣本）:")

    # 正樣本示例
    positive_samples = [
        (i, t_indices[i], entry_indices[i])
        for i in range(len(labels))
        if labels[i] == 1
    ]
    print(f"\n正樣本示例:")
    for idx, (i, t_idx, entry_idx) in enumerate(positive_samples[:5], 1):
        timestamp = df.iloc[entry_idx]["timestamp"]
        entry_price = df.iloc[entry_idx]["close"]
        is_true_reversal = (
            "✅ 真實反轉點" if t_idx in true_reversal_set else "💰 止盈條件"
        )
        print(f"   {idx}. {timestamp} | 入場價: {entry_price:.2f} | {is_true_reversal}")

    # 負樣本示例
    negative_samples = [
        (i, t_indices[i], entry_indices[i])
        for i in range(len(labels))
        if labels[i] == 0
    ]
    print(f"\n負樣本示例:")
    for idx, (i, t_idx, entry_idx) in enumerate(negative_samples[:5], 1):
        timestamp = df.iloc[entry_idx]["timestamp"]
        entry_price = df.iloc[entry_idx]["close"]
        print(f"   {idx}. {timestamp} | 入場價: {entry_price:.2f} | ❌ 負樣本")

    return t_indices, entry_indices, labels


def extract_features(
    df: pd.DataFrame, t_indices: list, entry_indices: list, labels: list
) -> pd.DataFrame:
    """
    提取特徵：計算變化率特徵

    特徵列表（12 個）：
    1. price_2nd 變化率: [t-1] -> [t], [t] -> [entry]
    2. macd_2nd 變化率: [t-1] -> [t]
    3. vol_2nd 變化率: [t-1] -> [t], [t] -> [entry]
    4. macd_histogram 變化率: [t] -> [entry]
    5. stochrsi_k 變化率: [t-1] -> [t]
    6. sar_distance_pct 變化率: [t-1] -> [t], [t] -> [entry]
    7. price_change_3: [t] 的值
    8. price_change_5: [t] 的值
    9. price_atr_ratio_1h: [t] 的值

    Args:
        df: 完整數據
        t_indices: 極值點索引列表
        entry_indices: 入場點索引列表
        labels: 標籤列表

    Returns:
        特徵 DataFrame
    """
    print("\n" + "=" * 80)
    print("🔧 (4) 提取特徵 - 變化率特徵")
    print("=" * 80)

    print(f"\n特徵策略: 計算關鍵指標的變化率")
    print(f"   共 12 個特徵 (移除 bb_position):")
    print(f"   1. price_2nd: [t-1]->[t], [t]->[entry]")
    print(f"   2. macd_2nd: [t-1]->[t]")
    print(f"   3. vol_2nd: [t-1]->[t], [t]->[entry]")
    print(f"   4. macd_histogram: [t]->[entry]")
    print(f"   5. stochrsi_k: [t-1]->[t]")
    print(f"   6. sar_distance_pct: [t-1]->[t], [t]->[entry]")
    print(f"   7. price_change_3: [t] 的值")
    print(f"   8. price_change_5: [t] 的值")
    print(f"   9. price_atr_ratio_1h: [t] 的值")

    feature_samples = []

    for i in range(len(t_indices)):
        t_idx = t_indices[i]
        entry_idx = entry_indices[i]
        label = labels[i]

        # 確保有 t-1 的數據
        if t_idx < 1:
            continue

        features = {}

        # 1. price_2nd 變化率
        price_2nd_t_minus_1 = df.iloc[t_idx - 1]["price_2nd"]
        price_2nd_t = df.iloc[t_idx]["price_2nd"]
        price_2nd_entry = df.iloc[entry_idx]["price_2nd"]

        if abs(price_2nd_t_minus_1) > 1e-10:
            features["price_2nd_change_t_minus_1_to_t"] = abs(
                price_2nd_t - price_2nd_t_minus_1
            ) / abs(price_2nd_t_minus_1)
        else:
            features["price_2nd_change_t_minus_1_to_t"] = 0.0

        if abs(price_2nd_t) > 1e-10:
            features["price_2nd_change_t_to_entry"] = abs(
                price_2nd_entry - price_2nd_t
            ) / abs(price_2nd_t)
        else:
            features["price_2nd_change_t_to_entry"] = 0.0

        # 2. macd_2nd 變化率
        macd_2nd_t_minus_1 = df.iloc[t_idx - 1]["macd_2nd"]
        macd_2nd_t = df.iloc[t_idx]["macd_2nd"]

        if abs(macd_2nd_t_minus_1) > 1e-10:
            features["macd_2nd_change_t_minus_1_to_t"] = abs(
                macd_2nd_t - macd_2nd_t_minus_1
            ) / abs(macd_2nd_t_minus_1)
        else:
            features["macd_2nd_change_t_minus_1_to_t"] = 0.0

        # 3. vol_2nd 變化率
        vol_2nd_t_minus_1 = df.iloc[t_idx - 1]["vol_2nd"]
        vol_2nd_t = df.iloc[t_idx]["vol_2nd"]
        vol_2nd_entry = df.iloc[entry_idx]["vol_2nd"]

        if abs(vol_2nd_t_minus_1) > 1e-10:
            features["vol_2nd_change_t_minus_1_to_t"] = abs(
                vol_2nd_t - vol_2nd_t_minus_1
            ) / abs(vol_2nd_t_minus_1)
        else:
            features["vol_2nd_change_t_minus_1_to_t"] = 0.0

        if abs(vol_2nd_t) > 1e-10:
            features["vol_2nd_change_t_to_entry"] = abs(
                vol_2nd_entry - vol_2nd_t
            ) / abs(vol_2nd_t)
        else:
            features["vol_2nd_change_t_to_entry"] = 0.0

        # 4. macd_histogram 變化率
        macd_hist_t = df.iloc[t_idx]["macd_histogram"]
        macd_hist_entry = df.iloc[entry_idx]["macd_histogram"]

        if abs(macd_hist_t) > 1e-10:
            features["macd_histogram_change_t_to_entry"] = abs(
                macd_hist_entry - macd_hist_t
            ) / abs(macd_hist_t)
        else:
            features["macd_histogram_change_t_to_entry"] = 0.0

        # 5. stochrsi_k 變化率
        stochrsi_k_t_minus_1 = df.iloc[t_idx - 1]["stochrsi_k"]
        stochrsi_k_t = df.iloc[t_idx]["stochrsi_k"]

        if abs(stochrsi_k_t_minus_1) > 1e-10:
            features["stochrsi_k_change_t_minus_1_to_t"] = abs(
                stochrsi_k_t - stochrsi_k_t_minus_1
            ) / abs(stochrsi_k_t_minus_1)
        else:
            features["stochrsi_k_change_t_minus_1_to_t"] = 0.0

        # 6. sar_distance_pct 變化率
        sar_dist_t_minus_1 = df.iloc[t_idx - 1]["sar_distance_pct"]
        sar_dist_t = df.iloc[t_idx]["sar_distance_pct"]
        sar_dist_entry = df.iloc[entry_idx]["sar_distance_pct"]

        if abs(sar_dist_t_minus_1) > 1e-10:
            features["sar_distance_pct_change_t_minus_1_to_t"] = abs(
                sar_dist_t - sar_dist_t_minus_1
            ) / abs(sar_dist_t_minus_1)
        else:
            features["sar_distance_pct_change_t_minus_1_to_t"] = 0.0

        if abs(sar_dist_t) > 1e-10:
            features["sar_distance_pct_change_t_to_entry"] = abs(
                sar_dist_entry - sar_dist_t
            ) / abs(sar_dist_t)
        else:
            features["sar_distance_pct_change_t_to_entry"] = 0.0

        # 7. price_change_3, price_change_5, price_atr_ratio_1h (移除 bb_position)
        features["price_change_3"] = df.iloc[t_idx]["price_change_3"]
        features["price_change_5"] = df.iloc[t_idx]["price_change_5"]
        features["price_atr_ratio_1h"] = df.iloc[t_idx]["price_atr_ratio_1h"]

        # 添加標籤和時間戳
        features["label"] = label
        features["timestamp"] = df.iloc[entry_idx]["timestamp"]

        feature_samples.append(features)

    # 轉換為 DataFrame
    features_df = pd.DataFrame(feature_samples)

    print(f"\n✅ 特徵提取完成:")
    print(f"   樣本數: {len(features_df)}")
    print(f"   特徵數: 9 個")

    # 檢查缺失值
    feature_cols = [
        col for col in features_df.columns if col not in ["label", "timestamp"]
    ]
    missing_counts = features_df[feature_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) > 0:
        print(f"\n⚠️  發現 {len(cols_with_missing)} 個欄位有缺失值")
        print(f"   移除有缺失值的樣本...")
        features_df = features_df.dropna(subset=feature_cols)
        print(f"   清理後樣本數: {len(features_df)} 個")
    else:
        print(f"\n✅ 無缺失值")

    # 最終統計
    final_positive = features_df["label"].sum()
    final_negative = len(features_df) - final_positive

    print(f"\n📊 最終樣本分佈:")
    print(
        f"   正樣本: {final_positive} 個 ({final_positive/len(features_df)*100:.2f}%)"
    )
    print(
        f"   負樣本: {final_negative} 個 ({final_negative/len(features_df)*100:.2f}%)"
    )

    # 顯示特徵統計
    print(f"\n📊 特徵統計:")
    for col in feature_cols:
        mean_val = features_df[col].mean()
        std_val = features_df[col].std()
        print(f"   {col}: 平均={mean_val:.4f}, 標準差={std_val:.4f}")

    return features_df


def balance_samples(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    平衡樣本 - 下採樣多數類使正負樣本 1:1

    Args:
        features_df: 原始特徵 DataFrame

    Returns:
        平衡後的 DataFrame
    """
    print("\n" + "=" * 80)
    print("⚖️  (5) 平衡樣本")
    print("=" * 80)

    positive_samples = features_df[features_df["label"] == 1]
    negative_samples = features_df[features_df["label"] == 0]

    n_positive = len(positive_samples)
    n_negative = len(negative_samples)

    print(f"\n原始樣本分佈:")
    print(f"   正樣本: {n_positive} 個")
    print(f"   負樣本: {n_negative} 個")
    print(f"   比例: 1:{n_negative/n_positive:.2f}")

    # 下採樣多數類
    if n_positive > n_negative:
        print(f"\n下採樣正樣本...")
        positive_samples_balanced = positive_samples.sample(
            n=n_negative, random_state=42
        )
        negative_samples_balanced = negative_samples
        removed = n_positive - n_negative
    else:
        print(f"\n下採樣負樣本...")
        positive_samples_balanced = positive_samples
        negative_samples_balanced = negative_samples.sample(
            n=n_positive, random_state=42
        )
        removed = n_negative - n_positive

    # 合併並打亂
    balanced_df = pd.concat([positive_samples_balanced, negative_samples_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✅ 平衡後樣本分佈:")
    print(f"   正樣本: {len(positive_samples_balanced)} 個 (50.00%)")
    print(f"   負樣本: {len(negative_samples_balanced)} 個 (50.00%)")
    print(f"   總樣本數: {len(balanced_df)} 個")
    print(f"   移除樣本: {removed} 個")

    return balanced_df


def train_and_validate(features_df: pd.DataFrame, output_dir: str = "models/reversal"):
    """
    訓練和驗證 RandomForest 模型

    Args:
        features_df: 特徵 DataFrame
        output_dir: 模型輸出目錄
    """
    print("\n" + "=" * 80)
    print("🤖 (6) 訓練和驗證模型 - RandomForest")
    print("=" * 80)

    # 準備特徵和標籤
    feature_cols = [
        col for col in features_df.columns if col not in ["label", "timestamp"]
    ]
    X = features_df[feature_cols]
    y = features_df["label"]

    print(f"\n數據準備:")
    print(f"   特徵數: {len(feature_cols)}")
    print(f"   樣本數: {len(X)}")
    print(f"   正樣本: {y.sum()} 個 ({y.sum()/len(y)*100:.2f}%)")
    print(f"   負樣本: {len(y) - y.sum()} 個 ({(len(y) - y.sum())/len(y)*100:.2f}%)")

    # 分割訓練和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n數據分割:")
    print(f"   訓練集: {len(X_train)} 個樣本")
    print(f"   測試集: {len(X_test)} 個樣本")

    # 訓練 RandomForest 模型
    print(f"\n🌲 訓練 RandomForest 模型...")
    print(f"   最終優化版本:")
    print(f"   - n_estimators=500 (增加樹的數量)")
    print(f"   - max_depth=5 (降低深度減少過擬合)")
    print(f"   - min_samples_split=15 (提高分裂門檻)")
    print(f"   - min_samples_leaf=8 (提高葉節點門檻)")
    print(f"   - max_features='sqrt' (使用 sqrt(n_features))")
    print(f"   - max_samples=0.75 (每棵樹用 75% 樣本，增加多樣性)")

    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features="sqrt",
        max_samples=0.75,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # 預測
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    y_train_proba = rf_model.predict_proba(X_train)[:, 1]
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]

    # 評估指標
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n📊 模型性能:")
    print(f"   訓練集準確率: {train_accuracy:.4f}")
    print(f"   測試集準確率: {test_accuracy:.4f}")
    print(f"   訓練集 AUC: {train_auc:.4f}")
    print(f"   測試集 AUC: {test_auc:.4f}")

    # 詳細分類報告
    print(f"\n📋 測試集分類報告:")
    print(classification_report(y_test, y_test_pred))

    # 混淆矩陣
    print(f"\n📊 測試集混淆矩陣:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"   真負例: {cm[0,0]}, 假正例: {cm[0,1]}")
    print(f"   假負例: {cm[1,0]}, 真正例: {cm[1,1]}")

    # 特徵重要性
    print(f"\n🎯 特徵重要性 (Top 5):")
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
        print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "reversal_model_v6_rf.pkl")

    print(f"\n💾 保存模型...")
    joblib.dump(rf_model, model_path)
    print(f"   模型已保存: {model_path}")

    # 保存特徵列表
    feature_list_path = os.path.join(output_dir, "reversal_model_v6_rf_features.txt")
    with open(feature_list_path, "w") as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    print(f"   特徵列表已保存: {feature_list_path}")

    # 保存特徵重要性
    importance_path = os.path.join(output_dir, "reversal_model_v6_rf_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"   特徵重要性已保存: {importance_path}")

    print(f"\n✅ 模型訓練完成！")
    print(f"   最終測試準確率: {test_accuracy:.4f}")
    print(f"   最終測試 AUC: {test_auc:.4f}")


def main():
    """主訓練流程"""
    print("=" * 80)
    print("🤖 ML Reversal V6 模型訓練")
    print("=" * 80)

    # (1) 載入數據
    data_file = "training_data/reversal_training_data_2023-12-01_to_2025-01-01_15m.csv"
    df = load_data(data_file, year=2024)

    # (2) 篩選候選樣本 - 找出極值點
    extreme_indices = filter_candidates(df)

    # (3) 生成標籤 - 基於入場點和止損止盈
    t_indices, entry_indices, labels = generate_labels(df, extreme_indices)

    # (4) 提取特徵
    features_df = extract_features(df, t_indices, entry_indices, labels)

    # (4.5) 平衡樣本
    balanced_df = balance_samples(features_df)

    # (5) 訓練和驗證
    train_and_validate(balanced_df)

    print("\n" + "=" * 80)
    print("✅ 訓練完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
