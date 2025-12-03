#!/usr/bin/env python3
"""
ML Reversal V5 模型訓練腳本 - 簡化特徵集

基於 V4，使用更簡化的特徵集提升模型效率

改進：
- 使用 Top 10 核心特徵 + price_change_3/5
- 移除複雜的組合特徵，專注於最重要的指標
- 同時訓練 Random Forest, LightGBM, CatBoost
- 比較各模型準確度和 ROC AUC
- 預測時間：1 小時（快速短期反轉）
- 使用 2024 年完整數據訓練

特徵選擇策略：
- 基於 V2 的 Top 10 特徵（累積重要性 60.7%）
- 新增 price_change_3/5 捕捉短期動量
- 總共 12 個特徵，更快的訓練和推理速度
"""

import sys
import os

# 必須在其他 import 之前修改 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    roc_auc_score,
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
    df = df[
        (df["timestamp"] >= f"{year}-01-01") & (df["timestamp"] < f"{year+1}-01-01")
    ].copy()

    print(f"   {year} 年數據: {len(df)} 行")
    print(f"   時間範圍: {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    # 檢查必要欄位
    required_columns = ["timestamp", "close", "volume", "reverse"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"缺少必要欄位: {missing_columns}")

    # 統計反轉點
    reversal_count = df["reverse"].sum()
    reversal_pct = reversal_count / len(df) * 100

    print(f"   反轉點數量: {reversal_count} 個 ({reversal_pct:.2f}%)")
    print(f"   欄位數: {len(df.columns)}")

    return df


def filter_candidates(df: pd.DataFrame) -> list:
    """
    篩選候選樣本 - 使用局部極值檢測

    使用 find_local_extremes 找出潛在的反轉點 [t]
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
        prices, window=20, prominence_factor=0
    )

    print(f"\n✅ 找到極值點:")
    print(f"   峰值點: {len(peaks)} 個")
    print(f"   谷值點: {len(valleys)} 個")

    # 添加 StochRSI 過濾條件
    print(f"\n🔍 應用過濾條件:")
    print(f"   峰值: StochRSI > 90")
    print(f"   谷值: StochRSI < 10")

    filtered_peaks = set()
    filtered_valleys = set()

    for peak_idx in peaks:
        if peak_idx < len(df):
            stochrsi_k = df.iloc[peak_idx].get("stochrsi_k", 50)
            if stochrsi_k > 90:
                filtered_peaks.add(peak_idx)

    for valley_idx in valleys:
        if valley_idx < len(df):
            stochrsi_k = df.iloc[valley_idx].get("stochrsi_k", 50)
            if stochrsi_k < 10:
                filtered_valleys.add(valley_idx)

    print(f"\n   過濾後峰值點: {len(filtered_peaks)} 個 (原 {len(peaks)} 個)")
    print(f"   過濾後谷值點: {len(filtered_valleys)} 個 (原 {len(valleys)} 個)")

    print(
        f"\n✅ 最終極值點: {len(filtered_peaks) + len(filtered_valleys)} 個 ({(len(filtered_peaks) + len(filtered_valleys))/len(df)*100:.2f}%)"
    )

    return filtered_peaks, filtered_valleys


def generate_labels(df: pd.DataFrame, peaks: set, valleys: set) -> tuple:
    """
    生成訓練標籤 - V5 簡化邏輯

    邏輯：
    1. 反轉點 [t] = 局部極值點（由 find_local_extremes + StochRSI 過濾）
    2. 入場點 [t+2] = 固定在反轉點後第2根K線
    3. 止損：跌破/漲破 [t] 的價格
       - 峰值做空：價格漲破 t_price → 止損
       - 谷值做多：價格跌破 t_price → 止損
    4. 止盈：2.5小時內獲利 >= 0.4%
    5. 正樣本：達到止盈
    6. 負樣本：觸及止損
    7. 排除：既沒達止盈也沒觸止損的樣本

    Args:
        df: 完整數據
        peaks: 峰值點索引集合
        valleys: 谷值點索引集合

    Returns:
        (t_indices, entry_indices, labels) 元組
    """
    print("\n" + "=" * 80)
    print("🏷️  (3) 生成標籤 - V5 簡化邏輯")
    print("=" * 80)

    print(f"\n標籤邏輯:")
    print(f"   反轉點: [t] = 局部極值點")
    print(f"   入場點: [t+2] = 固定在反轉點後第2根K線")
    print(f"   止損: 跌破/漲破 [t] 的價格")
    print(f"     - 峰值做空：價格漲破 t_price")
    print(f"     - 谷值做多：價格跌破 t_price")
    print(f"   止盈: 2.5小時內 >= 0.4% (10根15分鐘K線)")
    print(f"   正樣本: 達到止盈")
    print(f"   負樣本: 觸及止損")
    print(f"   排除: 既沒達止盈也沒觸止損")

    labels = []
    t_indices = []  # 極值點索引
    entry_indices = []  # 入場點索引
    excluded_count = 0  # 統計被排除的樣本數

    take_profit_pct = 0.4
    max_holding_bars = 10  # 2.5小時 = 10根15分鐘K線

    # 處理峰值（做空）
    for t_idx in peaks:
        # 確保有足夠的未來數據
        if t_idx + 2 + max_holding_bars >= len(df):
            continue

        # 反轉點價格
        t_price = df.iloc[t_idx]["close"]

        # 入場點固定為 [t+2]
        entry_idx = t_idx + 2
        entry_price = df.iloc[entry_idx]["close"]

        # 做空條件：[t+2] 價格要低於 [t]（價格已經下跌）
        if entry_price >= t_price:
            excluded_count += 1
            continue

        # 模擬做空交易
        hit_stop_loss = False
        hit_take_profit = False

        for i in range(1, max_holding_bars + 1):
            future_idx = entry_idx + i
            if future_idx >= len(df):
                break

            future_price = df.iloc[future_idx]["close"]

            # 止損：價格漲破 t_price
            if future_price > t_price:
                hit_stop_loss = True
                break
            # 止盈：獲利 >= 0.4%
            pnl_pct = (entry_price - future_price) / entry_price * 100
            if pnl_pct >= take_profit_pct:
                hit_take_profit = True
                break

        # 生成標籤
        if hit_take_profit:
            labels.append(1)  # 正樣本
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
        elif hit_stop_loss:
            labels.append(0)  # 負樣本
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
        else:
            excluded_count += 1

    # 處理谷值（做多）
    for t_idx in valleys:
        # 確保有足夠的未來數據
        if t_idx + 2 + max_holding_bars >= len(df):
            continue

        # 反轉點價格
        t_price = df.iloc[t_idx]["close"]

        # 入場點固定為 [t+2]
        entry_idx = t_idx + 2
        entry_price = df.iloc[entry_idx]["close"]

        # 做多條件：[t+2] 價格要高於 [t]（價格已經上漲）
        if entry_price <= t_price:
            excluded_count += 1
            continue

        # 模擬做多交易
        hit_stop_loss = False
        hit_take_profit = False

        for i in range(1, max_holding_bars + 1):
            future_idx = entry_idx + i
            if future_idx >= len(df):
                break

            future_price = df.iloc[future_idx]["close"]

            # 止損：價格跌破 t_price
            if future_price < t_price:
                hit_stop_loss = True
                break
            # 止盈：獲利 >= 0.4%
            pnl_pct = (future_price - entry_price) / entry_price * 100
            if pnl_pct >= take_profit_pct:
                hit_take_profit = True
                break

        # 生成標籤
        if hit_take_profit:
            labels.append(1)  # 正樣本
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
        elif hit_stop_loss:
            labels.append(0)  # 負樣本
            t_indices.append(t_idx)
            entry_indices.append(entry_idx)
        else:
            excluded_count += 1

    # 統計標籤分佈
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    total_candidates = len(labels) + excluded_count

    print(f"\n📊 標籤分佈:")
    print(
        f"   正樣本（達到止盈）: {positive_count} 個 ({positive_count/len(labels)*100:.2f}%)"
    )
    print(
        f"   負樣本（觸及止損）: {negative_count} 個 ({negative_count/len(labels)*100:.2f}%)"
    )
    print(
        f"   排除樣本（不確定）: {excluded_count} 個 ({excluded_count/total_candidates*100:.2f}%)"
    )
    print(f"   總樣本數: {len(labels)} 個（排除後）")
    print(f"   候選總數: {total_candidates} 個（排除前）")

    # 顯示樣本示例
    print(f"\n📋 樣本示例（前 5 個）:")
    for i in range(min(5, len(entry_indices))):
        t_idx = t_indices[i]
        entry_idx = entry_indices[i]
        label = labels[i]

        t_price = df.iloc[t_idx]["close"]
        entry_price = df.iloc[entry_idx]["close"]
        timestamp = df.iloc[entry_idx]["timestamp"]

        # 判斷方向（峰值做空，谷值做多）
        is_peak = t_idx in peaks
        direction = "做空" if is_peak else "做多"
        result = "✅ 達到止盈" if label == 1 else "❌ 觸及止損"

        print(
            f"   {i+1}. {timestamp} | {direction} | 反轉價: {t_price:.2f} | 入場價: {entry_price:.2f} | {result}"
        )

    return t_indices, entry_indices, labels


def extract_features(
    df: pd.DataFrame, t_indices: list, entry_indices: list, labels: list
) -> pd.DataFrame:
    """
    提取特徵：組合 [t] 和 [entry] 的特徵

    Args:
        df: 完整數據
        t_indices: 極值點索引列表
        entry_indices: 入場點索引列表
        labels: 標籤列表

    Returns:
        特徵 DataFrame
    """
    print("\n" + "=" * 80)
    print("🔧 (4) 提取特徵")
    print("=" * 80)

    print(f"\n策略: 組合 [t] 和 [entry] 的所有欄位作為特徵")
    print(f"   [t]: 極值點")
    print(f"   [entry]: 入場點 (t+1 或 t+2)")

    # V5: 全部 46 個數值欄位 × 3 個時間點 = 138 個特徵（排除 datetime）
    base_features = [
        "period",
        "close_price",
        "ema_100",
        "ema_50",
        "avg_amplitude",
        "avg_volume",
        "avg_macd",
        "avg_rsi",
        "price_above_sar_ratio",
        "avg_bb_position",
        "volatility",
        "price_change_pct",
        "price_atr_ratio_1h",
        "volume_trend",
        "lis",
        "lds",
        "macd_2nd",
        "price_2nd",
        "vol_2nd",
        "atr_2nd",
        "lis_2nd",
        "lds_2nd",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amplitude",
        "macd",
        "macd_signal",
        "macd_histogram",
        "rsi",
        "stochrsi_k",
        "stochrsi_d",
        "atr",
        "sar",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_position",
        "price_above_sar",
        "volume_change",
        "volume_sma",
        "volume_ratio",
        "price_change_3",
        "price_change_5",
    ]

    # 生成所有時間點的特徵
    extended_features = []
    for base_feat in base_features:
        for time_suffix in ["t", "t1", "t2"]:
            extended_features.append(f"{base_feat}_{time_suffix}")

    # 從特徵名稱反推需要的基礎欄位和時間點
    feature_map = {}
    for feat in extended_features:
        # 解析特徵名稱，例如 "price_2nd_t2" -> ("price_2nd", 2)
        parts = feat.rsplit("_", 1)
        base_col = parts[0]
        time_suffix = parts[1]

        if time_suffix == "t":
            offset = 0
        elif time_suffix == "t1":
            offset = 1
        elif time_suffix == "t2":
            offset = 2
        else:
            continue

        feature_map[feat] = (base_col, offset)

    print(f"   V5 特徵策略: 全部數值欄位特徵")
    print(f"   46 個數值欄位 × 3 個時間點 [t, t+1, t+2]")
    print(f"   包含: 訓練數據中的所有數值欄位（排除 timestamp, datetime, reverse）")
    print(f"   總特徵數: {len(extended_features)} 個")

    # 為每個樣本提取擴展特徵
    feature_samples = []

    for i in range(len(t_indices)):
        t_idx = t_indices[i]
        entry_idx = entry_indices[i]  # entry_idx = t_idx + 2
        label = labels[i]

        features = {}

        # 提取擴展特徵
        for feature_name, (base_col, offset) in feature_map.items():
            idx = t_idx + offset
            if idx < len(df):
                features[feature_name] = df.iloc[idx][base_col]

        # 添加標籤和時間戳
        features["label"] = label
        features["timestamp"] = df.iloc[entry_idx]["timestamp"]

        feature_samples.append(features)

    # 轉換為 DataFrame
    features_df = pd.DataFrame(feature_samples)

    print(f"\n✅ 特徵提取完成:")
    print(f"   樣本數: {len(features_df)}")
    print(f"   特徵數: {len(features_df.columns) - 2} 個 (排除 label 和 timestamp)")
    print(f"   擴展特徵組合（8 個指標 × 3 個時間點）")

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

    return features_df


def balance_samples(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    平衡樣本 - 下採樣負樣本使正負樣本 1:1

    Args:
        features_df: 原始特徵 DataFrame

    Returns:
        平衡後的 DataFrame
    """
    print("\n" + "=" * 80)
    print("⚖️  (4.5) 平衡樣本")
    print("=" * 80)

    positive_samples = features_df[features_df["label"] == 1]
    negative_samples = features_df[features_df["label"] == 0]

    print(f"\n原始樣本分佈:")
    print(f"   正樣本: {len(positive_samples)} 個")
    print(f"   負樣本: {len(negative_samples)} 個")
    print(f"   比例: 1:{len(negative_samples)/len(positive_samples):.2f}")

    # 下採樣負樣本
    n_positive = len(positive_samples)
    negative_samples_balanced = negative_samples.sample(n=n_positive, random_state=42)

    # 合併並打亂
    balanced_df = pd.concat([positive_samples, negative_samples_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✅ 平衡後樣本分佈:")
    print(f"   正樣本: {len(positive_samples)} 個 (50.00%)")
    print(f"   負樣本: {len(negative_samples_balanced)} 個 (50.00%)")
    print(f"   總樣本數: {len(balanced_df)} 個")
    print(f"   移除負樣本: {len(negative_samples) - len(negative_samples_balanced)} 個")

    return balanced_df


def train_random_forest(X_train, y_train, X_test, y_test):
    """訓練 Random Forest 模型"""
    print(f"\n🌲 訓練 Random Forest 模型...")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 評估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, test_proba)

    print(f"   訓練集準確率: {train_acc:.4f}")
    print(f"   測試集準確率: {test_acc:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "feature_importance": model.feature_importances_,
    }


def train_lightgbm(X_train, y_train, X_test, y_test):
    """訓練 LightGBM 模型"""
    print(f"\n💡 訓練 LightGBM 模型...")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    # 評估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, test_proba)

    print(f"   訓練集準確率: {train_acc:.4f}")
    print(f"   測試集準確率: {test_acc:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "feature_importance": model.feature_importances_,
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """訓練 XGBoost 模型"""
    print(f"\n🚀 訓練 XGBoost 模型...")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_train, y_train)

    # 評估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, test_proba)

    print(f"   訓練集準確率: {train_acc:.4f}")
    print(f"   測試集準確率: {test_acc:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "feature_importance": model.feature_importances_,
    }


def train_catboost(X_train, y_train, X_test, y_test):
    """訓練 CatBoost 模型"""
    print(f"\n🐱 訓練 CatBoost 模型...")

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
    )

    model.fit(X_train, y_train)

    # 評估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, test_proba)

    print(f"   訓練集準確率: {train_acc:.4f}")
    print(f"   測試集準確率: {test_acc:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "feature_importance": model.feature_importances_,
    }


def train_and_validate(features_df: pd.DataFrame, output_dir: str = "models/reversal"):
    """
    訓練和驗證多個模型，比較性能

    Args:
        features_df: 特徵 DataFrame
        output_dir: 模型輸出目錄
    """
    print("\n" + "=" * 80)
    print("🚀 (5) 訓練和驗證多個模型")
    print("=" * 80)

    # 準備特徵和標籤
    feature_cols = [
        col for col in features_df.columns if col not in ["label", "timestamp"]
    ]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    print(f"\n準備訓練數據:")
    print(f"   特徵數: {len(feature_cols)}")
    print(f"   樣本數: {len(X)}")

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n數據分割:")
    print(f"   訓練集: {len(X_train)} 個樣本")
    print(f"   測試集: {len(X_test)} 個樣本")

    # 訓練所有模型
    print(f"\n" + "=" * 80)
    print(f"🔧 訓練所有模型")
    print(f"=" * 80)

    model_results = {
        "Random Forest": train_random_forest(X_train, y_train, X_test, y_test),
        "LightGBM": train_lightgbm(X_train, y_train, X_test, y_test),
        "XGBoost": train_xgboost(X_train, y_train, X_test, y_test),
        "CatBoost": train_catboost(X_train, y_train, X_test, y_test),
    }

    # 比較結果
    print(f"\n" + "=" * 80)
    print(f"📊 模型比較")
    print(f"=" * 80)

    print(f"\n{'模型':<15} {'訓練準確率':<12} {'測試準確率':<12} {'ROC AUC':<10}")
    print(f"-" * 55)
    for name, result in model_results.items():
        print(
            f"{name:<15} {result['train_acc']:<12.4f} {result['test_acc']:<12.4f} {result['roc_auc']:<10.4f}"
        )

    # 選擇最佳模型（基於測試集 ROC AUC）
    best_model_name = max(model_results, key=lambda x: model_results[x]["roc_auc"])
    best_result = model_results[best_model_name]

    print(f"\n🏆 最佳模型: {best_model_name}")
    print(f"   測試集準確率: {best_result['test_acc']:.4f}")
    print(f"   ROC AUC: {best_result['roc_auc']:.4f}")

    # 使用各模型自己的 Top 20 特徵重新訓練
    print(f"\n" + "=" * 80)
    print(f"🔄 使用各模型自己的 Top 20 特徵重新訓練")
    print(f"=" * 80)

    # 獲取每個模型的 Top 20 特徵
    model_top20_features = {}

    for name, result in model_results.items():
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": result["feature_importance"]}
        ).sort_values("importance", ascending=False)

        top_20 = feature_importance.head(20)["feature"].tolist()
        model_top20_features[name] = top_20

        print(f"\n{name} Top 20 特徵:")
        for i, feat in enumerate(top_20, 1):
            print(f"   {i}. {feat}")

    # 分析共同特徵
    print(f"\n" + "=" * 80)
    print(f"🔍 分析共同特徵")
    print(f"=" * 80)

    rf_set = set(model_top20_features["Random Forest"])
    lgb_set = set(model_top20_features["LightGBM"])
    xgb_set = set(model_top20_features["XGBoost"])
    cat_set = set(model_top20_features["CatBoost"])

    # 四個模型都有的特徵
    common_all = rf_set & lgb_set & xgb_set & cat_set
    # 至少兩個模型有的特徵
    common_two = (
        (rf_set & lgb_set)
        | (rf_set & xgb_set)
        | (rf_set & cat_set)
        | (lgb_set & xgb_set)
        | (lgb_set & cat_set)
        | (xgb_set & cat_set)
    )

    print(f"\n四個模型共同的特徵 ({len(common_all)} 個):")
    for feat in sorted(common_all):
        print(f"   • {feat}")

    print(f"\n至少兩個模型共同的特徵 ({len(common_two)} 個):")
    for feat in sorted(common_two):
        models = []
        if feat in rf_set:
            models.append("RF")
        if feat in lgb_set:
            models.append("LGB")
        if feat in xgb_set:
            models.append("XGB")
        if feat in cat_set:
            models.append("CAT")
        print(f"   • {feat} ({'+'.join(models)})")

    # 使用各自的 Top 20 特徵重新訓練
    print(f"\n" + "=" * 80)
    print(f"🔧 使用各自 Top 20 特徵重新訓練")
    print(f"=" * 80)

    top20_results = {}

    for name in model_results.keys():
        print(f"\n訓練 {name}...")
        top_20 = model_top20_features[name]
        top_20_indices = [feature_cols.index(f) for f in top_20]

        X_train_top20 = X_train[:, top_20_indices]
        X_test_top20 = X_test[:, top_20_indices]

        if name == "Random Forest":
            result = train_random_forest(X_train_top20, y_train, X_test_top20, y_test)
        elif name == "LightGBM":
            result = train_lightgbm(X_train_top20, y_train, X_test_top20, y_test)
        elif name == "XGBoost":
            result = train_xgboost(X_train_top20, y_train, X_test_top20, y_test)
        else:  # CatBoost
            result = train_catboost(X_train_top20, y_train, X_test_top20, y_test)

        result["features"] = top_20
        top20_results[name] = result

    print(f"\n📊 各模型使用自己 Top 20 特徵的表現:")
    print(f"\n{'模型':<15} {'測試準確率':<12} {'ROC AUC':<10}")
    print(f"-" * 40)
    for name, result in top20_results.items():
        print(f"{name:<15} {result['test_acc']:<12.4f} {result['roc_auc']:<10.4f}")

    # 選擇最佳 Top20 模型
    best_top20_name = max(top20_results, key=lambda x: top20_results[x]["roc_auc"])
    best_top20_result = top20_results[best_top20_name]
    best_top20_model = best_top20_result["model"]
    best_top20_auc = best_top20_result["roc_auc"]
    best_top20_features = best_top20_result["features"]

    print(f"\n🏆 最佳 Top20 模型: {best_top20_name}")
    print(f"   測試準確率: {best_top20_result['test_acc']:.4f}")
    print(f"   ROC AUC: {best_top20_auc:.4f}")

    # 詳細評估最佳模型
    best_top20_indices = [feature_cols.index(f) for f in best_top20_features]
    X_test_best = X_test[:, best_top20_indices]
    y_test_pred_best = best_top20_model.predict(X_test_best)

    print(f"\n分類報告:")
    print(
        classification_report(y_test, y_test_pred_best, target_names=["未反轉", "反轉"])
    )

    cm_best = confusion_matrix(y_test, y_test_pred_best)
    print(f"\n混淆矩陣:")
    print(f"              預測未反轉  預測反轉")
    print(f"實際未反轉    {cm_best[0][0]:6d}      {cm_best[0][1]:6d}")
    print(f"實際反轉      {cm_best[1][0]:6d}      {cm_best[1][1]:6d}")

    # 保存所有三個 Top20 模型
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 保存所有 Top20 模型...")

    for name, result in top20_results.items():
        if name == "Random Forest":
            model_path = os.path.join(output_dir, "reversal_model_v5_rf_top12.pkl")
        elif name == "LightGBM":
            model_path = os.path.join(output_dir, "reversal_model_v5_lgb_top12.pkl")
        elif name == "XGBoost":
            model_path = os.path.join(output_dir, "reversal_model_v5_xgb_top12.pkl")
        else:  # CatBoost
            model_path = os.path.join(output_dir, "reversal_model_v5_cat_top12.pkl")

        joblib.dump(
            {
                "model": result["model"],
                "model_name": name,
                "feature_columns": result["features"],
                "test_accuracy": result["test_acc"],
                "roc_auc": result["roc_auc"],
            },
            model_path,
        )

        print(f"   ✅ {name}: {model_path}")

    # 額外保存最佳模型（向後兼容）
    best_model_path = os.path.join(output_dir, "reversal_model_v5_top12.pkl")

    joblib.dump(
        {
            "model": best_top20_model,
            "model_name": best_top20_name,
            "feature_columns": best_top20_features,
            "test_accuracy": best_top20_result["test_acc"],
            "roc_auc": best_top20_auc,
            "common_features": {
                "all_three": sorted(common_all),
                "at_least_two": sorted(common_two),
            },
        },
        best_model_path,
    )

    print(f"\n✅ 最佳模型已保存: {best_model_path}")
    print(f"   模型類型: {best_top20_name}")
    print(f"   特徵數: {len(best_top20_features)}")
    print(f"   三模型共同特徵: {len(common_all)} 個")
    print(f"   至少兩模型共同特徵: {len(common_two)} 個")

    # 測試使用每個模型的 Top 5 + 2 個共同特徵
    print(f"\n" + "=" * 80)
    print(f"🎯 測試使用每個模型的 Top 5 + 2 個共同特徵")
    print(f"=" * 80)

    # 提取每個模型的 Top 5
    top5_union = set()
    top5_union.add("amplitude_t2")  # 共同特徵 1
    top5_union.add("price_2nd_t1")  # 共同特徵 2

    for name in model_results.keys():
        top_5 = model_top20_features[name][:5]
        print(f"\n{name} Top 5:")
        for i, feat in enumerate(top_5, 1):
            print(f"   {i}. {feat}")
            top5_union.add(feat)

    top5_features = sorted(list(top5_union))
    print(f"\n合併後的特徵集 ({len(top5_features)} 個):")
    for i, feat in enumerate(top5_features, 1):
        print(f"   {i}. {feat}")

    top5_indices = [feature_cols.index(f) for f in top5_features]
    X_train_top5 = X_train[:, top5_indices]
    X_test_top5 = X_test[:, top5_indices]

    print(f"\n訓練所有模型...")

    top5_results = {
        "Random Forest": train_random_forest(
            X_train_top5, y_train, X_test_top5, y_test
        ),
        "LightGBM": train_lightgbm(X_train_top5, y_train, X_test_top5, y_test),
        "XGBoost": train_xgboost(X_train_top5, y_train, X_test_top5, y_test),
        "CatBoost": train_catboost(X_train_top5, y_train, X_test_top5, y_test),
    }

    print(f"\n📊 Top 5 聯合特徵模型比較:")
    print(f"\n{'模型':<15} {'測試準確率':<12} {'ROC AUC':<10}")
    print(f"-" * 40)
    for name, result in top5_results.items():
        print(f"{name:<15} {result['test_acc']:<12.4f} {result['roc_auc']:<10.4f}")

    best_top5_name = max(top5_results, key=lambda x: top5_results[x]["roc_auc"])
    best_top5_result = top5_results[best_top5_name]

    print(f"\n🏆 最佳 Top 5 聯合模型: {best_top5_name}")
    print(f"   測試準確率: {best_top5_result['test_acc']:.4f}")
    print(f"   ROC AUC: {best_top5_result['roc_auc']:.4f}")

    # 保存 Top 5 聯合模型
    model_path_top5 = os.path.join(output_dir, "reversal_model_v5_top5_union.pkl")

    joblib.dump(
        {
            "model": best_top5_result["model"],
            "model_name": best_top5_name,
            "feature_columns": top5_features,
            "test_accuracy": best_top5_result["test_acc"],
            "roc_auc": best_top5_result["roc_auc"],
        },
        model_path_top5,
    )

    print(f"\n✅ Top 5 聯合模型已保存: {model_path_top5}")
    print(f"   模型類型: {best_top5_name}")
    print(f"   特徵數: {len(top5_features)}")

    # 測試使用 Random Forest Top 20 + 額外特徵
    print(f"\n" + "=" * 80)
    print(f"🎯 測試使用 Random Forest Top 20 + 額外特徵")
    print(f"=" * 80)

    # Random Forest Top 20
    rf_top20 = model_top20_features["Random Forest"]

    # 額外特徵
    extra_features = [
        "amplitude_t",
        "amplitude_t1",
        "amplitude_t2",
        "price_2nd_t",
        "price_2nd_t1",
        "avg_amplitude_t2",
        "atr_2nd_t2",
        "macd_2nd_t2",
        "lis_2nd_t",
        "volume_change_t1",
        "bb_middle_t",
        "ema_50_t2",
        "sar_t1",
        "price_above_sar_t",
    ]

    # 合併並去重
    rf_extended = list(set(rf_top20 + extra_features))
    rf_extended.sort()

    print(f"\nRandom Forest Top 20 特徵數: {len(rf_top20)}")
    print(f"額外特徵數: {len(extra_features)}")
    print(f"合併後特徵數: {len(rf_extended)} 個")

    print(f"\n合併後的特徵:")
    for i, feat in enumerate(rf_extended, 1):
        print(f"   {i}. {feat}")

    rf_extended_indices = [feature_cols.index(f) for f in rf_extended]
    X_train_rf_ext = X_train[:, rf_extended_indices]
    X_test_rf_ext = X_test[:, rf_extended_indices]

    print(f"\n訓練所有模型...")

    rf_ext_results = {
        "Random Forest": train_random_forest(
            X_train_rf_ext, y_train, X_test_rf_ext, y_test
        ),
        "LightGBM": train_lightgbm(X_train_rf_ext, y_train, X_test_rf_ext, y_test),
        "XGBoost": train_xgboost(X_train_rf_ext, y_train, X_test_rf_ext, y_test),
        "CatBoost": train_catboost(X_train_rf_ext, y_train, X_test_rf_ext, y_test),
    }

    print(f"\n📊 RF Top 20 + 額外特徵模型比較:")
    print(f"\n{'模型':<15} {'測試準確率':<12} {'ROC AUC':<10}")
    print(f"-" * 40)
    for name, result in rf_ext_results.items():
        print(f"{name:<15} {result['test_acc']:<12.4f} {result['roc_auc']:<10.4f}")

    best_rf_ext_name = max(rf_ext_results, key=lambda x: rf_ext_results[x]["roc_auc"])
    best_rf_ext_result = rf_ext_results[best_rf_ext_name]

    print(f"\n🏆 最佳 RF 擴展模型: {best_rf_ext_name}")
    print(f"   測試準確率: {best_rf_ext_result['test_acc']:.4f}")
    print(f"   ROC AUC: {best_rf_ext_result['roc_auc']:.4f}")

    # 保存 RF 擴展模型
    model_path_rf_ext = os.path.join(output_dir, "reversal_model_v5_rf_extended.pkl")

    joblib.dump(
        {
            "model": best_rf_ext_result["model"],
            "model_name": best_rf_ext_name,
            "feature_columns": rf_extended,
            "test_accuracy": best_rf_ext_result["test_acc"],
            "roc_auc": best_rf_ext_result["roc_auc"],
        },
        model_path_rf_ext,
    )

    print(f"\n✅ RF 擴展模型已保存: {model_path_rf_ext}")
    print(f"   模型類型: {best_rf_ext_name}")
    print(f"   特徵數: {len(rf_extended)}")

    # 從 RF 擴展模型中選擇 Top 16
    print(f"\n" + "=" * 80)
    print(f"🎯 從 RF 擴展模型中選擇 Top 16 特徵")
    print(f"=" * 80)

    # 獲取 RF 擴展模型的特徵重要性
    rf_ext_model = rf_ext_results["Random Forest"]["model"]
    rf_ext_importance = rf_ext_model.feature_importances_

    # 創建特徵重要性 DataFrame
    rf_ext_feat_importance = pd.DataFrame(
        {"feature": rf_extended, "importance": rf_ext_importance}
    ).sort_values("importance", ascending=False)

    # 選擇 Top 16
    rf_top16 = rf_ext_feat_importance.head(16)["feature"].tolist()

    print(f"\nRF 擴展模型 Top 16 特徵:")
    for i, feat in enumerate(rf_top16, 1):
        importance = rf_ext_feat_importance[rf_ext_feat_importance["feature"] == feat][
            "importance"
        ].values[0]
        print(f"   {i}. {feat} (重要性: {importance:.4f})")

    rf_top16_indices = [feature_cols.index(f) for f in rf_top16]
    X_train_rf_top16 = X_train[:, rf_top16_indices]
    X_test_rf_top16 = X_test[:, rf_top16_indices]

    print(f"\n訓練所有模型...")

    rf_top16_results = {
        "Random Forest": train_random_forest(
            X_train_rf_top16, y_train, X_test_rf_top16, y_test
        ),
        "LightGBM": train_lightgbm(X_train_rf_top16, y_train, X_test_rf_top16, y_test),
        "XGBoost": train_xgboost(X_train_rf_top16, y_train, X_test_rf_top16, y_test),
        "CatBoost": train_catboost(X_train_rf_top16, y_train, X_test_rf_top16, y_test),
    }

    print(f"\n📊 RF Top 16 特徵模型比較:")
    print(f"\n{'模型':<15} {'測試準確率':<12} {'ROC AUC':<10}")
    print(f"-" * 40)
    for name, result in rf_top16_results.items():
        print(f"{name:<15} {result['test_acc']:<12.4f} {result['roc_auc']:<10.4f}")

    best_rf_top16_name = max(
        rf_top16_results, key=lambda x: rf_top16_results[x]["roc_auc"]
    )
    best_rf_top16_result = rf_top16_results[best_rf_top16_name]

    print(f"\n🏆 最佳 RF Top 16 模型: {best_rf_top16_name}")
    print(f"   測試準確率: {best_rf_top16_result['test_acc']:.4f}")
    print(f"   ROC AUC: {best_rf_top16_result['roc_auc']:.4f}")

    # 保存 RF Top 16 模型
    model_path_rf_top16 = os.path.join(output_dir, "reversal_model_v5_rf_top16.pkl")

    joblib.dump(
        {
            "model": best_rf_top16_result["model"],
            "model_name": best_rf_top16_name,
            "feature_columns": rf_top16,
            "test_accuracy": best_rf_top16_result["test_acc"],
            "roc_auc": best_rf_top16_result["roc_auc"],
        },
        model_path_rf_top16,
    )

    print(f"\n✅ RF Top 16 模型已保存: {model_path_rf_top16}")
    print(f"   模型類型: {best_rf_top16_name}")
    print(f"   特徵數: {len(rf_top16)}")

    # 測試使用核心共同特徵
    print(f"\n" + "=" * 80)
    print(f"🎯 測試使用核心共同特徵")
    print(f"=" * 80)

    core_features = sorted(common_all)

    if len(core_features) == 0:
        print(f"\n⚠️  四個模型沒有共同特徵，跳過核心特徵測試")
    else:
        core_indices = [feature_cols.index(f) for f in core_features]

        X_train_core = X_train[:, core_indices]
        X_test_core = X_test[:, core_indices]

        print(f"\n核心特徵:")
        for i, feat in enumerate(core_features, 1):
            print(f"   {i}. {feat}")

        print(f"\n訓練所有模型...")

        core_results = {
            "Random Forest": train_random_forest(
                X_train_core, y_train, X_test_core, y_test
            ),
            "LightGBM": train_lightgbm(X_train_core, y_train, X_test_core, y_test),
            "XGBoost": train_xgboost(X_train_core, y_train, X_test_core, y_test),
            "CatBoost": train_catboost(X_train_core, y_train, X_test_core, y_test),
        }

        print(f"\n📊 核心特徵模型比較:")
        print(f"\n{'模型':<15} {'測試準確率':<12} {'ROC AUC':<10}")
        print(f"-" * 40)
        for name, result in core_results.items():
            print(f"{name:<15} {result['test_acc']:<12.4f} {result['roc_auc']:<10.4f}")

        best_core_name = max(core_results, key=lambda x: core_results[x]["roc_auc"])
        best_core_result = core_results[best_core_name]

        print(f"\n🏆 最佳核心特徵模型: {best_core_name}")
        print(f"   測試準確率: {best_core_result['test_acc']:.4f}")
        print(f"   ROC AUC: {best_core_result['roc_auc']:.4f}")

        # 保存核心特徵模型
        model_path_core = os.path.join(output_dir, "reversal_model_v5_core9.pkl")

        joblib.dump(
            {
                "model": best_core_result["model"],
                "model_name": best_core_name,
                "feature_columns": core_features,
                "test_accuracy": best_core_result["test_acc"],
                "roc_auc": best_core_result["roc_auc"],
            },
            model_path_core,
        )

        print(f"\n✅ 核心特徵模型已保存: {model_path_core}")
        print(f"   模型類型: {best_core_name}")
        print(f"   特徵數: {len(core_features)}")

    # 總結對比
    print(f"\n" + "=" * 80)
    print(f"📊 最終總結")
    print(f"=" * 80)
    print(f"\n{'配置':<20} {'最佳模型':<15} {'準確率':<10} {'ROC AUC':<10}")
    print(f"-" * 60)
    print(
        f"{'全特徵 (82個)':<20} {best_model_name:<15} {best_result['test_acc']:<10.4f} {best_result['roc_auc']:<10.4f}"
    )
    print(
        f"{'各自Top20':<20} {best_top20_name:<15} {best_top20_result['test_acc']:<10.4f} {best_top20_auc:<10.4f}"
    )
    if len(core_features) > 0:
        print(
            f"{'核心8特徵':<20} {best_core_name:<15} {best_core_result['test_acc']:<10.4f} {best_core_result['roc_auc']:<10.4f}"
        )


def main():
    """主訓練流程"""
    print("=" * 80)
    print("🤖 ML Reversal V4 模型訓練 - 多模型比較")
    print("=" * 80)

    # (1) 載入數據
    data_file = "training_data/reversal_training_data_2023-01-01_to_2025-01-01_15m.csv"
    df = load_data(data_file, year=2024)

    # (2) 篩選候選樣本 - 找出極值點
    peaks, valleys = filter_candidates(df)

    # (3) 生成標籤 - 基於入場點和止損止盈
    t_indices, entry_indices, labels = generate_labels(df, peaks, valleys)

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
