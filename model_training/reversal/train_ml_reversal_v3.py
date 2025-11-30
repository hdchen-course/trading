#!/usr/bin/env python3
"""
ML Reversal V3 模型訓練腳本

基於 V2 的改進版本，使用更完整的訓練數據

改進：
- 使用 XGBoost 替代 Random Forest（更好的性能）
- 預測時間：1 小時（快速短期反轉）
- 使用 2024 年完整數據訓練
- Top 15 特徵選擇
"""

import sys
import os

# 必須在其他 import 之前修改 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
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
    參數：window=10, prominence_factor=0

    Args:
        df: 原始數據

    Returns:
        候選反轉點索引列表
    """
    print("\n" + "=" * 80)
    print("🎯 (2) 篩選候選樣本 - 局部極值檢測")
    print("=" * 80)

    print(f"\n使用 find_local_extremes:")
    print(f"   window = 10")
    print(f"   prominence_factor = 0")

    # 使用 find_local_extremes 找出極值點
    prices = df["close"].values
    extremes, peaks, valleys = find_local_extremes(
        prices, window=10, prominence_factor=0
    )

    # 極值點索引（peaks 和 valleys 的聯集）
    extreme_indices = sorted(peaks | valleys)

    print(f"\n✅ 找到極值點:")
    print(f"   峰值點: {len(peaks)} 個")
    print(f"   谷值點: {len(valleys)} 個")
    print(
        f"   總極值點: {len(extreme_indices)} 個 ({len(extreme_indices)/len(df)*100:.2f}%)"
    )

    return extreme_indices


def generate_labels(df: pd.DataFrame, extreme_indices: list) -> tuple:
    """
    生成訓練標籤 - 基於入場點和止損/止盈邏輯

    邏輯：
    1. 反轉點 [t] 已由 find_local_extremes 找出
    2. 入場點為 [t+1] 或 [t+2]
    3. 止損：0.18%
    4. 止盈：2.5小時內獲利 >= 0.4%
    5. 正樣本：達到止盈
    6. 負樣本：觸及止損
    7. 排除：既沒達止盈也沒觸止損的樣本（不確定的情況）

    Args:
        df: 完整數據
        extreme_indices: 極值點索引列表

    Returns:
        (entry_indices, labels) 元組
    """
    print("\n" + "=" * 80)
    print("🏷️  (3) 生成標籤 - 入場點 + 止損止盈")
    print("=" * 80)

    print(f"\n標籤邏輯:")
    print(f"   反轉點: [t] (由 find_local_extremes 找出)")
    print(f"   入場點: [t+1] 或 [t+2]")
    print(f"   止損: 0.18%")
    print(f"   止盈: 2.5小時內 >= 0.4% (10根15分鐘K線)")
    print(f"   正樣本: 達到止盈")
    print(f"   負樣本: 觸及止損")
    print(f"   排除: 既沒達止盈也沒觸止損（不確定的情況）")

    labels = []
    t_indices = []  # 極值點索引
    entry_indices = []  # 入場點索引
    excluded_count = 0  # 統計被排除的樣本數

    stop_loss_pct = 0.18
    take_profit_pct = 0.4  # 降低止盈目標，更容易達到
    max_holding_bars = 10  # 2.5小時 = 10根15分鐘K線

    for t_idx in extreme_indices:
        # 確保有足夠的未來數據
        if t_idx + 2 + max_holding_bars >= len(df):
            continue

        # 判斷是峰值還是谷值
        t_price = df.iloc[t_idx]["close"]

        # 嘗試 t+1 和 t+2 作為入場點
        for offset in [1, 2]:
            entry_idx = t_idx + offset
            entry_price = df.iloc[entry_idx]["close"]

            # 判斷交易方向（峰值做空，谷值做多）
            if t_idx > 0 and t_idx < len(df) - 1:
                prev_price = df.iloc[t_idx - 1]["close"]
                next_price = df.iloc[t_idx + 1]["close"]
                is_peak = t_price > prev_price and t_price > next_price
            else:
                continue

            # 模擬交易
            hit_stop_loss = False
            hit_take_profit = False

            for i in range(1, max_holding_bars + 1):
                future_idx = entry_idx + i
                if future_idx >= len(df):
                    break

                future_price = df.iloc[future_idx]["close"]

                if is_peak:  # 做空
                    pnl_pct = (entry_price - future_price) / entry_price * 100
                else:  # 做多
                    pnl_pct = (future_price - entry_price) / entry_price * 100

                # 檢查止損
                if pnl_pct <= -stop_loss_pct:
                    hit_stop_loss = True
                    break

                # 檢查止盈
                if pnl_pct >= take_profit_pct:
                    hit_take_profit = True
                    break

            # 生成標籤 - 只保留明確的正負樣本
            if hit_take_profit:
                label = 1  # 正樣本：達到止盈
                labels.append(label)
                t_indices.append(t_idx)
                entry_indices.append(entry_idx)
            elif hit_stop_loss:
                label = 0  # 負樣本：觸及止損
                labels.append(label)
                t_indices.append(t_idx)
                entry_indices.append(entry_idx)
            else:
                # 既沒達止盈也沒觸止損 -> 排除
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
        idx = entry_indices[i]
        label = labels[i]

        entry_price = df.iloc[idx]["close"]
        timestamp = df.iloc[idx]["timestamp"]

        # 找到對應的極值點
        t_idx = idx - 1 if idx - 1 in extreme_indices else idx - 2
        t_price = df.iloc[t_idx]["close"]

        # 判斷方向
        is_peak = t_price > entry_price
        direction = "做空" if is_peak else "做多"
        result = "✅ 達到止盈" if label == 1 else "❌ 觸及止損"

        print(
            f"   {i+1}. {timestamp} | {direction} | 入場價: {entry_price:.2f} | {result}"
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

    # 排除不用於訓練的欄位
    exclude_columns = [
        "timestamp",
        "datetime",
        "period",
        "reverse",
        "label",
        "price_change_abs",
    ]

    # 獲取所有可用的特徵欄位
    feature_columns_base = [col for col in df.columns if col not in exclude_columns]

    print(f"   基礎特徵欄位: {len(feature_columns_base)} 個")

    # 為每個樣本提取 [t] 和 [entry] 的特徵
    feature_samples = []

    for i in range(len(t_indices)):
        t_idx = t_indices[i]
        entry_idx = entry_indices[i]
        label = labels[i]

        features = {}

        # [t] 的特徵（極值點）
        for col in feature_columns_base:
            features[f"{col}_at_t"] = df.iloc[t_idx][col]

        # [entry] 的特徵（入場點）
        for col in feature_columns_base:
            features[f"{col}_at_entry"] = df.iloc[entry_idx][col]

        # 添加標籤和時間戳
        features["label"] = label
        features["timestamp"] = df.iloc[entry_idx]["timestamp"]

        feature_samples.append(features)

    # 轉換為 DataFrame
    features_df = pd.DataFrame(feature_samples)

    print(f"\n✅ 特徵提取完成:")
    print(f"   樣本數: {len(features_df)}")
    print(f"   特徵數: {len(features_df.columns) - 2} 個 (排除 label 和 timestamp)")
    print(f"   [t] 特徵: {len(feature_columns_base)} 個")
    print(f"   [entry] 特徵: {len(feature_columns_base)} 個")

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


def train_and_validate(features_df: pd.DataFrame, output_dir: str = "models/reversal"):
    """
    訓練和驗證模型

    Args:
        features_df: 特徵 DataFrame
        output_dir: 模型輸出目錄
    """
    print("\n" + "=" * 80)
    print("🚀 (5) 訓練和驗證模型")
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

    # 訓練 XGBoost 模型
    print(f"\n🔧 訓練 XGBoost 模型...")

    # 參數調整 4: 添加樣本和特徵採樣
    model = xgb.XGBClassifier(
        n_estimators=300,  # 更多樹
        max_depth=8,  # 更深的樹
        min_child_weight=1,  # 減少過擬合
        learning_rate=0.05,  # 更慢但更穩定
        reg_alpha=0.1,  # L1 正則化
        reg_lambda=1.0,  # L2 正則化
        gamma=0.1,  # 分裂所需最小損失減少
        subsample=0.8,  # 每棵樹使用 80% 樣本
        colsample_bytree=0.8,  # 每棵樹使用 80% 特徵
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    print(f"✅ 模型訓練完成")

    # 評估模型
    print(f"\n📊 模型評估:")

    # 訓練集評估
    y_train_pred = model.predict(X_train)
    train_accuracy = (y_train_pred == y_train).mean()
    print(f"\n訓練集準確率: {train_accuracy:.4f}")

    # 測試集評估
    y_test_pred = model.predict(X_test)
    test_accuracy = (y_test_pred == y_test).mean()
    print(f"測試集準確率: {test_accuracy:.4f}")

    # 詳細報告
    print(f"\n分類報告:")
    print(classification_report(y_test, y_test_pred, target_names=["未反轉", "反轉"]))

    # 混淆矩陣
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n混淆矩陣:")
    print(f"              預測未反轉  預測反轉")
    print(f"實際未反轉    {cm[0][0]:6d}      {cm[0][1]:6d}")
    print(f"實際反轉      {cm[1][0]:6d}      {cm[1][1]:6d}")

    # ROC AUC
    y_test_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f"\nROC AUC: {roc_auc:.4f}")

    # 特徵重要性
    print(f"\n📊 特徵重要性分析:")
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nTop 20 重要特徵:")
    for i, row in feature_importance.head(20).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

    # 使用 Top 15 特徵重新訓練
    print(f"\n" + "=" * 80)
    print(f"🔄 使用 Top 15 特徵重新訓練")
    print(f"=" * 80)

    top_15_features = feature_importance.head(15)["feature"].tolist()
    top_15_indices = [feature_cols.index(f) for f in top_15_features]

    X_train_top15 = X_train[:, top_15_indices]
    X_test_top15 = X_test[:, top_15_indices]

    print(f"\n特徵數: {len(top_15_features)} 個")
    print(f"Top 15 特徵:")
    for i, feat in enumerate(top_15_features, 1):
        print(f"   {i}. {feat}")

    # 重新訓練
    model_top15 = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        min_child_weight=1,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model_top15.fit(X_train_top15, y_train)

    # 評估 Top 15 模型
    y_train_pred_top15 = model_top15.predict(X_train_top15)
    y_test_pred_top15 = model_top15.predict(X_test_top15)

    train_accuracy_top15 = (y_train_pred_top15 == y_train).mean()
    test_accuracy_top15 = (y_test_pred_top15 == y_test).mean()

    print(f"\n📊 Top 15 特徵模型評估:")
    print(f"訓練集準確率: {train_accuracy_top15:.4f}")
    print(f"測試集準確率: {test_accuracy_top15:.4f}")

    print(f"\n分類報告:")
    print(
        classification_report(
            y_test, y_test_pred_top15, target_names=["未反轉", "反轉"]
        )
    )

    cm_top15 = confusion_matrix(y_test, y_test_pred_top15)
    print(f"\n混淆矩陣:")
    print(f"              預測未反轉  預測反轉")
    print(f"實際未反轉    {cm_top15[0][0]:6d}      {cm_top15[0][1]:6d}")
    print(f"實際反轉      {cm_top15[1][0]:6d}      {cm_top15[1][1]:6d}")

    y_test_proba_top15 = model_top15.predict_proba(X_test_top15)[:, 1]
    roc_auc_top15 = roc_auc_score(y_test, y_test_proba_top15)
    print(f"\nROC AUC: {roc_auc_top15:.4f}")

    # 保存 Top 15 特徵模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "reversal_model_v3_top15.pkl")

    joblib.dump(
        {
            "model": model_top15,
            "feature_columns": top_15_features,
            "feature_importance": feature_importance.head(15),
        },
        model_path,
    )

    print(f"\n✅ Top 15 特徵模型已保存: {model_path}")


def main():
    """主訓練流程"""
    print("=" * 80)
    print("🤖 ML Reversal V3 模型訓練")
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
