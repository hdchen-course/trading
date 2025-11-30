#!/usr/bin/env python3
"""
MACD V1 模型訓練腳本

基於 MACD 黃金交叉/死亡交叉的機器學習模型

訓練邏輯：
- 檢測 MACD 交叉點（macd_2nd 符號改變）
- 入場點：[t+1]（交叉後一根K線）
- 止損：1小時內 -0.18%
- 止盈：2.5小時內 >= 0.4%
- 使用 XGBoost 模型
- Top 15 特徵選擇
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
)
import joblib  # noqa: E402


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
    篩選候選樣本 - 檢測 MACD 交叉點

    條件：
    1. MACD 線和信號線的黃金交叉/死亡交叉
    2. 黃金交叉: EMA100 > EMA50
    3. 死亡交叉: EMA100 < EMA50

    Args:
        df: 原始數據（必須包含 macd, macd_signal, ema_50, ema_100 欄位）

    Returns:
        MACD 交叉點索引列表
    """
    print("\n" + "=" * 80)
    print("🎯 (2) 篩選候選樣本 - MACD 交叉點檢測")
    print("=" * 80)

    print(f"\n檢測邏輯:")
    print(f"   條件 1: 檢測 MACD 黃金交叉/死亡交叉")
    print(f"           黃金交叉: MACD 線從下方穿越信號線")
    print(f"           死亡交叉: MACD 線從上方穿越信號線")
    print(f"   條件 2: 黃金交叉時 EMA100 > EMA50")
    print(f"   條件 3: 死亡交叉時 EMA100 < EMA50")

    # 檢查必要欄位
    required_cols = ["macd", "macd_signal", "ema_50", "ema_100"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"數據中缺少 {col} 欄位，請先計算技術指標")

    cross_indices = []
    golden_crosses = 0  # 黃金交叉計數
    death_crosses = 0  # 死亡交叉計數
    filtered_by_ema = 0  # 被 EMA 條件過濾掉的數量

    # 檢測交叉點
    for i in range(len(df) - 1):
        macd_t = df.iloc[i]["macd"]
        macd_t1 = df.iloc[i + 1]["macd"]
        signal_t = df.iloc[i]["macd_signal"]
        signal_t1 = df.iloc[i + 1]["macd_signal"]
        ema_50_t = df.iloc[i]["ema_50"]
        ema_100_t = df.iloc[i]["ema_100"]

        # 跳過 NaN 值
        if (
            pd.isna(macd_t)
            or pd.isna(macd_t1)
            or pd.isna(signal_t)
            or pd.isna(signal_t1)
            or pd.isna(ema_50_t)
            or pd.isna(ema_100_t)
        ):
            continue

        # 檢測 MACD 交叉
        # 黃金交叉: MACD 從下方穿越信號線 (macd[t] < signal[t] 且 macd[t+1] > signal[t+1])
        # 死亡交叉: MACD 從上方穿越信號線 (macd[t] > signal[t] 且 macd[t+1] < signal[t+1])
        is_golden_cross = (macd_t < signal_t) and (macd_t1 > signal_t1)
        is_death_cross = (macd_t > signal_t) and (macd_t1 < signal_t1)

        if is_golden_cross or is_death_cross:
            # 黃金交叉: EMA100 > EMA50
            if is_golden_cross:
                if ema_100_t > ema_50_t:
                    cross_indices.append(i)
                    golden_crosses += 1
                else:
                    filtered_by_ema += 1

            # 死亡交叉: EMA100 < EMA50
            elif is_death_cross:
                if ema_100_t < ema_50_t:
                    cross_indices.append(i)
                    death_crosses += 1
                else:
                    filtered_by_ema += 1

    total_crosses = golden_crosses + death_crosses
    print(f"\n✅ 找到交叉點:")
    print(f"   黃金交叉: {golden_crosses} 個")
    print(f"   死亡交叉: {death_crosses} 個")
    print(f"   總交叉點: {total_crosses} 個 ({total_crosses/len(df)*100:.2f}%)")
    print(f"   被 EMA 條件過濾: {filtered_by_ema} 個")

    return cross_indices


def generate_labels(df: pd.DataFrame, cross_indices: list) -> tuple:
    """
    生成訓練標籤 - 基於 MACD 交叉點和止損/止盈邏輯

    邏輯：
    1. 交叉點 [t] 由 filter_candidates 找出
    2. 入場點為 [t+1]
    3. 止損：1小時內 -0.18% (4根15分鐘K線)
    4. 止盈：2.5小時內 >= 0.4% (10根15分鐘K線)
    5. 正樣本：達到止盈
    6. 負樣本：觸及止損
    7. 排除：既沒達止盈也沒觸止損的樣本（不確定的情況）

    Args:
        df: 完整數據
        cross_indices: MACD 交叉點索引列表

    Returns:
        (t_indices, entry_indices, labels) 元組
    """
    print("\n" + "=" * 80)
    print("🏷️  (3) 生成標籤 - MACD 交叉點 + 止損止盈")
    print("=" * 80)

    print(f"\n標籤邏輯:")
    print(f"   交叉點: [t] (MACD 黃金交叉/死亡交叉)")
    print(f"   入場點: [t+1] (交叉後一根K線)")
    print(f"   止損: 1小時內 -0.18% (4根15分鐘K線)")
    print(f"   止盈: 2.5小時內 >= 0.4% (10根15分鐘K線)")
    print(f"   正樣本: 達到止盈")
    print(f"   負樣本: 觸及止損")
    print(f"   排除: 既沒達止盈也沒觸止損（不確定的情況）")

    labels = []
    t_indices = []  # 交叉點索引
    entry_indices = []  # 入場點索引
    excluded_count = 0  # 統計被排除的樣本數

    stop_loss_pct = 0.18
    take_profit_pct = 0.4
    stop_loss_window = 4  # 1小時 (4根15分鐘K線)
    take_profit_window = 10  # 2.5小時 (10根15分鐘K線)

    for t_idx in cross_indices:
        # 入場點為 t+1
        entry_idx = t_idx + 1

        # 確保有足夠的未來數據
        if entry_idx + take_profit_window >= len(df):
            continue

        entry_price = df.iloc[entry_idx]["close"]

        # 先檢查1小時內的止損
        hit_stop_loss = False
        for i in range(1, stop_loss_window + 1):
            future_idx = entry_idx + i
            if future_idx >= len(df):
                break

            future_price = df.iloc[future_idx]["close"]
            pnl_pct = (future_price - entry_price) / entry_price * 100

            # 檢查止損（負向變化）
            if pnl_pct <= -stop_loss_pct:
                hit_stop_loss = True
                break

        # 如果沒有止損，檢查2.5小時內的止盈
        hit_take_profit = False
        if not hit_stop_loss:
            for i in range(1, take_profit_window + 1):
                future_idx = entry_idx + i
                if future_idx >= len(df):
                    break

                future_price = df.iloc[future_idx]["close"]
                pnl_pct = (future_price - entry_price) / entry_price * 100

                # 檢查止盈（正向變化）
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

        result = "✅ 達到止盈" if label == 1 else "❌ 觸及止損"

        print(f"   {i+1}. {timestamp} | 入場價: {entry_price:.2f} | {result}")

    return t_indices, entry_indices, labels


def extract_features(
    df: pd.DataFrame, t_indices: list, entry_indices: list, labels: list
) -> pd.DataFrame:
    """
    提取特徵：組合 [t] 和 [entry] 的特徵

    Args:
        df: 完整數據
        t_indices: MACD 交叉點索引列表
        entry_indices: 入場點索引列表
        labels: 標籤列表

    Returns:
        特徵 DataFrame
    """
    print("\n" + "=" * 80)
    print("🔧 (4) 提取特徵")
    print("=" * 80)

    print(f"\n策略: 組合 [t] 和 [entry] 的所有欄位作為特徵")
    print(f"   [t]: MACD 交叉點")
    print(f"   [entry]: 入場點 (t+1)")

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


def train_and_validate(features_df: pd.DataFrame):
    """
    訓練和驗證模型 - 使用 Random Forest 和全部 82 個特徵

    Args:
        features_df: 特徵 DataFrame
    """
    print("\n" + "=" * 80)
    print("🚀 (5) 訓練和驗證模型 - Random Forest")
    print("=" * 80)

    # 準備特徵和標籤
    feature_cols = [
        col for col in features_df.columns if col not in ["label", "timestamp"]
    ]
    X = features_df[feature_cols].values
    y = features_df["label"].values

    print(f"\n準備訓練數據:")
    print(f"   特徵數: {len(feature_cols)} 個（全部特徵）")
    print(f"   樣本數: {len(X)}")

    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n數據分割:")
    print(f"   訓練集: {len(X_train)} 個樣本")
    print(f"   測試集: {len(X_test)} 個樣本")

    # 訓練 Random Forest 模型
    print(f"\n🔧 訓練 Random Forest 模型（使用全部 {len(feature_cols)} 個特徵）...")

    model = RandomForestClassifier(
        n_estimators=200,  # 200 棵樹
        max_depth=15,  # 最大深度 15
        min_samples_split=10,  # 分裂所需最小樣本數
        min_samples_leaf=5,  # 葉節點最小樣本數
        max_features="sqrt",  # 每次分裂考慮 sqrt(n_features) 個特徵
        random_state=42,
        n_jobs=-1,  # 使用所有 CPU 核心
        verbose=1,
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

    # 保存全特徵模型
    os.makedirs("models/macd", exist_ok=True)
    model_path = os.path.join("models/macd", "macd_model_v1_rf_all.pkl")

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "feature_importance": feature_importance,
            "roc_auc": roc_auc,
            "test_accuracy": test_accuracy,
        },
        model_path,
    )

    print(f"\n✅ Random Forest 全特徵模型已保存: {model_path}")

    # 測試不同數量的 Top 特徵
    print("\n" + "=" * 80)
    print("🔬 測試不同數量的 Top 特徵")
    print("=" * 80)

    top_n_list = [20, 15, 10, 5]
    results = []

    for top_n in top_n_list:
        print(f"\n{'─' * 80}")
        print(f"📊 測試 Top {top_n} 特徵")
        print(f"{'─' * 80}")

        # 選擇 Top N 特徵
        top_features = feature_importance.head(top_n)["feature"].tolist()
        top_indices = [feature_cols.index(f) for f in top_features]

        X_train_topn = X_train[:, top_indices]
        X_test_topn = X_test[:, top_indices]

        print(f"\nTop {top_n} 特徵:")
        for i, feat in enumerate(top_features, 1):
            importance = feature_importance[feature_importance["feature"] == feat][
                "importance"
            ].values[0]
            print(f"   {i}. {feat}: {importance:.4f}")

        # 訓練模型
        model_topn = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        model_topn.fit(X_train_topn, y_train)

        # 評估
        y_train_pred_topn = model_topn.predict(X_train_topn)
        y_test_pred_topn = model_topn.predict(X_test_topn)

        train_accuracy_topn = (y_train_pred_topn == y_train).mean()
        test_accuracy_topn = (y_test_pred_topn == y_test).mean()

        y_test_proba_topn = model_topn.predict_proba(X_test_topn)[:, 1]
        roc_auc_topn = roc_auc_score(y_test, y_test_proba_topn)

        print(f"\n📈 性能指標:")
        print(f"   訓練集準確率: {train_accuracy_topn:.4f}")
        print(f"   測試集準確率: {test_accuracy_topn:.4f}")
        print(f"   ROC AUC: {roc_auc_topn:.4f}")

        # 混淆矩陣
        cm_topn = confusion_matrix(y_test, y_test_pred_topn)
        print(f"\n混淆矩陣:")
        print(f"              預測未反轉  預測反轉")
        print(f"實際未反轉    {cm_topn[0][0]:6d}      {cm_topn[0][1]:6d}")
        print(f"實際反轉      {cm_topn[1][0]:6d}      {cm_topn[1][1]:6d}")

        # 保存模型
        model_path_topn = os.path.join(
            "models/macd", f"macd_model_v1_rf_top{top_n}.pkl"
        )

        joblib.dump(
            {
                "model": model_topn,
                "feature_columns": top_features,
                "feature_importance": feature_importance.head(top_n),
                "roc_auc": roc_auc_topn,
                "test_accuracy": test_accuracy_topn,
            },
            model_path_topn,
        )

        print(f"\n✅ Top {top_n} 模型已保存: {model_path_topn}")

        # 記錄結果
        results.append(
            {
                "top_n": top_n,
                "train_acc": train_accuracy_topn,
                "test_acc": test_accuracy_topn,
                "roc_auc": roc_auc_topn,
            }
        )

    # 總結比較
    print("\n" + "=" * 80)
    print("📊 特徵數量比較總結")
    print("=" * 80)

    print(f"\n{'特徵數':<10} {'訓練準確率':<15} {'測試準確率':<15} {'ROC AUC':<10}")
    print("─" * 60)
    print(
        f"{'82 (全部)':<10} {train_accuracy:<15.4f} {test_accuracy:<15.4f} {roc_auc:<10.4f}"
    )
    for r in results:
        print(
            f"{r['top_n']:<10} {r['train_acc']:<15.4f} {r['test_acc']:<15.4f} {r['roc_auc']:<10.4f}"
        )

    # 找出最佳模型
    best_result = max(results, key=lambda x: x["roc_auc"])
    print(f"\n🏆 最佳模型: Top {best_result['top_n']} 特徵")
    print(f"   測試準確率: {best_result['test_acc']:.4f}")
    print(f"   ROC AUC: {best_result['roc_auc']:.4f}")


def main():
    """主訓練流程"""
    print("=" * 80)
    print("🤖 MACD V1 模型訓練")
    print("=" * 80)

    # (1) 載入數據
    data_file = "training_data/reversal_training_data_2023-12-01_to_2025-01-01_15m.csv"
    df = load_data(data_file, year=2024)

    # (2) 篩選候選樣本 - 找出 MACD 交叉點
    cross_indices = filter_candidates(df)

    # (3) 生成標籤 - 基於 MACD 交叉點和止損止盈
    t_indices, entry_indices, labels = generate_labels(df, cross_indices)

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
