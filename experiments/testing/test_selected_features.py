import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def create_features(data):
    """創建選定的技術指標"""
    df = data.copy()

    # EMA
    df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
    df["ema_100"] = talib.EMA(df["close"], timeperiod=100)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df["close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # 布林通道
    bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower

    # ATR
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"])

    # SAR
    df["sar"] = talib.SAR(df["high"], df["low"])

    # K線型態 - 全部型態
    df["doji"] = talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])
    df["hammer"] = talib.CDLHAMMER(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["engulfing"] = talib.CDLENGULFING(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["harami"] = talib.CDLHARAMI(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["shooting_star"] = talib.CDLSHOOTINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["hanging_man"] = talib.CDLHANGINGMAN(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["inverted_hammer"] = talib.CDLINVERTEDHAMMER(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["spinning_top"] = talib.CDLSPINNINGTOP(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["marubozu"] = talib.CDLMARUBOZU(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["piercing"] = talib.CDLPIERCING(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["dark_cloud"] = talib.CDLDARKCLOUDCOVER(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["morning_star"] = talib.CDLMORNINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["evening_star"] = talib.CDLEVENINGSTAR(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["three_white_soldiers"] = talib.CDL3WHITESOLDIERS(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["three_black_crows"] = talib.CDL3BLACKCROWS(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["abandoned_baby"] = talib.CDLABANDONEDBABY(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["advance_block"] = talib.CDLADVANCEBLOCK(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["belt_hold"] = talib.CDLBELTHOLD(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["breakaway"] = talib.CDLBREAKAWAY(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["closing_marubozu"] = talib.CDLCLOSINGMARUBOZU(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["concealing_baby_swallow"] = talib.CDLCONCEALBABYSWALL(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["counterattack"] = talib.CDLCOUNTERATTACK(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["dragonfly_doji"] = talib.CDLDRAGONFLYDOJI(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["gravestone_doji"] = talib.CDLGRAVESTONEDOJI(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["long_legged_doji"] = talib.CDLLONGLEGGEDDOJI(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["rickshaw_man"] = talib.CDLRICKSHAWMAN(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["rising_falling_three"] = talib.CDLRISEFALL3METHODS(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["separating_lines"] = talib.CDLSEPARATINGLINES(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["stick_sandwich"] = talib.CDLSTICKSANDWICH(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["takuri"] = talib.CDLTAKURI(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["tasuki_gap"] = talib.CDLTASUKIGAP(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["thrusting"] = talib.CDLTHRUSTING(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["tristar"] = talib.CDLTRISTAR(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["unique_3_river"] = talib.CDLUNIQUE3RIVER(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["upside_gap_two_crows"] = talib.CDLUPSIDEGAP2CROWS(
        df["open"], df["high"], df["low"], df["close"]
    )
    df["xside_gap_3_methods"] = talib.CDLXSIDEGAP3METHODS(
        df["open"], df["high"], df["low"], df["close"]
    )

    return df


def create_sequences(data, seq_length):
    """創建序列數據"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i])
        y.append(data[i, -1])
    return np.array(X), np.array(y)


# 載入數據
data = pd.read_csv("data/btc_12months_15m.csv")

# 創建特徵
data = create_features(data)

# 創建目標變量
data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

# 選擇特徵：OHLCV + 技術指標 + 全部K線型態
feature_cols = [
    # OHLCV
    "open",
    "high",
    "low",
    "close",
    "volume",
    # EMA
    "ema_50",
    "ema_100",
    # MACD
    "macd",
    "macd_signal",
    "macd_hist",
    # 布林通道
    "bb_upper",
    "bb_middle",
    "bb_lower",
    # ATR
    "atr",
    # SAR
    "sar",
    # 全部K線型態
    "doji",
    "hammer",
    "engulfing",
    "harami",
    "shooting_star",
    "hanging_man",
    "inverted_hammer",
    "spinning_top",
    "marubozu",
    "piercing",
    "dark_cloud",
    "morning_star",
    "evening_star",
    "three_white_soldiers",
    "three_black_crows",
    "abandoned_baby",
    "advance_block",
    "belt_hold",
    "breakaway",
    "closing_marubozu",
    "concealing_baby_swallow",
    "counterattack",
    "dragonfly_doji",
    "gravestone_doji",
    "long_legged_doji",
    "rickshaw_man",
    "rising_falling_three",
    "separating_lines",
    "stick_sandwich",
    "takuri",
    "tasuki_gap",
    "thrusting",
    "tristar",
    "unique_3_river",
    "upside_gap_two_crows",
    "xside_gap_3_methods",
    # 目標
    "target",
]

print(f"選定特徵數量: {len(feature_cols)-1}")  # 不包含target
print("特徵列表:", [col for col in feature_cols if col != "target"])

data = data[feature_cols].dropna()
print(f"處理後數據形狀: {data.shape}")

# 標準化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 使用最佳序列長度36步（9小時）
sequence_length = 36
X, y = create_sequences(scaled_data, sequence_length)

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 移除目標列
X_train = X_train[:, :, :-1]
X_test = X_test[:, :, :-1]

print(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")

# 建立模型
model = Sequential(
    [
        LSTM(
            64,
            return_sequences=True,
            input_shape=(sequence_length, X_train.shape[2]),
        ),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
print(f"模型參數數量: {model.count_params():,}")

# 訓練
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5),
]

print("開始訓練...")
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
)

# 評估
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n最終測試準確率: {test_accuracy:.4f}")

if test_accuracy >= 0.7:
    print("🎉 達到70%目標！")
else:
    print(f"未達到70%目標，當前準確率: {test_accuracy:.4f}")
