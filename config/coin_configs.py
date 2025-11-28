#!/usr/bin/env python3
"""
加密貨幣策略配置文件
每個幣種可以配置不同的策略組合和參數
"""

from strategies import (
    StrategyConfig,
    CoinConfig,
    MACDStrategy,
    MacdReversalStrategy,
    LSTMStrategy,
    MacdVolumeStrategy,
    MACDVolumeStrategy,
    BollingerStrategy,
    BollingerExtremeStrategy,
    ATRStrategy,
    SARStrategy,
    StochRSIStrategy,
    VolumeStrategy,
    TrendConfirmationStrategy,
    CandlestickStrategy,
    TrendMLStrategy,
    ReversalStrategy,
    MLReversalStrategy,
    TrendPredictionStrategy,
    SeparateTrendStrategy,
    CrashPredictionStrategy,
    BTCMACDBuyStrategy,
    TripleConfirmationStrategy,
    TripleConfirmationBuyStrategy,
    TripleConfirmationSellStrategy,
)


def create_coin_configs() -> list:
    """創建各幣種的策略配置"""

    # BTC - 基於180天回測結果優化 - Top 10 策略（平均勝率49.87%）
    # 新評分系統：勝率70% + 每次交易平均報酬30%
    btc_strategies = [
        # 🥇 超高勝率策略 - 合計37.63%權重
        ReversalStrategy(
            StrategyConfig(
                name="Reversal A/V Turn",
                params={},
                weight=0.2012,  # 100.00% 勝率, 41.70% 報酬, 平均3.47%/交易 - 完美勝率！
            )
        ),
        TripleConfirmationStrategy(
            StrategyConfig(
                name="Triple Confirmation Reversal",
                params={
                    "distance": 20,
                    "window_size": 16,
                    "signal_type": "both"
                },
                weight=0.1751,  # 85.71% 勝率, 985.21% 報酬, 平均6.12%/交易 - 最高質量！
            )
        ),
        
        # 🥈 高表現策略 - 合計26.27%權重
        MacdVolumeStrategy(
            StrategyConfig(
                name="MACD Volume Detection",
                params={},
                weight=0.0924,  # 46.43% 勝率, 12.42% 報酬, 平均0.44%/交易
            )
        ),
        MLReversalStrategy(
            StrategyConfig(
                name="ML Reversal A/V Turn",
                params={},
                weight=0.0906,  # 44.66% 勝率, 1754.14% 報酬, 平均2.46%/交易
            )
        ),
        MacdReversalStrategy(
            StrategyConfig(
                name="MACD Reversal Detection",
                params={},
                weight=0.0797,  # 40.00% 勝率, 23.07% 報酬, 平均0.51%/交易
            )
        ),
        LSTMStrategy(
            StrategyConfig(
                name="LSTM Detection",
                params={},
                weight=0.0762,  # 38.36% 勝率, 26.04% 報酬, 平均0.18%/交易
            )
        ),
        CrashPredictionStrategy(
            StrategyConfig(
                name="8H Crash Prediction",
                params={
                    "crash_threshold": 0.75,
                },
                weight=0.0736,  # 37.09% 勝率, 31.91% 報酬, 平均0.12%/交易
            )
        ),
        
        # 🥉 穩定策略 - 合計36.10%權重
        StochRSIStrategy(
            StrategyConfig(
                name="Stochastic RSI",
                params={
                    "rsi_period": 14,
                    "stoch_period": 14,
                    "k_period": 3,
                    "d_period": 3,
                    "oversold": 5,
                    "overbought": 95,
                },
                weight=0.0733,  # 36.94% 勝率, 156.83% 報酬, 平均0.16%/交易
            )
        ),
        ATRStrategy(
            StrategyConfig(
                name="ATR Breakout",
                params={"atr_period": 14, "multiplier": 2.0},
                weight=0.0695,  # 34.99% 勝率, 134.73% 報酬, 平均0.19%/交易
            )
        ),
        TrendMLStrategy(
            StrategyConfig(
                name="Trend ML",
                params={
                    "ema_fast": 20,
                    "ema_slow": 50,
                    "model_threshold": 0.7,
                },
                weight=0.0684,  # 34.48% 勝率, 67.43% 報酬, 平均0.05%/交易
            )
        ),
    ]

    # ETH - 偏向趨勢跟隨策略
    eth_strategies = [
        MACDStrategy(
            StrategyConfig(
                name="MACD Classic",
                params={"fast": 12, "slow": 26, "signal": 9},
                weight=0.4,
            )
        ),
        SARStrategy(
            StrategyConfig(
                name="Parabolic SAR",
                params={"start": 0.02, "increment": 0.02, "maximum": 0.2},
                weight=0.3,
            )
        ),
        ATRStrategy(
            StrategyConfig(
                name="ATR Breakout",
                params={"atr_period": 14, "multiplier": 2.0},
                weight=0.3,
            )
        ),
    ]

    # SOL - 高波動性策略
    sol_strategies = [
        BollingerExtremeStrategy(
            StrategyConfig(
                name="Bollinger Extreme",
                params={"period": 20, "std": 2.5},
                weight=0.4,
            )
        ),
        StochRSIStrategy(
            StrategyConfig(
                name="Stochastic RSI",
                params={
                    "rsi_period": 14,
                    "stoch_period": 14,
                    "k_period": 3,
                    "d_period": 3,
                    "oversold": 15,
                    "overbought": 85,
                },
                weight=0.3,
            )
        ),
        VolumeStrategy(
            StrategyConfig(
                name="Volume Breakout",
                params={
                    "volume_threshold": 2.0,
                    "price_change_threshold": 0.03,
                    "volume_ma_period": 20,
                },
                weight=0.3,
            )
        ),
    ]

    # SUI - 新興幣種保守策略
    sui_strategies = [
        MACDStrategy(
            StrategyConfig(
                name="MACD Conservative",
                params={"fast": 10, "slow": 21, "signal": 5},
                weight=0.5,
            )
        ),
        TrendConfirmationStrategy(
            StrategyConfig(
                name="Trend Confirmation",
                params={
                    "confirmation_hours": 4,
                    "bb_period": 20,
                    "bb_std": 2.0,
                },
                weight=0.3,
            )
        ),
        CandlestickStrategy(
            StrategyConfig(
                name="Candlestick Patterns",
                params={"threshold": 0.5},
                weight=0.2,
            )
        ),
    ]

    return [
        CoinConfig("Bitcoin", "BTCUSDT", btc_strategies, 10),  # 10倍槓桿
        CoinConfig("Ethereum", "ETHUSDT", eth_strategies, 8),  # 8倍槓桿
        CoinConfig("Solana", "SOLUSDT", sol_strategies, 6),  # 6倍槓桿
        CoinConfig("Sui", "SUIUSDT", sui_strategies, 5),  # 5倍槓桿
    ]


def get_test_config():
    """獲取測試配置 (僅 BTC)"""
    all_configs = create_coin_configs()
    # 只返回 BTC 配置
    return [config for config in all_configs if config.name == "Bitcoin"]
