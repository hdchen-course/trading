#!/usr/bin/env python3
"""
加密貨幣策略配置文件
每個幣種可以配置不同的策略組合和參數
"""

from strategies import (
    StrategyConfig,
    CoinConfig,
    MACDStrategy,
    ATRStrategy,
    SARStrategy,
    BollingerExtremeStrategy,
    StochRSIStrategy,
    VolumeStrategy,
    TrendConfirmationStrategy,
    CandlestickStrategy,
    ReversalStrategy,
    TripleConfirmationStrategy,
)
from strategies.ml import (
    MLReversalV2Strategy,
    MLReversalV3Strategy,
    MLReversalV4Strategy,
)


def create_coin_configs() -> list:
    """創建各幣種的策略配置"""

    # BTC - 核心策略組合（3個精選策略）
    # 權重根據 180 天回測勝率調整 (2025-06-03 至 2025-11-29)
    # 總勝率: 1.00 + 0.6338 + 0.6213 = 2.2551
    btc_strategies = [
        ReversalStrategy(
            StrategyConfig(
                name="Reversal A/V Turn",
                params={},
                weight=1.00
                / 2.2551,  # 100.00% 勝率 → 44.35% 權重 (12次交易, 報酬41.70%)
            )
        ),
        TripleConfirmationStrategy(
            StrategyConfig(
                name="Triple Confirmation Reversal",
                params={"distance": 10, "window_size": 16, "signal_type": "both"},
                weight=0.6338
                / 2.2551,  # 63.38% 勝率 → 28.11% 權重 (213次交易, 報酬439.76%)
            )
        ),
        MLReversalV2Strategy(
            StrategyConfig(
                name="ML Reversal V2",
                params={"distance": 10, "window_size": 16, "signal_type": "both"},
                weight=0.6213
                / 2.2551,  # 62.13% 勝率 → 27.55% 權重 (441次交易, 報酬2497.19%)
            )
        ),
        MLReversalV3Strategy(
            StrategyConfig(
                name="ML Reversal V3",
                params={"prob_threshold": 0.5, "signal_type": "both"},
                weight=0.0,  # 測試階段，不參與權重計算
            )
        ),
        MLReversalV4Strategy(
            StrategyConfig(
                name="ML Reversal V4",
                params={"prob_threshold": 0.5, "signal_type": "both"},
                weight=0.0,  # 測試階段，不參與權重計算
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
        CoinConfig(
            "Bitcoin", "BTCUSDT", btc_strategies, 10, enable_monitor=True
        ),  # 10倍槓桿，啟用監控
        CoinConfig(
            "Ethereum", "ETHUSDT", eth_strategies, 8, enable_monitor=False
        ),  # 8倍槓桿，暫不監控
        CoinConfig(
            "Solana", "SOLUSDT", sol_strategies, 6, enable_monitor=False
        ),  # 6倍槓桿，暫不監控
        CoinConfig(
            "Sui", "SUIUSDT", sui_strategies, 5, enable_monitor=False
        ),  # 5倍槓桿，暫不監控
    ]


def get_test_config():
    """獲取測試配置 (僅 BTC)"""
    all_configs = create_coin_configs()
    # 只返回 BTC 配置
    return [config for config in all_configs if config.name == "Bitcoin"]
