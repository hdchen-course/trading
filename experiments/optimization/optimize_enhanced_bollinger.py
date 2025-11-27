#!/usr/bin/env python3
"""
Enhanced Bollinger Strategy 參數優化
使用12個月BTC數據尋找最佳參數組合，要求夏普比率 >= 1
"""

import pandas as pd
import numpy as np
import logging
from itertools import product
from crypto_strategies import (
    EnhancedBollingerStrategy,
    StrategyConfig,
    CoinConfig,
)
from crypto_backtest import CryptoBacktester
from main import TradingConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_btc_data():
    """載入12個月BTC數據"""
    try:
        df = pd.read_csv(
            "/Volumes/workplace/crypto_trading_system/data/btc_12months_15m.csv"
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logging.info(f"✅ 載入數據: {len(df)} 筆記錄")
        return df
    except Exception as e:
        logging.error(f"❌ 載入數據失敗: {e}")
        return None


def optimize_enhanced_bollinger():
    """優化Enhanced Bollinger策略參數"""

    df = load_btc_data()
    if df is None:
        return

    # 參數範圍
    periods = [30, 40, 50, 60]
    stds = [2.0, 2.2, 2.4, 2.6]
    volume_thresholds = [1.2, 1.3, 1.4]
    pattern_strengths = [75, 80, 85]

    config = TradingConfig()
    backtester = CryptoBacktester(config)

    results = []
    valid_results = []
    total = (
        len(periods)
        * len(stds)
        * len(volume_thresholds)
        * len(pattern_strengths)
    )
    current = 0

    logging.info(f"🔍 開始優化，總共 {total} 種參數組合")

    for period, std, vol_threshold, pattern_strength in product(
        periods, stds, volume_thresholds, pattern_strengths
    ):
        current += 1

        try:
            strategy = EnhancedBollingerStrategy(
                StrategyConfig(
                    name=f"Enhanced_Bollinger_Test",
                    params={
                        "period": period,
                        "std": std,
                        "volume_period": 20,
                        "volume_threshold": vol_threshold,
                        "pattern_strength": pattern_strength,
                    },
                )
            )

            coin_config = CoinConfig("Bitcoin", "BTCUSDT", [strategy], 10)
            trades, total_return, win_rate = (
                backtester.backtest_strategy_vectorized(df, coin_config, "BTC")
            )

            if trades and len(trades) >= 10:  # 至少10筆交易
                profits = []
                for trade in trades:
                    if trade["exit_reason"] in [
                        "take_profit",
                        "stop_loss",
                        "signal",
                    ]:
                        entry_price = trade["entry_price"]
                        exit_price = trade["exit_price"]
                        if trade.get("position_type") == "short":
                            profit_pct = (
                                entry_price - exit_price
                            ) / entry_price
                        else:
                            profit_pct = (
                                exit_price - entry_price
                            ) / entry_price
                        profits.append(profit_pct)

                if len(profits) > 1:
                    sharpe_ratio = (
                        np.mean(profits) / np.std(profits)
                        if np.std(profits) > 0
                        else 0
                    )
                else:
                    sharpe_ratio = 0

                result = {
                    "period": period,
                    "std": std,
                    "volume_threshold": vol_threshold,
                    "pattern_strength": pattern_strength,
                    "trades": len(trades),
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "score": total_return * win_rate,
                }

                results.append(result)

                # 夏普比率 >= 1 的結果
                if sharpe_ratio >= 1.0:
                    valid_results.append(result)
                    logging.info(
                        f"✅ 找到符合條件: 期間{period}, 標準差{std}, 成交量{vol_threshold}, 型態{pattern_strength} "
                        f"-> 報酬{total_return:.2%}, 勝率{win_rate:.2%}, 夏普{sharpe_ratio:.2f}"
                    )

            if current % 20 == 0:
                logging.info(
                    f"📊 進度: {current}/{total} ({current/total*100:.1f}%), 符合條件: {len(valid_results)}"
                )

        except Exception as e:
            logging.error(f"❌ 參數組合失敗: {e}")

    # 排序結果
    results.sort(key=lambda x: x["score"], reverse=True)
    valid_results.sort(key=lambda x: x["score"], reverse=True)

    logging.info(f"\n📊 優化完成:")
    logging.info(f"   總測試組合: {len(results)}")
    logging.info(f"   符合夏普比率>=1: {len(valid_results)}")

    if valid_results:
        logging.info(f"\n🏆 符合條件的前5名:")
        for i, result in enumerate(valid_results[:5], 1):
            logging.info(
                f"{i}. 期間{result['period']}, 標準差{result['std']}, 成交量{result['volume_threshold']}, 型態{result['pattern_strength']} "
                f"-> 交易{result['trades']}次, 報酬{result['total_return']:.2%}, 勝率{result['win_rate']:.2%}, 夏普{result['sharpe_ratio']:.2f}"
            )

        best = valid_results[0]
        logging.info(f"\n🎯 推薦參數:")
        logging.info(f"   period: {best['period']}")
        logging.info(f"   std: {best['std']}")
        logging.info(f"   volume_threshold: {best['volume_threshold']}")
        logging.info(f"   pattern_strength: {best['pattern_strength']}")

        return best
    else:
        logging.info(f"\n⚠️  沒有找到夏普比率>=1的組合，顯示前5名:")
        for i, result in enumerate(results[:5], 1):
            logging.info(
                f"{i}. 期間{result['period']}, 標準差{result['std']}, 成交量{result['volume_threshold']}, 型態{result['pattern_strength']} "
                f"-> 交易{result['trades']}次, 報酬{result['total_return']:.2%}, 勝率{result['win_rate']:.2%}, 夏普{result['sharpe_ratio']:.2f}"
            )

        return results[0] if results else None


if __name__ == "__main__":
    optimize_enhanced_bollinger()
