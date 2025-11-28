#!/usr/bin/env python3
"""
加密貨幣交易系統單元測試模組
包含停損停利測試和回測功能測試
"""

import logging
import pandas as pd
from trading.backtest import CryptoBacktester
from config.coin_configs import get_test_config
from strategies import BaseStrategy, StrategyConfig, CoinConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# 簡單的測試配置類（給測試案例用）
class Config:
    def __init__(self):
        self.initial_capital = 10000
        self.position_size_ratio = 0.1  # 每次使用10%資金
        self.stop_loss_pct = 0.018  # 1.8%
        self.take_profit_pct = 0.1  # 10%
        self.enable_stop_loss = True
        self.enable_take_profit = True
        self.timeframe_minutes = 15
        self.open_fee_rate = 0.0002  # 開倉手續費率（0.02%）
        self.close_fee_rate = 0.0005  # 平倉手續費率（0.05%）


class MockStrategy(BaseStrategy):
    def __init__(self, buy_signals=None, sell_signals=None):
        super().__init__(StrategyConfig("Mock", {}))
        self.buy_signals = buy_signals or []
        self.sell_signals = sell_signals or []

    def generate_signals(self, df):
        signals = pd.Series(0, index=df.index)
        for i in self.buy_signals:
            if i < len(signals):
                signals.iloc[i] = 1
        for i in self.sell_signals:
            if i < len(signals):
                signals.iloc[i] = -1
        return signals

    def get_description(self):
        return "Mock Strategy for Testing"


def debug_mode(days: int = 30):
    """Debug模式：先執行單元測試，然後執行BTC一個月回測"""

    # 執行單元測試
    logging.info("=== 執行單元測試 ===")
    logging.info("=== 做多測試 ===")
    test_long_case_1()  # 直接停損
    test_long_case_2()  # 急漲跳階直接平倉
    test_long_case_3()  # 移動停利後觸發停利
    test_long_case_4()  # 移動停損後觸發停損

    # 做空測試
    logging.info("\n=== 做空測試 ===")
    test_short_case_1()  # 直接停損
    test_short_case_2()  # 直接停利
    test_short_case_3()  # 移動停利後觸發停利
    test_short_case_4()  # 移動停損後觸發停損

    # 執行BTC回測
    period = days
    logging.info("\n" + "=" * period)
    logging.info(f"=== BTC {days}天回測 (個別策略 + 綜合策略) ===")
    logging.info("=" * period)

    from config import TradingConfig

    config = TradingConfig()
    backtester = CryptoBacktester(config)

    # 只測試 BTC (使用優化配置)
    btc_config = get_test_config()

    # 2. 所有策略回測 (debug mode)
    logging.info("\n2. 執行所有策略回測 (僅 BTC)...")
    backtester.run_backtest(btc_config, period, debug_mode=True)

    # 3. 綜合策略回測
    logging.info("\n3. 執行綜合策略回測 (僅 BTC)...")
    backtester.run_backtest(btc_config, period, debug_mode=False)

    logging.info("\n✅ Debug 模式完成")


def test_long_case_1():
    """做多測試1: 直接停損"""
    logging.info("Test Case 1: 做多直接停損")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=55, freq="15min")
    # 價格下跌超過0.18%觸發停損: 100 -> 99.82 -> 99.8
    prices = [100] * 50 + [100, 99.9, 99.82, 99.8, 99.8]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(buy_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "stop_loss":
        logging.info("✅ 測試通過: 觸發停損")
    else:
        logging.error("❌ 測試失敗")


def test_long_case_2():
    """做多測試2: 急漲跳階直接平倉"""
    logging.info("Test Case 2: 做多急漲跳階直接平倉")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=55, freq="15min")
    # 價格直接從100跳到102 (跳兩階梯) 應該直接平倉
    prices = [100] * 50 + [100, 102, 102.1, 102.1, 102.1]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(buy_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "take_profit":
        logging.info("✅ 測試通過: 觸發停利")
    else:
        logging.error("❌ 測試失敗")


def test_long_case_3():
    """做多測試3: 移動停利後觸發停利"""
    logging.info("Test Case 3: 做多移動停利後觸發停利")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=65, freq="15min")
    # 每根K線上漲0.4%，5根後到102，停利移動到2.5%，下一根K線到102.5觸發停利
    prices = [100] * 50 + [
        100,
        100.4,
        100.8,
        101.2,
        101.6,
        102.5,
        102.5,
        102.5,
        102.5,
        102.5,
        102.5,
        102.5,
        102.5,
        102.5,
        102.5,
    ]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(buy_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "take_profit":
        # 檢查停利價格是否正確 - 應該在102.5(2.5%)觸發停利
        expected_take_profit = 102.5
        actual_exit_price = trades[0]["exit_price"]
        if abs(actual_exit_price - expected_take_profit) < 0.1:
            logging.info(
                f"✅ 測試通過: 移動停利後觸發停利，停利價格{actual_exit_price:.1f}正確"
            )
        else:
            logging.error(
                f"❌ 停利價格錯誤: 預期{expected_take_profit}, "
                f"實際{actual_exit_price:.1f}"
            )
    else:
        logging.error("❌ 測試失敗")


def test_long_case_4():
    """做多測試4: 移動停損後觸發停損"""
    logging.info("Test Case 4: 做多移動停損後觸發停損")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=65, freq="15min")
    # 複製test case 3的價格序列，但101.6之後跌到100.9觸發停損（停損點應該在101.0）
    prices = [100] * 50 + [
        100,
        100.4,
        100.8,
        101.2,
        101.6,
        100.9,
        100.9,
        100.9,
        100.9,
        100.9,
        100.9,
        100.9,
        100.9,
        100.9,
        100.9,
    ]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(buy_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "stop_loss":
        # 檢查停損價格是否正確 - 應該在101.0觸發停損（1.6%利潤時停損移動到1.0%）
        expected_stop_loss = 101.0
        actual_exit_price = trades[0]["exit_price"]
        if abs(actual_exit_price - expected_stop_loss) < 0.1:
            logging.info(
                f"✅ 測試通過: 移動停損後觸發停損，停損價格{actual_exit_price:.1f}正確"
            )
        else:
            logging.error(
                f"❌ 停損價格錯誤: 預期{expected_stop_loss}, 實際{actual_exit_price:.1f}"
            )
    else:
        exit_reason = trades[0]["exit_reason"] if trades else "no trades"
        logging.error(f"❌ 測試失敗: 預期stop_loss，實際{exit_reason}")


def test_short_case_1():
    """做空測試1: 直接停損"""
    logging.info("Test Case 1: 做空直接停損")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=55, freq="15min")
    prices = [100] * 50 + [100, 101, 102, 103, 104]  # 上漲到102觸發停損

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(sell_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "stop_loss":
        logging.info("✅ 測試通過: 觸發停損")
    else:
        logging.error("❌ 測試失敗")


def test_short_case_2():
    """做空測試2: 直接停利"""
    logging.info("Test Case 2: 做空直接停利")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=55, freq="15min")
    # 直接下跌1%觸發停利，不要觸發移動停損（避免0.5%的移動停損觸發點）
    prices = [100] * 50 + [100, 99.8, 99, 99, 99]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(sell_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "take_profit":
        logging.info("✅ 測試通過: 觸發停利")
    else:
        logging.error("❌ 測試失敗")


def test_short_case_3():
    """做空測試3: 移動停利後觸發停利"""
    logging.info("Test Case 3: 做空移動停利後觸發停利")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=60, freq="15min")
    # 做多測試3的反向：下跌0.4%每根K線，最後觸發停利在97.5
    prices = [100] * 50 + [
        100,
        99.6,
        99.2,
        98.8,
        98.4,
        97.5,
        97.5,
        97.5,
        97.5,
        97.5,
    ]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(sell_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "take_profit":
        # 檢查停利價格是否正確 - 應該在97.5觸發停利
        expected_take_profit = 97.5
        actual_exit_price = trades[0]["exit_price"]
        if abs(actual_exit_price - expected_take_profit) < 0.1:
            logging.info(
                f"✅ 測試通過: 移動停利後觸發停利，停利價格{actual_exit_price:.1f}正確"
            )
        else:
            logging.error(
                f"❌ 停利價格錯誤: 預期{expected_take_profit}, "
                f"實際{actual_exit_price:.1f}"
            )
    else:
        logging.error("❌ 測試失敗")


def test_short_case_4():
    """做空測試4: 移動停損後觸發停損"""
    logging.info("Test Case 4: 做空移動停損後觸發停損")

    config = Config()
    dates = pd.date_range("2024-01-01", periods=60, freq="15min")
    # 做多測試4的反向：下跌到98.4後上漲到99.1觸發停損（停損應該在99.0）
    prices = [100] * 50 + [
        100,
        99.6,
        99.2,
        98.8,
        98.4,
        99.1,
        99.1,
        99.1,
        99.1,
        99.1,
    ]

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1000000.0] * len(prices),
        }
    )
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    strategy = MockStrategy(sell_signals=[50])
    coin_config = CoinConfig("Test", "TESTUSDT", [strategy], 10)

    backtester = CryptoBacktester(config)
    trades, _, _ = backtester.backtest_strategy_vectorized(df, coin_config, "TEST")

    if trades and trades[0]["exit_reason"] == "stop_loss":
        # 檢查停損價格是否正確 - 應該在99.0觸發停損（1.6%利潤時停損移動到1.0%）
        expected_stop_loss = 99.0
        actual_exit_price = trades[0]["exit_price"]
        if abs(actual_exit_price - expected_stop_loss) < 0.1:
            logging.info(
                f"✅ 測試通過: 移動停損後觸發停損，停損價格{actual_exit_price:.1f}正確"
            )
        else:
            logging.error(
                f"❌ 停損價格錯誤: 預期{expected_stop_loss}, 實際{actual_exit_price:.1f}"
            )
    else:
        logging.error("❌ 測試失敗")


if __name__ == "__main__":
    debug_mode()
