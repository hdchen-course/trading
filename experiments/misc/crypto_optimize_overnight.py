import os
import pickle
import hashlib
import pandas as pd
import itertools
import time
from crypto_provider import DataProvider, BinanceDataProvider
from crypto_backtest import CryptoBacktester
from crypto_config import create_coin_configs
from main import TradingConfig


class CachedDataProvider(DataProvider):
    def __init__(self, cache_dir="data/cache", base_url="https://api.binance.com"):
        self.cache_dir = cache_dir
        self.binance_provider = BinanceDataProvider(base_url)
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, symbol: str, interval: str, limit: int) -> str:
        """生成快取檔案名稱"""
        key_str = f"{symbol}_{interval}_{limit}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """取得快取檔案路徑"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def get_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """取得資料，優先使用快取"""
        cache_key = self._get_cache_key(symbol, interval, limit)
        cache_path = self._get_cache_path(cache_key)

        # 檢查快取是否存在
        if os.path.exists(cache_path):
            print(f"📁 使用快取資料: {symbol} {interval} {limit}筆")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # 快取不存在，從API獲取
        print(f"🌐 從API獲取資料: {symbol} {interval} {limit}筆")
        df = self.binance_provider.get_data(symbol, interval, limit)

        # 儲存到快取
        if not df.empty:
            with open(cache_path, "wb") as f:
                pickle.dump(df, f)
            print(f"💾 資料已快取: {cache_path}")

        return df


class PrecomputedStrategyManager:
    """使用預計算信號的策略管理器"""

    def __init__(self, coin_config, df, original_strategy_manager):
        self.coin_config = coin_config
        self.original_strategy_manager = original_strategy_manager
        self.precomputed_signals = {}

        # 預計算所有策略信號 (weight=1)
        print("📊 預計算策略信號...")
        for strategy in coin_config.strategies:
            print(f"🔄 計算 {strategy.name}")
            signals = strategy.generate_signals(df)
            self.precomputed_signals[strategy.name] = signals
        print("✅ 信號預計算完成")

    def set_weights(self, weights):
        """設定策略權重"""
        self.weights = weights

    def generate_combined_signal_vectorized(self, df: pd.DataFrame, coin_config):
        """使用預計算信號生成綜合策略信號序列"""
        if not hasattr(self, "weights"):
            # 如果沒有設定權重，使用預設權重
            self.weights = [1.0] * len(coin_config.strategies)

        # 使用預計算信號進行權重合併
        strategy_signals = []
        for i, strategy in enumerate(coin_config.strategies):
            if strategy.name in self.precomputed_signals:
                signals = self.precomputed_signals[strategy.name]
                weight = self.weights[i]
                strategy_signals.append((pd.Series(signals, index=df.index), weight))

        if not strategy_signals:
            return pd.Series(0, index=df.index), 0, 0

        # 向量化計算綜合信號
        signals_df = pd.DataFrame(
            {f"signal_{i}": signal for i, (signal, _) in enumerate(strategy_signals)},
            index=df.index,
        )

        weights_df = pd.DataFrame(
            {f"weight_{i}": weight for i, (_, weight) in enumerate(strategy_signals)},
            index=df.index,
        )

        # 計算加權平均信號
        weighted_signals = signals_df.multiply(weights_df.values, axis=1)
        total_weights = weights_df.sum(axis=1)
        combined_signals = weighted_signals.sum(axis=1) / total_weights.replace(0, 1)

        return combined_signals, 0, 0


def optimize_strategy_weights(backtester, coin_configs, period_days=60):
    """優化策略權重 - 使用預計算信號"""
    print("🔍 開始策略權重優化...")

    coin_config = coin_configs[0]  # BTC

    # 獲取資料
    df = backtester.get_binance_data(
        coin_config.symbol,
        backtester.config.timeframe_str,
        period_days * 24 * 60 // backtester.config.timeframe_minutes,
    )

    if df.empty:
        print("❌ 無法獲取資料")
        return None, 0

    # 創建預計算策略管理器（一次性計算所有信號）
    original_manager = backtester.strategy_manager
    precomputed_manager = PrecomputedStrategyManager(coin_config, df, original_manager)
    backtester.set_strategy_manager(precomputed_manager)

    # 優化權重
    weight_options = [i / 10 for i in range(11)]
    best_return = 0
    best_weights = None
    total_combinations = 0

    for weights in itertools.product(
        weight_options, repeat=len(coin_config.strategies)
    ):
        if sum(weights) == 0:
            continue

        total_combinations += 1

        # 每1000次組合顯示進度
        if total_combinations % 1000 == 0:
            print(f"📈 已測試 {total_combinations} 組合，目前最佳: {best_return:.2f}%")

        # 設定新權重（不重新計算信號）
        precomputed_manager.set_weights(weights)

        try:
            # 執行回測
            trades, total_return, win_rate = backtester.backtest_strategy_vectorized(
                df, coin_config, coin_config.symbol
            )

            if total_return > best_return:
                best_return = total_return
                best_weights = weights
                print(f"🎯 新最佳: {weights}, 報酬率: {total_return:.2f}%")

        except Exception:
            continue

    # 恢復原始管理器
    backtester.set_strategy_manager(original_manager)

    print(f"✅ 測試了 {total_combinations} 種組合")
    print(f"🏆 最佳權重: {best_weights}")
    print(f"🏆 最佳報酬率: {best_return:.2f}%")

    return best_weights, best_return


# 策略權重優化
if __name__ == "__main__":
    # 創建配置
    config = TradingConfig()
    coin_configs = create_coin_configs()

    # 只取BTC
    btc_configs = [c for c in coin_configs if "BTC" in c.symbol]

    # 創建回測器並注入快取資料提供者
    backtester = CryptoBacktester(config)
    backtester.set_data_provider(CachedDataProvider())

    print("🚀 開始BTC策略權重優化...")
    print(f"策略數量: {len(btc_configs[0].strategies)}")

    # 執行優化
    start_time = time.time()
    best_weights, best_return = optimize_strategy_weights(backtester, btc_configs, 60)
    end_time = time.time()

    print(f"⏱️ 優化耗時: {end_time - start_time:.2f}秒")
    print("✅ 優化完成！")
