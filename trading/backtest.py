#!/usr/bin/env python3
"""
加密貨幣回測模組

功能:
- 向量化回測引擎，支援多策略並行測試
- 完整的資金管理和風險控制
- 複利效果和破產保護機制
- 詳細的交易記錄和統計分析
- 圖表生成和結果可視化
- 支援依賴注入的資料提供者

核心特性:
1. 使用配置比例的資金管理（position_size_ratio）
2. 動態停損停利機制
3. 複利計算和資金追蹤
4. 交易時間統計（以小時為單位）
5. 支援個別策略和綜合策略回測
"""

import logging
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple
from strategies import CoinConfig, StrategyManager
from data.providers.binance import DataProvider, BinanceDataProvider


class CryptoBacktester:
    """
    加密貨幣回測器

    提供完整的回測功能，包括：
    - 多策略回測
    - 資金管理
    - 風險控制
    - 結果分析
    - 圖表生成
    """

    def __init__(self, config):
        """
        初始化回測器

        Args:
            config: 交易配置對象，包含所有回測參數
        """
        self.config = config
        self.strategy_manager = StrategyManager()
        self.data_provider = BinanceDataProvider()

    def set_data_provider(self, data_provider: DataProvider):
        """
        設定資料提供者（依賴注入）

        Args:
            data_provider: 實現 DataProvider 介面的資料提供者
                          可以是 BinanceDataProvider、CachedDataProvider 等
        """
        self.data_provider = data_provider

    def set_strategy_manager(self, strategy_manager):
        """
        設定策略管理器（依賴注入）

        Args:
            strategy_manager: 策略管理器實例
                            可以是 StrategyManager 或 PrecomputedStrategyManager
        """
        self.strategy_manager = strategy_manager

    def get_binance_data(
        self, symbol: str, interval: str, limit: int = 1000
    ) -> pd.DataFrame:
        """
        從資料提供者獲取市場數據

        Args:
            symbol: 交易對符號（如 "BTCUSDT"）
            interval: 時間間隔（如 "15m"）
            limit: 數據筆數限制

        Returns:
            pd.DataFrame: 包含 OHLCV 數據的 DataFrame
        """
        return self.data_provider.get_data(symbol, interval, limit)

    def backtest_strategy_vectorized(
        self, df: pd.DataFrame, coin_config: CoinConfig, symbol: str
    ) -> Tuple[List, float, float]:
        """
        向量化回測引擎 - 核心回測邏輯

        實現完整的交易模擬，包括：
        1. 策略信號生成和合併
        2. 進場出場邏輯
        3. 動態停損停利
        4. 資金管理和複利計算
        5. 破產保護機制

        資金管理特性：
        - 使用 config.position_size_ratio 控制每次交易資金比例
        - 實現複利效果：獲利後下次交易金額增加
        - 破產保護：資金歸零時停止交易

        Args:
            df: 市場數據 DataFrame，包含 OHLCV 數據
            coin_config: 幣種配置，包含策略和槓桿設定
            symbol: 交易對符號

        Returns:
            Tuple[List, float, float]: (交易記錄列表, 總報酬率, 勝率)
        """
        """向量化回測策略 - 高性能版本"""
        
        # 統一計算所有技術指標（避免每個策略重複計算）
        from utils.indicators import calculate_technical_indicators, analyze_cumulative_metrics
        df = calculate_technical_indicators(df)
        df = analyze_cumulative_metrics(df, window_size=16)

        # 一次性計算所有信號 (向量化)
        if len(coin_config.strategies) == 1:
            # 單一策略直接計算
            strategy = coin_config.strategies[0]
            signals = strategy.generate_signals(df)
            # 單一策略的信號強度就是權重
            signal_strength = pd.Series(
                [strategy.weight if s != 0 else 0.0 for s in signals],
                index=df.index
            )
        else:
            # 多策略使用策略管理器
            signals, signal_strength, _, _ = self.strategy_manager.generate_combined_signal_vectorized(
                df, coin_config
            )
            
            # 信號強度閾值過濾 - 只保留強度 >= 0.3 的信號（僅用於多策略組合）
            SIGNAL_STRENGTH_THRESHOLD = 0.18
            signals = signals.where(signal_strength >= SIGNAL_STRENGTH_THRESHOLD, 0)

        # 向量化回測邏輯
        trades = []
        position = None
        capital = self.config.initial_capital

        # 槓桿後的停損停利比例
        actual_stop_loss = self.config.stop_loss_pct / coin_config.leverage
        actual_take_profit = self.config.take_profit_pct / coin_config.leverage

        for i in range(50, len(df)):  # 跳過前50根用於指標計算
            current_price = df.iloc[i]["close"]
            timestamp = df.iloc[i]["timestamp"]
            signal = signals.iloc[i]

            if signal == 1 and position is None:  # 買入信號 (做多)
                position = {
                    "entry_price": current_price,
                    "entry_time": timestamp,
                    "entry_idx": i,
                    "stop_loss_price": current_price * (1 - actual_stop_loss),
                    "take_profit_price": current_price * (1 + actual_take_profit),
                    "position_type": "long",
                    "original_stop_loss": actual_stop_loss,
                    "original_take_profit": actual_take_profit,
                    "level": 0,
                    "last_adjusted_level": -1,
                    "entry_signal_strength": signal_strength.iloc[i],  # 記錄進場信號強度
                }

            elif signal == -1 and position is None:  # 賣出信號 (做空)
                position = {
                    "entry_price": current_price,
                    "entry_time": timestamp,
                    "entry_idx": i,
                    "stop_loss_price": current_price * (1 + actual_stop_loss),
                    "take_profit_price": current_price * (1 - actual_take_profit),
                    "position_type": "short",
                    "original_stop_loss": actual_stop_loss,
                    "original_take_profit": actual_take_profit,
                    "level": 0,
                    "last_adjusted_level": -1,
                    "entry_signal_strength": signal_strength.iloc[i],  # 記錄進場信號強度
                }

            elif position is not None:  # 檢查平倉條件
                exit_reason = None
                exit_price = current_price

                # 先檢查停損停利條件
                if self.config.enable_stop_loss and (
                    (
                        position["position_type"] == "long"
                        and current_price <= position["stop_loss_price"]
                    )
                    or (
                        position["position_type"] == "short"
                        and current_price >= position["stop_loss_price"]
                    )
                ):
                    exit_reason = "stop_loss"
                    exit_price = position["stop_loss_price"]

                # 停利檢查
                elif self.config.enable_take_profit and (
                    (
                        position["position_type"] == "long"
                        and current_price >= position["take_profit_price"]
                    )
                    or (
                        position["position_type"] == "short"
                        and current_price <= position["take_profit_price"]
                    )
                ):
                    exit_reason = "take_profit"
                    exit_price = position["take_profit_price"]

                # 智能反向信號檢查 - 只有在停損未移動到獲利區時才允許signal exit
                elif (position["position_type"] == "long" and signal == -1) or (
                    position["position_type"] == "short" and signal == 1
                ):
                    # 檢查停損是否已移動到獲利區
                    if position["position_type"] == "long":
                        # 做多：停損價格高於進場價格表示已獲利
                        stop_loss_profitable = (
                            position["stop_loss_price"] > position["entry_price"]
                        )
                    else:
                        # 做空：停損價格低於進場價格表示已獲利
                        stop_loss_profitable = (
                            position["stop_loss_price"] < position["entry_price"]
                        )

                    # 只有在停損未移動到獲利區時才允許signal exit
                    if not stop_loss_profitable:
                        exit_reason = "signal"
                        exit_price = current_price

                # 如果沒有平倉，則進行動態調整
                if not exit_reason:
                    trailing_result = self._update_trailing_stops(
                        position, current_price
                    )
                    if trailing_result == "rapid_move":
                        exit_reason = "rapid_move"
                        exit_price = current_price

                # 平倉
                if exit_reason:
                    if position["position_type"] == "long":
                        price_change_pct = (
                            exit_price - position["entry_price"]
                        ) / position["entry_price"]
                    else:  # short
                        price_change_pct = (
                            position["entry_price"] - exit_price
                        ) / position["entry_price"]

                    # === 資金管理和複利計算 ===
                    leveraged_pnl = price_change_pct * coin_config.leverage

                    # 使用配置的資金比例進行交易
                    # 這實現了動態資金管理：隨著資金增減，交易金額也相應調整
                    position_size = capital * self.config.position_size_ratio

                    # 計算手續費：基於槓桿後的實際交易金額
                    leveraged_position_size = position_size * coin_config.leverage
                    total_fee_rate = (
                        self.config.open_fee_rate + self.config.close_fee_rate
                    )
                    fee_cost = leveraged_position_size * total_fee_rate

                    # 計算扣除手續費後的實際盈虧
                    gross_pnl = leveraged_pnl * position_size
                    pnl = gross_pnl - fee_cost
                    capital += pnl

                    # 破產保護機制：資金歸零時停止交易
                    if capital <= 0:
                        capital = 0
                        break

                    # 記錄詳細的交易信息
                    trades.append(
                        {
                            "type": position["position_type"],  # 交易類型（long/short）
                            "entry_time": position["entry_time"],  # 進場時間
                            "exit_time": timestamp,  # 出場時間
                            "entry_price": position["entry_price"],  # 進場價格
                            "exit_price": exit_price,  # 出場價格
                            "pnl": pnl,  # 絕對盈虧（美元）
                            "return_pct": price_change_pct,  # 價格變動百分比
                            "leveraged_return_pct": leveraged_pnl,  # 槓桿後報酬率
                            "exit_reason": exit_reason,  # 出場原因
                            "duration_hours": (
                                i - position["entry_idx"]
                            )  # 持倉時間（小時）
                            * self.config.timeframe_minutes
                            / 60,
                            "capital_before": capital - pnl,  # 交易前資金
                            "capital_after": capital,  # 交易後資金
                            "signal_strength": position.get("entry_signal_strength", 0.0),  # 進場信號強度
                        }
                    )

                    # 清除持倉狀態，準備下次交易
                    position = None

        # === 計算回測統計指標 ===
        # 總報酬率：(最終資金 - 初始資金) / 初始資金
        total_return = (
            capital - self.config.initial_capital
        ) / self.config.initial_capital

        # 勝率：獲利交易數 / 總交易數
        win_rate = (
            len([t for t in trades if t["pnl"] > 0]) / len(trades)
            if trades  # 避免除零錯誤
            else 0
        )

        return trades, total_return, win_rate

    def _update_trailing_stops(self, position, current_price):
        """
        階梯式動態停損停利機制

        根據獲利程度動態調整停損停利價格：
        - 獲利越多，停損價格越接近當前價格（保護獲利）
        - 實現階梯式調整，避免頻繁變動
        - 支援做多和做空兩種方向

        Args:
            position: 當前持倉信息字典
            current_price: 當前市場價格

        Returns:
            str: "rapid_move" 如果觸發快速移動平倉，否則 None
        """
        entry_price = position["entry_price"]
        pos_type = position.get("position_type") or position.get("type")
        is_long = pos_type in ["long", "LONG"]

        # 計算當前獲利百分比
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # === 階梯式停損停利設計 ===
        # 基於原始停利目標 (通常是 10%) 建立多層階梯
        # 隨著獲利增加，逐步提高停損價格以保護獲利
        x = position["original_take_profit"]  # 基準停利百分比 (如 10%)

        # 定義階梯獲利門檻
        data = [
            x * 0.25,  # 2.5% 獲利時開始調整
            x * 0.5,  # 5% 獲利
            x * 1,  # 10% 獲利 (原始停利目標)
            x * 1.5,  # 15% 獲利
            x * 2,  # 20% 獲利
            x * 2.5,  # 25% 獲利
            x * 3,  # 30% 獲利
            x * 3.5,  # 35% 獲利
            x * 4,  # 40% 獲利
            x * 4.5,  # 45% 獲利 - 前10組：細緻階梯
            x * 5,  # 50% 獲利
            x * 6,  # 60% 獲利
            x * 7,  # 70% 獲利
            x * 8,  # 80% 獲利
            x * 9,  # 90% 獲利
            x * 10,  # 100% 獲利
            x * 11,  # 110% 獲利
            x * 12,  # 120% 獲利
            x * 13,  # 130% 獲利
            x * 14,  # 140% 獲利 - 中10組：1x間距
            x * 15,
            x * 16,
            x * 17,
            x * 18,
            x * 19,
            x * 20,
            x * 21,
            x * 22,
            x * 23,
            x * 24,  # 後10組：1%間距
        ]

        # 檢查是否達到0.5%漲幅觸發移動停損停利
        if profit_pct >= data[1] and not position.get(
            "trailing_activated", False
        ):  # 0.5%
            position["trailing_activated"] = True
            position["current_level"] = 1

            if is_long:
                # 停損 = data[0] (0.25%), 停利 = data[3] (1.5%)
                position["stop_loss_price"] = entry_price * (1 + data[0])
                position["take_profit_price"] = entry_price * (1 + data[3])
            else:
                position["stop_loss_price"] = entry_price * (1 - data[0])
                position["take_profit_price"] = entry_price * (1 - data[3])

        # 繼續檢查後續階梯移動 - 檢測跳階情況
        if position.get("trailing_activated", False):
            current_level = position.get("current_level", 1)

            # 檢查跳了幾個階梯
            levels_jumped = 0
            target_level = current_level
            for level in range(current_level + 1, len(data)):
                if profit_pct >= data[level]:
                    target_level = level
                else:
                    break

            levels_jumped = target_level - current_level

            if levels_jumped > 0:
                if levels_jumped > 1:
                    # 跳超過一階 = 急漲急跌，直接平倉
                    return "rapid_move"
                else:
                    # 只跳一階 = 正常移動，調整停損停利
                    next_level = current_level + 1
                    position["current_level"] = next_level

                    if is_long:
                        # 停損 = 前一階梯, 停利 = 當前階梯 + 1%
                        position["stop_loss_price"] = entry_price * (
                            1 + data[next_level - 1]
                        )
                        take_profit_pct = (
                            data[next_level] + position["original_take_profit"]
                        )
                        position["take_profit_price"] = entry_price * (
                            1 + take_profit_pct
                        )
                    else:
                        position["stop_loss_price"] = entry_price * (
                            1 - data[next_level - 1]
                        )
                        take_profit_pct = (
                            data[next_level] + position["original_take_profit"]
                        )
                        position["take_profit_price"] = entry_price * (
                            1 - take_profit_pct
                        )

        return None

    def plot_backtest_results(
        self,
        df: pd.DataFrame,
        trades: List,
        symbol: str,
        strategy_name: str = "Combined",
    ):
        """
        生成回測結果圖表和交易記錄

        功能包括：
        1. 價格走勢圖 + 交易點標記（綠色=進場，紅色=出場）
        2. 資金曲線圖（顯示資金變化軌跡）
        3. 交易統計表格
        4. 圖表保存為 PNG 文件
        5. 交易記錄保存為 CSV 文件
        6. 詳細交易日誌輸出

        Args:
            df: 市場數據 DataFrame，包含時間戳和價格數據
            trades: 交易記錄列表，每個元素包含完整交易信息
            symbol: 交易對符號（如 "BTCUSDT"）
            strategy_name: 策略名稱，用於文件命名和圖表標題
        """
        # 確保 output 目錄存在
        os.makedirs("output", exist_ok=True)

        # 創建 2x1 子圖：上方價格圖，下方資金圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # === 上方圖：價格走勢 + 交易點 ===
        ax1.plot(df["timestamp"], df["close"], label="Price", linewidth=1)

        # === 標記交易點在價格圖上 ===
        if trades:
            # 按交易類型分類
            long_trades = [
                t for t in trades if t.get("type", "LONG") in ["LONG", "long"]
            ]
            short_trades = [t for t in trades if t.get("type") in ["SHORT", "short"]]

            # 標記做多交易點（綠色向上三角形）
            if long_trades:
                buy_times = [t["entry_time"] for t in long_trades]
                buy_prices = [t["entry_price"] for t in long_trades]
                ax1.scatter(
                    buy_times,
                    buy_prices,
                    color="green",  # 綠色表示買入
                    marker="^",  # 向上三角形
                    s=100,  # 點的大小
                    label="BUY (Long)",  # 圖例標籤
                    alpha=0.7,  # 透明度
                )

            # 標記做空開倉點
            if short_trades:
                sell_times = [t["entry_time"] for t in short_trades]
                sell_prices = [t["entry_price"] for t in short_trades]
                ax1.scatter(
                    sell_times,
                    sell_prices,
                    color="red",
                    marker="v",
                    s=100,
                    label="SELL (Short)",
                    alpha=0.7,
                )

        # 分類退出點
        signal_exits = [t for t in trades if t.get("exit_reason") == "signal"]
        stop_loss_exits = [t for t in trades if t.get("exit_reason") == "stop_loss"]
        take_profit_exits = [t for t in trades if t.get("exit_reason") == "take_profit"]

        # 標記信號退出點
        if signal_exits:
            exit_times = [t["exit_time"] for t in signal_exits]
            exit_prices = [t["exit_price"] for t in signal_exits]
            ax1.scatter(
                exit_times,
                exit_prices,
                color="orange",
                marker="x",
                s=100,
                label="Signal Exit",
                alpha=0.7,
            )

        # 標記停損點
        if stop_loss_exits:
            sl_times = [t["exit_time"] for t in stop_loss_exits]
            sl_prices = [t["exit_price"] for t in stop_loss_exits]
            ax1.scatter(
                sl_times,
                sl_prices,
                color="purple",
                marker="x",
                s=100,
                label="Stop Loss",
                alpha=0.7,
            )

        # 標記停利點
        if take_profit_exits:
            tp_times = [t["exit_time"] for t in take_profit_exits]
            tp_prices = [t["exit_price"] for t in take_profit_exits]
            ax1.scatter(
                tp_times,
                tp_prices,
                color="purple",
                marker="*",
                s=100,
                label="Take Profit",
                alpha=0.7,
            )

        # 設置上方圖表屬性
        ax1.set_title(f"{symbol} - {strategy_name}")  # 圖表標題
        ax1.set_ylabel("Price")  # Y軸標籤
        ax1.legend()  # 顯示圖例
        ax1.grid(True, alpha=0.3)  # 添加網格線

        # === 下方圖：資金曲線圖 ===
        # 顯示投資組合價值隨時間的變化
        if trades:
            # 計算每筆交易後的資金變化
            portfolio_values = [self.config.initial_capital]  # 起始資金
            trade_times = [df["timestamp"].iloc[0]]  # 起始時間

            # 遍歷每筆交易，累積計算資金變化
            for trade in trades:
                current_value = (
                    portfolio_values[-1] + trade["pnl"]
                )  # 當前資金 = 上次資金 + 本次盈虧
                portfolio_values.append(current_value)
                trade_times.append(trade["exit_time"])

            # 繪製資金曲線
            ax2.plot(
                trade_times,
                portfolio_values,
                label="Portfolio Value",  # 投資組合價值
                color="blue",
            )

            # 添加初始資本參考線（灰色虛線）
            ax2.axhline(
                y=self.config.initial_capital,
                color="gray",
                linestyle="--",
                label="Initial Capital",  # 初始資本線標籤
                alpha=0.7,
            )

        # 設置下方圖表屬性
        ax2.set_title("Portfolio Value")  # 圖表標題
        ax2.set_ylabel("Value")  # Y軸標籤：價值
        ax2.set_xlabel("Time")  # X軸標籤：時間
        ax2.legend()  # 顯示圖例
        ax2.grid(True, alpha=0.3)  # 添加網格線

        # 調整子圖間距，避免重疊
        plt.tight_layout()

        # === 保存圖表為 PNG 文件 ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 時間戳
        safe_strategy_name = strategy_name.replace(" ", "_").replace(
            "/", "_"
        )  # 安全的文件名
        filename = f"output/backtest_{symbol}_{safe_strategy_name}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")  # 高解析度保存
        plt.close()  # 關閉圖表釋放記憶體

        logging.info(f"📊 圖表已保存: {filename}")

        # === 保存交易記錄為 CSV 文件 ===
        if trades:
            # 按進場時間排序，便於分析
            sorted_trades = sorted(trades, key=lambda x: x["entry_time"])
            # 轉換為 DataFrame 便於保存和分析
            trades_df = pd.DataFrame(sorted_trades)

            # 生成 CSV 文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_clean = strategy_name.replace(" ", "_").replace("/", "_")
            csv_file = f"output/trades_{symbol}_{strategy_clean}_{timestamp}.csv"

            # 保存為 CSV 文件
            trades_df.to_csv(csv_file, index=False)
            logging.info(f"💾 交易紀錄: {csv_file}")

            # === 輸出詳細交易日誌到控制台 ===
            logging.info("\n=== 交易日誌 (按時間排序) ===")
            logging.info("總交易次數: %d", len(sorted_trades))

            # 表格標題行
            logging.info(
                "序號 進場時間              出場時間              "
                "進場價格   出場價格   盈虧%     退出原因"
            )
            logging.info("-" * 95)  # 分隔線

            # 顯示前15筆交易的詳細信息
            for i, trade in enumerate(sorted_trades[:15]):
                profit_pct = trade["return_pct"]
                profit_symbol = "📈" if profit_pct > 0 else "📉"  # 盈虧符號

                # 格式化時間顯示（只顯示到分鐘）
                entry_time = str(trade["entry_time"])[:16]
                exit_time = str(trade["exit_time"])[:16]

                # 格式化輸出每筆交易
                logging.info(
                    "%2d. %s %s %8.2f %8.2f %s%+6.2f%% %10s",
                    i + 1,  # 序號
                    entry_time,  # 進場時間
                    exit_time,
                    trade["entry_price"],
                    trade["exit_price"],
                    profit_symbol,
                    profit_pct,
                    trade["exit_reason"],
                )

            if len(sorted_trades) > 15:
                logging.info("... 還有 %d 筆交易", len(sorted_trades) - 15)

            logging.info("\n完整交易日誌已保存至: %s", csv_file)

    def run_backtest(
        self,
        coin_configs: List[CoinConfig],
        period_days: int = 30,
        debug_mode: bool = False,
    ):
        """
        執行完整的回測流程

        這是回測系統的主要入口點，協調所有回測活動：

        功能流程：
        1. 遍歷所有配置的幣種
        2. 獲取市場數據
        3. 根據模式執行回測：
           - Debug模式：測試每個策略的個別表現
           - 正常模式：測試綜合策略表現
        4. 生成圖表和統計報告
        5. 保存結果到文件

        Args:
            coin_configs: 幣種配置列表，每個包含策略和參數
            period_days: 回測期間天數，用於計算需要的數據量
            debug_mode: 是否為調試模式
                       True: 測試每個策略的個別表現
                       False: 測試綜合策略表現

        Returns:
            List[dict]: 回測結果列表，包含所有策略的詳細表現數據
        """
        start_time = time.time()  # 記錄開始時間
        os.makedirs("output", exist_ok=True)  # 確保輸出目錄存在
        all_results = []  # 存儲所有回測結果

        # 遍歷每個幣種配置
        for coin_config in coin_configs:
            coin_start_time = time.time()

            # 先獲取數據，不管是哪種模式都需要
            df = self.get_binance_data(
                coin_config.symbol,
                self.config.timeframe_str,
                period_days * 24 * 60 // self.config.timeframe_minutes,
            )

            if df.empty:
                logging.error(f"❌ 無法取得 {coin_config.symbol} 數據")
                continue

            if debug_mode:
                # Debug 模式：測試所有個別策略
                logging.info(
                    f"🐛 Debug 回測 {coin_config.name} "
                    f"({coin_config.symbol}) - 所有策略"
                )

                for strategy in coin_config.strategies:
                    strategy_start_time = time.time()
                    logging.info(
                        f"🔹 測試策略: {strategy.name} - "
                        f"{strategy.get_description()}"
                    )

                    # 創建單一策略的臨時配置
                    temp_config = CoinConfig(
                        coin_config.name,
                        coin_config.symbol,
                        [strategy],
                        coin_config.leverage,
                    )

                    trades, total_return, win_rate = self.backtest_strategy_vectorized(
                        df, temp_config, coin_config.symbol
                    )

                    strategy_time = time.time() - strategy_start_time

                    result = {
                        "coin": coin_config.name,
                        "symbol": coin_config.symbol,
                        "strategy": strategy.name,
                        "trades": len(trades),
                        "total_return": total_return,
                        "win_rate": win_rate,
                        "trade_details": trades,
                        "execution_time": strategy_time,
                    }

                    all_results.append(result)
                    logging.info(
                        f"   {strategy.name}: "
                        f"交易{len(trades)}次, "
                        f"報酬{total_return:.2%}, "
                        f"勝率{win_rate:.2%}, "
                        f"耗時{strategy_time:.2f}秒"
                    )

                    # 為個別策略生成圖表
                    if trades:
                        self.plot_backtest_results(
                            df, trades, coin_config.symbol, strategy.name
                        )
            else:
                # 正常模式：綜合策略
                logging.info(
                    f"📊 回測 {coin_config.name} " f"({coin_config.symbol}) - 綜合策略"
                )

                trades, total_return, win_rate = self.backtest_strategy_vectorized(
                    df, coin_config, coin_config.symbol
                )

                coin_time = time.time() - coin_start_time

                result = {
                    "coin": coin_config.name,
                    "symbol": coin_config.symbol,
                    "strategy": "Combined",
                    "trades": len(trades),
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "trade_details": trades,
                    "execution_time": coin_time,
                }

                all_results.append(result)

                logging.info(f"   交易次數: {len(trades)}")
                logging.info(f"   總報酬率: {total_return:.2%}")
                logging.info(f"   勝率: {win_rate:.2%}")
                logging.info(f"   耗時: {coin_time:.2f}秒")

                if trades:
                    self.plot_backtest_results(df, trades, coin_config.symbol)

        # 保存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_debug" if debug_mode else ""
        results_df = pd.DataFrame(
            [
                {
                    "coin": r["coin"],
                    "symbol": r["symbol"],
                    "strategy": r["strategy"],
                    "trades": r["trades"],
                    "total_return": r["total_return"],
                    "win_rate": r["win_rate"],
                    "execution_time": r["execution_time"],
                }
                for r in all_results
            ]
        )

        results_df.to_csv(
            f"output/backtest_results{mode_suffix}_{timestamp}.csv", index=False
        )

        total_time = time.time() - start_time
        logging.info(
            f"📄 結果已保存: " f"output/backtest_results{mode_suffix}_{timestamp}.csv"
        )
        logging.info(f"🏁 回測完成，總耗時: {total_time:.2f}秒")

        return all_results
