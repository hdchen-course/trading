#!/usr/bin/env python3
"""
測試 Triple Confirmation Strategy

單獨測試新策略的表現，不影響其他策略
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies import TripleConfirmationStrategy, StrategyConfig
from utils.indicators import calculate_technical_indicators


def load_test_data(days=30):
    """載入測試數據"""
    df = pd.read_csv("training_data/btc_15m_2022_2025.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 使用最近N天的數據
    end_date = df["timestamp"].max()
    start_date = end_date - timedelta(days=days)
    
    mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
    test_df = df[mask].copy().reset_index(drop=True)
    
    print(f"測試數據: {len(test_df)} 筆記錄")
    print(f"時間範圍: {test_df['timestamp'].min()} 到 {test_df['timestamp'].max()}")
    
    return test_df


def backtest_strategy(df, strategy):
    """簡單回測"""
    # 生成信號
    signals = strategy.generate_signals(df)
    
    # 統計信號
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    total_signals = buy_signals + sell_signals
    
    print(f"\n信號統計:")
    print(f"  買入信號: {buy_signals}")
    print(f"  賣出信號: {sell_signals}")
    print(f"  總信號數: {total_signals}")
    
    if total_signals == 0:
        print("\n⚠️  沒有產生任何信號")
        return
    
    # 簡單回測：每次信號交易
    capital = 10000  # 初始資金
    position = 0  # 持倉
    trades = []
    
    for i in range(len(df)):
        if signals.iloc[i] == 1 and position == 0:
            # 買入
            position = capital / df['close'].iloc[i]
            entry_price = df['close'].iloc[i]
            entry_time = df['timestamp'].iloc[i]
            trades.append({
                'type': 'BUY',
                'time': entry_time,
                'price': entry_price,
                'position': position
            })
        
        elif signals.iloc[i] == -1 and position > 0:
            # 賣出
            exit_price = df['close'].iloc[i]
            exit_time = df['timestamp'].iloc[i]
            capital = position * exit_price
            profit = capital - 10000
            profit_pct = (capital / 10000 - 1) * 100
            
            trades.append({
                'type': 'SELL',
                'time': exit_time,
                'price': exit_price,
                'profit': profit,
                'profit_pct': profit_pct
            })
            position = 0
    
    # 如果還有持倉，以最後價格平倉
    if position > 0:
        exit_price = df['close'].iloc[-1]
        capital = position * exit_price
        profit = capital - 10000
        profit_pct = (capital / 10000 - 1) * 100
        
        trades.append({
            'type': 'SELL (CLOSE)',
            'time': df['timestamp'].iloc[-1],
            'price': exit_price,
            'profit': profit,
            'profit_pct': profit_pct
        })
    
    # 統計交易結果
    completed_trades = [t for t in trades if 'profit' in t]
    
    if len(completed_trades) == 0:
        print("\n⚠️  沒有完成的交易")
        return
    
    profits = [t['profit'] for t in completed_trades]
    profit_pcts = [t['profit_pct'] for t in completed_trades]
    
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    print(f"\n交易結果:")
    print(f"  完成交易: {len(completed_trades)} 次")
    print(f"  獲利交易: {len(winning_trades)} 次")
    print(f"  虧損交易: {len(losing_trades)} 次")
    print(f"  勝率: {len(winning_trades)/len(completed_trades)*100:.2f}%")
    
    print(f"\n收益統計:")
    print(f"  總收益: ${sum(profits):.2f}")
    print(f"  平均收益: ${np.mean(profits):.2f}")
    print(f"  平均收益率: {np.mean(profit_pcts):.2f}%")
    print(f"  最大收益: ${max(profits):.2f} ({max(profit_pcts):.2f}%)")
    print(f"  最大虧損: ${min(profits):.2f} ({min(profit_pcts):.2f}%)")
    
    print(f"\n最終資金: ${capital:.2f}")
    print(f"總報酬率: {(capital/10000 - 1)*100:.2f}%")
    
    # 顯示前5筆交易
    print(f"\n前5筆交易:")
    for i, trade in enumerate(completed_trades[:5]):
        print(f"  {i+1}. {trade['time']}: {trade['type']} @ ${trade['price']:.2f} "
              f"-> 收益 ${trade['profit']:.2f} ({trade['profit_pct']:.2f}%)")


def test_different_params():
    """測試不同參數組合"""
    print("=" * 60)
    print("測試不同參數組合")
    print("=" * 60)
    
    df = load_test_data(days=90)
    
    param_combinations = [
        {"distance": 15, "window_size": 12, "signal_type": "both"},
        {"distance": 20, "window_size": 16, "signal_type": "both"},
        {"distance": 25, "window_size": 20, "signal_type": "both"},
        {"distance": 20, "window_size": 16, "signal_type": "buy"},
        {"distance": 20, "window_size": 16, "signal_type": "sell"},
    ]
    
    results = []
    
    for params in param_combinations:
        print(f"\n{'='*60}")
        print(f"測試參數: {params}")
        print(f"{'='*60}")
        
        config = StrategyConfig(
            name="Triple Confirmation Test",
            params=params
        )
        
        strategy = TripleConfirmationStrategy(config)
        
        # 生成信號
        signals = strategy.generate_signals(df)
        
        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()
        total_signals = buy_signals + sell_signals
        
        print(f"信號數: 買入={buy_signals}, 賣出={sell_signals}, 總計={total_signals}")
        
        results.append({
            'params': params,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': total_signals
        })
    
    # 總結
    print(f"\n{'='*60}")
    print("參數測試總結")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n參數: {result['params']}")
        print(f"  信號數: {result['total_signals']} (買入={result['buy_signals']}, 賣出={result['sell_signals']})")


def main():
    """主測試函數"""
    print("=" * 60)
    print("Triple Confirmation Strategy 測試")
    print("=" * 60)
    
    # 測試1: 基本功能測試
    print("\n測試 1: 基本功能測試 (30天)")
    print("-" * 60)
    
    df = load_test_data(days=30)
    
    config = StrategyConfig(
        name="Triple Confirmation Reversal",
        params={
            "distance": 20,
            "window_size": 16,
            "signal_type": "both"
        }
    )
    
    strategy = TripleConfirmationStrategy(config)
    print(f"\n策略描述: {strategy.get_description()}")
    
    backtest_strategy(df, strategy)
    
    # 測試2: 不同時間範圍
    print("\n\n測試 2: 不同時間範圍")
    print("-" * 60)
    
    for days in [7, 30, 90]:
        print(f"\n{'='*60}")
        print(f"測試 {days} 天數據")
        print(f"{'='*60}")
        
        df = load_test_data(days=days)
        backtest_strategy(df, strategy)
    
    # 測試3: 不同參數組合
    print("\n\n測試 3: 不同參數組合")
    print("-" * 60)
    test_different_params()
    
    print("\n" + "=" * 60)
    print("✅ 所有測試完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
