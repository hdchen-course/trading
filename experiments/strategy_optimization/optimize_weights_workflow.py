#!/usr/bin/env python3
"""
策略權重優化工作流程

完整流程：
1. 回測180天，獲取每個策略的表現（勝率、報酬率）
2. 根據表現重新計算權重
3. 使用新權重再次回測
4. 分析信號強度與盈虧的關係
5. 生成優化建議
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from config.trading_config import TradingConfig
from config.coin_configs import create_coin_configs
from trading.backtest import CryptoBacktester
from strategies import StrategyConfig, CoinConfig

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def step1_initial_backtest(days=180):
    """
    步驟1: 初始回測 - 測試每個策略的個別表現
    
    Args:
        days: 回測天數
        
    Returns:
        dict: 每個幣種的策略表現數據
    """
    print("=" * 80)
    print("步驟 1: 初始回測 - 測試每個策略的個別表現")
    print("=" * 80)
    
    config = TradingConfig()
    coin_configs = create_coin_configs()
    backtester = CryptoBacktester(config)
    
    # 使用 debug 模式測試每個策略
    results = backtester.run_backtest(coin_configs, period_days=days, debug_mode=True)
    
    # 整理結果
    strategy_performance = {}
    
    for result in results:
        coin = result['coin']
        if coin not in strategy_performance:
            strategy_performance[coin] = []
        
        strategy_performance[coin].append({
            'strategy_name': result['strategy'],
            'trades': result['trades'],
            'total_return': result['total_return'],
            'win_rate': result['win_rate'],
            'trade_details': result['trade_details']
        })
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for coin, strategies in strategy_performance.items():
        df = pd.DataFrame([{
            'strategy': s['strategy_name'],
            'trades': s['trades'],
            'total_return': s['total_return'],
            'win_rate': s['win_rate']
        } for s in strategies])
        
        filename = f"output/step1_initial_backtest_{coin}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ {coin} 策略表現已保存: {filename}")
        
        # 顯示表現
        print(f"\n{coin} 策略表現:")
        print(df.to_string(index=False))
    
    return strategy_performance


def step2_calculate_new_weights(strategy_performance, method='balanced'):
    """
    步驟2: 根據表現計算新權重
    
    Args:
        strategy_performance: 步驟1的結果
        method: 權重計算方法
            - 'balanced': 平衡勝率和報酬率
            - 'win_rate': 只看勝率
            - 'return': 只看報酬率
            - 'sharpe': 使用夏普比率（需要更多數據）
            
    Returns:
        dict: 新的權重配置
    """
    print("\n" + "=" * 80)
    print("步驟 2: 計算新權重")
    print("=" * 80)
    
    new_weights = {}
    
    for coin, strategies in strategy_performance.items():
        print(f"\n{coin} 權重計算:")
        print("-" * 60)
        
        # 過濾掉沒有交易或表現太差的策略
        valid_strategies = [
            s for s in strategies 
            if s['trades'] > 0 and s['total_return'] > -0.5  # 虧損不超過50%
        ]
        
        if not valid_strategies:
            print(f"⚠️  {coin} 沒有有效策略")
            continue
        
        # 計算權重
        if method == 'balanced':
            # 平衡勝率和報酬率
            scores = []
            for s in valid_strategies:
                # 標準化勝率 (0-1)
                win_rate_score = s['win_rate']
                
                # 標準化報酬率 (將負值轉為0，正值保持)
                return_score = max(0, s['total_return'])
                
                # 綜合分數 (60% 勝率 + 40% 報酬率)
                combined_score = win_rate_score * 0.6 + return_score * 0.4
                scores.append(combined_score)
            
        elif method == 'win_rate':
            scores = [s['win_rate'] for s in valid_strategies]
            
        elif method == 'return':
            scores = [max(0, s['total_return']) for s in valid_strategies]
        
        # 歸一化權重
        total_score = sum(scores)
        if total_score > 0:
            weights = [score / total_score for score in scores]
        else:
            # 如果所有分數都是0，平均分配
            weights = [1.0 / len(valid_strategies)] * len(valid_strategies)
        
        # 保存新權重
        new_weights[coin] = []
        for strategy, weight in zip(valid_strategies, weights):
            new_weights[coin].append({
                'strategy_name': strategy['strategy_name'],
                'old_weight': 0.0,  # 需要從原始配置獲取
                'new_weight': weight,
                'win_rate': strategy['win_rate'],
                'total_return': strategy['total_return'],
                'trades': strategy['trades']
            })
            
            print(f"  {strategy['strategy_name']:40s} "
                  f"勝率={strategy['win_rate']:.2%} "
                  f"報酬={strategy['total_return']:+.2%} "
                  f"→ 新權重={weight:.4f}")
    
    # 保存權重配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for coin, weights_list in new_weights.items():
        df = pd.DataFrame(weights_list)
        filename = f"output/step2_new_weights_{coin}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ {coin} 新權重已保存: {filename}")
    
    return new_weights


def step3_backtest_with_new_weights(new_weights, days=180):
    """
    步驟3: 使用新權重回測
    
    Args:
        new_weights: 步驟2計算的新權重
        days: 回測天數
        
    Returns:
        dict: 回測結果
    """
    print("\n" + "=" * 80)
    print("步驟 3: 使用新權重回測")
    print("=" * 80)
    
    config = TradingConfig()
    backtester = CryptoBacktester(config)
    
    # 創建新的配置（使用新權重）
    original_configs = create_coin_configs()
    new_configs = []
    
    for coin_config in original_configs:
        coin_name = coin_config.name
        
        if coin_name not in new_weights:
            print(f"⚠️  {coin_name} 沒有新權重，跳過")
            continue
        
        # 創建新的策略列表（使用新權重）
        new_strategies = []
        weight_map = {w['strategy_name']: w['new_weight'] for w in new_weights[coin_name]}
        
        for strategy in coin_config.strategies:
            if strategy.name in weight_map:
                # 更新權重
                new_config = StrategyConfig(
                    name=strategy.name,
                    params=strategy.params,
                    weight=weight_map[strategy.name]
                )
                # 重新創建策略實例
                strategy_class = type(strategy)
                new_strategy = strategy_class(new_config)
                new_strategies.append(new_strategy)
        
        if new_strategies:
            new_coin_config = CoinConfig(
                coin_config.name,
                coin_config.symbol,
                new_strategies,
                coin_config.leverage
            )
            new_configs.append(new_coin_config)
    
    # 執行回測
    results = backtester.run_backtest(new_configs, period_days=days, debug_mode=False)
    
    # 整理結果
    backtest_results = {}
    for result in results:
        backtest_results[result['coin']] = {
            'trades': result['trades'],
            'total_return': result['total_return'],
            'win_rate': result['win_rate'],
            'trade_details': result['trade_details']
        }
    
    # 保存結果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_data = []
    
    for coin, result in backtest_results.items():
        summary_data.append({
            'coin': coin,
            'trades': result['trades'],
            'total_return': result['total_return'],
            'win_rate': result['win_rate']
        })
    
    df = pd.DataFrame(summary_data)
    filename = f"output/step3_optimized_backtest_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ 優化後回測結果已保存: {filename}")
    print("\n優化後表現:")
    print(df.to_string(index=False))
    
    return backtest_results


def step4_analyze_signal_strength(backtest_results):
    """
    步驟4: 分析信號強度與盈虧的關係
    
    Args:
        backtest_results: 步驟3的回測結果
    """
    print("\n" + "=" * 80)
    print("步驟 4: 分析信號強度與盈虧關係")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for coin, result in backtest_results.items():
        trades = result['trade_details']
        
        if not trades:
            print(f"\n⚠️  {coin} 沒有交易記錄")
            continue
        
        print(f"\n{coin} 信號強度分析:")
        print("-" * 60)
        
        # 轉換為 DataFrame
        df = pd.DataFrame(trades)
        
        # 檢查是否有 signal_strength 欄位
        if 'signal_strength' not in df.columns:
            print(f"⚠️  交易記錄中沒有 signal_strength 欄位")
            continue
        
        # 分析盈虧與信號強度的關係
        df['is_profit'] = df['pnl'] > 0
        
        # 按信號強度分組分析
        strength_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
        strength_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '>1.0']
        
        df['strength_bin'] = pd.cut(df['signal_strength'], bins=strength_bins, labels=strength_labels)
        
        # 統計每個區間的表現
        analysis = df.groupby('strength_bin').agg({
            'pnl': ['count', 'mean', 'sum'],
            'is_profit': 'mean',
            'return_pct': 'mean'
        }).round(4)
        
        print("\n信號強度區間分析:")
        print(analysis)
        
        # 找出虧損交易的信號強度分布
        losing_trades = df[df['pnl'] < 0]
        if len(losing_trades) > 0:
            print(f"\n虧損交易 ({len(losing_trades)} 筆):")
            print(f"  平均信號強度: {losing_trades['signal_strength'].mean():.4f}")
            print(f"  信號強度範圍: {losing_trades['signal_strength'].min():.4f} - {losing_trades['signal_strength'].max():.4f}")
            print(f"  中位數: {losing_trades['signal_strength'].median():.4f}")
            
            # 信號強度分布
            print("\n虧損交易信號強度分布:")
            print(losing_trades['strength_bin'].value_counts().sort_index())
        
        # 找出獲利交易的信號強度分布
        profit_trades = df[df['pnl'] > 0]
        if len(profit_trades) > 0:
            print(f"\n獲利交易 ({len(profit_trades)} 筆):")
            print(f"  平均信號強度: {profit_trades['signal_strength'].mean():.4f}")
            print(f"  信號強度範圍: {profit_trades['signal_strength'].min():.4f} - {profit_trades['signal_strength'].max():.4f}")
            print(f"  中位數: {profit_trades['signal_strength'].median():.4f}")
        
        # 繪製圖表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 圖1: 信號強度分布（盈虧對比）
        axes[0, 0].hist([profit_trades['signal_strength'], losing_trades['signal_strength']], 
                       bins=20, label=['獲利', '虧損'], alpha=0.7)
        axes[0, 0].set_xlabel('信號強度')
        axes[0, 0].set_ylabel('交易數量')
        axes[0, 0].set_title('信號強度分布（盈虧對比）')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 圖2: 信號強度 vs 盈虧
        axes[0, 1].scatter(df['signal_strength'], df['pnl'], alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('信號強度')
        axes[0, 1].set_ylabel('盈虧 (USD)')
        axes[0, 1].set_title('信號強度 vs 盈虧')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 圖3: 信號強度區間的勝率
        win_rate_by_strength = df.groupby('strength_bin')['is_profit'].mean()
        win_rate_by_strength.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_xlabel('信號強度區間')
        axes[1, 0].set_ylabel('勝率')
        axes[1, 0].set_title('各信號強度區間的勝率')
        axes[1, 0].grid(True, alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # 圖4: 信號強度區間的平均報酬
        avg_return_by_strength = df.groupby('strength_bin')['return_pct'].mean()
        avg_return_by_strength.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_xlabel('信號強度區間')
        axes[1, 1].set_ylabel('平均報酬率')
        axes[1, 1].set_title('各信號強度區間的平均報酬率')
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存圖表
        chart_file = f"output/step4_signal_strength_analysis_{coin}_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ 圖表已保存: {chart_file}")
        
        # 保存詳細數據
        detail_file = f"output/step4_signal_strength_details_{coin}_{timestamp}.csv"
        df.to_csv(detail_file, index=False)
        print(f"✅ 詳細數據已保存: {detail_file}")


def step5_generate_recommendations(backtest_results):
    """
    步驟5: 生成優化建議
    
    Args:
        backtest_results: 步驟3的回測結果
    """
    print("\n" + "=" * 80)
    print("步驟 5: 生成優化建議")
    print("=" * 80)
    
    recommendations = []
    
    for coin, result in backtest_results.items():
        trades = result['trade_details']
        
        if not trades:
            continue
        
        df = pd.DataFrame(trades)
        
        if 'signal_strength' not in df.columns:
            continue
        
        print(f"\n{coin} 優化建議:")
        print("-" * 60)
        
        # 分析虧損交易
        losing_trades = df[df['pnl'] < 0]
        profit_trades = df[df['pnl'] > 0]
        
        if len(losing_trades) > 0 and len(profit_trades) > 0:
            # 計算最佳信號強度閾值
            losing_avg_strength = losing_trades['signal_strength'].mean()
            profit_avg_strength = profit_trades['signal_strength'].mean()
            
            # 建議1: 信號強度過濾
            if losing_avg_strength < profit_avg_strength:
                threshold = (losing_avg_strength + profit_avg_strength) / 2
                filtered_df = df[df['signal_strength'] >= threshold]
                
                if len(filtered_df) > 0:
                    new_win_rate = (filtered_df['pnl'] > 0).mean()
                    original_win_rate = (df['pnl'] > 0).mean()
                    
                    print(f"\n建議 1: 設置信號強度閾值")
                    print(f"  建議閾值: {threshold:.4f}")
                    print(f"  過濾前勝率: {original_win_rate:.2%}")
                    print(f"  過濾後勝率: {new_win_rate:.2%}")
                    print(f"  保留交易數: {len(filtered_df)}/{len(df)} ({len(filtered_df)/len(df):.1%})")
                    
                    recommendations.append({
                        'coin': coin,
                        'type': 'signal_strength_filter',
                        'threshold': threshold,
                        'improvement': new_win_rate - original_win_rate
                    })
            
            # 建議2: 識別最差表現區間
            df['strength_bin'] = pd.cut(df['signal_strength'], bins=5)
            worst_bin = df.groupby('strength_bin')['pnl'].mean().idxmin()
            worst_bin_win_rate = (df[df['strength_bin'] == worst_bin]['pnl'] > 0).mean()
            
            print(f"\n建議 2: 避免特定信號強度區間")
            print(f"  最差區間: {worst_bin}")
            print(f"  該區間勝率: {worst_bin_win_rate:.2%}")
            print(f"  建議: 過濾掉該區間的信號")
    
    # 保存建議
    if recommendations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_rec = pd.DataFrame(recommendations)
        filename = f"output/step5_recommendations_{timestamp}.csv"
        df_rec.to_csv(filename, index=False)
        print(f"\n✅ 優化建議已保存: {filename}")


def main():
    """主流程"""
    print("=" * 80)
    print("策略權重優化工作流程")
    print("=" * 80)
    
    try:
        # 步驟1: 初始回測
        strategy_performance = step1_initial_backtest(days=180)
        
        # 步驟2: 計算新權重
        new_weights = step2_calculate_new_weights(strategy_performance, method='balanced')
        
        # 步驟3: 使用新權重回測
        backtest_results = step3_backtest_with_new_weights(new_weights, days=180)
        
        # 步驟4: 分析信號強度
        step4_analyze_signal_strength(backtest_results)
        
        # 步驟5: 生成建議
        step5_generate_recommendations(backtest_results)
        
        print("\n" + "=" * 80)
        print("✅ 完整工作流程執行完成！")
        print("=" * 80)
        print("\n所有結果已保存在 output/ 目錄")
        
    except Exception as e:
        print(f"\n❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
