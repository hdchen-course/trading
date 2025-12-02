#!/usr/bin/env python3
"""
加密貨幣策略交易系統 - 主程式

功能:
- 多策略技術分析回測系統
- 實時監控與信號通知
- 支援個別策略和綜合策略測試
- 可配置的資金管理和風險控制
"""

import argparse
import glob
import logging
import os
from datetime import datetime, timezone

import matplotlib

from config.coin_configs import create_coin_configs
from config.trading_config import TradingConfig
from tests.test_backtest import debug_mode
from trading.backtest import CryptoBacktester
from trading.monitor import CryptoMonitor

# 設置 matplotlib 後端和字體配置
# 使用 Agg 後端避免 GUI 依賴，適合服務器環境
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Liberation Sans",  # Linux 系統字體
    "Arial",  # Windows 系統字體
    "DejaVu Sans",  # 跨平台字體
    "sans-serif",  # 後備字體
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題


def setup_logging():
    """
    設置系統日誌配置

    配置日誌格式、級別和輸出方式
    使用 INFO 級別記錄重要操作和結果
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("trading_system.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def show_menu():
    """
    顯示主選單界面

    提供用戶選擇系統功能的交互界面
    包含回測、監控和退出選項
    """
    print("\n" + "=" * 50)
    print("🚀 加密貨幣策略交易系統")
    print("=" * 50)
    print("1. 📊 執行回測")
    print("2. 👁️  開始監控")
    print("3. ❌ 退出")
    print("=" * 50)


def main():
    """
    主程式入口點

    處理命令行參數，根據參數決定執行模式：
    - --debug: 執行單元測試和詳細回測
    - 無參數: 進入交互式選單模式

    功能包括：
    1. 單元測試驗證
    2. 個別策略回測
    3. 綜合策略回測
    4. 實時監控
    """
    setup_logging()

    parser = argparse.ArgumentParser(
        description="加密貨幣策略交易系統",
        epilog="""
使用範例:
  1. 交互式模式（默認）:
     python main.py

  2. Debug 回測模式:
     python main.py --debug --days 7

  3. 分析模式:
     python main.py --analyze --start 2025-11-01 --end 2025-11-10
     python main.py --analyze --start 2025-11-01 --end 2025-11-10
         --output analysis.csv

  4. 視覺化模式:
     python main.py --visualize --start 2025-11-01 --end 2025-11-10
     python main.py --visualize --start 2025-11-01 --end 2025-11-10
         --methods macd lstm --output chart.png

  5. 生成訓練數據:
     python main.py --generate-data
     python main.py --generate-data --start 2022-01-01 --end 2025-09-01
         --interval 15m --window 16

  6. 分析信號強度:
     python main.py --analyze-strength latest
     python main.py --analyze-strength
         output/trades_BTCUSDT_Combined_20251128_210145.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Debug 模式")
    parser.add_argument("--days", type=int, default=30, help="回測天數 (默認30天)")
    parser.add_argument(
        "--analyze", action="store_true", help="分析模式 - 分析指定日期的技術指標"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="視覺化模式 - 顯示檢測信號圖表"
    )
    parser.add_argument(
        "--start",
        "--start-date",
        type=str,
        dest="start_date",
        help="開始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", "--end-date", type=str, dest="end_date", help="結束日期 (YYYY-MM-DD)"
    )
    parser.add_argument("--output", type=str, help="輸出文件路徑 (CSV 或 PNG)")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=[
            "macd",
            "macdv1",
            "lstm",
            "volume",
            "btc",
            "peak",
            "triple",
            "ml",
            "mlv3",
            "mlv4",
        ],
        help="檢測方法 (visualize 模式使用，默認全部)",
    )
    parser.add_argument("--generate-data", action="store_true", help="生成訓練數據模式")
    parser.add_argument(
        "--interval",
        type=str,
        default="15m",
        choices=["15m", "1h", "4h", "1d"],
        help="K線間隔 (generate-data 模式使用，默認 15m)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=16,
        help="窗口大小 (generate-data 模式使用，默認 16)",
    )
    parser.add_argument(
        "--analyze-strength",
        type=str,
        metavar="CSV_FILE",
        help="分析交易記錄的信號強度 (指定 CSV 文件路徑，或使用 'latest' 自動選擇最新文件)",
    )
    parser.add_argument(
        "--debug_monitor",
        action="store_true",
        help="Debug 監控模式 - 測試特定時間點的信號檢測",
    )
    parser.add_argument(
        "--now",
        type=str,
        help=(
            "指定當前時間點 (格式: YYYY-MM-DD HH:MM 或 "
            "YYYY-MM-DD HH:MM:SS)，用於 debug_monitor 模式"
        ),
    )
    args = parser.parse_args()

    if args.debug:
        debug_mode(args.days)
        return

    if args.debug_monitor:
        if not args.now:
            print("❌ 錯誤: --debug_monitor 模式需要 --now 參數")
            print("範例: python main.py --debug_monitor --now '2024-12-01 22:30'")
            return

        # 解析時間（明確指定為 UTC）
        try:
            if len(args.now.split(":")) == 2:
                # 格式: YYYY-MM-DD HH:MM
                debug_time = datetime.strptime(args.now, "%Y-%m-%d %H:%M")
            else:
                # 格式: YYYY-MM-DD HH:MM:SS
                debug_time = datetime.strptime(args.now, "%Y-%m-%d %H:%M:%S")

            # 明確設置為 UTC 時區
            debug_time = debug_time.replace(tzinfo=timezone.utc)
        except ValueError as e:
            print(f"❌ 時間格式錯誤: {e}")
            print("正確格式: 'YYYY-MM-DD HH:MM' 或 'YYYY-MM-DD HH:MM:SS'")
            print("範例: '2024-12-01 22:30' 或 '2024-12-01 22:30:00'")
            return

        print("=" * 80)
        print("🔍 Debug Monitor - 測試特定時間點的信號檢測")
        print("=" * 80)
        print(f"\n📅 目標時間: {debug_time.replace(tzinfo=None)} (UTC)")

        config = TradingConfig()
        coin_configs = create_coin_configs()

        # 只檢測 BTC
        btc_config = None
        for c in coin_configs:
            if c.symbol == "BTCUSDT":
                btc_config = c
                break

        if not btc_config:
            print("❌ 找不到 BTCUSDT 配置")
            return

        print(f"\n🔍 檢測幣種: {btc_config.name} ({btc_config.symbol})")

        # 創建 debug 模式的 monitor
        monitor = CryptoMonitor(config, debug_mode=True, debug_time=debug_time)

        # 檢查信號
        monitor.check_signals(btc_config)

        print("\n" + "=" * 80)
        print("✅ Debug Monitor 完成")
        print("=" * 80)
        return

    if args.analyze_strength:
        from utils.signal_strength_analyzer import analyze_signal_strength

        if args.analyze_strength.lower() == "latest":
            # 自動選擇最新的 Combined 文件
            files = glob.glob("output/trades_*_Combined_*.csv")
            if files:
                csv_file = max(files, key=os.path.getctime)
                print(f"📁 使用最新的交易記錄: {csv_file}\n")
            else:
                print("❌ 找不到交易記錄文件")
                return
        else:
            csv_file = args.analyze_strength

        analyze_signal_strength(csv_file)
        return

    if args.analyze:
        if not args.start_date or not args.end_date:
            print("❌ 錯誤: --analyze 模式需要 --start 和 --end 參數")
            print("範例: python main.py --analyze --start 2025-11-01 --end 2025-11-10")
            return
        from scripts.analyze import run_analysis

        run_analysis(args.start_date, args.end_date, args.output)
        return

    if args.visualize:
        if not args.start_date or not args.end_date:
            print("❌ 錯誤: --visualize 模式需要 --start 和 --end 參數")
            print(
                "範例: python main.py --visualize --start 2025-11-01 --end 2025-11-10"
            )
            return
        from scripts.visualize import run_visualization

        run_visualization(args.start_date, args.end_date, args.methods, args.output)
        return

    if args.generate_data:
        from utils.generate_training_data import main as generate_data_main

        start_date = args.start_date or "2022-01-01"
        end_date = args.end_date or "2025-09-01"
        generate_data_main(
            start_date=start_date,
            end_date=end_date,
            interval=args.interval,
            window_size=args.window,
        )
        return

    config = TradingConfig()
    coin_configs = create_coin_configs()

    while True:
        show_menu()

        try:
            choice = input("請選擇功能 (1-3): ").strip()

            if choice == "1":
                print("\n📊 選擇回測期間:")
                print("1. 一週")
                print("2. 一個月")
                print("3. 三個月")

                period_choice = input("請選擇回測期間 (1-3): ").strip()

                if period_choice == "1":
                    days = 7
                elif period_choice == "2":
                    days = 30
                elif period_choice == "3":
                    days = 90
                else:
                    print("❌ 無效選擇")
                    continue

                print(f"\n📊 開始 {days} 天回測...")
                backtester = CryptoBacktester(config)
                results = backtester.run_backtest(coin_configs, days)

                logging.info("\n=== 回測總結 ===")
                for result in results:
                    logging.info(
                        f"{result['coin']}: "
                        f"交易{result['trades']}次, "
                        f"報酬{result['total_return']:.2%}, "
                        f"勝率{result['win_rate']:.2%}"
                    )

            elif choice == "2":
                print("\n👁️  開始監控...")

                # 過濾啟用監控的幣種
                enabled_configs = [c for c in coin_configs if c.enable_monitor]

                if not enabled_configs:
                    print("❌ 沒有啟用監控的幣種")
                    print("請在 config/coin_configs.py 中設置 enable_monitor=True")
                    continue

                print("\n🔍 監控以下幣種:")
                for c in enabled_configs:
                    print(f"  - {c.name} ({c.symbol})")

                monitor = CryptoMonitor(config)
                monitor.monitor_coins(enabled_configs)

            elif choice == "3":
                print("\n👋 感謝使用！")
                break

            else:
                print("❌ 無效選擇，請重新輸入")

        except KeyboardInterrupt:
            print("\n👋 程式已中斷")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")


if __name__ == "__main__":
    main()
