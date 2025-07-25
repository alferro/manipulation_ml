import asyncio
import logging
from typing import Dict, Any
import time
from datetime import datetime # Added for timestamp conversion

logger = logging.getLogger(__name__)


class DebugMonitor:
    def __init__(self, db_manager, trade_processor, orderbook_analyzer, check_interval_sec=180):
        self.db_manager = db_manager
        self.trade_processor = trade_processor
        self.orderbook_analyzer = orderbook_analyzer
        self.check_interval_sec = check_interval_sec

        self.start_time = time.time()
        self.last_stats = {}

    async def start_monitoring(self):
        """Запускает мониторинг системы в фоновом режиме"""
        logger.info("Debug monitor started")

        while True:
            try:
                await self.log_system_stats()
                await asyncio.sleep(self.check_interval_sec)
            except asyncio.CancelledError:
                logger.info("Debug monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in debug monitor: {e}")
                await asyncio.sleep(60)

    async def log_system_stats(self):
        """Выводит детальную статистику работы системы"""
        current_time = time.time()
        uptime_hours = (current_time - self.start_time) / 3600

        trade_stats = self.trade_processor.get_trade_stats()
        analysis_stats = self.orderbook_analyzer.get_analysis_stats()

        try:
            recent_alerts = await self.db_manager.get_alerts(limit=10, offset=0, symbol=None, alert_type=None,
                                                             status=None)
            # Ensure 'created_at' is a datetime object before calling timestamp()
            alert_count_24h = len([a for a in recent_alerts if current_time - (
                a['created_at'].timestamp() if isinstance(a['created_at'], datetime) else 0) < 86400])
        except Exception as e:
            logger.error(f"Error getting alert stats: {e}")
            alert_count_24h = 0

        logger.info("=" * 50)
        logger.info(f"SYSTEM STATUS (Uptime: {uptime_hours:.1f}h)")
        logger.info("=" * 50)

        total_trades = sum(trade_stats["processed_trades"].values())
        total_wash_alerts = sum(trade_stats.get("wash_trade_alerts", {}).values())
        total_ping_pong_alerts = sum(trade_stats.get("ping_pong_alerts", {}).values())
        total_ramping_alerts = sum(trade_stats.get("ramping_alerts", {}).values())

        logger.info(f"TRADES: Total processed: {total_trades}")
        logger.info(
            f"TRADE ALERTS: Wash={total_wash_alerts}, PingPong={total_ping_pong_alerts}, Ramping={total_ramping_alerts}")

        top_trade_symbols = sorted(trade_stats["processed_trades"].items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"TOP TRADE SYMBOLS: {top_trade_symbols}")

        total_analysis = sum(analysis_stats["analysis_counts"].values())
        total_alerts = sum(analysis_stats["alert_counts"].values())

        logger.info(f"ORDERBOOK ANALYSIS: Total: {total_analysis}")
        logger.info(f"ORDERBOOK ALERTS: Total={total_alerts}")

        top_analysis_symbols = sorted(analysis_stats["analysis_counts"].items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"TOP ANALYSIS SYMBOLS: {top_analysis_symbols}")

        logger.info(f"ALERTS LAST 24H: {alert_count_24h}")

        logger.info("-" * 30)
        logger.info("DETAILED SYMBOL STATS:")

        all_symbols = set(list(trade_stats["processed_trades"].keys()) + list(analysis_stats["analysis_counts"].keys()))

        for symbol in sorted(all_symbols)[:10]:
            trades = trade_stats["processed_trades"].get(symbol, 0)
            analyses = analysis_stats["analysis_counts"].get(symbol, 0)
            alerts = analysis_stats["alert_counts"].get(symbol, 0)
            wash_alerts = trade_stats.get("wash_trade_alerts", {}).get(symbol, 0)
            ping_pong_alerts = trade_stats.get("ping_pong_alerts", {}).get(symbol, 0)

            if trades > 0 or analyses > 0:
                logger.info(f"{symbol}: trades={trades}, analyses={analyses}, "
                            f"ob_alerts={alerts}, wash={wash_alerts}, ping_pong={ping_pong_alerts}")

        symbols_without_trades = [s for s in analysis_stats["analysis_counts"].keys()
                                  if trade_stats["processed_trades"].get(s, 0) == 0]

        if symbols_without_trades:
            logger.warning(f"SYMBOLS WITHOUT TRADES: {symbols_without_trades[:10]}")

        symbols_without_analysis = [s for s in trade_stats["processed_trades"].keys()
                                    if analysis_stats["analysis_counts"].get(s, 0) == 0]

        if symbols_without_analysis:
            logger.warning(f"SYMBOLS WITHOUT ANALYSIS: {symbols_without_analysis[:10]}")

        if total_trades > 10000 and total_alerts == 0:
            logger.warning("HIGH TRADE VOLUME BUT NO ALERTS - CHECK DETECTION THRESHOLDS!")

        logger.info("=" * 50)

    async def check_alert_generation(self):
        """Проверяет, генерируются ли алерты"""
        try:
            recent_alerts = await self.db_manager.get_alerts(limit=1, offset=0, symbol=None, alert_type=None,
                                                             status=None)
            if recent_alerts:
                last_alert_time = recent_alerts[0]['created_at']
                time_since_last = time.time() - (
                    last_alert_time.timestamp() if isinstance(last_alert_time, datetime) else 0)

                if time_since_last > 3600:
                    logger.warning(f"No alerts generated for {time_since_last / 3600:.1f} hours!")
                    return False
            else:
                logger.warning("No alerts found in database!")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking alert generation: {e}")
            return False
