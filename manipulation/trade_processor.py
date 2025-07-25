import logging
import time
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional # Added Optional
import asyncio
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TradeProcessor:
    def __init__(self, db_manager,
                 volume_window_minutes: int = int(os.getenv("VOLUME_WINDOW_MINUTES", 5)),
                 volume_multiplier: float = float(os.getenv("VOLUME_MULTIPLIER", 3.0)),
                 min_volume_usdt: float = float(os.getenv("MIN_VOLUME_USDT", 100000)),
                 wash_trade_threshold_ratio: float = float(os.getenv("WASH_TRADE_THRESHOLD_RATIO", 0.75)),
                 ping_pong_window_sec: int = int(os.getenv("PING_PONG_WINDOW_SEC", 45)),
                 ramping_window_sec: int = int(os.getenv("RAMPING_WINDOW_SEC", 90)),
                 consecutive_long_count: int = int(os.getenv("CONSECUTIVE_LONG_COUNT", 5)),
                 alert_grouping_minutes: int = int(os.getenv("ALERT_GROUPING_MINUTES", 5)),
                 data_retention_hours: int = int(os.getenv("DATA_RETENTION_HOURS", 24)),
                 volume_type: str = os.getenv("VOLUME_TYPE", "long")):

        self.db_manager = db_manager

        self.volume_window_minutes = volume_window_minutes
        self.volume_multiplier = volume_multiplier
        self.min_volume_usdt = min_volume_usdt
        self.wash_trade_threshold_ratio = wash_trade_threshold_ratio
        self.ping_pong_window_sec = ping_pong_window_sec
        self.ramping_window_sec = ramping_window_sec
        self.consecutive_long_count = consecutive_long_count
        self.alert_grouping_minutes = alert_grouping_minutes
        self.data_retention_hours = data_retention_hours
        self.volume_type = volume_type

        self.trade_history = defaultdict(lambda: deque(maxlen=10000))

        self.price_volume_profile = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.trade_sequences = defaultdict(lambda: deque(maxlen=100))
        self.volume_clusters = defaultdict(lambda: deque(maxlen=50))

        self.kline_data = defaultdict(lambda: {'open': 0.0, 'high': 0.0, 'low': 0.0, 'close': 0.0, 'volume': 0.0, 'start_time': 0.0, 'is_closed': False})
        self.consecutive_long_counts = defaultdict(int)

        self.processed_trades = defaultdict(int)
        self.wash_trade_alerts = defaultdict(int)
        self.ping_pong_alerts = defaultdict(int)
        self.ramping_alerts = defaultdict(int)

        self.recent_alerts = defaultdict(lambda: defaultdict(float))
        self.alert_grouping_seconds = self.alert_grouping_minutes * 60

        logger.info("TradeProcessor initialized with parameters:")
        logger.info(f"  Volume Window: {self.volume_window_minutes} min, Multiplier: {self.volume_multiplier}x, Min USDT: {self.min_volume_usdt}")
        logger.info(f"  Wash Trade Threshold: {self.wash_trade_threshold_ratio}, Ping-Pong Window: {self.ping_pong_window_sec}s")
        logger.info(f"  Ramping Window: {self.ramping_window_sec}s, Consecutive Long Count: {self.consecutive_long_count}")
        logger.info(f"  Alert Grouping: {self.alert_grouping_minutes} min, Data Retention: {self.data_retention_hours} hours")


    async def add_trade(self, symbol: str, trade_data: dict):
        """Adds a trade to history and triggers all analyzers"""
        try:
            timestamp_ms = trade_data.get('T')
            price_str = trade_data.get('p')
            size_str = trade_data.get('v')
            side = trade_data.get('S', 'unknown')
            trade_id = trade_data.get('i', f"{timestamp_ms}_{price_str}")

            if not all([timestamp_ms, price_str, size_str]):
                logger.debug(f"Incomplete trade data for {symbol}: {trade_data}")
                return

            timestamp_sec = timestamp_ms / 1000.0
            price = float(price_str)
            size = float(size_str)
            volume_usdt = price * size

            trade_record = {
                'timestamp': timestamp_sec,
                'price': price,
                'size': size,
                'volume_usdt': volume_usdt,
                'side': side,
                'trade_id': trade_id,
                'price_level': round(price, 8)
            }

            self.trade_history[symbol].append(trade_record)
            self.processed_trades[symbol] += 1

            self._update_price_volume_profile(symbol, trade_record)
            self._update_trade_sequences(symbol, trade_record)
            self._update_volume_clusters(symbol, trade_record)
            self._update_kline_data(symbol, trade_record)

            if self.processed_trades[symbol] % 1000 == 0:
                logger.info(f"Processed {self.processed_trades[symbol]} trades for {symbol}, "
                            f"alerts: wash={self.wash_trade_alerts[symbol]}, "
                            f"ping_pong={self.ping_pong_alerts[symbol]}, "
                            f"ramping={self.ramping_alerts[symbol]}")

            await self._run_all_trade_detectors(symbol, trade_record)

        except (ValueError, TypeError) as e:
            logger.error(f"Error processing trade for {symbol}: {e}, data: {trade_data}")
        except Exception as e:
            logger.error(f"Unexpected error processing trade for {symbol}: {e}")

    def _update_price_volume_profile(self, symbol: str, trade: dict):
        """Updates price-volume profile"""
        price_level = trade['price_level']
        self.price_volume_profile[symbol][price_level].append({
            'timestamp': trade['timestamp'],
            'size': trade['size'],
            'side': trade['side']
        })

    def _update_trade_sequences(self, symbol: str, trade: dict):
        """Updates trade sequences for pattern analysis"""
        sequence_info = {
            'timestamp': trade['timestamp'],
            'price': trade['price'],
            'size': trade['size'],
            'side': trade['side'],
            'volume_usdt': trade['volume_usdt']
        }
        self.trade_sequences[symbol].append(sequence_info)

    def _update_volume_clusters(self, symbol: str, trade: dict):
        """Updates volume clusters for analysis"""
        current_time = trade['timestamp']

        window_size = 10
        window_start = int(current_time // window_size) * window_size

        if (not self.volume_clusters[symbol] or
                self.volume_clusters[symbol][-1]['window_start'] != window_start):
            self.volume_clusters[symbol].append({
                'window_start': window_start,
                'total_volume': 0,
                'buy_volume': 0,
                'sell_volume': 0,
                'trade_count': 0,
                'unique_prices': set(),
                'avg_trade_size': 0
            })

        current_cluster = self.volume_clusters[symbol][-1]
        current_cluster['total_volume'] += trade['size']
        current_cluster['trade_count'] += 1
        current_cluster['unique_prices'].add(trade['price_level'])

        if trade['side'] == 'Buy':
            current_cluster['buy_volume'] += trade['size']
        else:
            current_cluster['sell_volume'] += trade['size']

        current_cluster['avg_trade_size'] = (
                current_cluster['total_volume'] / current_cluster['trade_count']
        ) if current_cluster['trade_count'] > 0 else 0

    def _update_kline_data(self, symbol: str, trade_record: Dict[str, Any]):
        """
        Updates 1-minute kline data based on trades.
        This is a simplified in-memory kline generation.
        """
        current_minute = int(trade_record['timestamp'] // 60) * 60

        if self.kline_data[symbol]['start_time'] != current_minute:
            if self.kline_data[symbol]['start_time'] != 0 and not self.kline_data[symbol]['is_closed']:
                self.kline_data[symbol]['is_closed'] = True
                asyncio.create_task(self.detect_consecutive_long(symbol, self.kline_data[symbol]))

            self.kline_data[symbol] = {
                'open': trade_record['price'],
                'high': trade_record['price'],
                'low': trade_record['price'],
                'close': trade_record['price'],
                'volume': trade_record['size'],
                'start_time': current_minute,
                'is_closed': False
            }
        else:
            self.kline_data[symbol]['high'] = max(self.kline_data[symbol]['high'], trade_record['price'])
            self.kline_data[symbol]['low'] = min(self.kline_data[symbol]['low'], trade_record['price'])
            self.kline_data[symbol]['close'] = trade_record['price']
            self.kline_data[symbol]['volume'] += trade_record['size']

    async def _run_all_trade_detectors(self, symbol: str, current_trade: dict):
        """Runs all trade manipulation detectors"""
        try:
            await self.analyze_volume_spike(symbol, current_trade)
            await self.detect_advanced_wash_trading(symbol, current_trade)
            await self.detect_ping_pong_trading(symbol, current_trade)
            await self.detect_ramping_dumping(symbol, current_trade)

            await self.detect_volume_clustering(symbol, current_trade)
            await self.detect_trade_timing_manipulation(symbol, current_trade)
            await self.detect_size_manipulation(symbol, current_trade)

        except Exception as e:
            logger.error(f"Error in trade detectors for {symbol}: {e}")

    async def analyze_volume_spike(self, symbol: str, trade_record: Dict[str, Any]):
        """
        Detects significant volume spikes.
        """
        current_time = trade_record['timestamp']
        window_start = current_time - (self.volume_window_minutes * 60)

        recent_trades = [t for t in self.trade_history[symbol] if t['timestamp'] >= window_start]
        current_window_volume_usdt = sum(t['volume_usdt'] for t in recent_trades)

        if current_window_volume_usdt < self.min_volume_usdt:
            return

        longer_window_start = current_time - (self.volume_window_minutes * 60 * 10)
        historical_trades = [t for t in self.trade_history[symbol] if t['timestamp'] >= longer_window_start and t['timestamp'] < window_start]

        if not historical_trades:
            return

        average_volume_usdt = sum(t['volume_usdt'] for t in historical_trades) / 9

        if average_volume_usdt > 0 and current_window_volume_usdt > average_volume_usdt * self.volume_multiplier:
            volume_ratio = current_window_volume_usdt / average_volume_usdt
            message = (f"Объем {symbol} ({self.volume_window_minutes} мин) превысил средний в {volume_ratio:.2f}x. "
                       f"Текущий: ${current_window_volume_usdt:,.0f}, Средний: ${average_volume_usdt:,.0f}")

            await self._create_alert(
                symbol,
                "volume_spike",
                trade_record['price'],
                message,
                details={
                    "volume_ratio": volume_ratio,
                    "current_volume_usdt": current_window_volume_usdt,
                    "average_volume_usdt": average_volume_usdt
                },
                trade_history=list(self.trade_history[symbol]) # Pass trade history
            )

    async def detect_consecutive_long(self, symbol: str, kline: Dict[str, Any]):
        """
        Detects consecutive long (green) candles.
        """
        if not kline['is_closed']:
            return

        if kline['close'] > kline['open']:
            self.consecutive_long_counts[symbol] += 1
            if self.consecutive_long_counts[symbol] >= self.consecutive_long_count:
                message = (f"Обнаружено {self.consecutive_long_counts[symbol]} последовательных LONG свечей для {symbol}. "
                           f"Последняя цена: {kline['close']:.2f}")
                await self._create_alert(
                    symbol,
                    "consecutive_long",
                    kline['close'],
                    message,
                    details={"consecutive_count": self.consecutive_long_counts[symbol], "candle_data": kline},
                    trade_history=list(self.trade_history[symbol]) # Pass trade history
                )
        else:
            self.consecutive_long_counts[symbol] = 0

    async def detect_advanced_wash_trading(self, symbol: str, current_trade: dict):
        """Improved wash trading detection with multiple criteria"""
        current_time = current_trade['timestamp']
        window_start = current_time - (self.volume_window_minutes * 60)

        recent_trades = [
            trade for trade in self.trade_history[symbol]
            if trade['timestamp'] >= window_start
        ]

        if len(recent_trades) < 15:
            return

        wash_signals = {
            'repeated_volumes': 0,
            'price_level_concentration': 0,
            'timing_patterns': 0,
            'side_alternation': 0,
            'total_score': 0
        }

        wash_signals['repeated_volumes'] = self._analyze_repeated_volumes(recent_trades)
        wash_signals['price_level_concentration'] = self._analyze_price_concentration(recent_trades)
        wash_signals['timing_patterns'] = self._analyze_timing_patterns(recent_trades)
        wash_signals['side_alternation'] = self._analyze_side_alternation(recent_trades)

        wash_signals['total_score'] = (
                wash_signals['repeated_volumes'] * 0.3 +
                wash_signals['price_level_concentration'] * 0.25 +
                wash_signals['timing_patterns'] * 0.25 +
                wash_signals['side_alternation'] * 0.2
        )

        if wash_signals['total_score'] > self.wash_trade_threshold_ratio:
            await self._create_alert(
                symbol,
                "Advanced Wash Trading",
                current_trade['price'],
                (f"Advanced Wash Trading: Score={wash_signals['total_score']:.2f}, "
                 f"Repeated volumes={wash_signals['repeated_volumes']:.2f}, "
                 f"Price concentration={wash_signals['price_level_concentration']:.2f}"),
                details={
                    "detection_score": wash_signals['total_score'],
                    "repeated_volumes_score": wash_signals['repeated_volumes'],
                    "price_concentration_score": wash_signals['price_level_concentration'],
                    "timing_patterns_score": wash_signals['timing_patterns'],
                    "side_alternation_score": wash_signals['side_alternation'],
                    "confidence": min(wash_signals['total_score'], 1.0),
                    "analysis_window_minutes": self.volume_window_minutes
                },
                trade_history=list(self.trade_history[symbol]) # Pass trade history
            )
            self.wash_trade_alerts[symbol] += 1
            logger.warning(f"ADVANCED WASH TRADING ALERT for {symbol}: score={wash_signals['total_score']:.2f}")


    def _analyze_repeated_volumes(self, trades: List[Dict]) -> float:
        """Analyzes repeated volumes"""
        volumes = [trade['size'] for trade in trades]
        volume_counts = defaultdict(int)

        for vol in volumes:
            rounded_vol = round(vol, 8)
            volume_counts[rounded_vol] += 1

        if not volume_counts:
            return 0

        max_repeats = max(volume_counts.values())
        total_trades = len(volumes)

        repeat_ratio = max_repeats / total_trades

        if repeat_ratio > 0.3 and max_repeats >= 5:
            return min(repeat_ratio * 2, 1.0)

        return 0

    def _analyze_price_concentration(self, trades: List[Dict]) -> float:
        """Analyzes trade concentration on price levels"""
        price_counts = defaultdict(int)

        for trade in trades:
            price_counts[trade['price_level']] += 1

        if len(price_counts) <= 1:
            return 0

        total_trades = len(trades)
        max_concentration = max(price_counts.values())

        concentration_ratio = max_concentration / total_trades

        if concentration_ratio > 0.4:
            return min(concentration_ratio * 1.5, 1.0)

        return 0

    def _analyze_timing_patterns(self, trades: List[Dict]) -> float:
        """Analyzes timing patterns between trades"""
        if len(trades) < 5:
            return 0

        intervals = []
        for i in range(1, len(trades)):
            interval = trades[i]['timestamp'] - trades[i - 1]['timestamp']
            intervals.append(interval)

        if not intervals:
            return 0

        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if avg_interval == 0:
            return 0

        cv = std_interval / avg_interval

        if cv < 0.3 and avg_interval < 10:
            return min((0.3 - cv) * 3, 1.0)

        return 0

    def _analyze_side_alternation(self, trades: List[Dict]) -> float:
        """Analyzes side alternation (buy/sell)"""
        if len(trades) < 6:
            return 0

        sides = [trade['side'] for trade in trades if trade['side'] != 'unknown']

        if len(sides) < 6:
            return 0

        alternations = 0
        for i in range(1, len(sides)):
            if sides[i] != sides[i - 1]:
                alternations += 1

        alternation_ratio = alternations / (len(sides) - 1)

        if alternation_ratio > 0.7:
            return min(alternation_ratio * 1.2, 1.0)

        return 0

    async def detect_ping_pong_trading(self, symbol: str, current_trade: dict):
        """Detects ping-pong trading between two prices"""
        current_time = current_trade['timestamp']
        window_start = current_time - self.ping_pong_window_sec

        recent_trades = [
            trade for trade in self.trade_history[symbol]
            if trade['timestamp'] >= window_start
        ]

        if len(recent_trades) < 10:
            return

        price_counts = defaultdict(int)
        for trade in recent_trades:
            price_counts[trade['price_level']] += 1

        sorted_prices = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_prices) >= 2:
            price1, count1 = sorted_prices[0]
            price2, count2 = sorted_prices[1]

            total_trades = len(recent_trades)
            dominance = (count1 + count2) / total_trades

            if dominance > 0.7 and count1 >= 5 and count2 >= 5:
                alternation_score = self._check_price_alternation(recent_trades, price1, price2)

                if alternation_score > 0.6:
                    await self._create_alert(
                        symbol,
                        "Ping-Pong Trading",
                        current_trade['price'],
                        (f"Ping-Pong Trading: Between {price1} and {price2}, "
                         f"dominance={dominance:.2f}, alternation={alternation_score:.2f}"),
                        details={
                            "price_1": price1,
                            "price_2": price2,
                            "dominance_ratio": dominance,
                            "alternation_score": alternation_score,
                            "price_spread": abs(price1 - price2),
                            "spread_percent": abs(price1 - price2) / min(price1, price2) * 100 if min(price1, price2) > 0 else 0,
                            "analysis_window_sec": self.ping_pong_window_sec
                        },
                        trade_history=list(self.trade_history[symbol]) # Pass trade history
                    )
                    self.ping_pong_alerts[symbol] += 1
                    logger.warning(f"PING-PONG TRADING ALERT for {symbol}: {price1} <-> {price2}")

    def _check_price_alternation(self, trades: List[Dict], price1: float, price2: float) -> float:
        """Checks alternation between two prices"""
        relevant_trades = [
            trade for trade in trades
            if trade['price_level'] in [price1, price2]
        ]

        if len(relevant_trades) < 4:
            return 0

        alternations = 0
        for i in range(1, len(relevant_trades)):
            if relevant_trades[i]['price_level'] != relevant_trades[i - 1]['price_level']:
                alternations += 1

        return alternations / (len(relevant_trades) - 1)

    async def detect_ramping_dumping(self, symbol: str, current_trade: dict):
        """Detects ramping (pump) and dumping (dump)"""
        current_time = current_trade['timestamp']
        window_start = current_time - self.ramping_window_sec

        recent_trades = [
            trade for trade in self.trade_history[symbol]
            if trade['timestamp'] >= window_start
        ]

        if len(recent_trades) < 20:
            return

        prices = [trade['price'] for trade in recent_trades]
        volumes = [trade['volume_usdt'] for trade in recent_trades]

        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

        total_volume = sum(volumes)
        first_half_volume = sum(volumes[:len(volumes) // 2])
        second_half_volume = sum(volumes[len(volumes) // 2:])

        volume_imbalance = abs(first_half_volume - second_half_volume) / total_volume if total_volume > 0 else 0

        if (price_change > 0.02 and volume_imbalance > 0.3 and
                first_half_volume > second_half_volume):
            await self._create_alert(
                symbol,
                "Price Ramping",
                current_trade['price'],
                (f"Price Ramping: Price change={price_change * 100:.2f}%, "
                 f"volume imbalance={volume_imbalance:.2f}"),
                details={
                    "manipulation_type": "ramping",
                    "price_change_percent": price_change * 100,
                    "volume_imbalance": volume_imbalance,
                    "analysis_window_sec": self.ramping_window_sec,
                    "confidence": min(abs(price_change) * 10 + volume_imbalance, 1.0)
                },
                trade_history=list(self.trade_history[symbol]) # Pass trade history
            )
            self.ramping_alerts[symbol] += 1
            logger.warning(f"RAMPING ALERT for {symbol}: {price_change * 100:.2f}%")

        elif (price_change < -0.02 and volume_imbalance > 0.3 and
              second_half_volume > first_half_volume):
            await self._create_alert(
                symbol,
                "Price Dumping",
                current_trade['price'],
                (f"Price Dumping: Price change={price_change * 100:.2f}%, "
                 f"volume imbalance={volume_imbalance:.2f}"),
                details={
                    "manipulation_type": "dumping",
                    "price_change_percent": price_change * 100,
                    "volume_imbalance": volume_imbalance,
                    "analysis_window_sec": self.ramping_window_sec,
                    "confidence": min(abs(price_change) * 10 + volume_imbalance, 1.0)
                },
                trade_history=list(self.trade_history[symbol]) # Pass trade history
            )
            self.ramping_alerts[symbol] += 1
            logger.warning(f"DUMPING ALERT for {symbol}: {price_change * 100:.2f}%")

    async def detect_volume_clustering(self, symbol: str, current_trade: dict):
        """Detects suspicious volume clustering"""
        if len(self.volume_clusters[symbol]) < 5:
            return

        recent_clusters = list(self.volume_clusters[symbol])[-5:]

        suspicious_patterns = 0

        for cluster in recent_clusters:
            if len(self.volume_clusters[symbol]) > 1:
                prev_clusters = list(self.volume_clusters[symbol])[:-1]
                avg_volume = np.mean([c['total_volume'] for c in prev_clusters]) if prev_clusters else 0

                if avg_volume > 0 and cluster['total_volume'] > avg_volume * self.volume_multiplier:
                    suspicious_patterns += 1

            if len(cluster['unique_prices']) == 1 and cluster['trade_count'] > 10 and cluster['total_volume'] * cluster['avg_trade_size'] > self.min_volume_usdt:
                suspicious_patterns += 1

        if suspicious_patterns >= 2:
            await self._create_alert(
                symbol,
                "Volume Clustering",
                current_trade['price'],
                f"Volume Clustering: {suspicious_patterns} suspicious patterns detected",
                details={
                    "suspicious_pattern_count": suspicious_patterns,
                    "analysis_clusters": len(self.volume_clusters[symbol]),
                    "confidence": min(suspicious_patterns / 5.0, 1.0)
                },
                trade_history=list(self.trade_history[symbol]) # Pass trade history
            )
            logger.warning(f"VOLUME CLUSTERING ALERT for {symbol}: {suspicious_patterns} patterns")

    async def detect_trade_timing_manipulation(self, symbol: str, current_trade: dict):
        """Detects trade timing manipulation"""
        if len(self.trade_sequences[symbol]) < 20:
            return

        recent_trades = list(self.trade_sequences[symbol])[-20:]

        intervals = []
        for i in range(1, len(recent_trades)):
            interval = recent_trades[i]['timestamp'] - recent_trades[i - 1]['timestamp']
            intervals.append(interval)

        if not intervals:
            return

        std_interval = np.std(intervals)
        mean_interval = np.mean(intervals)

        if mean_interval > 0:
            cv = std_interval / mean_interval

            if cv < 0.1 and mean_interval < 5:
                await self._create_alert(
                    symbol,
                    "Trade Timing Manipulation",
                    current_trade['price'],
                    f"Trade Timing Manipulation: CV={cv:.3f}, interval={mean_interval:.1f}s",
                    details={
                        "coefficient_of_variation": cv,
                        "mean_interval_sec": mean_interval,
                        "regularity_score": 1 - cv,
                        "confidence": min((0.3 - cv) * 3, 1.0)
                    },
                    trade_history=list(self.trade_history[symbol]) # Pass trade history
                )
                logger.warning(f"TIMING MANIPULATION ALERT for {symbol}: CV={cv:.3f}")

    async def detect_size_manipulation(self, symbol: str, current_trade: dict):
        """Detects trade size manipulation"""
        current_time = current_trade['timestamp']
        window_start = current_time - 120

        recent_trades = [
            trade for trade in self.trade_history[symbol]
            if trade['timestamp'] >= window_start
        ]

        if len(recent_trades) < 15:
            return

        sizes = [trade['size'] for trade in recent_trades]

        size_counts = defaultdict(int)
        for size in sizes:
            rounded_size = round(size, 6)
            size_counts[rounded_size] += 1

        max_count = max(size_counts.values())
        total_trades = len(sizes)

        if max_count / total_trades > 0.6 and max_count >= 8:
            dominant_size = max(size_counts.items(), key=lambda x: x[1])[0]
            ratio = max_count / total_trades
            await self._create_alert(
                symbol,
                "Trade Size Manipulation",
                current_trade['price'],
                f"Size Manipulation: {max_count}/{total_trades} trades with size {dominant_size} ({ratio:.1%})",
                details={
                    "dominant_size": dominant_size,
                    "occurrence_count": max_count,
                    "total_trades": total_trades,
                    "dominance_ratio": ratio,
                    "confidence": min(ratio * 1.5, 1.0)
                },
                trade_history=list(self.trade_history[symbol]) # Pass trade history
            )
            self.wash_trade_alerts[symbol] += 1
            logger.warning(f"SIZE MANIPULATION ALERT for {symbol}: {ratio:.1%} same size")

    async def _create_alert(self, symbol: str, alert_type: str, price: float, message: str,
                           details: Dict[str, Any],
                           order_book_snapshot: Optional[Dict] = None,
                           trade_history: Optional[List[Dict]] = None
                           ):
        """
        Helper to create and insert an alert, with cooldown and grouping logic.
        """
        current_time = time.time()
        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        last_alert_time = self.recent_alerts[symbol].get(alert_type, 0)
        if (current_time - last_alert_time) < self.alert_grouping_seconds:
            logger.debug(f"Alert for {symbol} ({alert_type}) on cooldown. Skipping.")
            return

        await self.db_manager.insert_alert(
            symbol=symbol,
            alert_type=alert_type,
            price=price,
            message=message,
            details=details,
            alert_start_time=alert_start_time,
            order_book_snapshot=order_book_snapshot,
            trade_history=trade_history
        )
        self.recent_alerts[symbol][alert_type] = current_time
        logger.info(f"Alert generated: {alert_type} for {symbol} at {price}")


    def get_trade_stats(self):
        """Returns extended trade processing statistics"""
        return {
            "processed_trades": dict(self.processed_trades),
            "wash_trade_alerts": dict(self.wash_trade_alerts),
            "ping_pong_alerts": dict(self.ping_pong_alerts),
            "ramping_alerts": dict(self.ramping_alerts),
            "trade_history_sizes": {
                symbol: len(history) for symbol, history in self.trade_history.items()
            },
            "volume_cluster_sizes": {
                symbol: len(clusters) for symbol, clusters in self.volume_clusters.items()
            },
            "kline_data_symbols": len(self.kline_data),
            "consecutive_long_counts": dict(self.consecutive_long_counts),
            "recent_alerts_count": {
                symbol: len(alerts) for symbol, alerts in self.recent_alerts.items()
            }
        }
