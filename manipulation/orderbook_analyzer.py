from collections import defaultdict, deque
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    def __init__(self, db_manager, ob_history_depth=20,
                 iceberg_trade_window_sec=15,
                 iceberg_volume_ratio_threshold: float = 1.6,
                 iceberg_min_trade_count: int = 4,
                 layering_distance_from_market_percent: float = 0.003,
                 layering_min_size_change_abs: float = 150,
                 layering_ob_window_sec: int = 8,
                 spoofing_cancel_ratio: float = 0.65,
                 momentum_ignition_threshold: float = 0.015,
                 enabled: bool = False,
                 snapshot_on_alert: bool = False,
                 iceberg_min_volume_ratio: float = 0.1,
                 iceberg_hidden_volume_multiplier: float = 5.0,
                 iceberg_price_tolerance: float = 0.0001,
                 layering_spoofing_depth: int = 5,
                 layering_spoofing_threshold: float = 0.7,
                 layering_spoofing_time_window_sec: int = 30,
                 liquidity_detection_window_sec: int = 300,
                 liquidity_change_threshold: float = 0.2,
                 toxic_order_flow_window_sec: int = 10,
                 toxic_order_flow_threshold: float = 0.7,
                 cross_market_anomaly_enabled: bool = False,
                 cross_market_anomaly_threshold: float = 0.001,
                 spread_manipulation_enabled: bool = False,
                 spread_manipulation_threshold: float = 0.005,
                 spread_manipulation_time_window_sec: int = 10,
                 imbalance_threshold: float = 0.6,
                 trade_processor=None # Added: Reference to TradeProcessor
                 ):

        self.db_manager = db_manager
        self.trade_processor = trade_processor # Added: Store TradeProcessor reference
        self.current_orderbooks = defaultdict(lambda: {'bids': {}, 'asks': {}})
        self.orderbook_history = defaultdict(lambda: deque(maxlen=ob_history_depth))
        self.iceberg_trade_history = defaultdict(lambda: deque(maxlen=2000))
        self.price_history = defaultdict(lambda: deque(maxlen=100))
        self.volume_profile = defaultdict(lambda: defaultdict(float))
        self.order_flow_imbalance = defaultdict(lambda: deque(maxlen=50))

        self.iceberg_trade_window_sec = iceberg_trade_window_sec
        self.iceberg_volume_ratio_threshold = iceberg_volume_ratio_threshold
        self.iceberg_min_trade_count = iceberg_min_trade_count
        self.layering_distance_from_market_percent = layering_distance_from_market_percent
        self.layering_min_size_change_abs = layering_min_size_change_abs
        self.layering_ob_window_sec = layering_ob_window_sec
        self.spoofing_cancel_ratio = spoofing_cancel_ratio
        self.momentum_ignition_threshold = momentum_ignition_threshold

        self.enabled = enabled
        self.snapshot_on_alert = snapshot_on_alert
        self.history_depth = ob_history_depth
        self.imbalance_threshold = imbalance_threshold
        self.iceberg_min_volume_ratio = iceberg_min_volume_ratio
        self.iceberg_hidden_volume_multiplier = iceberg_hidden_volume_multiplier
        self.iceberg_price_tolerance = iceberg_price_tolerance
        self.layering_spoofing_depth = layering_spoofing_depth
        self.layering_spoofing_threshold = layering_spoofing_threshold
        self.layering_spoofing_time_window_sec = layering_spoofing_time_window_sec
        self.liquidity_detection_window_sec = liquidity_detection_window_sec
        self.liquidity_change_threshold = liquidity_change_threshold
        self.toxic_order_flow_window_sec = toxic_order_flow_window_sec
        self.toxic_order_flow_threshold = toxic_order_flow_threshold
        self.cross_market_anomaly_enabled = cross_market_anomaly_enabled
        self.cross_market_anomaly_threshold = cross_market_anomaly_threshold
        self.spread_manipulation_enabled = spread_manipulation_enabled
        self.spread_manipulation_threshold = spread_manipulation_threshold
        self.spread_manipulation_time_window_sec = spread_manipulation_time_window_sec


        self.analysis_counts = defaultdict(int)
        self.alert_counts = defaultdict(int)
        self.last_best_prices = defaultdict(lambda: {'bid': 0, 'ask': 0})

        self.recent_alerts = defaultdict(lambda: defaultdict(float))
        self.alert_cooldown_sec = 30

        self.order_patterns = defaultdict(lambda: deque(maxlen=30))
        self.liquidity_events = defaultdict(lambda: deque(maxlen=20))

    async def update_orderbook(self, symbol: str, bids: List[List[float]], asks: List[List[float]]):
        """Обновляет ордербук и сохраняет историю изменений"""
        current_time = time.time()

        prev_best_bid = self.last_best_prices[symbol]['bid']
        prev_best_ask = self.last_best_prices[symbol]['ask']

        self.current_orderbooks[symbol]['bids'] = bids
        self.current_orderbooks[symbol]['asks'] = asks
        self.orderbook_history[symbol].append((current_time, {'bids': bids, 'asks': asks}))

        if bids and asks:
            current_best_bid = bids[0][0]
            current_best_ask = asks[0][0]

            self.last_best_prices[symbol]['bid'] = current_best_bid
            self.last_best_prices[symbol]['ask'] = current_best_ask

            mid_price = (current_best_bid + current_best_ask) / 2
            self.price_history[symbol].append((current_time, mid_price))

            spread = current_best_ask - current_best_bid
            spread_pct = spread / mid_price if mid_price > 0 else 0

            if len(self.price_history[symbol]) > 5:
                recent_spreads = []
                for i in range(max(0, len(self.orderbook_history[symbol]) - 5), len(self.orderbook_history[symbol])):
                    _, ob = self.orderbook_history[symbol][i]
                    if ob['bids'] and ob['asks']:
                        s = ob['asks'][0][0] - ob['bids'][0][0]
                        recent_spreads.append(s)

                if recent_spreads:
                    avg_spread = np.mean(recent_spreads)
                    if spread > avg_spread * 3 and spread_pct > 0.001:
                        await self._detect_spread_manipulation(symbol, spread, avg_spread, current_time)

    def add_trade_for_iceberg(self, symbol: str, trade_data: dict):
        """Добавляет трейд для анализа iceberg ордеров с улучшенной обработкой"""
        timestamp_ms = trade_data.get('T')
        price_str = trade_data.get('p')
        size_str = trade_data.get('v')
        side = trade_data.get('S', 'unknown')

        if not timestamp_ms or price_str is None or size_str is None:
            logger.debug(f"Trade data for iceberg detection missing essential fields: {trade_data}")
            return

        try:
            timestamp_sec = timestamp_ms / 1000.0
            price = float(price_str)
            size = float(size_str)

            trade_record = {
                'timestamp': timestamp_sec,
                'price': price,
                'size': size,
                'side': side,
                'volume_usd': price * size
            }

            self.iceberg_trade_history[symbol].append(trade_record)

            price_level = round(price, 6)
            self.volume_profile[symbol][price_level] += size

            self._update_order_flow_imbalance(symbol, trade_record)

        except (ValueError, TypeError) as e:
            logger.error(f"Error converting trade data for {symbol}: {e}")

    def _update_order_flow_imbalance(self, symbol: str, trade: dict):
        """Обновляет метрики дисбаланса order flow"""
        current_time = trade['timestamp']

        window_size = 5
        window_start = int(current_time // window_size) * window_size

        if not self.order_flow_imbalance[symbol] or self.order_flow_imbalance[symbol][-1]['window'] != window_start:
            self.order_flow_imbalance[symbol].append({
                'window': window_start,
                'buy_volume': 0,
                'sell_volume': 0,
                'buy_count': 0,
                'sell_count': 0,
                'total_volume': 0
            })

        current_window = self.order_flow_imbalance[symbol][-1]

        if trade['side'] == 'Buy':
            current_window['buy_volume'] += trade['size']
            current_window['buy_count'] += 1
        else:
            current_window['sell_volume'] += trade['size']
            current_window['sell_count'] += 1

        current_window['total_volume'] += trade['size']

    async def analyze_orderbook(self, symbol: str):
        """Главная функция анализа с множественными детекторами"""
        self.analysis_counts[symbol] += 1

        current_ob = self.current_orderbooks[symbol]

        if not self._validate_orderbook(symbol, current_ob):
            return

        if self.analysis_counts[symbol] % 500 == 0:
            logger.info(f"Analysis milestone for {symbol}: {self.analysis_counts[symbol]} analyses, "
                        f"alerts: {self.alert_counts[symbol]}")

        await self._run_all_detectors(symbol, current_ob)

    def _validate_orderbook(self, symbol: str, current_ob: dict) -> bool:
        """Валидация ордербука"""
        if not current_ob['bids'] or not current_ob['asks']:
            return False

        if (current_ob['bids'][0][0] <= 0 or current_ob['asks'][0][0] <= 0 or
                current_ob['bids'][0][1] <= 0 or current_ob['asks'][0][1] <= 0):
            return False

        if current_ob['bids'][0][0] >= current_ob['asks'][0][0]:
            logger.warning(
                f"Invalid spread for {symbol}: bid={current_ob['bids'][0][0]}, ask={current_ob['asks'][0][0]}")
            return False

        return True

    async def _run_all_detectors(self, symbol: str, current_ob: dict):
        """Запускает все детекторы манипуляций"""
        try:
            await self.detect_advanced_layering_spoofing(symbol, current_ob)
            await self.detect_advanced_iceberg(symbol, current_ob)
            await self.detect_momentum_ignition(symbol, current_ob)
            await self.detect_quote_stuffing(symbol, current_ob)
            await self.detect_liquidity_detection(symbol, current_ob)
            await self.detect_order_flow_toxicity(symbol, current_ob)
            await self.detect_cross_market_manipulation(symbol, current_ob)

        except Exception as e:
            logger.error(f"Error in detectors for {symbol}: {e}")

    async def detect_advanced_layering_spoofing(self, symbol: str, current_ob: dict):
        """Улучшенное детектирование layering/spoofing с анализом паттернов"""
        if len(self.orderbook_history[symbol]) < 5:
            return

        current_time = time.time()
        analysis_window = []

        for ts, snapshot in reversed(list(self.orderbook_history[symbol])):
            if current_time - ts <= self.layering_ob_window_sec:
                analysis_window.append((ts, snapshot))
            else:
                break

        if len(analysis_window) < 3:
            return

        best_bid = current_ob['bids'][0][0]
        best_ask = current_ob['asks'][0][0]

        layering_signals_bid = self._analyze_layering_patterns(
            analysis_window, 'bids', best_bid, is_bid=True
        )

        layering_signals_ask = self._analyze_layering_patterns(
            analysis_window, 'asks', best_ask, is_bid=False
        )

        if layering_signals_bid['strength'] > self.layering_spoofing_threshold:
            await self._create_layering_alert(symbol, 'Bid', layering_signals_bid, best_bid, current_ob)

        if layering_signals_ask['strength'] > self.layering_spoofing_threshold:
            await self._create_layering_alert(symbol, 'Ask', layering_signals_ask, best_ask, current_ob)

    def _analyze_layering_patterns(self, analysis_window: List[Tuple], side: str, best_price: float,
                                   is_bid: bool) -> Dict:
        """Анализирует паттерны layering на одной стороне ордербука"""
        pattern_signals = {
            'large_orders_far': 0,
            'rapid_changes': 0,
            'size_progression': 0,
            'cancellation_pattern': 0,
            'strength': 0
        }

        if len(analysis_window) < 2:
            return pattern_signals

        for i in range(len(analysis_window) - 1):
            current_snapshot = analysis_window[i][1][side]
            prev_snapshot = analysis_window[i + 1][1][side]

            far_orders = self._detect_far_large_orders(current_snapshot, best_price, is_bid)
            pattern_signals['large_orders_far'] += len(far_orders)

            rapid_changes = self._detect_rapid_size_changes(current_snapshot, prev_snapshot)
            pattern_signals['rapid_changes'] += len(rapid_changes)

            size_progression = self._analyze_size_progression(current_snapshot, best_price, is_bid)
            pattern_signals['size_progression'] += size_progression

        total_signals = (pattern_signals['large_orders_far'] * 0.4 +
                         pattern_signals['rapid_changes'] * 0.3 +
                         pattern_signals['size_progression'] * 0.3)

        pattern_signals['strength'] = min(total_signals / 10, 1.0)

        return pattern_signals

    def _detect_far_large_orders(self, orders: List[List[float]], best_price: float, is_bid: bool) -> List[Dict]:
        """Детектирует большие ордера далеко от лучшей цены"""
        far_orders = []

        for price, size in orders:
            if best_price <= 0:
                continue

            distance = abs(price - best_price) / best_price

            if distance > self.layering_distance_from_market_percent:
                avg_size = np.mean([s for p, s in orders[:10]]) if len(orders) >= 10 else size

                if size > avg_size * 2 and size > self.layering_min_size_change_abs:
                    far_orders.append({
                        'price': price,
                        'size': size,
                        'distance_pct': distance * 100,
                        'size_ratio': size / avg_size if avg_size > 0 else 1
                    })

        return far_orders

    def _detect_rapid_size_changes(self, current: List[List[float]], previous: List[List[float]]) -> List[Dict]:
        """Детектирует быстрые изменения размеров ордеров"""
        changes = []

        current_dict = {price: size for price, size in current}
        prev_dict = {price: size for price, size in previous}

        for price in set(list(current_dict.keys()) + list(prev_dict.keys())):
            current_size = current_dict.get(price, 0)
            prev_size = prev_dict.get(price, 0)

            size_change = abs(current_size - prev_size)

            if size_change > self.layering_min_size_change_abs:
                changes.append({
                    'price': price,
                    'size_change': size_change,
                    'prev_size': prev_size,
                    'current_size': current_size,
                    'change_type': 'increase' if current_size > prev_size else 'decrease'
                })

        return changes

    def _analyze_size_progression(self, orders: List[List[float]], best_price: float, is_bid: bool) -> float:
        """Анализирует прогрессию размеров ордеров (характерно для layering)"""
        if len(orders) < 5 or best_price <= 0:
            return 0

        levels = orders[:10]
        sizes = [size for price, size in levels]

        progression_score = 0

        for i in range(1, len(sizes)):
            if sizes[i] > sizes[i - 1] * 1.2:
                progression_score += 1
            elif sizes[i] < sizes[i - 1] * 0.8:
                progression_score -= 0.5

        return max(0, progression_score)

    async def _create_layering_alert(self, symbol: str, side: str, signals: Dict, price: float, current_ob: Dict):
        """Создает алерт о layering/spoofing"""
        current_time = time.time()
        alert_type = f"Advanced Layering/Spoofing ({side})"

        if self._is_alert_on_cooldown(symbol, alert_type, current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        message = (f"Advanced Layering/Spoofing ({side}): "
                   f"Strength={signals['strength']:.2f}, "
                   f"Far orders={signals['large_orders_far']}, "
                   f"Rapid changes={signals['rapid_changes']}")

        details = {
            "detection_strength": signals['strength'],
            "large_orders_far": signals['large_orders_far'],
            "rapid_changes": signals['rapid_changes'],
            "size_progression": signals['size_progression'],
            "side": side.lower(),
            "analysis_window_sec": self.layering_ob_window_sec,
            "confidence": signals['strength']
        }

        await self.db_manager.insert_alert(
            symbol, alert_type, price, message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, alert_type, current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"ADVANCED LAYERING ALERT for {symbol} ({side}): strength={signals['strength']:.2f}")

    async def detect_advanced_iceberg(self, symbol: str, current_ob: dict):
        """Улучшенное детектирование iceberg ордеров"""
        current_time = time.time()
        window_start = current_time - self.iceberg_trade_window_sec

        recent_trades = [
            trade for trade in self.iceberg_trade_history[symbol]
            if trade['timestamp'] >= window_start
        ]

        if len(recent_trades) < self.iceberg_min_trade_count:
            return

        best_bid_price = current_ob['bids'][0][0]
        best_bid_size = current_ob['bids'][0][1]
        best_ask_price = current_ob['asks'][0][0]
        best_ask_size = current_ob['asks'][0][1]

        price_tolerance = best_bid_price * self.iceberg_price_tolerance

        trade_clusters = self._cluster_trades_by_price(recent_trades, price_tolerance)

        for cluster_price, cluster_trades in trade_clusters.items():
            iceberg_signals = self._analyze_iceberg_cluster(
                cluster_trades, cluster_price, best_bid_price, best_ask_price,
                best_bid_size, best_ask_size
            )

            if iceberg_signals['is_iceberg']:
                await self._create_iceberg_alert(symbol, iceberg_signals, current_ob)

    def _cluster_trades_by_price(self, trades: List[Dict], tolerance: float) -> Dict[float, List[Dict]]:
        """Группирует трейды по ценовым уровням с учетом толерантности"""
        clusters = {}

        for trade in trades:
            price = trade['price']

            cluster_found = False
            for cluster_price in clusters.keys():
                if abs(price - cluster_price) <= tolerance:
                    clusters[cluster_price].append(trade)
                    cluster_found = True
                    break

            if not cluster_found:
                clusters[price] = [trade]

        return clusters

    def _analyze_iceberg_cluster(self, trades: List[Dict], cluster_price: float,
                                 best_bid: float, best_ask: float,
                                 best_bid_size: float, best_ask_size: float) -> Dict:
        """Анализирует кластер трейдов на предмет iceberg паттерна"""

        total_volume = sum(trade['size'] for trade in trades)
        trade_count = len(trades)
        time_span = max(trade['timestamp'] for trade in trades) - min(trade['timestamp'] for trade in trades)

        bid_distance = abs(cluster_price - best_bid) / best_bid if best_bid > 0 else float('inf')
        ask_distance = abs(cluster_price - best_ask) / best_ask if best_ask > 0 else float('inf')

        is_bid_side = bid_distance < ask_distance
        relevant_visible_size = best_bid_size if is_bid_side else best_ask_size

        volume_ratio = total_volume / relevant_visible_size if relevant_visible_size > 0 else 0

        avg_trade_size = total_volume / trade_count
        size_consistency = self._check_trade_size_consistency(trades)
        price_stability = bid_distance < 0.001 or ask_distance < 0.001

        is_iceberg = (
                volume_ratio > self.iceberg_volume_ratio_threshold and
                trade_count >= self.iceberg_min_trade_count and
                price_stability and
                size_consistency > 0.3
        )

        return {
            'is_iceberg': is_iceberg,
            'cluster_price': cluster_price,
            'total_volume': total_volume,
            'trade_count': trade_count,
            'volume_ratio': volume_ratio,
            'visible_size': relevant_visible_size,
            'side': 'bid' if is_bid_side else 'ask',
            'time_span': time_span,
            'avg_trade_size': avg_trade_size,
            'size_consistency': size_consistency,
            'price_stability': price_stability
        }

    def _check_trade_size_consistency(self, trades: List[Dict]) -> float:
        """Проверяет консистентность размеров трейдов (характерно для iceberg)"""
        if len(trades) < 2:
            return 0

        sizes = [trade['size'] for trade in trades]

        mean_size = np.mean(sizes)
        std_size = np.std(sizes)

        if mean_size == 0:
            return 0

        cv = std_size / mean_size

        consistency = max(0, 1 - cv)

        return consistency

    async def _create_iceberg_alert(self, symbol: str, signals: Dict, current_ob: Dict):
        """Создает алерт об iceberg ордере"""
        current_time = time.time()
        side = signals['side'].title()
        alert_type = f"Advanced Iceberg ({side} Side)"

        if self._is_alert_on_cooldown(symbol, alert_type, current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        message = (f"Advanced Iceberg ({side}): "
                   f"Volume ratio={signals['volume_ratio']:.2f}, "
                   f"Trades={signals['trade_count']}, "
                   f"Consistency={signals['size_consistency']:.2f}")

        details = {
            "total_volume": signals['total_volume'],
            "visible_size": signals['visible_size'],
            "volume_ratio": signals['volume_ratio'],
            "trade_count": signals['trade_count'],
            "avg_trade_size": signals['avg_trade_size'],
            "size_consistency": signals['size_consistency'],
            "time_span_sec": signals['time_span'],
            "side": signals['side'],
            "detection_confidence": min(signals['volume_ratio'] / 3, 1.0),
            "confidence": min(signals['volume_ratio'] / 3, 1.0)
        }

        await self.db_manager.insert_alert(
            symbol, alert_type, signals['cluster_price'], message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, alert_type, current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"ADVANCED ICEBERG ALERT for {symbol} ({side}): ratio={signals['volume_ratio']:.2f}")

    async def detect_momentum_ignition(self, symbol: str, current_ob: dict):
        """Детектирует momentum ignition - искусственное создание momentum"""
        if len(self.price_history[symbol]) < 20:
            return

        current_time = time.time()
        recent_prices = [
            (ts, price) for ts, price in self.price_history[symbol]
            if current_time - ts <= 30
        ]

        if len(recent_prices) < 10:
            return

        prices = [price for ts, price in recent_prices]
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

        if abs(price_change) > self.momentum_ignition_threshold:
            volume_spike = await self._check_volume_spike(symbol, current_time)
            order_imbalance = self._check_order_imbalance(symbol)

            if volume_spike or order_imbalance:
                await self._create_momentum_ignition_alert(
                    symbol, price_change, volume_spike, order_imbalance, prices[-1], current_ob
                )

    async def _check_volume_spike(self, symbol: str, current_time: float) -> bool:
        """Проверяет наличие всплеска объема"""
        recent_trades = [
            trade for trade in self.iceberg_trade_history[symbol]
            if current_time - trade['timestamp'] <= 10
        ]

        if len(recent_trades) < 5:
            return False

        recent_volume = sum(trade['volume_usd'] for trade in recent_trades)

        prev_trades = [
            trade for trade in self.iceberg_trade_history[symbol]
            if current_time - 20 <= trade['timestamp'] <= current_time - 10
        ]

        if not prev_trades:
            return False

        prev_volume = sum(trade['volume_usd'] for trade in prev_trades)

        return recent_volume > prev_volume * 3

    def _check_order_imbalance(self, symbol: str) -> bool:
        """Проверяет дисбаланс ордеров"""
        if not self.order_flow_imbalance[symbol]:
            return False

        recent_window = self.order_flow_imbalance[symbol][-1]

        total_volume = recent_window['total_volume']
        if total_volume == 0:
            return False

        buy_ratio = recent_window['buy_volume'] / total_volume

        return buy_ratio > 0.8 or buy_ratio < 0.2

    async def _create_momentum_ignition_alert(self, symbol: str, price_change: float,
                                              volume_spike: bool, order_imbalance: bool, price: float, current_ob: Dict):
        """Создает алерт о momentum ignition"""
        current_time = time.time()

        if self._is_alert_on_cooldown(symbol, "Momentum Ignition", current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        message = (f"Momentum Ignition: Price change={price_change * 100:.2f}%, "
                   f"Volume spike={volume_spike}, Imbalance={order_imbalance}")

        details = {
            "price_change_percent": price_change * 100,
            "has_volume_spike": volume_spike,
            "has_order_imbalance": order_imbalance,
            "detection_confidence": 0.7 + (0.15 if volume_spike else 0) + (0.15 if order_imbalance else 0),
            "confidence": 0.7 + (0.15 if volume_spike else 0) + (0.15 if order_imbalance else 0)
        }

        await self.db_manager.insert_alert(
            symbol, "Momentum Ignition", price, message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, "Momentum Ignition", current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"MOMENTUM IGNITION ALERT for {symbol}: change={price_change * 100:.2f}%")

    async def detect_quote_stuffing(self, symbol: str, current_ob: dict):
        """Детектирует quote stuffing - избыточное количество котировок"""
        current_time = time.time()

        recent_updates = [
            ts for ts, _ in self.orderbook_history[symbol]
            if current_time - ts <= 10
        ]

        update_rate = len(recent_updates) / 10

        if update_rate > 20:
            meaningful_changes = self._count_meaningful_price_changes(symbol, current_time)

            if meaningful_changes < update_rate * 0.1:
                await self._create_quote_stuffing_alert(symbol, update_rate, meaningful_changes, current_ob)

    def _count_meaningful_price_changes(self, symbol: str, current_time: float) -> int:
        """Считает количество значимых изменений цены"""
        meaningful_changes = 0
        prev_best_bid = None
        prev_best_ask = None

        for ts, snapshot in self.orderbook_history[symbol]:
            if current_time - ts > 10:
                continue

            if snapshot['bids'] and snapshot['asks']:
                current_best_bid = snapshot['bids'][0][0]
                current_best_ask = snapshot['asks'][0][0]

                if prev_best_bid is not None and prev_best_ask is not None:
                    bid_change = abs(current_best_bid - prev_best_bid) / prev_best_bid
                    ask_change = abs(current_best_ask - prev_best_ask) / prev_best_ask

                    if bid_change > 0.0001 or ask_change > 0.0001:
                        meaningful_changes += 1

            prev_best_bid = current_best_bid
            prev_best_ask = current_best_ask

        return meaningful_changes

    async def _create_quote_stuffing_alert(self, symbol: str, update_rate: float, meaningful_changes: int, current_ob: Dict):
        """Создает алерт о quote stuffing"""
        current_time = time.time()

        if self._is_alert_on_cooldown(symbol, "Quote Stuffing", current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        price = (current_ob['bids'][0][0] + current_ob['asks'][0][0]) / 2

        message = (f"Quote Stuffing: {update_rate:.1f} updates/sec, "
                   f"only {meaningful_changes} meaningful changes")

        details = {
            "update_rate_per_sec": update_rate,
            "meaningful_changes": meaningful_changes,
            "noise_ratio": 1 - (meaningful_changes / (update_rate * 10)) if update_rate > 0 else 0,
            "confidence": min(update_rate / 30, 1.0)
        }

        await self.db_manager.insert_alert(
            symbol, "Quote Stuffing", price, message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, "Quote Stuffing", current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"QUOTE STUFFING ALERT for {symbol}: {update_rate:.1f} updates/sec")

    async def detect_liquidity_detection(self, symbol: str, current_ob: dict):
        """Детектирует liquidity detection - зондирование ликвидности"""
        current_time = time.time()

        small_order_threshold = 100

        if len(self.orderbook_history[symbol]) < 5:
            return

        depth_changes = self._analyze_depth_changes(symbol, current_time, small_order_threshold)

        if depth_changes['suspicious_patterns'] > 3:
            await self._create_liquidity_detection_alert(symbol, depth_changes, current_ob)

    def _analyze_depth_changes(self, symbol: str, current_time: float, threshold: float) -> Dict:
        """Анализирует изменения в глубине ордербука"""
        patterns = {
            'small_order_appearances': 0,
            'small_order_cancellations': 0,
            'depth_probing': 0,
            'suspicious_patterns': 0
        }

        if len(self.orderbook_history[symbol]) < 2:
            return patterns

        current_snapshot = self.orderbook_history[symbol][-1][1]
        prev_snapshot = self.orderbook_history[symbol][-2][1]

        self._analyze_side_depth_changes(
            current_snapshot['bids'], prev_snapshot['bids'], threshold, patterns
        )

        self._analyze_side_depth_changes(
            current_snapshot['asks'], prev_snapshot['asks'], threshold, patterns
        )

        patterns['suspicious_patterns'] = (
                patterns['small_order_appearances'] +
                patterns['small_order_cancellations'] +
                patterns['depth_probing']
        )

        return patterns

    def _analyze_side_depth_changes(self, current_side: List, prev_side: List,
                                    threshold: float, patterns: Dict):
        """Анализирует изменения на одной стороне ордербука"""
        current_dict = {price: size for price, size in current_side}
        prev_dict = {price: size for price, size in prev_side}

        for price, size in current_dict.items():
            if price not in prev_dict and size <= threshold:
                patterns['small_order_appearances'] += 1

        for price, size in prev_dict.items():
            if price not in current_dict and size <= threshold:
                patterns['small_order_cancellations'] += 1

    async def _create_liquidity_detection_alert(self, symbol: str, patterns: Dict, current_ob: Dict):
        """Создает алерт о liquidity detection"""
        current_time = time.time()

        if self._is_alert_on_cooldown(symbol, "Liquidity Detection", current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        price = (current_ob['bids'][0][0] + current_ob['asks'][0][0]) / 2

        message = (f"Liquidity Detection: {patterns['suspicious_patterns']} suspicious patterns, "
                   f"small orders: +{patterns['small_order_appearances']} "
                   f"-{patterns['small_order_cancellations']}")

        details = {
            **patterns,
            "confidence": min(patterns['suspicious_patterns'] / 5.0, 1.0)
        }

        await self.db_manager.insert_alert(
            symbol, "Liquidity Detection", price, message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, "Liquidity Detection", current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"LIQUIDITY DETECTION ALERT for {symbol}: {patterns['suspicious_patterns']} patterns")

    async def detect_order_flow_toxicity(self, symbol: str, current_ob: dict):
        """Детектирует токсичный order flow"""
        if len(self.order_flow_imbalance[symbol]) < 5:
            return

        recent_windows = list(self.order_flow_imbalance[symbol])[-5:]

        toxicity_score = 0

        for window in recent_windows:
            if window['total_volume'] == 0:
                continue

            buy_ratio = window['buy_volume'] / window['total_volume']

            if buy_ratio > self.toxic_order_flow_threshold or buy_ratio < (1 - self.toxic_order_flow_threshold):
                toxicity_score += 2
            elif buy_ratio > 0.8 or buy_ratio < 0.2:
                toxicity_score += 1

        if toxicity_score >= 5:
            await self._create_toxicity_alert(symbol, toxicity_score, recent_windows, current_ob)

    async def _create_toxicity_alert(self, symbol: str, score: int, windows: List[Dict], current_ob: Dict):
        """Создает алерт о токсичном order flow"""
        current_time = time.time()

        if self._is_alert_on_cooldown(symbol, "Toxic Order Flow", current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        price = (current_ob['bids'][0][0] + current_ob['asks'][0][0]) / 2

        total_volume = sum(w['total_volume'] for w in windows)
        total_buy = sum(w['buy_volume'] for w in windows)
        overall_buy_ratio = total_buy / total_volume if total_volume > 0 else 0

        message = f"Toxic Order Flow: Score={score}, Buy ratio={overall_buy_ratio:.2f}"

        details = {
            "toxicity_score": score,
            "overall_buy_ratio": overall_buy_ratio,
            "analysis_windows": len(windows),
            "total_volume": total_volume,
            "confidence": min(score / 10.0, 1.0)
        }

        await self.db_manager.insert_alert(
            symbol, "Toxic Order Flow", price, message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, "Toxic Order Flow", current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"TOXIC ORDER FLOW ALERT for {symbol}: score={score}")

    async def detect_cross_market_manipulation(self, symbol: str, current_ob: dict):
        """Детектирует кросс-рыночные манипуляции (упрощенная версия)"""
        current_time = time.time()

        if len(self.orderbook_history[symbol]) < 10:
            return

        spreads = []
        for ts, snapshot in self.orderbook_history[symbol]:
            if current_time - ts > 60:
                continue
            if snapshot['bids'] and snapshot['asks']:
                spread = snapshot['asks'][0][0] - snapshot['bids'][0][0]
                spreads.append(spread)

        if len(spreads) < 5:
            return

        current_spread = spreads[-1]
        avg_spread = np.mean(spreads[:-1])

        if self.cross_market_anomaly_enabled and (current_spread > avg_spread * 5 or current_spread < avg_spread * 0.2):
            await self._create_cross_market_alert(symbol, current_spread, avg_spread, current_ob)

    async def _create_cross_market_alert(self, symbol: str, current_spread: float, avg_spread: float, current_ob: Dict):
        """Создает алерт о возможной кросс-рыночной манипуляции"""
        current_time = time.time()

        if self._is_alert_on_cooldown(symbol, "Cross-Market Anomaly", current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        price = (current_ob['bids'][0][0] + current_ob['asks'][0][0]) / 2

        spread_ratio = current_spread / avg_spread if avg_spread > 0 else 0

        message = f"Cross-Market Anomaly: Spread ratio={spread_ratio:.2f}"

        details = {
            "current_spread": current_spread,
            "average_spread": avg_spread,
            "spread_ratio": spread_ratio,
            "anomaly_type": "wide_spread" if spread_ratio > 1 else "narrow_spread",
            "confidence": min(abs(spread_ratio - 1) * 2, 1.0)
        }

        await self.db_manager.insert_alert(
            symbol, "Cross-Market Anomaly", price, message, details,
            alert_start_time=alert_start_time,
            order_book_snapshot=current_ob, # Pass current OB snapshot
            trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
        )

        self._update_alert_cooldown(symbol, "Cross-Market Anomaly", current_time)

        self.alert_counts[symbol] += 1
        logger.warning(f"CROSS-MARKET ANOMALY ALERT for {symbol}: ratio={spread_ratio:.2f}")

    async def _detect_spread_manipulation(self, symbol: str, current_spread: float,
                                          avg_spread: float, current_time: float):
        """Детектирует манипуляции со спредом"""
        if self._is_alert_on_cooldown(symbol, "Spread Manipulation", current_time):
            return

        alert_start_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')

        spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1

        if self.spread_manipulation_enabled and spread_ratio > self.spread_manipulation_threshold:
            current_ob = self.current_orderbooks[symbol]
            price = (current_ob['bids'][0][0] + current_ob['asks'][0][0]) / 2

            message = f"Spread Manipulation: Spread increased {spread_ratio:.1f}x"

            details = {
                "current_spread": current_spread,
                "average_spread": avg_spread,
                "spread_ratio": spread_ratio,
                "manipulation_type": "spread_widening",
                "confidence": min(spread_ratio / 5.0, 1.0)
            }

            await self.db_manager.insert_alert(
                symbol, "Spread Manipulation", price, message, details,
                alert_start_time=alert_start_time,
                order_book_snapshot=current_ob, # Pass current OB snapshot
                trade_history=list(self.trade_processor.trade_history.get(symbol, deque())) if self.trade_processor else None # Pass trade history
            )

            self._update_alert_cooldown(symbol, "Spread Manipulation", current_time)

            self.alert_counts[symbol] += 1
            logger.warning(f"SPREAD MANIPULATION ALERT for {symbol}: {spread_ratio:.1f}x increase")

    def _is_alert_on_cooldown(self, symbol: str, alert_type: str, current_time: float) -> bool:
        """Проверяет, находится ли алерт в периоде cooldown"""
        last_alert_time = self.recent_alerts[symbol].get(alert_type, 0)
        return (current_time - last_alert_time) < self.alert_cooldown_sec

    def _update_alert_cooldown(self, symbol: str, alert_type: str, current_time: float):
        """Обновляет время последнего алерта"""
        self.recent_alerts[symbol][alert_type] = current_time

    def get_analysis_stats(self):
        """Возвращает расширенную статистику анализа"""
        return {
            "analysis_counts": dict(self.analysis_counts),
            "alert_counts": dict(self.alert_counts),
            "trade_history_sizes": {
                symbol: len(history) for symbol, history in self.iceberg_trade_history.items()
            },
            "orderbook_history_sizes": {
                symbol: len(history) for symbol, history in self.orderbook_history.items()
            },
            "volume_profile_sizes": {
                symbol: len(profile) for symbol, profile in self.volume_profile.items()
            },
            "recent_alerts_count": {
                symbol: len(alerts) for symbol, alerts in self.recent_alerts.items()
            }
        }
