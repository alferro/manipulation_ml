import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from collections import deque
import time # ADDED: Import time for time.time()

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, trade_processor=None, orderbook_analyzer=None):
        self.trade_processor = trade_processor
        self.orderbook_analyzer = orderbook_analyzer
        logger.info("FeatureEngineer initialized.")

    def extract_features(self, symbol: str, order_book_snapshot: Dict, trade_history: List[Dict], timestamp_sec: float) -> Optional[Dict[str, Any]]:
        """
        Extracts features from order book and trade data for a given symbol at a specific timestamp.
        Args:
            symbol: The trading pair symbol (e.g., "BTCUSDT").
            order_book_snapshot: A dictionary with 'bids' and 'asks' lists.
            trade_history: A list of trade dictionaries.
            timestamp_sec: The timestamp (in seconds) around which to extract features.
        Returns:
            A dictionary of extracted features, or None if data is insufficient.
        """
        if not order_book_snapshot or not trade_history:
            logger.warning(f"Insufficient data for feature engineering for {symbol}.")
            return None

        features = {}

        # --- Order Book Features ---
        bids = order_book_snapshot.get('bids', [])
        asks = order_book_snapshot.get('asks', [])

        if not bids or not asks:
            logger.warning(f"Order book snapshot is empty for {symbol}.")
            return None

        best_bid_price = bids[0][0]
        best_bid_size = bids[0][1]
        best_ask_price = asks[0][0]
        best_ask_size = asks[0][1]

        features['spread'] = best_ask_price - best_bid_price
        features['mid_price'] = (best_bid_price + best_ask_price) / 2
        features['relative_spread'] = features['spread'] / features['mid_price'] if features['mid_price'] > 0 else 0

        # Liquidity at different depths
        depth_levels = [5, 10, 20] # Number of levels to sum liquidity
        for level in depth_levels:
            features[f'bid_liquidity_depth_{level}'] = sum(s for p, s in bids[:level])
            features[f'ask_liquidity_depth_{level}'] = sum(s for p, s in asks[:level])
            features[f'total_liquidity_depth_{level}'] = features[f'bid_liquidity_depth_{level}'] + features[f'ask_liquidity_depth_{level}']
            features[f'liquidity_imbalance_depth_{level}'] = (features[f'bid_liquidity_depth_{level}'] - features[f'ask_liquidity_depth_{level}']) / features[f'total_liquidity_depth_{level}'] if features[f'total_liquidity_depth_{level}'] > 0 else 0

        # Order book imbalance (overall)
        total_bid_volume = sum(s for p, s in bids)
        total_ask_volume = sum(s for p, s in asks)
        total_ob_volume = total_bid_volume + total_ask_volume
        features['orderbook_imbalance'] = (total_bid_volume - total_ask_volume) / total_ob_volume if total_ob_volume > 0 else 0

        # --- Trade Features (recent history) ---
        # Filter trades within a recent window (e.g., last 60 seconds)
        recent_trades = [trade for trade in trade_history if timestamp_sec - trade['timestamp'] <= 60]

        if not recent_trades:
            logger.warning(f"No recent trades for feature engineering for {symbol}.")
            # Return features collected so far, or None if critical trade features are missing
            return features if features else None

        total_trade_volume_usd = sum(trade['volume_usdt'] for trade in recent_trades)
        buy_volume_usd = sum(trade['volume_usdt'] for trade in recent_trades if trade['side'] == 'Buy')
        sell_volume_usd = sum(trade['volume_usdt'] for trade in recent_trades if trade['side'] == 'Sell')

        features['total_trade_volume_usd'] = total_trade_volume_usd
        features['buy_trade_volume_usd'] = buy_volume_usd
        features['sell_trade_volume_usd'] = sell_volume_usd
        features['trade_count'] = len(recent_trades)

        features['trade_volume_imbalance'] = (buy_volume_usd - sell_volume_usd) / total_trade_volume_usd if total_trade_volume_usd > 0 else 0
        features['avg_trade_size_usd'] = total_trade_volume_usd / len(recent_trades) if len(recent_trades) > 0 else 0

        # Price volatility (e.g., standard deviation of prices in the window)
        recent_prices = [trade['price'] for trade in recent_trades]
        if len(recent_prices) > 1:
            features['price_volatility'] = np.std(recent_prices)
            features['price_change_recent'] = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        else:
            features['price_volatility'] = 0
            features['price_change_recent'] = 0

        # --- Interaction Features (if managers are available) ---
        # These features would typically come from the detectors themselves,
        # indicating the strength of a detected pattern.
        # For this setup, we'll simulate or use simplified versions.

        # Example: Simplified wash trading signal (if trade_processor is linked)
        if self.trade_processor and symbol in self.trade_processor.processed_trades:
            # This is a simplification. A real feature would be a score from the detector.
            features['wash_trade_signal'] = self.trade_processor.wash_trade_alerts.get(symbol, 0) > 0
            features['ping_pong_signal'] = self.trade_processor.ping_pong_alerts.get(symbol, 0) > 0
            features['ramping_signal'] = self.trade_processor.ramping_alerts.get(symbol, 0) > 0
        else:
            features['wash_trade_signal'] = False
            features['ping_pong_signal'] = False
            features['ramping_signal'] = False

        # Example: Simplified iceberg signal (if orderbook_analyzer is linked)
        if self.orderbook_analyzer and symbol in self.orderbook_analyzer.analysis_counts:
            # This is a simplification. A real feature would be a score from the detector.
            features['iceberg_signal'] = self.orderbook_analyzer.alert_counts.get(symbol, 0) > 0 # Placeholder
            features['layering_spoofing_signal'] = self.orderbook_analyzer.alert_counts.get(symbol, 0) > 0 # Placeholder
        else:
            features['iceberg_signal'] = False
            features['layering_spoofing_signal'] = False

        # Convert boolean features to int (0 or 1) for ML models
        for key, value in features.items():
            if isinstance(value, bool):
                features[key] = int(value)

        return features

    def get_feature_names(self) -> List[str]:
        """Возвращает список всех возможных имен признаков."""
        dummy_symbol = "BTCUSDT"
        dummy_order_book_snapshot = {
            'bids': [[99.9, 100], [99.8, 200]],
            'asks': [[100.1, 150], [100.2, 250]]
        }
        dummy_trade_history = [
            {'timestamp': 1, 'price': 100, 'size': 10, 'side': 'Buy', 'volume_usdt': 1000},
            {'timestamp': 2, 'price': 100.01, 'size': 12, 'side': 'Sell', 'volume_usdt': 1200}
        ]
        dummy_timestamp_sec = time.time()

        dummy_features = self.extract_features(dummy_symbol, dummy_order_book_snapshot, dummy_trade_history, dummy_timestamp_sec)
        if dummy_features:
            return sorted(list(dummy_features.keys()))
        return []
