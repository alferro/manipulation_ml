import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from collections import deque  # Added deque for consistency, though not directly used here
from time import time
from cryptoscan.backand.bybit.ML.ml_model import MLModel
from cryptoscan.backand.bybit.ML.feature_engineering import FeatureEngineer
from cryptoscan.backand.bybit.db_manager import DBManager
from cryptoscan.backand.bybit.manipulation.trade_processor import TradeProcessor
from cryptoscan.backand.bybit.manipulation.orderbook_analyzer import OrderBookAnalyzer

logger = logging.getLogger(__name__)


class MLPricePredictor:
    def __init__(self, db_manager: DBManager, ml_model: MLModel, feature_engineer: FeatureEngineer,
                 trade_processor: TradeProcessor, orderbook_analyzer: OrderBookAnalyzer):
        self.db_manager = db_manager
        self.ml_model = ml_model
        self.feature_engineer = feature_engineer
        self.trade_processor = trade_processor
        self.orderbook_analyzer = orderbook_analyzer

        # Link managers to feature engineer for data access
        self.feature_engineer.trade_processor = self.trade_processor
        self.feature_engineer.orderbook_analyzer = self.orderbook_analyzer

        logger.info("MLPricePredictor initialized and managers linked to FeatureEngineer.")

    async def predict_for_alert(self, symbol: str, alert_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Predicts price change and direction for a given alert.
        This function is called by MLDataCollector.
        """
        alert_timestamp_ms = alert_data.get('alert_timestamp_ms')
        if not alert_timestamp_ms:
            logger.warning(f"Alert data for {symbol} missing timestamp, cannot predict.")
            return 0.0, "neutral"

        # Retrieve relevant historical data from managers
        # This is a critical part: ensure the managers hold enough history
        # or can retrieve it efficiently for the feature engineering window.

        # Get current order book snapshot (or from alert if provided)
        order_book_snapshot = alert_data.get('order_book_snapshot')
        if not order_book_snapshot and self.orderbook_analyzer:
            current_ob = self.orderbook_analyzer.current_orderbooks.get(symbol)
            if current_ob:
                # Convert internal dict representation to list for feature engineer
                order_book_snapshot = {
                    'bids': sorted([[p, s] for p, s in current_ob['bids'].items()], key=lambda x: x[0], reverse=True),
                    'asks': sorted([[p, s] for p, s in current_ob['asks'].items()], key=lambda x: x[0])
                }

        # Get trade history (or from alert if provided)
        trade_history = alert_data.get('trade_history')
        if not trade_history and self.trade_processor:
            trade_history = list(self.trade_processor.trade_history.get(symbol, deque()))

        if not order_book_snapshot or not trade_history:
            logger.warning(f"Insufficient real-time data from managers for ML prediction for {symbol}.")
            return 0.0, "neutral"

        try:
            features = self.feature_engineer.extract_features(
                symbol,
                order_book_snapshot,
                trade_history,
                alert_timestamp_ms / 1000  # Convert to seconds
            )
            if not features:
                logger.warning(f"No features extracted for {symbol} for prediction.")
                return 0.0, "neutral"

            predicted_price_change, predicted_direction = self.ml_model.predict(features)
            return predicted_price_change, predicted_direction

        except Exception as e:
            logger.error(f"Error during ML prediction for alert {symbol}: {e}")
            return 0.0, "neutral"

    async def get_prediction_for_current_state(self, symbol: str) -> Dict[str, Any]:
        """
        Provides a real-time prediction based on the current market state.
        This can be used for API endpoints.
        """
        if not self.ml_model.get_model_status()['price_model_trained']:
            return {"error": "ML models are not trained yet."}

        current_ob = self.orderbook_analyzer.current_orderbooks.get(symbol)
        trade_history = list(self.trade_processor.trade_history.get(symbol, deque()))
        current_timestamp = time.time()

        if not current_ob or not trade_history:
            return {"error": f"No real-time data available for {symbol}."}

        # Convert internal dict representation to list for feature engineer
        current_ob_for_fe = {
            'bids': sorted([[p, s] for p, s in current_ob['bids'].items()], key=lambda x: x[0], reverse=True),
            'asks': sorted([[p, s] for p, s in current_ob['asks'].items()], key=lambda x: x[0])
        }

        try:
            features = self.feature_engineer.extract_features(
                symbol,
                current_ob_for_fe,
                trade_history,
                current_timestamp
            )
            if not features:
                return {"error": f"Could not extract features for {symbol}."}

            predicted_price_change, predicted_direction = self.ml_model.predict(features)

            return {
                "symbol": symbol,
                "predicted_price_change": predicted_price_change,
                "predicted_direction": predicted_direction,
                "timestamp": datetime.fromtimestamp(current_timestamp).isoformat(),
                "model_status": self.ml_model.get_model_status()
            }
        except Exception as e:
            logger.error(f"Error getting real-time prediction for {symbol}: {e}")
            return {"error": f"Failed to get real-time prediction: {e}"}
