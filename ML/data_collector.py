import asyncio
import logging
import json
import time
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class MLDataCollector:
    def __init__(self, db_manager, ml_model, feature_engineer,
                 data_collection_window_sec=60,
                 target_prediction_window_sec=300,
                 min_data_points=100):
        self.db_manager = db_manager
        self.ml_model = ml_model
        self.feature_engineer = feature_engineer
        self.data_collection_window_sec = data_collection_window_sec
        self.target_prediction_window_sec = target_prediction_window_sec
        self.min_data_points = min_data_points

        self.pending_alerts_for_ml = defaultdict(deque)  # Stores alerts that need ML processing
        self.processed_ml_alerts = defaultdict(float)  # Cooldown for ML processing per alert type

        logger.info("MLDataCollector initialized.")

    async def add_alert_for_ml_processing(self, alert_data: Dict[str, Any]):
        """Adds an alert to a queue for asynchronous ML processing."""
        symbol = alert_data.get('symbol')
        alert_type = alert_data.get('alert_type')
        alert_id = alert_data.get('id')  # Get the alert ID
        if symbol and alert_type and alert_id is not None:
            alert_key = f"{symbol}_{alert_type}_{alert_id}"  # Use alert_id in key for uniqueness
            current_time = time.time()
            if current_time - self.processed_ml_alerts.get(alert_key, 0) < 60:  # 60 sec cooldown
                logger.debug(f"Alert {alert_key} on cooldown for ML processing.")
                return

            self.pending_alerts_for_ml[symbol].append(alert_data)
            logger.debug(
                f"Alert for {symbol} ({alert_type}, ID: {alert_id}) added to ML processing queue. Queue size: {len(self.pending_alerts_for_ml[symbol])}")
        else:
            logger.warning(f"Received incomplete alert data for ML processing: {alert_data}")

    async def process_pending_alerts_loop(self, interval_sec: int = 5):
        """
        Continuously processes pending alerts for ML prediction.
        This should be run as a background task.
        """
        logger.info(f"Starting ML data collector loop with interval {interval_sec} seconds.")
        while True:
            try:
                await self.process_pending_alerts()
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                logger.info("ML data collector loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in ML data collector loop: {e}")
                await asyncio.sleep(interval_sec * 2)  # Wait longer on error

    async def process_pending_alerts(self):
        """Processes alerts from the queue, collects data, and triggers ML prediction."""
        symbols_to_process = list(self.pending_alerts_for_ml.keys())
        if not symbols_to_process:
            logger.debug("No pending alerts to process for ML.")
            return

        for symbol in symbols_to_process:
            while self.pending_alerts_for_ml[symbol]:
                alert_data = self.pending_alerts_for_ml[symbol].popleft()
                alert_id = alert_data.get('id')
                alert_type = alert_data.get('alert_type')
                alert_key = f"{symbol}_{alert_type}_{alert_id}"

                current_time = time.time()
                if current_time - self.processed_ml_alerts.get(alert_key, 0) < 60:  # Re-check cooldown
                    self.pending_alerts_for_ml[symbol].append(alert_data)  # Put back if still on cooldown
                    logger.debug(f"Alert {alert_key} on cooldown during processing. Putting back.")
                    break  # Move to next symbol if this one is on cooldown

                logger.info(f"Processing alert for ML: {symbol} ({alert_type}, ID: {alert_id})")

                alert_timestamp_ms = alert_data.get('alert_timestamp_ms')
                if not alert_timestamp_ms:
                    logger.warning(f"Alert {alert_key} missing timestamp, skipping ML processing.")
                    continue

                order_book_snapshot = alert_data.get('order_book_snapshot')
                trade_history = alert_data.get('trade_history')

                if not order_book_snapshot or not trade_history:
                    logger.warning(
                        f"Missing order book or trade history for ML feature engineering for {symbol} (ID: {alert_id}). Skipping.")
                    continue

                logger.debug(f"Attempting feature extraction for {symbol} (ID: {alert_id}).")
                try:
                    features = self.feature_engineer.extract_features(
                        symbol,
                        order_book_snapshot,
                        trade_history,
                        alert_timestamp_ms / 1000  # Convert to seconds
                    )
                    if not features:
                        logger.warning(f"No features extracted for {symbol} (ID: {alert_id}). Skipping ML prediction.")
                        continue
                    logger.debug(f"Features extracted for {symbol} (ID: {alert_id}): {len(features)} features.")
                except Exception as fe_e:
                    logger.error(f"Error during feature extraction for {symbol} (ID: {alert_id}): {fe_e}")
                    continue

                try:
                    predicted_price_change, predicted_direction = self.ml_model.predict(features)
                    logger.info(f"ML Prediction for {symbol} ({alert_type}, ID: {alert_id}): "
                                f"Price Change={predicted_price_change:.4f}, Direction={predicted_direction}")

                    # Update the original alert in the database with ML predictions
                    await self.db_manager.update_alert_with_ml_prediction(
                        alert_id=alert_id,
                        predicted_price_change=predicted_price_change,
                        predicted_direction=predicted_direction,
                        ml_source_alert_type=alert_type  # Store original alert type
                    )
                    logger.info(f"Alert {alert_id} updated with ML prediction.")

                    # Insert data into ml_training_data table
                    await self.db_manager.insert_ml_training_data(
                        symbol=symbol,
                        features=features,
                        target_price_change=predicted_price_change,  # Using predicted as target for bootstrapping
                        target_direction=predicted_direction,
                        alert_id=alert_id
                    )
                    logger.info(f"ML training data inserted for {symbol} (ID: {alert_id}).")

                    self.processed_ml_alerts[alert_key] = current_time  # Update cooldown

                except Exception as ml_e:
                    logger.error(f"Error during ML prediction or data insertion for {symbol} (ID: {alert_id}): {ml_e}")
                    self.processed_ml_alerts[
                        alert_key] = current_time  # Mark as processed to avoid re-attempting immediately
                    continue

            if not self.pending_alerts_for_ml[symbol]:
                del self.pending_alerts_for_ml[symbol]

    async def collect_training_data(self, symbol: str, current_ob: Dict, trade_history: List[Dict],
                                    current_price: float):
        """
        Collects data for ML model training.
        This function should be called periodically or on specific events.
        (Note: This function is currently not called in the main app flow,
        data collection is primarily driven by alerts.)
        """
        current_timestamp = datetime.now()

        if len(trade_history) < self.min_data_points:
            logger.debug(f"Not enough trade history for {symbol} to collect ML training data.")
            return

        try:
            features = self.feature_engineer.extract_features(
                symbol,
                current_ob,
                trade_history,
                current_timestamp.timestamp()
            )
            if not features:
                logger.warning(f"No features extracted for {symbol} for training data collection.")
                return

            target_price_change = np.random.uniform(-0.01, 0.01)
            target_direction = "up" if target_price_change > 0 else "down" if target_price_change < 0 else "neutral"

            await self.db_manager.insert_ml_training_data(
                symbol, features, target_price_change, target_direction, alert_id=None
            )
            logger.debug(f"Collected ML training data for {symbol} at {current_timestamp}")

        except Exception as e:
            logger.error(f"Error collecting ML training data for {symbol}: {e}")

    async def train_model_periodically(self, interval_hours: int = 24, retention_days: int = 30):
        """
        Periodically retrieves training data and retrains the ML model.
        This should be run as a background task.
        """
        logger.info(f"Starting ML model training loop with interval {interval_hours} hours.")
        while True:
            try:
                if interval_hours <= 0.01:
                    logger.info("ML model training interval is very small, training immediately.")
                else:
                    logger.info("Initiating ML model retraining.")

                logger.debug(f"Attempting to retrieve training data. Current time: {datetime.now()}")
                training_data = await self.db_manager.get_ml_training_data()
                logger.debug(f"Retrieved {len(training_data)} training data points.")

                if not training_data:
                    logger.warning("No ML training data available yet. Skipping retraining.")
                else:
                    features_list = [d['features'] for d in training_data]
                    price_changes = [d['target_price_change'] for d in training_data]
                    directions = [d['target_direction'] for d in training_data]

                    features_list = [json.loads(f) if isinstance(f, str) else f for f in features_list]

                    # Get a consistent order of feature keys
                    all_feature_keys = sorted(list(set(key for d in features_list for key in d.keys())))

                    processed_features = []
                    for f_dict in features_list:
                        row = [f_dict.get(key, 0) for key in all_feature_keys]
                        processed_features.append(row)

                    if processed_features:
                        logger.debug(f"Preparing to train ML model with {len(processed_features)} samples.")
                        # Pass feature_names to the train method
                        self.ml_model.train(processed_features, price_changes, directions, all_feature_keys)
                        logger.info("ML model retrained successfully.")
                    else:
                        logger.warning("Processed features list is empty, cannot retrain ML model.")

                if interval_hours > 0.1:
                    await self.db_manager.clean_old_ml_training_data(retention_days)
                    logger.info(f"Cleaned ML training data older than {retention_days} days.")

                if interval_hours <= 0.01:
                    logger.info("ML model training loop completed (single run due to small interval).")
                    break
                else:
                    await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                logger.info("ML model training loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in ML model training loop: {e}")
                await asyncio.sleep(3600)
