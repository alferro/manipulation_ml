import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler # Added: Import StandardScaler
import joblib # For saving/loading models
import os
from typing import List, Tuple, Dict, Any, Optional # Добавлен импорт Optional

logger = logging.getLogger(__name__)


class MLModel:
    def __init__(self, model_path="ml_model.joblib"):
        self.price_model: Optional[RandomForestRegressor] = None
        self.direction_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None # Added: StandardScaler instance
        self.feature_names: Optional[List[str]] = None # Added: Store feature names for consistent prediction
        self.model_path = model_path
        self._load_models()
        logger.info("MLModel initialized.")

    def _load_models(self):
        """Loads pre-trained models or initializes new ones."""
        if os.path.exists(self.model_path):
            try:
                models = joblib.load(self.model_path)
                self.price_model = models['price_model']
                self.direction_model = models['direction_model']
                self.scaler = models.get('scaler') # Load scaler
                self.feature_names = models.get('feature_names') # Load feature names
                logger.info("ML models and scaler loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}. Initializing new models.")
                self._initialize_models()
        else:
            logger.info("No existing ML models found. Initializing new models.")
            self._initialize_models()

    def _initialize_models(self):
        """Initializes new RandomForest models and a new scaler."""
        self.price_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.direction_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler() # Initialize new scaler
        self.feature_names = None # Reset feature names
        logger.info("New RandomForest models and StandardScaler initialized.")

    def train(self, features: List[List[float]], target_price_change: List[float], target_direction: List[str], feature_names: List[str]):
        """
        Trains the ML models.
        Args:
            features: List of lists, where each inner list is a feature vector.
            target_price_change: List of float, actual price changes.
            target_direction: List of str, actual price directions ('up', 'down', 'neutral').
            feature_names: List of strings, names of features in the order they appear in 'features'.
        """
        if not features or len(features) < 2:
            logger.warning("Not enough data to train ML models.")
            return

        X = np.array(features)
        y_price = np.array(target_price_change)
        y_direction = np.array(target_direction)

        self.feature_names = feature_names # Store feature names for consistent prediction

        # Split data for validation
        X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = \
            train_test_split(X, y_price, y_direction, test_size=0.2, random_state=42, stratify=y_direction)

        # Fit and transform features using the scaler
        logger.info("Fitting StandardScaler and transforming features...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info("Features scaled.")

        logger.info(f"Training price prediction model with {len(X_train_scaled)} samples...")
        self.price_model.fit(X_train_scaled, y_price_train)
        price_predictions = self.price_model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_price_test, price_predictions))
        logger.info(f"Price model trained. Test RMSE: {rmse:.4f}")

        logger.info(f"Training direction prediction model with {len(X_train_scaled)} samples...")
        self.direction_model.fit(X_train_scaled, y_direction_train)
        direction_predictions = self.direction_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_direction_test, direction_predictions)
        logger.info(f"Direction model trained. Test Accuracy: {accuracy:.4f}")

        # Cross-validation for more robust evaluation (using all data, scaled)
        X_scaled_all = self.scaler.transform(X)
        price_cv_scores = cross_val_score(self.price_model, X_scaled_all, y_price, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        logger.info(f"Price model CV RMSE: {np.sqrt(-price_cv_scores).mean():.4f}")

        direction_cv_scores = cross_val_score(self.direction_model, X_scaled_all, y_direction, cv=5, scoring='accuracy', n_jobs=-1)
        logger.info(f"Direction model CV Accuracy: {direction_cv_scores.mean():.4f}")

        self._save_models()
        logger.info("ML models training complete and saved.")

    def predict(self, features: Dict[str, Any]) -> Tuple[float, str]:
        """
        Makes a prediction using the trained models.
        Args:
            features: A dictionary of features for a single data point.
        Returns:
            A tuple of (predicted_price_change, predicted_direction).
        """
        if not self.price_model or not self.direction_model or not self.scaler or not self.feature_names:
            logger.warning("ML models or scaler not trained/loaded. Returning default prediction.")
            return 0.0, "neutral"

        # Ensure feature order consistency with training data using stored feature_names
        feature_vector = np.array([[features.get(key, 0) for key in self.feature_names]])

        # Scale the feature vector before prediction
        feature_vector_scaled = self.scaler.transform(feature_vector)

        predicted_price_change = self.price_model.predict(feature_vector_scaled)[0]
        predicted_direction = self.direction_model.predict(feature_vector_scaled)[0]

        return float(predicted_price_change), str(predicted_direction)

    def _save_models(self):
        """Saves the trained models and scaler to disk."""
        try:
            models = {
                'price_model': self.price_model,
                'direction_model': self.direction_model,
                'scaler': self.scaler, # Save the scaler
                'feature_names': self.feature_names # Save feature names
            }
            joblib.dump(models, self.model_path)
            logger.info(f"ML models and scaler saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")

    def get_model_status(self) -> Dict[str, bool]:
        """Returns the training status of the models and scaler."""
        return {
            "price_model_trained": self.price_model is not None,
            "direction_model_trained": self.direction_model is not None,
            "scaler_fitted": self.scaler is not None and hasattr(self.scaler, 'scale_') # Check if scaler is fitted
        }
