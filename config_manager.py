import os
from dotenv import load_dotenv, dotenv_values
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define all default settings
DEFAULT_ENV_SETTINGS = {
    "LOG_LEVEL": "INFO",
    "LOG_FILE": "cryptoscan.log",

    "DB_USER": "your_db_user",
    "DB_PASSWORD": "your_db_password",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "cryptoscan_db",

    "LOCAL_WS_PORT": "8766",
    "WS_PING_INTERVAL": "20",
    "WS_PING_TIMEOUT": "10",
    "WS_CLOSE_TIMEOUT": "10",
    "WS_MAX_SIZE": "10000000",

    "VOLUME_WINDOW_MINUTES": "5",
    "VOLUME_MULTIPLIER": "3.5",
    "MIN_VOLUME_USDT": "25000",
    "WASH_TRADE_THRESHOLD_RATIO": "0.88",
    "PING_PONG_WINDOW_SEC": "25",
    "RAMPING_WINDOW_SEC": "50",
    "CONSECUTIVE_LONG_COUNT": "7",
    "ALERT_GROUPING_MINUTES": "10",
    "DATA_RETENTION_HOURS": "24",
    "VOLUME_TYPE": "long",

    "ORDERBOOK_ENABLED": "true",
    "ORDERBOOK_SNAPSHOT_ON_ALERT": "true",
    "OB_HISTORY_DEPTH": "50",
    "ICEBERG_WINDOW_SEC": "15",
    "ICEBERG_VOLUME_RATIO": "2.2",
    "ICEBERG_MIN_COUNT": "6",
    "LAYERING_DISTANCE_PERCENT": "0.0015",
    "LAYERING_MIN_CHANGE": "250",
    "LAYERING_WINDOW_SEC": "5",
    "SPOOFING_CANCEL_RATIO": "0.8",
    "MOMENTUM_IGNITION_THRESHOLD": "0.025",
    "OB_IMBALANCE_THRESHOLD": "0.75",
    "OB_ICEBERG_MIN_VOLUME_RATIO": "0.1",
    "OB_ICEBERG_HIDDEN_VOLUME_MULTIPLIER": "5.0",
    "OB_ICEBERG_PRICE_TOLERANCE": "0.0001",
    "LAYERING_SPOOFING_DEPTH": "5",
    "LAYERING_SPOOFING_THRESHOLD": "0.7",
    "LAYERING_SPOOFING_TIME_WINDOW_SEC": "30",
    "LIQUIDITY_DETECTION_WINDOW_SEC": "300",
    "LIQUIDITY_CHANGE_THRESHOLD": "0.2",
    "TOXIC_ORDER_FLOW_WINDOW_SEC": "10",
    "TOXIC_ORDER_FLOW_THRESHOLD": "0.85",
    "CROSS_MARKET_ANOMALY_ENABLED": "true",
    "CROSS_MARKET_ANOMALY_THRESHOLD": "0.0007",
    "SPREAD_MANIPULATION_ENABLED": "true",
    "SPREAD_MANIPULATION_THRESHOLD": "0.01",
    "SPREAD_MANIPULATION_TIME_WINDOW_SEC": "10",
    "ALERT_COOLDOWN_SEC": "60",

    "ML_DATA_COLLECTOR_INTERVAL_SEC": "5",
    "ML_MODEL_TRAINING_INTERVAL_HOURS": "24"
}

ENV_FILE_PATH = os.path.join(os.getcwd(), '.env')
_current_config: Dict[str, Any] = {}


def ensure_env_file():
    """
    Ensures the .env file exists and contains all default settings.
    Adds missing settings without overwriting existing ones.
    """
    existing_values = {}
    if os.path.exists(ENV_FILE_PATH):
        existing_values = dotenv_values(ENV_FILE_PATH)
        logger.info(f"Existing .env file found at {ENV_FILE_PATH}. Checking for missing settings.")
    else:
        logger.info(f"No .env file found at {ENV_FILE_PATH}. Creating with default settings.")

    # Read existing lines to preserve comments and order if possible, but prioritize defaults
    lines_to_write = []
    existing_lines_map = {}  # To track if a key from DEFAULT_ENV_SETTINGS was already in the file

    if os.path.exists(ENV_FILE_PATH):
        with open(ENV_FILE_PATH, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    key_value = stripped_line.split('=', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        existing_values[key] = key_value[1].strip()
                        existing_lines_map[key] = True  # Mark as found in file
                lines_to_write.append(line.rstrip())  # Keep original line ending

    # Add missing default settings or update existing ones if they are not present in the file
    for key, default_value in DEFAULT_ENV_SETTINGS.items():
        if key not in existing_values:
            # Add new default setting
            lines_to_write.append(f"{key}={default_value}")
            logger.info(f"Added missing setting to .env: {key}={default_value}")
        elif key not in existing_lines_map:
            # This case handles if a default key was in existing_values but not explicitly in lines_to_write
            # (e.g., if it was commented out or part of a complex line).
            # For simplicity, we'll just ensure it's written at the end if not found.
            pass  # It's already in existing_values, will be handled by final write

    # Reconstruct the file content, prioritizing existing values for default keys
    final_content = []
    # Add comments and headers
    final_content.append("# ==============================================================================")
    final_content.append("# This file is automatically managed by CryptoScan Backend.")
    final_content.append("# Missing settings will be added with default values.")
    final_content.append("# Existing values will be preserved.")
    final_content.append("# ==============================================================================")
    final_content.append("")

    # Add default settings, using existing values if present
    for key, default_value in DEFAULT_ENV_SETTINGS.items():
        final_content.append(f"{key}={existing_values.get(key, default_value)}")

    # Add any custom keys that were in the original .env but not in DEFAULT_ENV_SETTINGS
    custom_keys_added = set()
    for key, value in existing_values.items():
        if key not in DEFAULT_ENV_SETTINGS:
            if key not in custom_keys_added:  # Avoid duplicates
                final_content.append(f"{key}={value}")
                custom_keys_added.add(key)

    with open(ENV_FILE_PATH, 'w') as f:  # Open in write mode to overwrite
        f.write("\n".join(final_content) + "\n")
    logger.info(".env file ensured and updated with default settings.")


def load_config() -> Dict[str, Any]:
    """
    Loads environment variables from .env file and updates os.environ.
    Returns a dictionary of loaded configurations.
    """
    global _current_config
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)  # Load into os.environ

    # Read values directly from .env file to get the actual values, not just what's in os.environ
    file_values = dotenv_values(ENV_FILE_PATH)

    # Populate _current_config with values from file, falling back to defaults
    for key, default_value in DEFAULT_ENV_SETTINGS.items():
        _current_config[key] = file_values.get(key, default_value)
        # Also ensure os.environ is updated for consistency
        os.environ[key] = str(_current_config[key])  # Ensure it's a string for os.environ

    logger.info("Configuration loaded from .env file.")
    return _current_config


def get_current_config() -> Dict[str, Any]:
    """Returns the currently loaded configuration."""
    return _current_config


def reload_config():
    """Reloads the configuration from the .env file."""
    logger.info("Reloading configuration from .env file...")
    load_config()  # This will update _current_config and os.environ
    logger.info("Configuration reloaded.")
