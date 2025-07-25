import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

import config_manager

from db_manager import DBManager
from manipulation.trade_processor import TradeProcessor
from manipulation.orderbook_analyzer import OrderBookAnalyzer
from debug_monitor import DebugMonitor
from ML.ml_model import MLModel
from ML.feature_engineering import FeatureEngineer
from ML.data_collector import MLDataCollector
from ML.ml_price_predictor import MLPricePredictor
import bybit_websocket_client as bwc

logger = logging.getLogger(__name__)


class CryptoScanApp:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager: Optional[DBManager] = None
        self.trade_processor: Optional[TradeProcessor] = None
        self.orderbook_analyzer: Optional[OrderBookAnalyzer] = None
        self.debug_monitor: Optional[DebugMonitor] = None
        self.ml_model: Optional[MLModel] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.ml_data_collector: Optional[MLDataCollector] = None
        self.ml_price_predictor: Optional[MLPricePredictor] = None

        self.bybit_stream_task: Optional[asyncio.Task] = None
        self.debug_monitor_task: Optional[asyncio.Task] = None
        self.ml_data_collector_task: Optional[asyncio.Task] = None
        self.ml_model_training_task: Optional[asyncio.Task] = None

        self._setup_logging()
        logger.info("CryptoScanApp instance initialized.")

    def _setup_logging(self):
        """Sets up logging based on current configuration."""
        log_level = self.config.get("LOG_LEVEL", "INFO").upper()
        log_file = self.config.get("LOG_FILE", "cryptoscan.log")

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        logger.setLevel(log_level)
        logging.getLogger("websockets").setLevel(logging.WARNING)
        logging.getLogger("asyncpg").setLevel(logging.WARNING)
        logger.info(f"Logging configured to level {log_level} and file {log_file}.")

    async def _initialize_components(self):
        """Initializes all core components of the application."""
        logger.info("Initializing application components...")
        try:
            # Initialize DBManager first
            self.db_manager = DBManager()
            await self.db_manager.init_db_connection()

            # Initialize ML components
            self.ml_model = MLModel()
            self.feature_engineer = FeatureEngineer()
            self.ml_data_collector = MLDataCollector(
                db_manager=self.db_manager,  # Pass db_manager to MLDataCollector
                ml_model=self.ml_model,
                feature_engineer=self.feature_engineer
            )
            # Set MLDataCollector in DBManager after it's initialized
            self.db_manager.set_ml_data_collector(self.ml_data_collector)

            self.trade_processor = TradeProcessor(
                self.db_manager,
                volume_window_minutes=int(self.config.get("VOLUME_WINDOW_MINUTES", 5)),
                wash_trade_threshold_ratio=float(self.config.get("WASH_TRADE_THRESHOLD_RATIO", 0.75)),
                ping_pong_window_sec=int(self.config.get("PING_PONG_WINDOW_SEC", 45)),
                ramping_window_sec=int(self.config.get("RAMPING_WINDOW_SEC", 90)),
                volume_multiplier=float(self.config.get("VOLUME_MULTIPLIER", 2.0)),
                min_volume_usdt=float(self.config.get("MIN_VOLUME_USDT", 1000)),
                volume_type=self.config.get("VOLUME_TYPE", "long")
            )

            self.orderbook_analyzer = OrderBookAnalyzer(
                self.db_manager,
                trade_processor=self.trade_processor,  # Pass trade_processor to orderbook_analyzer
                ob_history_depth=int(self.config.get("OB_HISTORY_DEPTH", 25)),
                iceberg_trade_window_sec=int(self.config.get("ICEBERG_WINDOW_SEC", 15)),
                iceberg_volume_ratio_threshold=float(self.config.get("ICEBERG_VOLUME_RATIO", 1.6)),
                iceberg_min_trade_count=int(self.config.get("ICEBERG_MIN_COUNT", 4)),
                layering_distance_from_market_percent=float(self.config.get("LAYERING_DISTANCE_PERCENT", 0.003)),
                layering_min_size_change_abs=float(self.config.get("LAYERING_MIN_CHANGE", 150)),
                layering_ob_window_sec=int(self.config.get("LAYERING_WINDOW_SEC", 8)),
                spoofing_cancel_ratio=float(self.config.get("SPOOFING_CANCEL_RATIO", 0.65)),
                momentum_ignition_threshold=float(self.config.get("MOMENTUM_IGNITION_THRESHOLD", 0.015)),
                enabled=self.config.get("ORDERBOOK_ENABLED", "False").lower() == "true",
                snapshot_on_alert=self.config.get("ORDERBOOK_SNAPSHOT_ON_ALERT", "False").lower() == "true",
                imbalance_threshold=float(self.config.get("OB_IMBALANCE_THRESHOLD", 0.6)),
                iceberg_min_volume_ratio=float(self.config.get("OB_ICEBERG_MIN_VOLUME_RATIO", 0.1)),
                iceberg_hidden_volume_multiplier=float(self.config.get("OB_ICEBERG_HIDDEN_VOLUME_MULTIPLIER", 5.0)),
                iceberg_price_tolerance=float(self.config.get("OB_ICEBERG_PRICE_TOLERANCE", 0.0001)),
                layering_spoofing_depth=int(self.config.get("LAYERING_SPOOFING_DEPTH", 5)),
                layering_spoofing_threshold=float(self.config.get("LAYERING_SPOOFING_THRESHOLD", 0.7)),
                layering_spoofing_time_window_sec=int(self.config.get("LAYERING_SPOOFING_TIME_WINDOW_SEC", 30)),
                liquidity_detection_window_sec=int(self.config.get("LIQUIDITY_DETECTION_WINDOW_SEC", 300)),
                liquidity_change_threshold=float(self.config.get("LIQUIDITY_CHANGE_THRESHOLD", 0.2)),
                toxic_order_flow_window_sec=int(self.config.get("TOXIC_ORDER_FLOW_WINDOW_SEC", 10)),
                toxic_order_flow_threshold=float(self.config.get("TOXIC_ORDER_FLOW_THRESHOLD", 0.7)),
                cross_market_anomaly_enabled=self.config.get("CROSS_MARKET_ANOMALY_ENABLED", "False").lower() == "true",
                cross_market_anomaly_threshold=float(self.config.get("CROSS_MARKET_ANOMALY_THRESHOLD", 0.001)),
                spread_manipulation_enabled=self.config.get("SPREAD_MANIPULATION_ENABLED", "False").lower() == "true",
                spread_manipulation_threshold=float(self.config.get("SPREAD_MANIPULATION_THRESHOLD", 0.005)),
                spread_manipulation_time_window_sec=int(self.config.get("SPREAD_MANIPULATION_TIME_WINDOW_SEC", 10))
            )

            self.ml_price_predictor = MLPricePredictor(
                db_manager=self.db_manager,
                ml_model=self.ml_model,
                feature_engineer=self.feature_engineer,
                trade_processor=self.trade_processor,
                orderbook_analyzer=self.orderbook_analyzer
            )

            self.debug_monitor = DebugMonitor(
                self.db_manager,
                self.trade_processor,
                self.orderbook_analyzer
            )

            bwc.set_managers(self.db_manager, self.trade_processor, self.orderbook_analyzer, self.ml_data_collector)
            logger.info("Application components initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize application components: {e}")
            raise

    async def _start_background_tasks(self):
        """Starts all background asyncio tasks."""
        logger.info("Starting background tasks...")
        self.bybit_stream_task = asyncio.create_task(bwc.start_bybit_data_stream(), name="bybit_stream_task")
        self.debug_monitor_task = asyncio.create_task(self.debug_monitor.start_monitoring(), name="debug_monitor_task")

        ml_data_collector_interval = int(self.config.get("ML_DATA_COLLECTOR_INTERVAL_SEC", 5))
        self.ml_data_collector_task = asyncio.create_task(
            self.ml_data_collector.process_pending_alerts_loop(ml_data_collector_interval),
            name="ml_data_collector_task"
        )
        logger.info(f"ML Data Collector loop started with interval {ml_data_collector_interval} seconds.")

        ml_model_training_interval_hours = int(self.config.get("ML_MODEL_TRAINING_INTERVAL_HOURS", 24))
        self.ml_model_training_task = asyncio.create_task(
            self.ml_data_collector.train_model_periodically(ml_model_training_interval_hours),
            name="ml_model_training_task"
        )
        logger.info(f"ML Model Training loop started with interval {ml_model_training_interval_hours} hours.")
        logger.info("All background tasks started successfully.")

    async def _stop_background_tasks(self):
        """Cancels all running background asyncio tasks."""
        logger.info("Stopping background tasks...")
        tasks_to_cancel = [
            self.bybit_stream_task,
            self.debug_monitor_task,
            self.ml_data_collector_task,
            self.ml_model_training_task
        ]

        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                    logger.info(f"Background task {task.get_name()} successfully cancelled.")
                except asyncio.CancelledError:
                    logger.info(f"Background task {task.get_name()} confirmed cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation for {task.get_name()}: {e}")
        logger.info("All background tasks stopped.")

    async def start(self):
        """Starts the application: initializes components and background tasks."""
        logger.info("Starting CryptoScanApp...")
        await self._initialize_components()
        await self._start_background_tasks()
        logger.info("CryptoScanApp started.")

    async def stop(self):
        """Stops the application: cancels tasks and closes connections."""
        logger.info("Stopping CryptoScanApp...")
        await self._stop_background_tasks()
        if self.db_manager:
            await self.db_manager.close_db_connection()
        logger.info("CryptoScanApp stopped.")

    async def reconfigure(self, new_config: Dict[str, Any]):
        """
        Reconfigures the application with new settings.
        Stops current operations, updates config, and restarts.
        """
        logger.info("Reconfiguring CryptoScanApp with new settings...")
        await self.stop()
        self.config = new_config
        self._setup_logging()
        await self.start()
        logger.info("CryptoScanApp reconfigured and restarted.")


app_instance: Optional[CryptoScanApp] = None
observer: Optional[Observer] = None


class ConfigFileEventHandler(FileSystemEventHandler):
    """Handles events for the .env file changes."""

    def __init__(self, app: CryptoScanApp):
        super().__init__()
        self.app = app
        self.last_modified_time = 0

    def on_modified(self, event):
        if event.is_directory:
            return
        if os.path.basename(event.src_path) == '.env':
            current_time = time.time()
            if current_time - self.last_modified_time < 1.0:
                return
            self.last_modified_time = current_time

            logger.info(f".env file modified: {event.src_path}. Reloading configuration...")
            config_manager.reload_config()
            asyncio.create_task(self.app.reconfigure(config_manager.get_current_config()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for managing the lifespan of the FastAPI application.
    Initializes resources on startup and cleans them up on shutdown.
    """
    global app_instance, observer

    logger.info("Application startup: Ensuring .env file and initializing CryptoScanApp...")
    try:
        config_manager.ensure_env_file()
        initial_config = config_manager.load_config()

        app_instance = CryptoScanApp(initial_config)
        await app_instance.start()

        event_handler = ConfigFileEventHandler(app_instance)
        observer = Observer()
        observer.schedule(event_handler, path='.', recursive=False)
        observer.start()
        logger.info("Watchdog observer started for .env file changes.")

        app.state.db_manager = app_instance.db_manager
        app.state.trade_processor = app_instance.trade_processor
        app.state.orderbook_analyzer = app_instance.orderbook_analyzer
        app.state.ml_price_predictor = app_instance.ml_price_predictor
        app.state.debug_monitor = app_instance.debug_monitor

        yield

    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        if observer and observer.is_alive():
            observer.stop()
            observer.join()
        raise

    logger.info("Application shutdown: Cleaning up resources...")
    if observer and observer.is_alive():
        observer.stop()
        observer.join()
        logger.info("Watchdog observer stopped.")

    if app_instance:
        await app_instance.stop()
    logger.info("Application shutdown complete.")


app = FastAPI(
    title="CryptoScan Backend",
    description="API for real-time crypto market data analysis and manipulation detection.",
    version="1.0.0",
    lifespan=lifespan
)

from _api_watchlist import setup_watchlist_routes
from _api_alerts import setup_alerts_routes

watchlist_router = setup_watchlist_routes(None)
alerts_router = setup_alerts_routes(None)

app.include_router(watchlist_router)
app.include_router(alerts_router)


@app.middleware("http")
async def add_managers_to_request(request: Request, call_next):
    if app_instance:
        request.state.db_manager = app_instance.db_manager
        request.state.trade_processor = app_instance.trade_processor
        request.state.orderbook_analyzer = app_instance.orderbook_analyzer
        request.state.ml_price_predictor = app_instance.ml_price_predictor
        request.state.debug_monitor = app_instance.debug_monitor
    else:
        logger.error("app_instance is not initialized in middleware!")
        raise HTTPException(status_code=503, detail="Application not fully initialized")
    response = await call_next(request)
    return response


@app.get("/debug/stats")
async def get_debug_stats(request: Request):
    if not all([request.state.trade_processor, request.state.orderbook_analyzer, request.state.debug_monitor]):
        raise HTTPException(status_code=503, detail="System not fully initialized")

    trade_stats = request.state.trade_processor.get_trade_stats()
    analysis_stats = request.state.orderbook_analyzer.get_analysis_stats()

    return {
        "trade_processor": trade_stats,
        "orderbook_analyzer": analysis_stats,
        "uptime_hours": (
                            asyncio.get_event_loop().time() - request.state.debug_monitor.start_time if request.state.debug_monitor else 0) / 3600
    }


@app.get("/predict_price/{symbol}")
async def predict_price(symbol: str, request: Request):
    ml_predictor = request.state.ml_price_predictor
    if not ml_predictor:
        raise HTTPException(status_code=503, detail="ML Price Predictor not initialized.")

    prediction = await ml_predictor.get_prediction_for_current_state(symbol)
    if "error" in prediction:
        raise HTTPException(status_code=500, detail=prediction["error"])
    return prediction
