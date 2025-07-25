import asyncio
import json
import websockets
from collections import defaultdict, deque
import bisect
import logging
import os
from typing import Dict, Any, Optional, List
from websockets.legacy.client import WebSocketClientProtocol
import random

# Import config_manager for standalone run
import config_manager

from cryptoscan.backand.bybit.db_manager import DBManager
from cryptoscan.backand.bybit.manipulation.trade_processor import TradeProcessor
from cryptoscan.backand.bybit.manipulation.orderbook_analyzer import OrderBookAnalyzer
from cryptoscan.backand.bybit.ML.data_collector import MLDataCollector
from cryptoscan.backand.bybit.ML.feature_engineering import FeatureEngineer
from cryptoscan.backand.bybit.ML.ml_model import MLModel

# Logging configuration will be set by CryptoScanApp or config_manager for standalone run
logger = logging.getLogger(__name__)

# Global state for order books by symbol
orderbooks_by_symbol = defaultdict(lambda: {'bids': {}, 'asks': {}})

bybit_ws_list: List[WebSocketClientProtocol] = []
subscribed_symbols = set()

client_subscriptions = {}
connected_clients = set()

_db_manager: Optional[DBManager] = None
_trade_processor: Optional[TradeProcessor] = None
_orderbook_analyzer: Optional[OrderBookAnalyzer] = None
_ml_data_collector: Optional[MLDataCollector] = None


def set_managers(db_m: DBManager, trade_p: TradeProcessor, ob_a: OrderBookAnalyzer, ml_dc: MLDataCollector):
    global _db_manager, _trade_processor, _orderbook_analyzer, _ml_data_collector
    _db_manager = db_m
    _trade_processor = trade_p
    _orderbook_analyzer = ob_a
    _ml_data_collector = ml_dc


# Function to update order book (for delta) using dictionaries
def update_orderbook(symbol: str, side: str, data: List[List[str]]):
    current_orderbook_side = orderbooks_by_symbol[symbol][side]

    for price_str, size_str in data:
        price = float(price_str)
        size = float(size_str)
        if size == 0:
            if price in current_orderbook_side:
                del current_orderbook_side[price]
        else:
            current_orderbook_side[price] = size


async def handle_message(symbol: str, message: Dict[str, Any]):
    """Обработка сообщений от Bybit с улучшенной обработкой ошибок"""
    try:
        data = message.get('data', {})
        topic = message.get('topic', '')

        if "orderbook.500" in topic:
            await _handle_orderbook_message(symbol, message, data)
        elif "publicTrade" in topic:
            await _handle_trade_message(symbol, data)
        else:
            logger.debug(f"Skipped message for topic: {topic}")
    except Exception as e:
        logger.error(f"Error handling message for {symbol}: {e}")


async def _handle_orderbook_message(symbol: str, message: Dict[str, Any], data: Dict[str, Any]):
    """Обработка сообщений ордербука"""
    try:
        if message.get('type') == 'snapshot':
            orderbooks_by_symbol[symbol]['bids'] = {float(p): float(s) for p, s in data.get('b', [])}
            orderbooks_by_symbol[symbol]['asks'] = {float(p): float(s) for p, s in data.get('a', [])}
        elif message.get('type') == 'delta':
            update_orderbook(symbol, 'bids', data.get('b', []))
            update_orderbook(symbol, 'asks', data.get('a', []))

        if _orderbook_analyzer:
            # Конвертируем словари в отсортированные списки для OrderBookAnalyzer
            sorted_bids = sorted([[p, s] for p, s in orderbooks_by_symbol[symbol]['bids'].items()], key=lambda x: x[0],
                                 reverse=True)
            sorted_asks = sorted([[p, s] for p, s in orderbooks_by_symbol[symbol]['asks'].items()], key=lambda x: x[0])

            # Теперь update_orderbook и analyze_orderbook являются async методами
            await _orderbook_analyzer.update_orderbook(
                symbol,
                sorted_bids,
                sorted_asks
            )
            await _orderbook_analyzer.analyze_orderbook(symbol)

        await broadcast_orderbook(symbol)
    except Exception as e:
        logger.error(f"Error handling orderbook message for {symbol}: {e}")


async def _handle_trade_message(symbol: str, data: List[Dict[str, Any]]):
    """Обработка сообщений о трейдах"""
    try:
        if _trade_processor and _orderbook_analyzer:
            for trade in data:
                if not _validate_trade_data(trade):
                    logger.warning(f"Invalid trade data for {symbol}: {trade}")
                    continue

                await _trade_processor.add_trade(symbol, trade)
                _orderbook_analyzer.add_trade_for_iceberg(symbol, trade)
    except Exception as e:
        logger.error(f"Error handling trade message for {symbol}: {e}")


def _validate_trade_data(trade: Dict[str, Any]) -> bool:
    """Валидация данных трейда"""
    required_fields = ['T', 'p', 'v', 'S']

    for field in required_fields:
        if field not in trade:
            return False

    try:
        float(trade['p'])
        float(trade['v'])
        int(trade['T'])
        return True
    except (ValueError, TypeError):
        return False


async def broadcast_orderbook(symbol: str):
    # Конвертируем словари в отсортированные списки для отправки клиентам
    sorted_bids = sorted([[p, s] for p, s in orderbooks_by_symbol[symbol]['bids'].items()], key=lambda x: x[0],
                         reverse=True)
    sorted_asks = sorted([[p, s] for p, s in orderbooks_by_symbol[symbol]['asks'].items()], key=lambda x: x[0])

    message = json.dumps({'bids': sorted_bids, 'asks': sorted_asks})
    tasks = []
    for client_ws in connected_clients:
        if client_subscriptions.get(client_ws) == symbol:
            tasks.append(client_ws.send(message))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def connect_to_bybit_chunk(symbols: List[str], testnet: bool = False):
    """Подключение к Bybit с улучшенной обработкой ошибок и переподключением"""
    url = "wss://stream-testnet.bybit.com/v5/public/linear" if testnet else "wss://stream.bybit.com/v5/public/linear"
    depth = 500
    reconnect_delay = 1
    max_delay = 60
    jitter_factor = 0.5
    max_reconnect_attempts = 10
    reconnect_attempts = 0

    # Use os.getenv, values are set by config_manager
    ws_ping_interval = int(os.getenv("WS_PING_INTERVAL", 20))
    ws_ping_timeout = int(os.getenv("WS_PING_TIMEOUT", 10))
    ws_close_timeout = int(os.getenv("WS_CLOSE_TIMEOUT", 10))
    ws_max_size = int(os.getenv("WS_MAX_SIZE", 10000000))

    while True:
        try:
            logger.info(f"Connecting to Bybit WS for batch {symbols[:3]}... (attempt {reconnect_attempts + 1})")
            async with websockets.connect(
                    url,
                    ping_interval=ws_ping_interval,
                    ping_timeout=ws_ping_timeout,
                    open_timeout=30,
                    close_timeout=ws_close_timeout,
                    max_size=ws_max_size
            ) as ws:
                bybit_ws_list.append(ws)
                logger.info(f"Connected to Bybit WS for batch of {len(symbols)} symbols")

                reconnect_attempts = 0
                reconnect_delay = 1

                args = [f"orderbook.{depth}.{s}" for s in symbols] + [f"publicTrade.{s}" for s in symbols]
                subscribe_msg = json.dumps({
                    "op": "subscribe",
                    "args": args
                })
                await ws.send(subscribe_msg)
                logger.info(f"Subscription sent for batch symbols: {symbols}")

                subscribed_symbols.update(symbols)

                async for msg in ws:
                    try:
                        data = json.loads(msg)

                        if "ping" in data:
                            pong_response = json.dumps({"pong": data["ping"]})
                            await ws.send(pong_response)
                            logger.debug(f"Received ping {data['ping']}, sent pong: {pong_response}")
                            continue

                        if "success" in data:
                            if data["success"]:
                                logger.debug(f"Subscription success: {data.get('ret_msg', '')}")
                            else:
                                logger.warning(f"Subscription failed: {data.get('ret_msg', '')}")
                            continue

                        topic = data.get('topic', '')
                        if '.' in topic:
                            parts = topic.split('.')
                            symbol = parts[-1] if len(parts) > 2 else None
                            if symbol:
                                await handle_message(symbol, data)
                    except json.JSONDecodeError:
                        logger.error(f"JSON decoding error: {msg}")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")

                logger.info("Connection closed for batch")
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}. Retrying...")
            reconnect_attempts += 1
        except Exception as e:
            logger.error(f"Connection error for batch {symbols[:5]}...: {e}. Reconnecting in {reconnect_delay} sec...")
            reconnect_attempts += 1

        if reconnect_attempts >= max_reconnect_attempts:
            logger.critical(f"Max reconnection attempts ({max_reconnect_attempts}) reached for batch {symbols[:3]}...")
            break

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2 + random.uniform(0, reconnect_delay * jitter_factor), max_delay)


async def connect_to_bybit_all(testnet: bool = False):
    symbols = await _db_manager.get_watchlist_symbols() if _db_manager else ['BTCUSDT', 'ETHUSDT']
    if not symbols:
        symbols = ['BTCUSDT']
        logger.warning("Watchlist empty; using default symbol BTCUSDT.")

    batch_size = 30
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

    tasks = []
    for batch in batches:
        task = asyncio.create_task(connect_to_bybit_chunk(batch, testnet))
        tasks.append(task)

    await asyncio.gather(*tasks)


async def ws_handler(websocket):
    connected_clients.add(websocket)
    logger.info(f"New local client connected. Total: {len(connected_clients)}")

    try:
        async for message in websocket:
            logger.debug(f"Local client message: {message}")
            try:
                data = json.loads(message)
                if data.get('op') == 'subscribe_symbol':
                    new_symbol = data.get('symbol')
                    if new_symbol:
                        logger.info(f"Client {websocket.remote_address} requested symbol: {new_symbol}")
                        client_subscriptions[websocket] = new_symbol

                        if new_symbol not in subscribed_symbols:
                            if bybit_ws_list and bybit_ws_list[0].open:
                                args = [f"orderbook.500.{new_symbol}", f"publicTrade.{new_symbol}"]
                                await bybit_ws_list[0].send(json.dumps({"op": "subscribe", "args": args}))
                                subscribed_symbols.add(new_symbol)
                                logger.info(f"Dynamic subscription added for {new_symbol}")

                        # Send current snapshot (converted from dict to list)
                        if new_symbol in orderbooks_by_symbol and (
                                orderbooks_by_symbol[new_symbol]['bids'] or orderbooks_by_symbol[new_symbol]['asks']):
                            sorted_bids = sorted([[p, s] for p, s in orderbooks_by_symbol[new_symbol]['bids'].items()],
                                                 key=lambda x: x[0], reverse=True)
                            sorted_asks = sorted([[p, s] for p, s in orderbooks_by_symbol[new_symbol]['asks'].items()],
                                                 key=lambda x: x[0])
                            await websocket.send(json.dumps({'bids': sorted_bids, 'asks': sorted_asks}))
                            logger.info(f"Sent initial snapshot for {new_symbol} to client {websocket.remote_address}")
                        else:
                            logger.info(f"Order book for {new_symbol} is empty, snapshot not sent")
                    else:
                        logger.warning(f"Client {websocket.remote_address} sent empty symbol.")
                else:
                    logger.warning(f"Unknown message from client {websocket.remote_address}: {data}")
            except json.JSONDecodeError:
                logger.error(f"JSON decoding error from client {websocket.remote_address}: {message}")
            except Exception as e:
                logger.error(f"Error processing message from client {websocket.remote_address}: {e}")
    finally:
        connected_clients.remove(websocket)
        if websocket in client_subscriptions:
            subscribed_symbol = client_subscriptions.pop(websocket)
            logger.info(f"Local client disconnected. Subscription to {subscribed_symbol} removed.")

            if subscribed_symbol not in client_subscriptions.values() and bybit_ws_list:
                for ws in bybit_ws_list:
                    if ws.open:
                        args = [f"orderbook.500.{subscribed_symbol}", f"publicTrade.{subscribed_symbol}"]
                        await ws.send(json.dumps({"op": "unsubscribe", "args": args}))
                        subscribed_symbols.discard(subscribed_symbol)
                        logger.info(f"Unsubscribed from {subscribed_symbol}")
                        break


async def start_bybit_data_stream(direct_run_symbols: Optional[List[str]] = None):
    logger.info("Starting Bybit data stream background task.")
    try:
        bybit_task = asyncio.create_task(connect_to_bybit_all())

        # REMOVED ML data collector and training tasks from here
        # These tasks are now managed by CryptoScanApp in app.py

        local_ws_port = int(os.getenv("LOCAL_WS_PORT", 8766))
        logger.info(f"Starting local WS server on ws://localhost:{local_ws_port}")
        async with websockets.serve(ws_handler, "localhost", local_ws_port):
            await asyncio.Future()  # Keep the server running indefinitely
    except asyncio.CancelledError:
        logger.info("Bybit data stream cancelled.")
    except Exception as e:
        logger.critical(f"Critical error in Bybit data stream: {e}")
        raise


if __name__ == "__main__":
    # This block is for direct execution of bybit_websocket_client.py
    # It should also use config_manager to load settings.
    # For a full application, app.py is the main entry point.
    # This block is mostly for testing bybit_websocket_client in isolation.

    # Ensure .env is handled for standalone run
    config_manager.ensure_env_file()
    config = config_manager.load_config()

    # Setup basic logging for standalone run
    logging.basicConfig(
        level=config.get("LOG_LEVEL", "INFO").upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.get("LOG_FILE", "bybit_websocket.log"), encoding='utf-8')
        ]
    )
    logger.setLevel(config.get("LOG_LEVEL", "INFO").upper())
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)

    db_manager_instance = DBManager()

    trade_processor_instance = TradeProcessor(
        db_manager_instance,
        volume_window_minutes=int(config.get("VOLUME_WINDOW_MINUTES", 5)),
        wash_trade_threshold_ratio=float(config.get("WASH_TRADE_THRESHOLD_RATIO", 0.75)),
        ping_pong_window_sec=int(config.get("PING_PONG_WINDOW_SEC", 45)),
        ramping_window_sec=int(config.get("RAMPING_WINDOW_SEC", 90)),
        consecutive_long_count=int(config.get("CONSECUTIVE_LONG_COUNT", 5)),
        alert_grouping_minutes=int(config.get("ALERT_GROUPING_MINUTES", 5)),
        data_retention_hours=int(config.get("DATA_RETENTION_HOURS", 24)),
        volume_multiplier=float(config.get("VOLUME_MULTIPLIER", 2.0)),
        min_volume_usdt=float(config.get("MIN_VOLUME_USDT", 1000)),
        volume_type=config.get("VOLUME_TYPE", "long")
    )

    orderbook_analyzer_instance = OrderBookAnalyzer(
        db_manager_instance,
        enabled=config.get("ORDERBOOK_ENABLED", "False").lower() == "true",
        snapshot_on_alert=config.get("ORDERBOOK_SNAPSHOT_ON_ALERT", "False").lower() == "true",
        ob_history_depth=int(config.get("OB_HISTORY_DEPTH", 100)),
        imbalance_threshold=float(config.get("OB_IMBALANCE_THRESHOLD", 0.6)),
        iceberg_min_volume_ratio=float(config.get("OB_ICEBERG_MIN_VOLUME_RATIO", 0.1)),
        iceberg_hidden_volume_multiplier=float(config.get("OB_ICEBERG_HIDDEN_VOLUME_MULTIPLIER", 5.0)),
        iceberg_price_tolerance=float(config.get("OB_ICEBERG_PRICE_TOLERANCE", 0.0001)),
        iceberg_trade_window_sec=int(config.get("ICEBERG_WINDOW_SEC", 60)),  # Use ICEBERG_WINDOW_SEC from config
        layering_spoofing_depth=int(config.get("LAYERING_SPOOFING_DEPTH", 5)),
        layering_spoofing_threshold=float(config.get("LAYERING_SPOOFING_THRESHOLD", 0.7)),
        layering_spoofing_time_window_sec=int(config.get("LAYERING_SPOOFING_TIME_WINDOW_SEC", 30)),
        liquidity_detection_window_sec=int(config.get("LIQUIDITY_DETECTION_WINDOW_SEC", 300)),
        liquidity_change_threshold=float(config.get("LIQUIDITY_CHANGE_THRESHOLD", 0.2)),
        toxic_order_flow_window_sec=int(config.get("TOXIC_ORDER_FLOW_WINDOW_SEC", 10)),
        toxic_order_flow_threshold=float(config.get("TOXIC_ORDER_FLOW_THRESHOLD", 0.7)),
        cross_market_anomaly_enabled=config.get("CROSS_MARKET_ANOMALY_ENABLED", "False").lower() == "true",
        cross_market_anomaly_threshold=float(config.get("CROSS_MARKET_ANOMALY_THRESHOLD", 0.001)),
        spread_manipulation_enabled=config.get("SPREAD_MANIPULATION_ENABLED", "False").lower() == "true",
        spread_manipulation_threshold=float(config.get("SPREAD_MANIPULATION_THRESHOLD", 0.005)),
        spread_manipulation_time_window_sec=int(config.get("SPREAD_MANIPULATION_TIME_WINDOW_SEC", 10)),
        iceberg_volume_ratio_threshold=float(config.get("ICEBERG_VOLUME_RATIO", 1.6)),
        iceberg_min_trade_count=int(config.get("ICEBERG_MIN_COUNT", 4)),
        layering_distance_from_market_percent=float(config.get("LAYERING_DISTANCE_PERCENT", 0.003)),
        layering_min_size_change_abs=float(config.get("LAYERING_MIN_CHANGE", 150)),
        layering_ob_window_sec=int(config.get("LAYERING_WINDOW_SEC", 8)),
        spoofing_cancel_ratio=float(config.get("SPOOFING_CANCEL_RATIO", 0.65)),
        momentum_ignition_threshold=float(config.get("MOMENTUM_IGNITION_THRESHOLD", 0.015))
    )

    ml_model_instance = MLModel()
    feature_engineer_instance = FeatureEngineer()

    ml_data_collector_instance = MLDataCollector(
        db_manager=db_manager_instance,
        ml_model=ml_model_instance,
        feature_engineer=feature_engineer_instance
    )

    set_managers(db_manager_instance, trade_processor_instance, orderbook_analyzer_instance, ml_data_collector_instance)


    async def main_run():
        try:
            await db_manager_instance.init_db_connection()
            # Only start the bybit stream and local WS server here
            await start_bybit_data_stream()
        finally:
            if db_manager_instance:
                await db_manager_instance.close_db_connection()


    try:
        asyncio.run(main_run())
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception during direct run: {e}")
