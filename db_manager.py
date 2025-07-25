import os
import asyncpg
import logging
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class DBManager:
    def __init__(self):
        self.conn_pool = None
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        if not all([db_user, db_password, db_host, db_port, db_name]):
            logger.error(
                "One or more database environment variables (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME) are not set.")
            raise ValueError("Missing database connection details.")

        self.database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        self._ml_data_collector = None  # Added: Placeholder for MLDataCollector

    def set_ml_data_collector(self, ml_data_collector_instance):
        """Sets the MLDataCollector instance for cross-module communication."""
        self._ml_data_collector = ml_data_collector_instance
        logger.info("MLDataCollector instance set in DBManager.")

    async def init_db_connection(self):
        """Initializes the database connection pool."""
        if not self.conn_pool:
            try:
                self.conn_pool = await asyncpg.create_pool(self.database_url)
                logger.info("Database connection pool created successfully.")
                await self.create_tables()
            except Exception as e:
                logger.error(f"Failed to connect to database or create pool: {e}")
                raise

    async def close_db_connection(self):
        """Closes the database connection pool."""
        if self.conn_pool:
            await self.conn_pool.close()
            self.conn_pool = None
            logger.info("Database connection pool closed.")

    async def create_tables(self):
        """Creates necessary tables if they do not exist."""
        async with self.conn_pool.acquire() as conn:
            try:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS watchlist (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT UNIQUE NOT NULL,
                        price_drop FLOAT,
                        current_price FLOAT,
                        historical_price FLOAT,
                        is_active BOOLEAN DEFAULT TRUE,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS alert_manipulation (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        price FLOAT,
                        alert_timestamp_ms BIGINT NOT NULL,
                        alert_start_time TIMESTAMP,
                        alert_end_time TIMESTAMP,
                        message TEXT NOT NULL,
                        volume_ratio FLOAT,
                        current_volume_usdt FLOAT,
                        average_volume_usdt FLOAT,
                        consecutive_count INTEGER,
                        grouped_alerts_count INTEGER DEFAULT 1,
                        is_grouped BOOLEAN DEFAULT FALSE,
                        group_id TEXT,
                        has_imbalance BOOLEAN DEFAULT FALSE,
                        imbalance_data JSONB,
                        candle_data JSONB,
                        order_book_snapshot JSONB,
                        trade_history JSONB, -- Added: Store trade history for ML
                        status TEXT DEFAULT 'new',
                        is_true_signal BOOLEAN,
                        predicted_price_change FLOAT,
                        predicted_direction TEXT,
                        ml_source_alert_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS ml_training_data (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        features JSONB NOT NULL,
                        target_price_change FLOAT NOT NULL,
                        target_direction TEXT NOT NULL,
                        alert_id INTEGER REFERENCES alert_manipulation(id) ON DELETE SET NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_symbol_type_time 
                    ON alert_manipulation (symbol, alert_type, alert_timestamp_ms)
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_alerts_group_id 
                    ON alert_manipulation (group_id) WHERE group_id IS NOT NULL
                ''')
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_ml_training_data_symbol 
                    ON ml_training_data (symbol)
                ''')

                logger.info("Tables created or already exist.")
            except Exception as e:
                logger.error(f"Error creating tables: {e}")
                raise

    # --- Watchlist Methods ---
    async def get_watchlist_details(self) -> List[Dict[str, Any]]:
        """Получение полного списка watchlist с деталями."""
        async with self.conn_pool.acquire() as conn:
            return await conn.fetch("SELECT * FROM watchlist ORDER BY symbol")

    async def get_watchlist_symbols(self) -> List[str]:
        """Получение только активных символов из watchlist."""
        async with self.conn_pool.acquire() as conn:
            symbols = await conn.fetch("SELECT symbol FROM watchlist WHERE is_active = TRUE")
            return [s['symbol'] for s in symbols]

    async def add_to_watchlist(self, symbol: str, price_drop: float, current_price: float, historical_price: float):
        """Добавление символа в watchlist."""
        async with self.conn_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO watchlist (symbol, price_drop, current_price, historical_price, is_active)
                VALUES ($1, $2, $3, $4, TRUE)
                ON CONFLICT (symbol) DO UPDATE SET
                    price_drop = EXCLUDED.price_drop,
                    current_price = EXCLUDED.current_price,
                    historical_price = EXCLUDED.historical_price,
                    is_active = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                """,
                symbol, price_drop, current_price, historical_price
            )

    async def update_watchlist_item(self, item_id: Optional[int] = None, symbol: Optional[str] = None,
                                    is_active: Optional[bool] = None):
        """Обновление элемента watchlist по ID или символу."""
        async with self.conn_pool.acquire() as conn:
            query = "UPDATE watchlist SET updated_at = CURRENT_TIMESTAMP"
            params = []
            param_idx = 1

            if is_active is not None:
                query += f", is_active = ${param_idx}"
                params.append(is_active)
                param_idx += 1

            if symbol:
                query += f", symbol = ${param_idx}"
                params.append(symbol)
                param_idx += 1

            if item_id is not None:
                query += f" WHERE id = ${param_idx}"
                params.append(item_id)
            elif symbol:
                query += f" WHERE symbol = ${param_idx}"
                params.append(symbol)
            else:
                raise ValueError("Either item_id or symbol must be provided for update.")

            result = await conn.execute(query, *params)
            return result == "UPDATE 1"

    async def remove_from_watchlist(self, item_id: Optional[int] = None, symbol: Optional[str] = None):
        """Удаление элемента из watchlist по ID или символу."""
        async with self.conn_pool.acquire() as conn:
            if item_id is not None:
                result = await conn.execute("DELETE FROM watchlist WHERE id = $1", item_id)
            elif symbol:
                result = await conn.execute("DELETE FROM watchlist WHERE symbol = $1", symbol)
            else:
                raise ValueError("Either item_id or symbol must be provided for removal.")
            return result == "DELETE 1"

    # --- Alerts Methods ---
    async def insert_alert(self, symbol: str, alert_type: str, price: float, message: str,
                           details: dict = None, alert_start_time: Optional[str] = None,
                           alert_end_time: Optional[str] = None,
                           order_book_snapshot: Optional[Dict] = None,  # Added for ML
                           trade_history: Optional[List[Dict]] = None  # Added for ML
                           ) -> Optional[int]:  # Changed return type to Optional[int]
        """Inserts a detected manipulation event as an alert and triggers ML processing."""
        async with self.conn_pool.acquire() as conn:
            alert_timestamp_ms = int(time.time() * 1000)
            try:
                if alert_start_time:
                    try:
                        alert_start_time = datetime.strptime(alert_start_time, '%Y-%m-%d %H:%M:%S')
                    except ValueError as ve:
                        logger.error(f"Invalid date format for alert_start_time: {alert_start_time}. Error: {ve}")
                        return None

                if alert_end_time:
                    try:
                        alert_end_time = datetime.strptime(alert_end_time, '%Y-%m-%d %H:%M:%S')
                    except ValueError as ve:
                        logger.error(f"Invalid date format for alert_end_time: {alert_end_time}. Error: {ve}")
                        return None

                group_info = await self._check_alert_grouping(conn, symbol, alert_type, alert_timestamp_ms)

                if group_info['should_group']:
                    # If grouped, we don't create a new alert, so no new ID for ML processing
                    await self._update_grouped_alert(conn, group_info['existing_alert_id'],
                                                     alert_timestamp_ms, message, details,
                                                     alert_end_time, None, None,
                                                     None)  # ML predictions are handled by MLDataCollector
                    logger.info(f"Alert grouped for {symbol} ({alert_type})")
                    return None

                group_id = f"{symbol}_{alert_type}_{alert_timestamp_ms}" if group_info['create_group'] else None

                alert_id = await conn.fetchval(  # Changed to fetchval to get ID
                    """
                    INSERT INTO alert_manipulation (
                        symbol, alert_type, price, alert_timestamp_ms, message,
                        alert_start_time, alert_end_time, group_id,
                        volume_ratio, current_volume_usdt, average_volume_usdt,
                        consecutive_count, has_imbalance, imbalance_data,
                        candle_data, order_book_snapshot, trade_history
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    RETURNING id
                    """,
                    symbol,
                    alert_type,
                    price,
                    alert_timestamp_ms,
                    message,
                    alert_start_time,
                    alert_end_time,
                    group_id,
                    details.get('volume_ratio'),
                    details.get('current_volume_usdt'),
                    details.get('average_volume_usdt'),
                    details.get('consecutive_count'),
                    details.get('has_imbalance', False),
                    json.dumps(details.get('imbalance_data')) if details and details.get('imbalance_data') else None,
                    json.dumps(details.get('candle_data')) if details and details.get('candle_data') else None,
                    json.dumps(order_book_snapshot) if order_book_snapshot else None,  # Store OB snapshot
                    json.dumps(trade_history) if trade_history else None  # Store trade history
                )
                logger.info(f"Alert inserted: {alert_type} for {symbol} at {price}, ID: {alert_id}")

                # Trigger ML processing for the new alert
                if self._ml_data_collector and alert_id:
                    alert_data_for_ml = {
                        'id': alert_id,
                        'symbol': symbol,
                        'alert_type': alert_type,
                        'price': price,
                        'alert_timestamp_ms': alert_timestamp_ms,
                        'message': message,
                        'details': details,
                        'alert_start_time': alert_start_time.isoformat() if alert_start_time else None,
                        'alert_end_time': alert_end_time.isoformat() if alert_end_time else None,
                        'order_book_snapshot': order_book_snapshot,
                        'trade_history': trade_history
                    }
                    await self._ml_data_collector.add_alert_for_ml_processing(alert_data_for_ml)
                    logger.debug(f"Alert {alert_id} sent to MLDataCollector for processing.")

                return alert_id

            except Exception as e:
                logger.error(f"Error inserting alert for {symbol} ({alert_type}): {e}")
                return None

    async def _check_alert_grouping(self, conn, symbol: str, alert_type: str, current_timestamp_ms: int) -> Dict[
        str, Any]:
        """Проверяет, нужно ли группировать алерт с существующими"""
        grouping_minutes = int(os.getenv("ALERT_GROUPING_MINUTES", 5))
        grouping_window_ms = grouping_minutes * 60 * 1000

        recent_alerts = await conn.fetch(
            """
            SELECT id, alert_timestamp_ms, group_id, grouped_alerts_count
            FROM alert_manipulation 
            WHERE symbol = $1 AND alert_type = $2 
            AND alert_timestamp_ms >= $3
            ORDER BY alert_timestamp_ms DESC
            LIMIT 1
            """,
            symbol, alert_type, current_timestamp_ms - grouping_window_ms
        )

        if recent_alerts:
            recent_alert = recent_alerts[0]
            time_diff_ms = current_timestamp_ms - recent_alert['alert_timestamp_ms']

            if time_diff_ms <= grouping_window_ms:
                return {
                    'should_group': True,
                    'existing_alert_id': recent_alert['id'],
                    'create_group': False
                }

        return {
            'should_group': False,
            'existing_alert_id': None,
            'create_group': True
        }

    async def _update_grouped_alert(self, conn, alert_id: int, new_timestamp_ms: int,
                                    new_message: str, new_details: dict, alert_end_time: Optional[str],
                                    predicted_price_change: Optional[float],
                                    # Kept for compatibility, but ML handled by data_collector
                                    predicted_direction: Optional[str],  # Kept for compatibility
                                    ml_source_alert_type: Optional[str]):  # Kept for compatibility
        """Обновляет сгруппированный алерт"""
        current_alert = await conn.fetchrow("SELECT * FROM alert_manipulation WHERE id = $1", alert_id)

        if not current_alert:
            return

        if alert_end_time:
            try:
                alert_end_time = datetime.strptime(alert_end_time, '%Y-%m-%d %H:%M:%S')
            except ValueError as ve:
                logger.error(f"Invalid date format for alert_end_time in update: {alert_end_time}. Error: {ve}")
                return
        else:
            alert_end_time = datetime.fromtimestamp(new_timestamp_ms / 1000)

        new_count = (current_alert['grouped_alerts_count'] or 1) + 1
        combined_message = f"{current_alert['message']} | {new_message} (всего: {new_count})"

        await conn.execute(
            """
            UPDATE alert_manipulation SET 
                grouped_alerts_count = $1,
                is_grouped = TRUE,
                alert_end_time = $2,
                message = $3,
                alert_timestamp_ms = $4
            WHERE id = $5
            """,
            new_count, alert_end_time, combined_message, new_timestamp_ms,
            alert_id
        )

    async def get_alerts(self, limit: int, offset: int, symbol: Optional[str], alert_type: Optional[str],
                         status: Optional[str]) -> List[Dict[str, Any]]:
        """Получение списка алертов с фильтрацией и пагинацией."""
        async with self.conn_pool.acquire() as conn:
            query = "SELECT * FROM alert_manipulation"
            params = []
            conditions = []
            param_idx = 1

            if symbol:
                conditions.append(f"symbol = ${param_idx}")
                params.append(symbol)
                param_idx += 1
            if alert_type:
                conditions.append(f"alert_type = ${param_idx}")
                params.append(alert_type)
                param_idx += 1
            if status:
                conditions.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([limit, offset])

            return await conn.fetch(query, *params)

    async def get_alert_by_id(self, alert_id: int) -> Optional[Dict[str, Any]]:
        """Получение алерта по ID."""
        async with self.conn_pool.acquire() as conn:
            return await conn.fetchrow("SELECT * FROM alert_manipulation WHERE id = $1", alert_id)

    async def update_alert_status(self, alert_id: int, status: Optional[str], is_true_signal: Optional[bool]) -> bool:
        """Обновление статуса и флага is_true_signal алерта."""
        async with self.conn_pool.acquire() as conn:
            query = "UPDATE alert_manipulation SET created_at = created_at"
            params = []
            param_idx = 1

            if status is not None:
                query += f", status = ${param_idx}"
                params.append(status)
                param_idx += 1
            if is_true_signal is not None:
                query += f", is_true_signal = ${param_idx}"
                params.append(is_true_signal)
                param_idx += 1

            if len(params) == 0:
                return False

            query += f" WHERE id = ${param_idx}"
            params.append(alert_id)

            result = await conn.execute(query, *params)
            return result == "UPDATE 1"

    async def update_alert_with_ml_prediction(self, alert_id: int, predicted_price_change: float,
                                              predicted_direction: str, ml_source_alert_type: str) -> bool:
        """Updates an existing alert with ML prediction results."""
        async with self.conn_pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE alert_manipulation SET
                    predicted_price_change = $1,
                    predicted_direction = $2,
                    ml_source_alert_type = $3,
                    status = 'ml_processed' -- Optionally update status
                WHERE id = $4
                """,
                predicted_price_change, predicted_direction, ml_source_alert_type, alert_id
            )
            return result == "UPDATE 1"

    async def delete_alert(self, alert_id: int) -> bool:
        """Удаление алерта."""
        async with self.conn_pool.acquire() as conn:
            result = await conn.execute("DELETE FROM alert_manipulation WHERE id = $1", alert_id)
            return result == "DELETE 1"

    async def get_alerts_statistics(self, days: int) -> Dict[str, Any]:
        """Получение статистики алертов за последние N дней."""
        async with self.conn_pool.acquire() as conn:
            query = """
            SELECT
                COUNT(*) AS total_alerts,
                COUNT(*) FILTER (WHERE alert_type IN ('Wash/Cross Trading', 'Iceberg (Bid Side)', 'Iceberg (Ask Side)')) AS manipulation_alerts,
                COUNT(*) FILTER (WHERE is_true_signal = TRUE) AS true_signals,
                COUNT(*) FILTER (WHERE is_true_signal = FALSE) AS false_signals,
                COUNT(*) FILTER (WHERE has_imbalance = TRUE) AS alerts_with_imbalance,
                AVG(volume_ratio) AS avg_volume_ratio
            FROM alert_manipulation
            WHERE created_at >= NOW() - INTERVAL '$1 days'
            """
            stats = await conn.fetchrow(query, days)

            accuracy_percentage = 0.0
            if stats['true_signals'] is not None and stats['false_signals'] is not None:
                total_classified = stats['true_signals'] + stats['false_signals']
                if total_classified > 0:
                    accuracy_percentage = (stats['true_signals'] / total_classified) * 100

            return {
                "total_alerts": stats['total_alerts'] or 0,
                "volume_alerts": 0,
                "consecutive_alerts": 0,
                "priority_alerts": 0,
                "true_signals": stats['true_signals'] or 0,
                "false_signals": stats['false_signals'] or 0,
                "alerts_with_imbalance": stats['alerts_with_imbalance'] or 0,
                "accuracy_percentage": round(accuracy_percentage, 2),
                "avg_volume_ratio": stats['avg_volume_ratio']
            }

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Generic query execution for total count in alerts API."""
        async with self.conn_pool.acquire() as conn:
            return await conn.fetch(query, *args)

    # --- ML Training Data Methods ---
    async def insert_ml_training_data(self, symbol: str, features: Dict[str, Any],
                                      target_price_change: float, target_direction: str,
                                      alert_id: Optional[int] = None) -> int:
        """Inserts data for ML model training."""
        async with self.conn_pool.acquire() as conn:
            record_id = await conn.fetchval(
                """
                INSERT INTO ml_training_data (symbol, features, target_price_change, target_direction, alert_id)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                symbol, json.dumps(features), target_price_change, target_direction, alert_id
            )
            logger.info(f"ML training data inserted for {symbol}, record ID: {record_id}")
            return record_id

    async def get_ml_training_data(self, symbol: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        """Retrieves ML training data."""
        async with self.conn_pool.acquire() as conn:
            query = "SELECT * FROM ml_training_data"
            params = []
            if symbol:
                query += " WHERE symbol = $1"
                params.append(symbol)
            query += f" ORDER BY created_at DESC LIMIT {limit}"
            return await conn.fetch(query, *params)

    async def get_latest_ml_training_data_timestamp(self) -> Optional[datetime]:
        """Gets the timestamp of the latest ML training data entry."""
        async with self.conn_pool.acquire() as conn:
            latest_timestamp = await conn.fetchval(
                "SELECT MAX(created_at) FROM ml_training_data"
            )
            return latest_timestamp

    async def clean_old_ml_training_data(self, retention_days: int):
        """Deletes ML training data older than retention_days."""
        async with self.conn_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM ml_training_data WHERE created_at < NOW() - INTERVAL '$1 days'",
                retention_days
            )
            logger.info(f"Cleaned ML training data older than {retention_days} days.")
