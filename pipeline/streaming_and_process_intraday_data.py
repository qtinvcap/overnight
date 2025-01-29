import websocket
import json
import pandas as pd
from datetime import datetime
import pytz
import threading
import time
import logging
import traceback
import requests
from pathlib import Path
import sys
from overnight.features.intraday.features_engineering import assemble_all_intraday_features
from overnight.features.utils import get_ticker_full_tickers_list


def test_api_key(api_key, logger):
    """
    Checks API key validity by querying Polygon for a single-day AAPL daily agg.
    """
    url = "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        logger.info("API key validation successful.")
        return True
    else:
        logger.error(f"API key validation failed: {response.status_code} - {response.text}")
        return False


class PolygonStreamProcessor:
    """
    Continuously connects to Polygon WebSocket to collect 1-minute bars all day.
    At 15:56, it calculates intraday features and calls 'intraday_features_callback' with the resulting DataFrame.
    """

    def __init__(
        self,
        api_key,
        logger=None,
        intraday_features_callback=None,  # <--- callback for final intraday features
    ):
        # Move logging setup to the start
        if logger is None:
            self.logger = logging.getLogger("streaming")
            self.logger.setLevel(logging.INFO)

            # Create separate log file for streaming
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            stream_log_file = log_dir / f"streaming_{datetime.now().strftime('%Y%m%d')}.log"

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            # Add handlers with the formatter
            file_handler = logging.FileHandler(stream_log_file)
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)

            # Remove any existing handlers to avoid duplicates
            self.logger.handlers = []

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        else:
            self.logger = logger

        self.api_key = api_key
        self.ws = None
        self.authenticated = False
        self.last_message_time = time.time()

        # Reconnect / watch-dog parameters
        self.reconnect_timeout = 70
        self.max_reconnect_attempts = 3
        self.reconnect_delay = 300
        self.connection_attempts = 0

        # Market time logic
        self.et_tz = pytz.timezone("US/Eastern")
        self.market_start_hour = 4
        self.market_end_hour = 20
        self.preconnect_minutes = 5

        # Data buffer
        self.data_buffer = []
        self.last_buffer_size = 0
        self.features_calculated = False

        # Callback for final intraday features
        self.intraday_features_callback = intraday_features_callback

        # Initialize tickers list once during startup
        try:
            self.logger.info("Fetching ticker list...")
            self.tickers_list = get_ticker_full_tickers_list()
            self.logger.info(f"Ticker list fetched: {len(self.tickers_list)} tickers")
        except Exception as e:
            self.logger.error(f"Error fetching ticker list: {str(e)}")
            self.tickers_list = None
            raise  # Fail fast if we can't get the ticker list

        # Validate API key before starting
        if not test_api_key(api_key, self.logger):
            raise ValueError("Invalid API key")

    def _wait_until_next_session(self):
        """
        If the current time is after the market_end_hour (8pm),
        sleep until 5 minutes before next day's pre-market (3:55am).
        """
        now = datetime.now(self.et_tz)
        if now.hour >= self.market_end_hour:
            next_session = now.replace(hour=self.market_start_hour, minute=0, second=0, microsecond=0)
            next_session += pd.Timedelta(days=1)
        else:
            next_session = now.replace(hour=self.market_start_hour, minute=0, second=0, microsecond=0)

        connect_time = next_session - pd.Timedelta(minutes=self.preconnect_minutes)
        wait_seconds = (connect_time - now).total_seconds()

        if wait_seconds > 0:
            self.logger.info(f"Waiting {wait_seconds/60:.1f} minutes until next session...")
            time.sleep(wait_seconds)

    def _monitor_connection(self):
        """
        Background thread that periodically checks the time since last received message.
        If it exceeds reconnect_timeout, it closes the WebSocket (forcing a reconnect).
        """
        while True:
            if self.ws and (time.time() - self.last_message_time) > self.reconnect_timeout:
                self.logger.warning(f"No messages received for {self.reconnect_timeout} seconds. Reconnecting...")
                self.ws.close()
            time.sleep(10)

    def _process_market_data(self, msg):
        """
        Convert the raw JSON message from Polygon into a dictionary (row) suitable for DataFrame appending.
        """
        try:
            start_time = pd.to_datetime(msg["s"], unit="ms", utc=True)
            end_time = pd.to_datetime(msg["e"], unit="ms", utc=True)
            start_time = start_time.tz_convert("US/Eastern")
            end_time = end_time.tz_convert("US/Eastern")

            hour = start_time.hour
            minute = start_time.minute
            if hour < 9 or (hour == 9 and minute < 30):
                market_status = "pre-market"
            elif hour == 16:
                market_status = "post-market"
            elif hour == 9 and minute == 30:
                market_status = "open"
            else:
                market_status = "intra-day"

            row = {
                "ticker": msg.get("sym"),
                "volume": msg.get("v"),
                "open": msg.get("o"),
                "close": msg.get("c"),
                "high": msg.get("h"),
                "low": msg.get("l"),
                "window_start": start_time,
                "vwap": msg.get("vw"),
                "avg_trade_size": msg.get("z"),
                "accumulated_volume": msg.get("av"),
                "market_status": market_status,
                "trade_date": start_time.date(),
            }

            # Estimate the number of trades
            if row["avg_trade_size"] and row["avg_trade_size"] > 0:
                row["transactions"] = int(row["volume"] / row["avg_trade_size"])
            else:
                row["transactions"] = 0

            return row

        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            return None

    def _process_features(self, now):
        """
        Process features in a separate thread to avoid blocking WebSocket messages.
        """
        try:
            self.logger.info("Starting end-of-day intraday feature calculation...")
            df = pd.DataFrame(self.data_buffer)

            df = df[df["ticker"].isin(self.tickers_list)]
            self.logger.info(f"Filtered to {len(df['ticker'].unique())} tickers from universe")

            # Minimal columns and transformations
            df = df[
                [
                    "ticker",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "market_status",
                    "trade_date",
                    "transactions",
                    "window_start",
                ]
            ]

            # Compute intraday features in memory
            features_df = assemble_all_intraday_features(df)

            # Save features_df for debugging
            debug_dir = Path("debug_data") / now.strftime("%Y%m%d")
            debug_dir.mkdir(parents=True, exist_ok=True)

            features_df.to_parquet(debug_dir / "intraday_features.parquet", engine="pyarrow", compression="snappy")
            self.logger.info(f"Saved intraday features to {debug_dir}/intraday_features.parquet")

            # Mark them with the final trade date
            features_df["trade_date"] = now.date()

            # Mark as done to prevent re-calculation
            self.features_calculated = True
            self.logger.info("Intraday features calculation done.")

            # Call the callback to hand over today's intraday features
            if self.intraday_features_callback:
                self.logger.info("Sending intraday features to pipeline callback...")
                self.intraday_features_callback(features_df)
            else:
                self.logger.warning("No intraday_features_callback provided. Can't hand off intraday data.")

        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _on_message(self, ws, message):
        """
        WebSocket callback triggered on any incoming message from Polygon.
        """
        try:
            self.last_message_time = time.time()
            data = json.loads(message)

            # Polygon sends messages in list form. We handle them in a loop:
            if isinstance(data, list):
                for msg in data:
                    if msg.get("ev") == "status":
                        status = msg.get("status")
                        if status == "connected":
                            self.logger.info("Connected Successfully.")
                            auth_data = {"action": "auth", "params": self.api_key}
                            ws.send(json.dumps(auth_data))
                        elif status == "auth_success":
                            self.logger.info("Authentication successful.")
                            self.authenticated = True
                            # Subscribe to all minute bars
                            subscribe_msg = {"action": "subscribe", "params": "AM.*"}
                            ws.send(json.dumps(subscribe_msg))
                            self.logger.info("Subscribed to market data.")
                    elif msg.get("ev") == "AM" and self.authenticated:
                        row = self._process_market_data(msg)
                        if row:
                            self.data_buffer.append(row)

                            now = datetime.now(self.et_tz)

                            # Check if it's 15:56 and we haven't computed features yet:
                            if (
                                now.hour == 15
                                and now.minute == 56
                                and now.second == 15
                                and not self.features_calculated
                                and now.date() == row["trade_date"]
                            ):
                                feature_thread = threading.Thread(
                                    target=self._process_features, args=(now,), daemon=True
                                )
                                feature_thread.start()

                            # Periodic logging every 100 records

                            # Periodic logging every 100 records
                            current_size = len(self.data_buffer)
                            if current_size % 1000 == 0 and current_size != self.last_buffer_size:
                                log_msg = (
                                    f"Buffer size: {current_size} records "
                                    f"| Latest record: {row['ticker']} at {row['window_start']} "
                                    f"| Current time: {datetime.now(self.et_tz)}"
                                )
                                self.logger.info(log_msg)
                                self.last_buffer_size = current_size

        except Exception as e:
            self.logger.error(f"Error in message handling: {str(e)}")
            self.logger.error(traceback.format_exc())

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {str(error)}")
        if "401" in str(error):
            self.logger.error("Authentication error - please check your API key.")
            self.authenticated = False

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.warning(f"Connection closed: {close_status_code} - {close_msg}")
        if not self.authenticated:
            self.logger.error("Connection closed before successful authentication.")
            time.sleep(5)

    def _on_open(self, ws):
        self.logger.info("Websocket connected. Sending auth request...")
        auth_data = {"action": "auth", "params": self.api_key}
        ws.send(json.dumps(auth_data))

    def _check_new_day(self):
        """
        Reset the feature calculation flag and data buffer when a new trading day starts.
        """
        while True:
            now = datetime.now(self.et_tz)
            # Reset after midnight
            if now.hour == 0 and now.minute == 1:
                self.features_calculated = False
                self.data_buffer = []
                self.logger.info("New trading day started - resetting intraday state.")
            time.sleep(60)

    def connect(self):
        """
        Creates the WebSocketApp object (not yet running).
        """
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://socket.polygon.io/stocks",
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )

    def run(self):
        """
        Main loop that:
         1. Waits if it's outside market hours,
         2. Connects to Polygon,
         3. On close/error, tries to reconnect until end of market day,
         4. Repeats the next day.
        Also runs helper threads to monitor connection and reset daily flags.
        """
        # Start helper threads
        for thread_target in [self._monitor_connection, self._check_new_day]:
            thread = threading.Thread(target=thread_target, daemon=True)
            thread.start()

        while True:
            try:
                now = datetime.now(self.et_tz)

                # If outside of 4am-8pm, wait until tomorrow 3:55am
                if now.hour < self.market_start_hour or now.hour >= self.market_end_hour:
                    self._wait_until_next_session()
                    continue

                self.logger.info("Establishing new WebSocket connection to Polygon...")
                self.authenticated = False
                self.connect()
                self.ws.run_forever(ping_interval=30, ping_timeout=10)

            except Exception as e:
                self.logger.error(f"Connection error: {str(e)}")
                self.logger.error(traceback.format_exc())

                self.connection_attempts += 1
                if self.connection_attempts >= self.max_reconnect_attempts:
                    self.logger.error(
                        f"Failed to reconnect after {self.max_reconnect_attempts} attempts. "
                        f"Waiting {self.reconnect_delay} seconds..."
                    )
                    time.sleep(self.reconnect_delay)
                    self.connection_attempts = 0
                else:
                    time.sleep(5)
