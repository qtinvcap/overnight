import logging
import sys
import time
import pickle
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import threading

import pytz
import pandas as pd
from overnight.features.daily.retrieve_price_data_api import PolygonHistoricalDailyData
from overnight.features.intraday.features_engineering import add_rolling_ratio_features
from overnight.features.combine_intraday_daily_features import (
    add_indicator_normalizations,
    merge_daily_and_intraday_data,
    add_temporal_features,
)
from overnight.features.vix import generate_vix_data_and_merge
from overnight.features.utils import get_ticker_full_tickers_list
from overnight.features.daily.features_engineering import compute_advanced_daily_features
from overnight.models.main_inference import get_trading_signals, load_best_ranges, score_features_df
from overnight.features.intraday.main_intraday import process_intraday_data
import pandas_market_calendars as mcal

# Import the streaming script class
from overnight.pipeline.streaming_and_process_intraday_data import PolygonStreamProcessor
import os


class TradingPipeline:
    """
    Encapsulates daily data preparation and end-of-day intraday processing.
    """

    def __init__(self, api_key, logger=None):
        self.api_key = api_key
        self.polygon = PolygonHistoricalDailyData(self.api_key)
        self.et_tz = pytz.timezone("US/Eastern")

        self.logger = logger or logging.getLogger("pipeline")

        # Data containers
        self.daily_features = None
        self.intraday_past_5_days = None
        self.ready_for_trading = False

    def wait_until_time(self, hour, minute):
        """
        Blocks until a specified hour/minute in ET time on the *same day*.
        If that time has passed, returns immediately.
        """
        now = datetime.now(self.et_tz)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < target:
            wait_s = (target - now).total_seconds()
            self.logger.info(f"Waiting {wait_s:.0f} seconds until {hour:02d}:{minute:02d} ET...")
            time.sleep(wait_s)

    def get_trading_calendar(self):
        """Get trading calendar and 5th latest trading day"""
        today = datetime.now().strftime("%Y-%m-%d")
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date="2010-01-02", end_date=today)
        self.trading_days = mcal.date_range(schedule, frequency="1D")
        self.trading_days = [d.date() for d in self.trading_days]
        # Get the 5th last trading day
        self.past_205_trading_day = self.trading_days[-205]  # Changed variable name and index
        self.past_5_trading_day = self.trading_days[-6]  # Changed variable name and index
        self.yesterday_trading_day = self.trading_days[-2]
        self.logger.info(f"Trading calendar prepared, using data from {self.past_205_trading_day}")

    def prepare_daily_data(self, test_date=None):
        """
        Called after market open (e.g., ~9:31am).
        Prepares daily data, VIX, plus loads 5 days of historical intraday.
        """
        try:
            self.logger.info("Starting daily data preparation...")
            start_time = time.time()

            today_str = test_date if test_date else datetime.now().strftime("%Y-%m-%d")

            # For example, define past_5_trading_days by subtracting 5 calendar days
            # (In a real scenario, use an exchange calendar for actual 5 trading days)
            self.get_trading_calendar()
            past_205_date = self.past_205_trading_day.strftime("%Y-%m-%d")
            past_5_date = self.past_5_trading_day.strftime("%Y-%m-%d")
            yesterday_date = self.yesterday_trading_day.strftime("%Y-%m-%d")

            self.logger.info("Getting updated ticker list...")
            tickers_list = get_ticker_full_tickers_list()
            self.logger.info(f"Collected {len(tickers_list)} tickers to process for daily data.")

            # Load daily data and compute features
            self.logger.info("Loading daily data from Polygon...")
            daily_data = self.polygon.get_historical_data(past_205_date, today_str)

            self.logger.info("Filtering on relevant tickers...")
            daily_data = daily_data[daily_data["ticker"].isin(tickers_list)]

            print(daily_data.tail())

            self.logger.info("Computing daily features...")
            self.daily_features = compute_advanced_daily_features(daily_data)
            self.daily_features = generate_vix_data_and_merge(self.daily_features)

            # Load historical intraday for the past 5 days
            self.logger.info("Loading last 5 days intraday data...")
            self.intraday_past_5_days = process_intraday_data(
                past_5_date, yesterday_date
            )  # Using yesterday because minute data is not yet available for today

            self.intraday_past_5_days = self.intraday_past_5_days[
                self.intraday_past_5_days["ticker"].isin(tickers_list)
            ]

            self.ready_for_trading = True
            proc_time = time.time() - start_time
            self.logger.info(f"Daily data preparation done in {proc_time:.2f} seconds.")

        except Exception as e:
            self.logger.error(f"Error in daily data preparation: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.ready_for_trading = False
            raise

    def process_intraday_features(self, today_intraday_features: pd.DataFrame):
        """
        This is called at 15:56 (via the streaming script callback).
        Merges today's intraday with the past intraday + daily data,
        computes final rolling features, and triggers trade logic.
        """
        if not self.ready_for_trading:
            raise RuntimeError("Pipeline not ready for trading - daily data not prepared.")

        try:
            start_time = time.time()
            self.logger.info("Merging today's intraday features with historical + daily data...")

            # Combine historical and today's intraday
            intraday_combined = pd.concat([self.intraday_past_5_days, today_intraday_features], ignore_index=True)
            intraday_combined = add_rolling_ratio_features(intraday_combined)

            # Merge intraday + daily
            final_merged = merge_daily_and_intraday_data(self.daily_features, intraday_combined)
            data_for_stock_selection = add_indicator_normalizations(final_merged)
            data_for_stock_selection = add_temporal_features(data_for_stock_selection)

            # Save data_for_stock_selection for debugging
            today_str = datetime.now().strftime("%Y%m%d")
            debug_dir = Path("debug_data") / today_str
            debug_dir.mkdir(parents=True, exist_ok=True)

            data_for_stock_selection.to_parquet(
                debug_dir / "data_for_stock_selection.parquet", engine="pyarrow", compression="snappy"
            )
            self.logger.info(f"Saved data_for_stock_selection to {debug_dir}/data_for_stock_selection.parquet")

            # At this point, we can do inference or signals to select tickers to trade
            selected_tickers = self.select_tickers(data_for_stock_selection)

            """
            sample output:
            ticker  |  score     |  price    |  position_size  |  quantity
            FTAI    |  0.765837  |  91.88    |  9923.04       |  108
            ALUR    |  0.759116  |  8.50     |  999 6.12       |  1176  
            ARQQ    |  0.753899  |  27.36    |  9986.40       |  365
            MBOT    |  0.747657  |  2.04     |  9999.67       |  4903
            EVTL    |  0.731899  |  5.54     |  9999.70       |  1805
            """

            if not selected_tickers.empty:
                selected_tickers.to_parquet(
                    debug_dir / "selected_tickers.parquet", engine="pyarrow", compression="snappy"
                )
                self.logger.info(f"Saved selected_tickers to {debug_dir}/selected_tickers.parquet")


            # TODO:
            # Execute trades (placeholder)
            # self.execute_trades(selected_tickers)

            proc_time = time.time() - start_time
            self.logger.info(f"End-of-day processing completed in {proc_time:.2f} seconds.")

            return selected_tickers

        except Exception as e:
            self.logger.error(f"Error in end-of-day trading pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def select_tickers(self, data):
        """
        Uses the main_inference module to select tickers based on scoring and trading signals.
        Returns a DataFrame with selected tickers and their trading details.
        """
        try:
            self.logger.info("Starting ticker selection using inference model...")

            # Load the best ranges from saved file
            best_ranges = load_best_ranges(
                "/home/aime/overnigh_strat/overnight/models/rules/best_ranges.pkl"
            )  # Update path as needed

            # Run main inference to get scored data
            scored_df = score_features_df(
                df=data,
                best_ranges=best_ranges,
                metric="trimmed_mean_std_ratio_performance",
                target_total_rules=160,
                target_ratio=0.5,
                weight_by_spread=True,
            )

            # Get trading signals
            selected = get_trading_signals(
                scored_df=scored_df,
                initial_capital=50000,  # TO PUT THE PORTFOLIO VAULE FROM IB GATEWAY HERE
                threshold=0.6,
                max_positions=5,
                liquidity_threshold=1000000,
                price_threshold=2,
            )

            self.logger.info(f"Selected {len(selected)} tickers for trading.")
            if len(selected) > 0:
                self.logger.info(f"Selected tickers: {', '.join(selected['ticker'].tolist())}")
                self.logger.info(f"Total capital required: ${selected['position_size'].sum():,.2f}")

            return selected

        except Exception as e:
            self.logger.error(f"Error in ticker selection: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()  # Return empty DataFrame on error

    def execute_trades(self, tickers):
        """
        Placeholder for your actual trade execution (IB API or other).
        """
        self.logger.info(f"Executing trades for {len(tickers)} tickers (placeholder).")
        # Implementation depends on your brokerage
        pass

    def save_intraday_data(self):
        """
        Saves the intraday data to parquet files after market close.
        """
        try:
            save_dir = Path("intraday_data") / datetime.now().strftime("%Y%m%d")
            save_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Starting to save intraday data...")
            start_time = time.time()

            # Save each dataset
            for name, data in self.intraday_data.items():
                if data is not None:
                    filepath = save_dir / f"{name}.parquet"
                    data.to_parquet(filepath, engine="pyarrow", compression="snappy")
                    self.logger.info(f"Saved {name} to {filepath}")

            proc_time = time.time() - start_time
            self.logger.info(f"Intraday data saving completed in {proc_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error saving intraday data: {str(e)}")
            self.logger.error(traceback.format_exc())


def intraday_features_callback_factory(pipeline: TradingPipeline):
    """
    Returns a callback function that the streaming processor will call at 15:56
    with the final intraday features DataFrame.
    """

    def callback(intraday_df: pd.DataFrame):
        logging.info("Intraday features callback invoked. Starting processing thread...")

        # Create a new thread for processing
        def process_thread():
            try:
                logging.info("Starting end-of-day processing in separate thread...")
                pipeline.process_intraday_features(intraday_df)
                logging.info("End-of-day thread processing completed.")
            except Exception as e:
                logging.error(f"Error in end-of-day processing thread: {str(e)}")
                logging.error(traceback.format_exc())

        # Start processing in separate thread
        processing_thread = threading.Thread(target=process_thread)
        processing_thread.start()

    return callback


def main():
    """
    Main entry point:
      1) Start streaming in a background thread
      2) Wait until ~9:31 to prepare daily data
      3) Remain alive to receive the 15:56 callback with intraday features
      4) Merge + finalize trades
    """
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    pipeline_log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    # Create pipeline logger
    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    pipeline_logger.handlers = []

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(pipeline_log_file)
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    pipeline_logger.addHandler(console_handler)

    pipeline_logger.info("===== Starting Trading Pipeline =====")

    # Initialize the TradingPipeline

    api_key = os.getenv("POLYGON_API_KEY")
    pipeline = TradingPipeline(api_key=api_key, logger=pipeline_logger)

    # Wait until 3:55 AM ET to start streaming (5 minutes before pre-market)
    et_tz = pytz.timezone("US/Eastern")
    now = datetime.now(et_tz)
    if now.hour < 4:
        pipeline.wait_until_time(3, 55)

    # Create the Polygon streaming processor with a callback to pipeline
    processor = PolygonStreamProcessor(
        api_key=api_key,
        logger=None,
        intraday_features_callback=intraday_features_callback_factory(pipeline),
    )

    # Start streaming in a background thread
    import threading

    streaming_thread = threading.Thread(target=processor.run, daemon=True)
    streaming_thread.start()
    logging.info("Streaming thread started for pre-market data collection")

    # Wait until ~9:31am to prepare daily data
    pipeline.wait_until_time(9, 55)
    pipeline.prepare_daily_data()

    # After daily data is ready, do nothing. We just wait for the 15:56 callback.
    # Keep the script alive until market close or beyond.
    logging.info("Daily data prepared. Waiting for end-of-day intraday features callback at 15:56...")

    # Keep running indefinitely (or at least past 16:00)
    # You can adjust or add logic to exit after 16:00 if you wish.
    while True:
        time.sleep(60)  # Sleep in 1-minute increments and let threads do their work.

        now = datetime.now(pipeline.et_tz)
        # Example: automatically exit after 16:10 if you want
        if now.hour == 16 and now.minute >= 10:
            logging.info("Reached 16:10 ET. Saving intraday data...")
            pipeline.save_intraday_data()
            logging.info("Intraday data saved. Exiting the pipeline.")
        if now.hour >= 20:
            logging.info("Reached end of post-market session. Exiting pipeline.")
            break


if __name__ == "__main__":
    main()


"""

# Create a new tmux session
tmux new -s trading

# Now inside tmux, run your script
python -m overnight.pipeline.main_trading_pipeline

# To detach from the session (script keeps running):
# Press Ctrl+B, then D

# Later, to reattach to the session from SSH:
tmux attach -t trading

# To list all sessions:
tmux ls

# Kill a specific session by name
tmux kill-session -t trading

# Kill all tmux sessions
tmux kill-server

# List all sessions
tmux ls

# Then kill the one you want
tmux kill-session -t session_name
"""


"""
2025-01-22 21:56:01,102 - INFO - Starting end-of-day intraday feature calculation...
2025-01-22 21:56:38,980 - INFO - Intraday features calculation done.
2025-01-22 21:56:38,980 - INFO - Sending intraday features to pipeline callback...
2025-01-22 21:56:38,980 - INFO - Intraday features callback invoked. Processing end-of-day logic...
2025-01-22 21:56:38,980 - INFO - Merging today's intraday features with historical + daily data...
2025-01-22 21:57:17,170 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:57:30,374 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:57:41,871 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:57:52,041 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:58:02,227 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:58:12,377 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:58:22,434 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:58:32,519 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:58:42,638 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:58:52,741 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:59:04,393 - WARNING - No messages received for 70 seconds. Reconnecting...
2025-01-22 21:59:10,817 - INFO - Selecting tickers for trading (placeholder)...
2025-01-22 21:59:10,817 - INFO - Selected 0 tickers for trading.
2025-01-22 21:59:10,817 - INFO - Executing trades for 0 tickers (placeholder).
2025-01-22 21:59:10,817 - INFO - End-of-day processing completed in 151.84 seconds.
2025-01-22 21:59:10,837 - ERROR - WebSocket error: ping/pong timed out
2025-01-22 21:59:10,837 - ERROR - ping/pong timed out - goodbye
2025-01-22 21:59:10,837 - WARNING - Connection closed: None - None
2025-01-22 21:59:10,838 - INFO - Establishing new WebSocket connection to Polygon...
2025-01-22 21:59:11,154 - INFO - Websocket connected
2025-01-22 21:59:11,155 - INFO - Websocket connected. Sending auth request...
2025-01-22 21:59:11,155 - INFO - Connected Successfully.
2025-01-22 21:59:11,251 - INFO - Authentication successful.
2025-01-22 21:59:11,251 - INFO - Subscribed to market data.


"""
