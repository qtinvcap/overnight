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
from overnight.features.combine_intraday_daily_features import (
    add_indicator_normalizations,
    merge_daily_and_intraday_data,
    add_temporal_features,
)
from overnight.features.vix import generate_vix_data_and_merge
from overnight.features.utils import get_ticker_full_tickers_list
from overnight.features.daily.features_engineering import compute_advanced_daily_features
from overnight.models.model_v1.main_inference import get_trading_signals, load_best_ranges, score_features_df
from overnight.features.intraday.main_intraday import process_intraday_data
from overnight.features.intraday.rolling_calcs import compute_historical_rolling_metrics, apply_rolling_metrics
import pandas_market_calendars as mcal

# Import the streaming script class
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
        """Get trading calendar and define needed dates."""
        today = datetime.now().strftime("%Y-%m-%d")
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date="2010-01-02", end_date=today)
        self.trading_days = mcal.date_range(schedule, frequency="1D")
        self.trading_days = [d.date() for d in self.trading_days]
        # 205th and 5th last trading day
        self.past_205_trading_day = self.trading_days[-205]
        self.past_5_trading_day = self.trading_days[-6]
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

            self.get_trading_calendar()
            past_205_date = self.past_205_trading_day.strftime("%Y-%m-%d")
            past_5_date = self.past_5_trading_day.strftime("%Y-%m-%d")
            yesterday_date = self.yesterday_trading_day.strftime("%Y-%m-%d")

            self.logger.info("Getting updated ticker list...")
            tickers_list = get_ticker_full_tickers_list()
            self.logger.info(f"Collected {len(tickers_list)} tickers for daily data.")

            # Load daily data and compute features
            self.logger.info("Loading daily data from Polygon...")
            daily_data = self.polygon.get_historical_data(past_205_date, today_str)

            self.logger.info("Filtering on relevant tickers...")
            daily_data = daily_data[daily_data["ticker"].isin(tickers_list)]

            self.logger.info("Computing daily features...")
            self.daily_features = compute_advanced_daily_features(daily_data)
            self.daily_features = generate_vix_data_and_merge(self.daily_features)

            # Load historical intraday for the past 5 days
            self.logger.info("Loading last 5 days intraday data...")
            self.intraday_past_5_days = process_intraday_data(past_5_date, yesterday_date)
            self.intraday_past_5_days = self.intraday_past_5_days[
                self.intraday_past_5_days["ticker"].isin(tickers_list)
            ]
            self.logger.info("Computing historical rolling metrics...")
            self.historical_rolling_metrics = compute_historical_rolling_metrics(self.intraday_past_5_days)
            self.logger.info(f"Computed rolling metrics for {len(self.historical_rolling_metrics)} tickers")

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
        Called at ~15:56 via the streaming script callback.
        Merges today's intraday with the past intraday + daily data,
        computes final features, and triggers trade logic.
        """
        if not self.ready_for_trading:
            raise RuntimeError("Pipeline not ready for trading - daily data not prepared.")

        try:
            start_time = time.time()
            # self.logger.info("Merging today's intraday features with historical + daily data...")

            # Combine historical and today's intraday
            # intraday_combined = pd.concat([self.intraday_past_5_days, today_intraday_features], ignore_index=True)
            # intraday_combined = add_rolling_ratio_features(intraday_combined)
            self.logger.info("Processing today's intraday features...")

            # Apply rolling metrics to today's data
            self.logger.info("Applying rolling metrics to today's data...")
            today_with_metrics = apply_rolling_metrics(today_intraday_features, self.historical_rolling_metrics)
            # Merge intraday + daily
            final_merged = merge_daily_and_intraday_data(self.daily_features, today_with_metrics)
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

            # Inference / stock selection
            selected_tickers = self.select_tickers(data_for_stock_selection)

            if not selected_tickers.empty:
                selected_tickers.to_parquet(
                    debug_dir / "selected_tickers.parquet", engine="pyarrow", compression="snappy"
                )
                self.logger.info(f"Saved selected_tickers to {debug_dir}/selected_tickers.parquet")

            # Placeholder for actual trades
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

            best_ranges_path = os.path.join(self.root_path, "overnight/models/model_v1/rules/best_ranges.pkl")
            best_ranges = load_best_ranges(best_ranges_path)

            # Score data
            scored_df = score_features_df(
                df=data,
                best_ranges=best_ranges,
                metric="trimmed_mean_std_ratio_performance",
                target_total_rules=100,
                target_ratio=0.5,
                weight_by_spread=False,
            )

            # Get signals
            selected = get_trading_signals(
                scored_df=scored_df,
                initial_capital=50000,  # example
                threshold=0.1,
                max_positions=5,
                liquidity_threshold=1_000_000,
                price_threshold=2,
            )

            self.logger.info(f"Selected {len(selected)} tickers for trading.")
            if len(selected) > 0:
                tickers_str = ", ".join(selected["ticker"].tolist())
                total_capital = selected["position_size"].sum()
                self.logger.info(f"Selected tickers: {tickers_str}")
                self.logger.info(f"Total capital required: ${total_capital:,.2f}")

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
        pass

    def save_intraday_data(self):
        """
        Saves intraday data to parquet files after market close.
        (You can decide how you'd like to implement or call this.)
        """
        # Example implementation if you have self.intraday_data dict
        try:
            save_dir = Path("intraday_data") / datetime.now().strftime("%Y%m%d")
            save_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Starting to save intraday data...")
            start_time = time.time()
            # If you store data in self.intraday_data:
            # for name, data in self.intraday_data.items():
            #     if data is not None:
            #         filepath = save_dir / f"{name}.parquet"
            #         data.to_parquet(filepath, engine="pyarrow", compression="snappy")
            #         self.logger.info(f"Saved {name} to {filepath}")

            proc_time = time.time() - start_time
            self.logger.info(f"Intraday data saving completed in {proc_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Error saving intraday data: {str(e)}")
            self.logger.error(traceback.format_exc())


def intraday_features_callback_factory(pipeline: TradingPipeline):
    """
    Returns a callback function that the streaming processor will call ~15:56
    with the final intraday features DataFrame.
    """

    def callback(intraday_df: pd.DataFrame):
        logging.info("Intraday features callback invoked. Starting processing thread...")

        def process_thread():
            try:
                logging.info("Starting end-of-day pipeline processing in a separate thread...")
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
      2) Wait until ~9:55 to prepare daily data
      3) Remain alive to receive the ~15:56 callback with intraday features
      4) Merge + finalize trades
    """
    # Setup logging for the pipeline
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    pipeline_log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"

    pipeline_logger = logging.getLogger("pipeline")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(pipeline_log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    pipeline_logger.addHandler(file_handler)
    pipeline_logger.addHandler(console_handler)

    pipeline_logger.info("===== Starting Trading Pipeline =====")

    api_key = os.getenv("POLYGON_API_KEY")
    pipeline = TradingPipeline(api_key=api_key, logger=pipeline_logger)

    # If you want to wait until 3:55 AM ET to start streaming, do so:
    et_tz = pytz.timezone("US/Eastern")
    now = datetime.now(et_tz)
    if now.hour < 4:
        pipeline.wait_until_time(3, 55)

    # Create the streaming processor with callback
    from overnight.pipeline.streaming_and_process_intraday_data_time_trigger import PolygonStreamProcessor

    processor = PolygonStreamProcessor(
        api_key=api_key,
        logger=None,  # We'll use the streaming script's default separate logger
        intraday_features_callback=intraday_features_callback_factory(pipeline),
    )

    # Start streaming in a background thread
    import threading

    streaming_thread = threading.Thread(target=processor.run, daemon=True)
    streaming_thread.start()
    pipeline_logger.info("Streaming thread started for pre-market data collection")

    # Wait until ~9:55am to prepare daily data
    pipeline.wait_until_time(9, 55)
    pipeline.prepare_daily_data()

    pipeline_logger.info("Daily data prepared. Waiting for the ~15:56 intraday callback...")

    # Keep running, e.g., until ~16:10 or later
    while True:
        time.sleep(60)
        now = datetime.now(pipeline.et_tz)

        # Example auto-exit after 16:10 if desired
        if now.hour == 17 and now.minute >= 50:
            pipeline_logger.info("Reached 16:10 ET. Saving intraday data...")
            pipeline.save_intraday_data()
            pipeline_logger.info("Intraday data saved. Exiting the pipeline.")
            break

    pipeline_logger.info("Trading Pipeline has ended.")


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
