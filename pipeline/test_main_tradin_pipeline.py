import logging
import time
from datetime import datetime
import sys
from pathlib import Path
import os
import pytz
import pandas_market_calendars as mcal

from overnight.pipeline.main_trading_pipeline_with_ib import TradingPipeline
from overnight.features.intraday.main_intraday import process_intraday_data
from overnight.features.intraday.rolling_calcs import apply_rolling_metrics
from overnight.features.combine_intraday_daily_features import (
    add_indicator_normalizations,
    merge_daily_and_intraday_data,
    add_temporal_features,
)


def setup_logging(test_date):
    """Setup logging configuration"""
    root_path = Path(os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~")))
    log_dir = root_path / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"pipeline_backtest_{test_date}.log"

    logger = logging.getLogger("pipeline_test")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


class BacktestPipeline(TradingPipeline):
    """Extended pipeline class for backtesting purposes"""

    def __init__(self, api_key, test_date, logger=None):
        super().__init__(api_key, logger)
        self.test_date = test_date

    def get_trading_calendar(self):
        """Override to use test date instead of current date"""
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date="2010-01-02", end_date=self.test_date)
        self.trading_days = mcal.date_range(schedule, frequency="1D")
        self.trading_days = [d.date() for d in self.trading_days]

        # Get the required trading days relative to test date
        self.past_205_trading_day = self.trading_days[-205]
        self.past_20_trading_day = self.trading_days[-21]
        self.yesterday_trading_day = self.trading_days[-2]

        self.logger.info(f"Trading calendar prepared for test date {self.test_date}")
        self.logger.info(f"Using data from {self.past_205_trading_day}")


def run_backtest(test_date: str):
    """
    Run a backtest of the trading pipeline for a specific historical date

    Args:
        test_date: Date string in format 'YYYY-MM-DD'
    """
    # Setup logging
    logger = setup_logging(test_date)
    logger.info(f"=== Starting Pipeline Backtest for {test_date} ===")

    # Validate test date format
    try:
        datetime.strptime(test_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD")

    # Initialize pipeline
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    pipeline = BacktestPipeline(api_key=api_key, test_date=test_date, logger=logger)

    try:
        # Step 1: Initialize trading calendar
        logger.info("Step 1: Initializing trading calendar...")
        pipeline.get_trading_calendar()

        # Step 2: Prepare daily data
        logger.info("Step 2: Preparing daily data...")
        start_time = time.time()
        pipeline.prepare_daily_data(test_date=test_date)
        daily_time = time.time() - start_time
        logger.info(f"Daily data preparation completed in {daily_time:.2f} seconds")

        retrieved_tickers = pipeline.daily_features.ticker.unique()
        logger.info(f"Retrieved {len(retrieved_tickers)} tickers from daily data")
        # Step 3: Process historical intraday data
        logger.info(f"Step 3: Processing intraday data for {test_date}...")
        start_time = time.time()
        intraday_features = process_intraday_data(start_date_str=test_date, end_date_str=test_date)
        intraday_features = intraday_features[intraday_features.ticker.isin(retrieved_tickers)]
        intraday_time = time.time() - start_time
        logger.info(f"Intraday data processing completed in {intraday_time:.2f} seconds")

        # Step 4: Process final features and select tickers
        logger.info("Step 4: Processing final features and selecting tickers...")
        start_time = time.time()

        # Apply rolling metrics (new step from updated pipeline)
        logger.info("Applying rolling metrics to today's data...")
        intraday_features = apply_rolling_metrics(intraday_features, pipeline.historical_rolling_metrics)

        final_merged = merge_daily_and_intraday_data(pipeline.daily_features, intraday_features)
        data_for_stock_selection = add_indicator_normalizations(final_merged)
        data_for_stock_selection = add_temporal_features(data_for_stock_selection)
        # Save data before temporal features for debugging
        root_path = Path(os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~")))
        debug_dir = root_path / "debug_data" / test_date
        debug_dir.mkdir(parents=True, exist_ok=True)
        data_for_stock_selection.to_parquet(
            debug_dir / "data_before_temporal.parquet", engine="pyarrow", compression="snappy"
        )
        logger.info(f"Saved data before temporal features to {debug_dir}/data_before_prediction.parquet")
        feature_time = time.time() - start_time
        logger.info(f"Feature processing and ticker selection completed in {feature_time:.2f} seconds")

        selected_tickers = pipeline.select_tickers(data_for_stock_selection)
        # Summary
        total_time = daily_time + intraday_time + feature_time
        logger.info("\n=== Backtest Summary ===")
        logger.info(f"Test Date: {test_date}")
        logger.info(f"Daily Data Processing Time: {daily_time:.2f}s")
        logger.info(f"Intraday Data Processing Time: {intraday_time:.2f}s")
        logger.info(f"Feature Processing Time: {feature_time:.2f}s")
        logger.info(f"Total Processing Time: {total_time:.2f}s")

        if selected_tickers is not None and not selected_tickers.empty:
            logger.info(f"Number of Selected Tickers: {len(selected_tickers)}")
            logger.info(f"Selected Tickers: {', '.join(selected_tickers['ticker'])}")
            logger.info(f"Total Position Size: ${selected_tickers['position_size'].sum():,.2f}")
        else:
            logger.info("No tickers selected for trading")

        logger.info("=== Backtest Complete ===")

        return selected_tickers

    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Use a past date that was a trading day
    test_date = "2025-02-06"  # Make sure this is a valid trading day
    run_backtest(test_date)

    # 2025-01-31 20:06:35,296 - INFO - Selected Tickers: ACON, MODV, WULF, BNGO, RZLV
