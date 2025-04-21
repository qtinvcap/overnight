import asyncio
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
from features.daily.retrieve_price_data_api import PolygonHistoricalDailyData
from features.combine_intraday_daily_features import (
    add_indicator_normalizations,
    merge_daily_and_intraday_data,
    add_temporal_features,
)
from features.vix import generate_vix_data_and_merge
from features.utils import get_ticker_full_tickers_list
from features.daily.features_engineering_with_etf import compute_advanced_daily_features
from models.model_v3.main_inference import main_inference
from features.intraday.main_intraday_with_etf import process_intraday_data
from features.intraday.rolling_calcs import compute_historical_rolling_metrics, apply_rolling_metrics
import pandas_market_calendars as mcal
from ib_execution.orders_management.utils import IBConnection, setup_logging
from ib_execution.orders_management.entry_orders import place_entry_orders
from ib_execution.orders_management.exit_orders import place_exit_orders_fixed, place_premarket_limit
from ib_execution.orders_management.cancel_orders import robust_cancel_all_orders
from ib_execution.portfolio_data.retrieve_portfolio_data import (
    get_available_cash,
    save_today_net_liquidation,
    saved_filled_positions_report,
    analyze_exit_trades,
)
from ib_execution.orders_management.open_positions import get_positions
from ib_execution.orders_management.pending_orders import get_pending_orders

# Import the streaming script class
import os


class TradingPipeline:
    """
    Encapsulates daily data preparation and end-of-day intraday processing.
    """

    def __init__(self, api_key, logger=None, client_id=None):
        self.api_key = api_key
        self.polygon = PolygonHistoricalDailyData(self.api_key)
        self.et_tz = pytz.timezone("US/Eastern")

        self.logger = logger or logging.getLogger("pipeline")

        self.root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))

        # Data containers
        self.daily_features = None
        self.intraday_past_5_days = None
        self.ready_for_trading = False
        self.ib = (
            IBConnection.get_instance(port=4002)
            if client_id is None
            else IBConnection.get_instance(port=4002, client_id=client_id)
        )
        self.ib_pipeline_logger = setup_logging()
        self.cash_available = None
        self.selected_tickers = pd.DataFrame()

        self.setup_execution_logging()

    def setup_execution_logging(self):
        """
        Prepares the execution logs folder and attaches an execution callback.
        """
        self.execution_log_dir = Path(self.root_path) / "execution_logs"
        self.execution_log_dir.mkdir(parents=True, exist_ok=True)
        # Attach the execution callback from IB-insync
        self.ib.execDetailsEvent += self.on_exec_details
        self.logger.info("Execution logging has been set up.")

    def on_exec_details(self, reqId, contract, execution):
        """
        Callback fired when an execution (fill) occurs.
        Logs execution details to a daily log file.
        """
        # Get current time in Eastern Time
        now_str = datetime.now(self.et_tz).strftime("%Y-%m-%d %H:%M:%S")
        log_message = (
            f"{now_str} - ReqID: {reqId}, Symbol: {contract.symbol}, SecType: {contract.secType}, "
            f"Exchange: {contract.exchange}, Action: {execution.side}, Price: {execution.price}, "
            f"Shares: {execution.shares}, ExecTime: {execution.time}\n"
        )
        # Log to the pipeline logger
        self.logger.info("Execution details: " + log_message.strip())
        # Append to daily execution log file
        today_str = datetime.now(self.et_tz).strftime("%Y%m%d")
        log_file = self.execution_log_dir / f"execution_{today_str}.log"
        try:
            with open(log_file, "a") as f:
                f.write(log_message)
        except Exception as e:
            self.logger.error(f"Failed to write execution log: {e}")

    def wait_until_time(self, hour, minute, second=0):
        """
        Blocks until a specified hour/minute in ET time on the *same day*.
        If that time has passed, returns immediately.
        """
        now = datetime.now(self.et_tz)
        target = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        if now < target:
            wait_s = (target - now).total_seconds()
            self.logger.info(f"Waiting {wait_s:.0f} seconds until {hour:02d}:{minute:02d}:{second:02d} ET...")
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
        self.past_20_trading_day = self.trading_days[-21]
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
            past_20_date = self.past_20_trading_day.strftime("%Y-%m-%d")
            yesterday_date = self.yesterday_trading_day.strftime("%Y-%m-%d")

            ETF_TICKERS = ["SPY", "QQQ", "QQQQ", "IWM"]

            self.logger.info("Getting updated ticker list...")
            tickers_list = get_ticker_full_tickers_list() + ETF_TICKERS
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
            self.logger.info("Loading last 20 days intraday data...")
            self.intraday_past_20_days = process_intraday_data(past_20_date, yesterday_date)
            self.intraday_past_20_days = self.intraday_past_20_days[
                self.intraday_past_20_days["ticker"].isin(tickers_list)
            ]
            self.logger.info("Computing historical rolling metrics...")
            self.historical_rolling_metrics = compute_historical_rolling_metrics(self.intraday_past_20_days)
            self.logger.info(f"Computed rolling metrics for {len(self.historical_rolling_metrics)} tickers")

            # Get available cash from IB
            self.cash_available = get_available_cash(self.ib)
            self.logger.info(f"Available cash: ${self.cash_available:,.2f}")

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
            debug_dir = Path(self.root_path) / "debug_data" / today_str
            debug_dir.mkdir(parents=True, exist_ok=True)

            data_for_stock_selection.to_parquet(
                debug_dir / "data_for_stock_selection.parquet", engine="pyarrow", compression="snappy"
            )
            self.logger.info(f"Saved data_for_stock_selection to {debug_dir}/data_for_stock_selection.parquet")

            # Inference / stock selection
            self.selected_tickers = self.select_tickers(data_for_stock_selection, cash_available=self.cash_available)

            if not self.selected_tickers.empty:
                self.selected_tickers.to_parquet(
                    debug_dir / "selected_tickers.parquet", engine="pyarrow", compression="snappy"
                )
                self.logger.info(f"Saved selected_tickers to {debug_dir}/selected_tickers.parquet")

            self.execute_entry_trades(self.selected_tickers)

            proc_time = time.time() - start_time
            self.logger.info(f"End-of-day processing completed in {proc_time:.2f} seconds.")

            return self.selected_tickers

        except Exception as e:
            self.logger.error(f"Error in end-of-day trading pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def select_tickers(self, data, cash_available=50000):
        """
        Uses the main_inference module to select tickers based on scoring and trading signals.
        Returns a DataFrame with selected tickers and their trading details.
        """
        try:
            self.logger.info("Starting ticker selection using inference model...")

            selected = main_inference(
                df=data,
                initial_capital=cash_available,
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

    def execute_entry_trades(self, selected_tickers):
        """
        Execute trades using IB API through TradingBot
        """
        if selected_tickers.empty:
            self.logger.info("No tickers selected for trading")
            return

        try:
            # Format orders for TradingBot
            orders_list = []
            for _, row in selected_tickers.iterrows():
                orders_list.append(
                    {
                        "ticker": row["ticker"],
                        "quantity": int(row["quantity"]),  # Ensure integer quantity
                        "action": "BUY",  # Assuming all entries are buys
                    }
                )

            self.logger.info(f"Executing trades for {len(orders_list)} positions...")

            # Place entry orders
            place_entry_orders(self.ib, self.ib_pipeline_logger, orders_list)

        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            self.logger.error(traceback.format_exc())

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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                pipeline.process_intraday_features(intraday_df)
                loop.close()
                logging.info("End-of-day thread processing completed.")

            except Exception as e:
                logging.error(f"Error in end-of-day processing thread: {str(e)}")
                logging.error(traceback.format_exc())

        # Start processing in separate thread
        processing_thread = threading.Thread(target=process_thread)
        processing_thread.start()

    return callback


def check_and_restart_ibgateway(logger):
    """
    Check IB Gateway connection and restart if needed.
    Returns True if connection is successful, False otherwise.
    """
    try:
        # Try to establish connection
        ib = IBConnection.get_instance(port=4002)
        if ib.isConnected():
            logger.info("IB Gateway connection is active")
            return True

        logger.warning("IB Gateway connection not active, attempting restart...")

        # Execute the restart script
        import subprocess

        restart_script = "/root/overnight/restart_ibgateway.sh"  # Update with actual path
        process = subprocess.Popen(["bash", restart_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for the script to complete (with timeout)
        try:
            stdout, stderr = process.communicate(timeout=60)
            logger.info(f"Restart script output: {stdout.decode()}")
            if stderr:
                logger.error(f"Restart script errors: {stderr.decode()}")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Restart script timed out after 60 seconds")
            return False

        # Wait for IB Gateway to initialize
        time.sleep(30)

        # Verify connection
        ib = IBConnection.get_instance(port=4002)
        if ib.isConnected():
            logger.info("IB Gateway successfully restarted and connected")
            return True
        else:
            logger.error("Failed to establish IB Gateway connection after restart")
            return False

    except Exception as e:
        logger.error(f"Error checking/restarting IB Gateway: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def wait_for_all_orders_filled(ib, logger, et_tz, timeout_minutes=10):
    """
    Waits until all sell orders are fully filled before proceeding, with an optional timeout.
    Returns the time when all orders were confirmed filled.
    """
    logger.info("Waiting for all sell orders to be fully filled...")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    # Initial request to sync order data
    ib.reqOpenOrders()
    ib.sleep(0.5)  # Brief delay to ensure response is processed

    while time.time() - start_time < timeout_seconds:
        open_orders = ib.openOrders()
        unfilled_sell_orders = [order for order in open_orders if order.action == "SELL"]
        if not unfilled_sell_orders:
            completion_time = datetime.now(et_tz)
            logger.info(f"All sell orders are fully filled at {completion_time.strftime('%H:%M:%S')}.")
            return completion_time
        else:
            tickers = ", ".join({order.contract.symbol for order in unfilled_sell_orders})
            logger.info(f"Waiting for {len(unfilled_sell_orders)} sell orders to fill: {tickers}")
            ib.reqOpenOrders()  # Refresh order status each iteration
            time.sleep(60)
    logger.error(f"Timeout: Not all sell orders filled after {timeout_minutes} minutes.")
    return None


def main():
    """
    Main entry point:
      1) Start streaming in a background thread
      2) Wait until ~9:55 to prepare daily data
      3) Remain alive to receive the ~15:56 callback with intraday features
      4) Merge + finalize trades
    """
    # Setup logging for the pipeline
    root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
    log_dir = Path(root_path) / "logs"
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

    # Check IB Gateway connection before proceeding
    if not check_and_restart_ibgateway(pipeline_logger):
        pipeline_logger.error("Unable to establish IB Gateway connection. Exiting.")
        sys.exit(1)

    api_key = os.getenv("POLYGON_API_KEY")
    pipeline = TradingPipeline(api_key=api_key, logger=pipeline_logger)

    pipeline.get_trading_calendar()
    today = datetime.now().date()
    if today not in pipeline.trading_days:
        pipeline_logger.info("Today is not a trading day. Exiting.")
        sys.exit(1)

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

    # Wait until 9:28:30am to cancel all limit premarket orders
    pipeline.wait_until_time(9, 28, 30)
    robust_cancel_all_orders(pipeline.ib, pipeline.ib_pipeline_logger)
    get_pending_orders(pipeline.ib, pipeline.ib_pipeline_logger)

    # Wait until 9:29:15am to schedule exit OPG orders for next morning
    pipeline.wait_until_time(9, 29, 15)
    pipeline_logger.info("Scheduling exit orders for next market open...")
    place_exit_orders_fixed(pipeline.ib, pipeline.ib_pipeline_logger, tif="OPG")

    # monitor positions after open before placing market orders for remaining positions
    pipeline.wait_until_time(9, 30, 7)
    robust_cancel_all_orders(pipeline.ib, pipeline.ib_pipeline_logger)
    get_positions(pipeline.ib, pipeline.ib_pipeline_logger)

    # Wait until 9:30:10am to place market orders for remaining positions
    pipeline_logger.info("placing orders after open if remaining positions after auction...")
    pipeline.wait_until_time(9, 30, 10)
    place_exit_orders_fixed(pipeline.ib, pipeline.ib_pipeline_logger, tif="DAY")

    # monitor positions after open after placing market orders for remaining positions
    pipeline.wait_until_time(9, 32)
    get_positions(pipeline.ib, pipeline.ib_pipeline_logger)

    # Wait until 10:00am to save today's net liquidation
    pipeline.wait_until_time(10, 00)
    save_today_net_liquidation(pipeline.ib)  # SAVE TODAY'S NET LIQUIDATION
    analyze_exit_trades(pipeline.ib)  # ANALYZE EXIT EXECUTIONS

    # Wait until ~10:05am to prepare daily data
    pipeline.wait_until_time(10, 5)
    pipeline.prepare_daily_data()
    pipeline_logger.info("Daily data prepared. Waiting for the ~15:56 intraday callback...")
    
    # Wait until 16:00:30 to place premarket limit orders and monitor filled positions
    pipeline.wait_until_time(16, 0, 10)
    get_pending_orders(pipeline.ib, pipeline.ib_pipeline_logger)
    saved_filled_positions_report(pipeline.ib, selected_tickers=pipeline.selected_tickers)

    # Wait until 16:01:00 to place premarket limit orders and monitor filled positions
    pipeline.wait_until_time(16, 0, 15)
    place_premarket_limit(pipeline.ib, pipeline.ib_pipeline_logger)

    # Keep running, e.g., until ~16:10 or later
    while True:
        time.sleep(60)
        now = datetime.now(pipeline.et_tz)

        # Example auto-exit after 16:10 if desired
        if now.hour == 20 and now.minute >= 00:
            pipeline_logger.info("Reached 20:00 ET end of after hours. Saving intraday data...")
            pipeline.save_intraday_data()
            pipeline_logger.info("Intraday data saved. Exiting the pipeline.")
            break

    pipeline_logger.info("Trading Pipeline has ended.")


if __name__ == "__main__":
    main()
