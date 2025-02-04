from ib_insync import *
from datetime import datetime
import pytz
import logging
from rich.logging import RichHandler
import os

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        logging.FileHandler("app.log"),
        RichHandler(rich_tracebacks=True)
    ]
)
logger = logging.getLogger()


class IBConnection:
    _instance = None
    _default_port = 7497  # Default TWS port
    
    @classmethod
    def get_instance(cls, port=None):
        """
        Get or create singleton IB connection
        Args:
            port: Optional port number. If not provided, uses default port
        """
        if cls._instance is None or not cls._instance.isConnected():
            cls._instance = cls._connect(port=port)
        return cls._instance
    
    @staticmethod
    def _connect(port=None):
        """
        Create new IB connection
        Args:
            port: Optional port number. If not provided, uses default port
        """
        ib = IB()
        port = port or IBConnection._default_port
        ib.connect('127.0.0.1', port, clientId=1)
        return ib
    
    @classmethod
    def set_default_port(cls, port):
        """Set the default port for future connections"""
        cls._default_port = port


def get_market_schedule(ib, contract):
    """Get market schedule and time until close/open."""
    details = ib.reqContractDetails(contract)
    if not details:
        return {
            "is_open": False,
            "time_to_close": None,
            "time_to_open": None,
            "current_session": None
        }

    exchange_tz = details[0].timeZoneId
    now = datetime.now(pytz.timezone(exchange_tz))
    trading_hours = details[0].liquidHours if details[0].liquidHours else details[0].tradingHours

    current_session = None
    next_open_time = None
    current_close_time = None

    for session in trading_hours.split(";"):
        times = session.split("-")
        if len(times) == 2:
            start_time, end_time = times
            start_dt = datetime.strptime(start_time, "%Y%m%d:%H%M")
            end_dt = datetime.strptime(end_time, "%Y%m%d:%H%M")
            start_dt = pytz.timezone(exchange_tz).localize(start_dt)
            end_dt = pytz.timezone(exchange_tz).localize(end_dt)

            if start_dt <= now <= end_dt:
                current_session = {"open": start_dt, "close": end_dt}
                current_close_time = end_dt
            elif start_dt > now and (next_open_time is None or start_dt < next_open_time):
                next_open_time = start_dt

    return {
        "is_open": current_close_time is not None,
        "time_to_close": None if not current_close_time else current_close_time - now,
        "time_to_open": None if not next_open_time else next_open_time - now,
        "current_session": current_session
    }


def is_near_market_close(ib=None, minutes_threshold=5):
    """
    Check if we're within specified minutes of market close.
    """
    ib = ib or IBConnection.get_instance()
    contract = Stock('SPY', 'SMART', 'USD')
    schedule = get_market_schedule(ib, contract)
    
    if not schedule['is_open']:
        return False, None
        
    time_to_close = schedule['time_to_close']
    if time_to_close is None:
        return False, None
        
    # Convert threshold to seconds
    threshold_seconds = minutes_threshold * 60
    
    # Check if we're within the threshold
    is_near_close = time_to_close.total_seconds() <= threshold_seconds
    
    return is_near_close, time_to_close

def calculate_liquidity_metrics(quotes_df):
    """
    Calculate key liquidity metrics from quote data
    
    Parameters:
    quotes_df: DataFrame with columns [ask_price, ask_size, bid_price, bid_size]
    
    Returns:
    dict: Dictionary of liquidity metrics
    """
    # Dollar volume 
    dollar_volume = ((quotes_df['bid_size'] * quotes_df['bid_price'] + 
                     quotes_df['ask_size'] * quotes_df['ask_price']) / 2).sum()
    
    # Spread metrics
    spread = quotes_df['ask_price'] - quotes_df['bid_price']
    relative_spread = spread / ((quotes_df['ask_price'] + quotes_df['bid_price']) / 2)
    avg_spread = spread.mean()
    avg_relative_spread = relative_spread.mean()
    
    # Depth metrics
    quote_depth = (quotes_df['bid_size'] + quotes_df['ask_size']) / 2
    avg_depth = quote_depth.mean()
    
    # Quote activity
    quote_updates = len(quotes_df)
    
    return {
        'dollar_volume_lh': dollar_volume,
        'avg_spread_lh': avg_spread,
        'avg_relative_spread_lh': avg_relative_spread,
        'avg_depth_lh': avg_depth,
        'quote_updates_lh': quote_updates,
        'avg_bid_size_lh': quotes_df['bid_size'].mean(),
        'avg_ask_size_lh': quotes_df['ask_size'].mean(),
        'avg_price_lh': ((quotes_df['ask_price'] + quotes_df['bid_price']) / 2).mean()
    }

from features.nbbo.retrieve_nbbo_data import fetch_quotes_in_window
import pandas_market_calendars as mcal
import pandas as pd

def gather_nbbo_liquidity_metrics(ticker, day_date):
    """
    Iterate over all valid NYSE trading days date_str,
    fetch NBBO quotes for two windows:
        14:55–15:55 (ET) to compute 'last-hour' features
    Return a DataFrame with liquidity metrics on window.

    :param ticker: e.g. "AAPL"
    :param date_str: e.g. "2024-01-01"
    :return: pd.DataFrame with columns for each metric, plus date
    """

    eastern_tz = pytz.timezone("US/Eastern")

    # Construct local datetimes in ET
    # Window 1: 14:55–15:55 ET
    start_local_1 = eastern_tz.localize(datetime(day_date.year, day_date.month, day_date.day, 14, 55, 0))
    end_local_1 = eastern_tz.localize(datetime(day_date.year, day_date.month, day_date.day, 15, 55, 0))


    # 2) Fetch quotes & compute features for each window
    quotes_1 = pd.DataFrame(fetch_quotes_in_window(ticker, start_local_1, end_local_1))
    liquidity_metrics = calculate_liquidity_metrics(quotes_1)

    return liquidity_metrics



if __name__ == "__main__":
    # Test the functions
    ib = IBConnection.get_instance()
    is_near_market_close(ib)
