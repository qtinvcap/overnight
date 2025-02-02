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
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton IB connection"""
        if cls._instance is None or not cls._instance.isConnected():
            cls._instance = cls._connect()
        return cls._instance
    
    @staticmethod
    def _connect():
        """Create new IB connection"""
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)  # Default TWS port
        return ib


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


if __name__ == "__main__":
    # Test the functions
    ib = IBConnection.get_instance()
    is_near_market_close(ib)
