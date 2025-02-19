from ib_insync import IB, util, Stock
import pandas as pd
import datetime
import pytz
import logging
import pytz
import time
import os
from pathlib import Path
from ib_execution.orders_management.open_positions import get_orders_and_positions
from ib_execution.orders_management.utils import setup_logging
def connect_to_ib():
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=56)  # Adjust connection details as needed
    return ib

def is_market_open_with_time(ib, contract):
    details = ib.reqContractDetails(contract)
    if not details:
        return (
            False,
            None,
        )  # No details, assume market is not open and no next open time

    exchange_tz = details[0].timeZoneId
    now = datetime.datetime.now(pytz.timezone(exchange_tz))
    trading_hours = details[0].liquidHours if details[0].liquidHours else details[0].tradingHours

    next_open_time = None
    for session in trading_hours.split(";"):
        times = session.split("-")
        if len(times) == 2:
            start_time, end_time = times
            start_dt = datetime.datetime.strptime(start_time, "%Y%m%d:%H%M")
            end_dt = datetime.datetime.strptime(end_time, "%Y%m%d:%H%M")
            start_dt = pytz.timezone(exchange_tz).localize(start_dt)
            end_dt = pytz.timezone(exchange_tz).localize(end_dt)
            if start_dt <= now <= end_dt:
                return True, None  # Market is open
            elif start_dt > now and (next_open_time is None or start_dt < next_open_time):
                next_open_time = start_dt

    return False, next_open_time


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_latest_price(ib, contract):
    # Check if the market is open

    is_open, _ = is_market_open_with_time(ib, contract)
    if is_open:
        # Market is open, fetch real-time data
        # contract = Stock("GOOG", "SMART", "USD")
        contract = Stock(contract.symbol, "SMART", "USD")
        ticker = ib.reqMktData(contract, "", False, False)
        ib.sleep(1)  # Allow some time for data to be returned
        if ticker:
            print("Market is open. Using real-time price.")
            market_price = ticker.last if ticker.last is not None else ticker.close
            return market_price
        else:
            return None
    else:
        print("Market is closed. Fetching last close price.")
        # Adjust to get the last close price properly
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        formatted_end_date = end_date.strftime("%Y%m%d 23:59:59")

        contract.exchange = "SMART"

        # Make sure to adjust this line if your data subscription requires
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=formatted_end_date,
            durationStr="2 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if bars:
            return bars[-1].close  # Last available closing price
        else:
            print("No historical data returned.")
            return None


def fetch_pending_orders(ib):
    # Fetching all open trades which may contain the STP and LMT orders.
    open_trades = ib.openTrades()
    tp_sl_orders = [
        {
            "ticker": trade.contract.symbol,
            "order_type": trade.order.orderType,
            "shares": trade.order.totalQuantity,  # - trade.order.filledQuantity,
            "price": (trade.order.auxPrice if trade.order.orderType == "STP" else trade.order.lmtPrice),
        }
        for trade in open_trades
        if trade.order.orderType in ["STP", "LMT"]
    ]
    return tp_sl_orders


def fetch_portfolio_dataframe(ib):
    portfolio = ib.portfolio()

    # Initialize a data dictionary for DataFrame creation
    portfolio_data = {
        "ticker": [],
        "position_size": [],
        "market_value": [],
        "average_cost": [],
        "latest_price": [],
        "unrealized_pnl": [],
    }

    # Loop through each position and process orders
    for position in portfolio:

        contract = position.contract
        latest_price = fetch_latest_price(ib, contract)

        # Populate the data dictionary
        portfolio_data["ticker"].append(position.contract.symbol)
        portfolio_data["position_size"].append(position.position)
        portfolio_data["market_value"].append(position.marketValue)
        portfolio_data["average_cost"].append(position.averageCost)
        portfolio_data["latest_price"].append(latest_price)
        portfolio_data["unrealized_pnl"].append(position.unrealizedPNL)

    portfolio_df = pd.DataFrame(portfolio_data)

    # Fetch account summary and create a summary DataFrame
    account_summary = ib.accountSummary()
    summary_data = {
        "net_liquidation": next(item.value for item in account_summary if item.tag == "NetLiquidation"),
        "total_cash_value": next(item.value for item in account_summary if item.tag == "TotalCashValue"),
    }
    summary_df = pd.DataFrame([summary_data])

    # Combine the DataFrames
    combined_df = pd.concat([portfolio_df, summary_df], axis=1)
    # Apply ffill only to the specified columns
    combined_df[["net_liquidation", "total_cash_value"]] = combined_df[["net_liquidation", "total_cash_value"]].ffill()

    return combined_df

def get_available_cash(ib):

    account_summary = ib.accountSummary()
    total_cash_value = float(next(item.value for item in account_summary if item.tag == "TotalCashValue"))
    return total_cash_value

def save_today_net_liquidation(ib):
    root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
    portfolio_data_dir = Path(root_path) / "portfolio_monitoring_data"
    series_file = portfolio_data_dir / "daily_net_liquidation.csv"
    
    # Create directory if it doesn't exist
    portfolio_data_dir.mkdir(parents=True, exist_ok=True)

    et_tz = pytz.timezone("US/Eastern")
    
    # Get the net liquidation value directly
    account_summary = ib.accountSummary()
    net_liquidation = float(next(item.value for item in account_summary if item.tag == "NetLiquidation"))
    today = datetime.datetime.now(et_tz).date()

    # Try to load an existing Series from file, or create a new one if not available
    try:
        performance_series = pd.read_csv(series_file, index_col=0, parse_dates=True, squeeze=True)
        performance_series.index = pd.to_datetime(performance_series.index).date
    except Exception:
        performance_series = pd.Series(dtype=float)

    performance_series.loc[today] = net_liquidation
    performance_series.sort_index(inplace=True)

    # Save the updated series to CSV so that it persists across sessions
    performance_series.to_csv(series_file, header=True)
    
    print(f"Recorded net liquidation value {net_liquidation} for {today}")
    return performance_series


def saved_filled_positions_report(ib, selected_tickers):

    pipeline_logger = setup_logging()
    root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
    portfolio_data_dir = Path(root_path) / "portfolio_monitoring_data"
    df_file = portfolio_data_dir / "filled_positions.csv"

    # Get positions from IB (using your existing function)
    positions_df = get_orders_and_positions(ib, pipeline_logger)
    
    # Rename 'symbol' column to 'ticker' in positions_df to ease merging.
    if not positions_df.empty:
        positions_df = positions_df.rename(columns={"symbol": "ticker", "quantity": "filled_position"})
    else:
        pipeline_logger.info("No filled positions reported by IB.")
        # Create an empty DataFrame with the expected columns.
        positions_df = pd.DataFrame(columns=["ticker", "filled_position", "market_value"])
    
    # Merge the selected_tickers (orders) with positions_df (filled positions) on ticker.
    merged_df = pd.merge(selected_tickers, positions_df, on="ticker", how="left")
    
    # Fill NaN values with 0 for filled_position and market_value (if no position was filled).
    merged_df["filled_position"] = merged_df["filled_position"].fillna(0)
    merged_df["market_value"] = merged_df["market_value"].fillna(0)
    
    # Compute market value based on the order details.
    # Here we assume the intended order price is in the 'price' column.
    merged_df["market_value_order"] = merged_df["quantity"] * merged_df["price"]
    
    # Add current date (using Eastern Time) to track when this record was captured.
    et_tz = pytz.timezone("US/Eastern")
    current_date = datetime.datetime.now(et_tz).date()
    merged_df["date"] = current_date

    # Rename 'quantity' to 'order_position' and 'market_value' to 'market_value_filled'
    merged_df = merged_df.rename(columns={"quantity": "order_position", "market_value": "market_value_filled"})
    
    # Select the desired columns in a preferred order.
    final_df = merged_df[["date", "ticker", "order_position", "filled_position", 
                          "market_value_order", "market_value_filled"]]
    
    # Save (or append) to CSV.
    csv_file = Path(df_file)
    if csv_file.exists():
        existing_df = pd.read_csv(csv_file)
        combined_df = pd.concat([existing_df, final_df], ignore_index=True)
        combined_df.to_csv(csv_file, index=False)
    else:
        final_df.to_csv(csv_file, index=False)
    
    pipeline_logger.info(f"Filled positions saved to {csv_file}")
    return final_df

if __name__ == "__main__":
    ib = connect_to_ib()
    #selected_tickers = pd.read_parquet("/root/overnight/debug_data/20250214/selected_tickers.parquet")
    save_today_net_liquidation(ib)
    #portfolio_df = fetch_portfolio_dataframe(ib)
    #print(portfolio_df)
    #cash_value = get_available_cash(ib)
    #print(cash_value)
    ib.disconnect()