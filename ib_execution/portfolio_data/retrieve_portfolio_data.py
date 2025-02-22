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


def get_official_closes(ib, tickers):

    pipeline_logger = setup_logging()
    et_tz = pytz.timezone("US/Eastern")

    closing_prices = []
    end_date = datetime.datetime.now(et_tz)
    
    for ticker in tickers:
        contract = Stock(ticker, 'SMART', 'USD')
        try:
            # Get today's historical data
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            
            if bars and len(bars) > 0:
                closing_prices.append({
                    'ticker': ticker,
                    'official_close': bars[-1].close
                })
            else:
                pipeline_logger.warning(f"No data received for {ticker}")
                
        except Exception as e:
            pipeline_logger.error(f"Error getting close price for {ticker}: {e}")
    
    closes_df = pd.DataFrame(closing_prices)
    return closes_df



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

    # Try to load existing data from file, or create a new Series if not available
    try:
        # Read as DataFrame first
        df = pd.read_csv(series_file)
        # Convert to Series properly
        performance_series = pd.Series(
            data=df['net_liquidation'].values,
            index=pd.to_datetime(df['date']).dt.date,
            name='net_liquidation'
        )
    except Exception:
        performance_series = pd.Series(name='net_liquidation', dtype=float)

    # Add today's value
    performance_series.loc[today] = net_liquidation
    performance_series.sort_index(inplace=True)

    # Save to CSV with explicit date column
    df_to_save = pd.DataFrame({
        'date': performance_series.index,
        'net_liquidation': performance_series.values
    })
    df_to_save.to_csv(series_file, index=False)
    
    print(f"Recorded net liquidation value {net_liquidation} for {today}")
    return performance_series


def saved_filled_positions_report(ib, selected_tickers):
    pipeline_logger = setup_logging()
    root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
    portfolio_data_dir = Path(root_path) / "portfolio_monitoring_data"
    df_file = portfolio_data_dir / "filled_positions_report.csv"


    et_tz = pytz.timezone("US/Eastern")

    # Get positions from IB
    positions_df = get_orders_and_positions(ib, pipeline_logger)
    
    # Add current date (using Eastern Time)
    et_tz = pytz.timezone("US/Eastern")
    current_date = datetime.datetime.now(et_tz).date()

    positions_df = get_orders_and_positions(ib, pipeline_logger)

    final_df = selected_tickers.copy()
    final_df["date"] = current_date
    final_df.rename(columns={"price": "price_at_1555"}, inplace=True)

    positions_data = {
        "ticker": positions_df["symbol"],
        "quantity_filled": positions_df["quantity"],
        "position_size_filled": positions_df["market_value"],
        "avg_cost": positions_df["avg_cost"]
    }

    positions_df_formatted = pd.DataFrame(positions_data)

    # Merge with selected_tickers
    final_df = pd.merge(
        final_df,
        positions_df_formatted,
        on="ticker",
        how="left"
    )

    closes_df = get_official_closes(ib, final_df['ticker'].unique())
    final_df = pd.merge(final_df, closes_df, on='ticker', how='left')

    columns_order = [
        "date", "ticker", "score", "price_at_1555", "official_close","avg_cost", "quantity", 
        "quantity_filled", "position_size", "position_size_filled"
    ]

    final_df = final_df[columns_order]

        
    # Append to existing CSV
    if df_file.exists():
        existing_df = pd.read_csv(df_file)
        # Remove any existing entries for today to avoid duplicates
        existing_df = existing_df[existing_df["date"] != str(current_date)]
        combined_df = pd.concat([existing_df, final_df], ignore_index=True)
        combined_df.to_csv(df_file, index=False)

    else:
        final_df.to_csv(df_file, index=False)
    

if __name__ == "__main__":
    ib = connect_to_ib()
    selected_tickers = pd.read_parquet("/root/overnight/debug_data/20250221/selected_tickers.parquet")
    saved_filled_positions_report(ib, selected_tickers)
    #save_today_net_liquidation(ib)
    #portfolio_df = fetch_portfolio_dataframe(ib)
    #print(portfolio_df)
    #cash_value = get_available_cash(ib)
    #print(cash_value)
    ib.disconnect()