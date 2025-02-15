from ib_insync import IB, util, Stock
import pandas as pd
import datetime
import pytz
import logging
import pytz


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


if __name__ == "__main__":
    ib = connect_to_ib()
    portfolio_df = fetch_portfolio_dataframe(ib)
    print(portfolio_df)
    #cash_value = get_available_cash(ib)
    #print(cash_value)
    ib.disconnect()