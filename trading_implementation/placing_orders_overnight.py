import pandas as pd
import numpy as np
import math
import logging
from utils import is_near_market_close, IBConnection
from ib_insync import IB, Stock, MarketOrder, Order, TagValue
from datetime import datetime
import time
import pytz


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TradingBot:
    def __init__(self, ib):
        self.ib = ib
        self.active_trades = {}
        self.retry_limit = 3  # Define max retries for placing orders
        self.lim_nb_minutes_before_market_close = 5
        self.max_wait_minutes = 10. # time to wait for orders to be filled
        self.nb_minutes_of_twap_order = 5. # number of minutes to wait for the VWAP order to be filled

    def is_within_limit_time_before_market_close(self):
        is_near_close, time_to_close = is_near_market_close(self.ib, self.lim_nb_minutes_before_market_close)
        if is_near_close:
            logging.info(f"Market close in {time_to_close} minutes. Placing orders...")
            return True
        else:
            logging.info(f"Market close in {time_to_close} minutes. Not placing orders...")
            return False

    def on_order_status(self, orderId, status, filled, remaining, avgFillPrice,
                       permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        logging.info(f"Order Status: {status} for Order ID: {orderId}")
        logging.info(f"Filled: {filled}, Remaining: {remaining}, Avg Fill Price: {avgFillPrice}")

        if status in ["Cancelled", "ApiCancelled", "Rejected"]:
            logging.error(f"Order {orderId} was not successful: {status}")

    def handle_order_failure(self, trade, error):
        # Custom logic based on error type or order details
        if "insufficient funds" in str(error).lower():
            logging.warning("Insufficient funds to place order. Adjusting trade size.")
            # Adjust trade size or skip
        elif "market closed" in str(error).lower():
            logging.warning("Market is closed. Will try again later.")

    def run(self, orders_list, twap=False, monitor=True):
        """
        Run the VWAP order placement process
        orders_list: list of dicts with keys: ticker, quantity, action
        """
        # Place orders
        if twap:
            logging.info("Starting TWAP order placement...")
            success = self.place_twap_orders(orders_list)
        else:
            logging.info("Starting Close Price order placement...")
            success = self.place_close_price_orders(orders_list, risk_aversion="Aggressive")
        
        if not success:
            logging.error("Failed to place TWAP orders")
            return False
        
        # Monitor orders until they're all filled or cancelled
        if monitor:
            logging.info("Monitoring orders...")
            nb_orders_cancelled = self.monitor_orders()  # This function name can stay the same as it's generic
            logging.info(f"{len(orders_list) - nb_orders_cancelled} / {len(orders_list)} TWAP orders processed")

        # current portfolio status
        tickers_open_trades, tickers_positions = self.check_trades_and_positions()
        orders_list_tickers = np.array([order['ticker'] for order in orders_list])
        for ticker in orders_list_tickers[~np.isin(orders_list_tickers, tickers_positions)]:
            logging.info(f"Order {ticker} not filled")

    def place_twap_order(self, ticker, quantity, action="BUY"):
        """Place a TWAP order for a given ticker starting now and ending 30 seconds later"""
        try:
            contract = Stock(ticker, 'SMART', 'USD')

            # Calculate start (now) and end time (self.nb_minutes_of_twap_order seconds later)
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            end_time = current_time + pd.Timedelta(minutes=self.nb_minutes_of_twap_order)
            
            # Format times in IB's expected format
            start_time = current_time.strftime('%H:%M:%S US/Eastern')
            end_time = end_time.strftime('%H:%M:%S US/Eastern')
            
            order = Order()
            order.orderType = "MKT"
            order.action = action
            order.totalQuantity = quantity
            order.tif = "DAY"
            order.algoStrategy = "Twap"  # Changed from Vwap to Twap
            order.algoParams = []
            order.algoParams.append(TagValue("startTime", start_time))
            order.algoParams.append(TagValue("endTime", end_time))
            order.algoParams.append(TagValue("allowPastEndTime", "0"))  # Allow to complete after end time if necessary
                    
            trade = self.ib.placeOrder(contract, order)
            
            self.active_trades[order.orderId] = {
                "trade": trade,
                "filled": False,
                "ticker": ticker,
                "quantity": quantity,
                "action": action
            }
            
            logging.info(f"Placed TWAP order for {ticker}: {action} {quantity} shares, starting at {start_time}, ending at {end_time}")
            return trade
            
        except Exception as e:
            logging.error(f"Error placing TWAP order for {ticker}: {str(e)}")
            self.handle_order_failure(None, e)
            return None

    def place_twap_orders(self, orders_list):
        """Place TWAP orders for a list of tickers"""
        for order in orders_list:
            ticker = order['ticker']
            quantity = order['quantity']
            action = order.get('action', 'BUY')
            
            self.place_twap_order(ticker, quantity, action)
            self.ib.sleep(0.1)  # Small delay between orders
        
        return True

    def robust_cancel_all_pending_orders(self, max_retries=3):
        """Cancel all pending orders with retries"""
        cancelled_count = 0
        retry_count = 0
        active_statuses = ['Submitted', 'PreSubmitted', 'PendingSubmit', 'PendingCancel']
        
        open_trades = self.ib.reqAllOpenOrders()
        while retry_count < max_retries:

            if not open_trades:
                logging.info("No open orders to cancel.")
                return 0
            
            for trade in open_trades:
                try:
                    if trade.orderStatus.status in active_statuses:
                        if trade.order.orderId != 0:  # Skip orders with ID 0
                            logging.info(f"Cancelling order {trade.order.orderId} for {trade.contract.symbol} "
                                       f"with status: {trade.orderStatus.status}")
                            self.ib.cancelOrder(trade.order)
                            cancelled_count += 1
                            self.ib.sleep(0.1)  # Small delay between cancellations
                        else:
                            logging.warning(f"Skipping order with ID 0 for {trade.contract.symbol}")
                except Exception as e:
                    logging.warning(f"Could not cancel order for {trade.contract.symbol}: {str(e)}")
                    continue
            
            # Check if there are still active orders
            # Wait a moment for orders to be registered
            self.ib.sleep(1)
            
            open_trades = self.ib.reqAllOpenOrders()
            if not any(trade.orderStatus.status in active_statuses for trade in open_trades):
                logging.info("Successfully cancelled all active orders")
                break
            
            retry_count += 1
            if retry_count < max_retries:
                logging.warning(f"Some orders still active, retrying... Attempt {retry_count + 1}/{max_retries}")
        
        logging.info(f"Cancelled {cancelled_count} pending orders")
        return cancelled_count

    def check_trades_and_positions(self):
        # Check open trades (pending orders)
        open_trades = self.ib.reqAllOpenOrders() # Retrieve all open trades 

        tickers_open_trades = []
        tickers_positions = []

        logging.info("\nPending Orders:")
        if not open_trades:
            logging.info("No pending orders")
        else:
            for trade in open_trades:
                logging.info(f"Order: {trade.contract.symbol}, Status: {trade.orderStatus.status}")
                tickers_open_trades.append(trade.contract.symbol)

        # Check positions (what you own)
        positions = self.ib.positions()
        logging.info("\nCurrent Positions:")
        if not positions:
            logging.info("No positions")
        else:
            for position in positions:
                logging.info(f"Symbol: {position.contract.symbol}, Quantity: {position.position}, Avg Cost: {position.avgCost}")                
                tickers_positions.append(position.contract.symbol)

        return tickers_open_trades, tickers_positions

    def monitor_orders(self):
        """Monitor the status of all orders and cancel any remaining after specified time        
        """
        start_time = time.time()
        max_wait_seconds = self.max_wait_minutes * 60
        nb_orders_cancelled = 0
        
        while self.active_trades:
            # Check if we've exceeded the maximum wait time
            if time.time() - start_time > max_wait_seconds:
                logging.warning(f"Reached maximum wait time of {self.max_wait_minutes} minutes. Cancelling remaining orders.")
                nb_orders_cancelled = self.robust_cancel_all_pending_orders()
                break
                
            for order_id, trade_info in list(self.active_trades.items()):
                trade = trade_info['trade']
                if trade.orderStatus.status in ['Filled', 'Cancelled', 'ApiCancelled']:
                    logging.info(f"TWAP order for {trade_info['ticker']} {trade.orderStatus.status}: "
                               f"{trade_info['action']} {trade_info['quantity']} shares")
                    del self.active_trades[order_id]
            
            if self.active_trades:
                self.ib.sleep(1)

        return nb_orders_cancelled
    
    def close_all_positions_at_open(self):
        """Close all positions at market open"""
        self.close_all_positions(tif='OPG')

    def robust_close_all_positions(self, max_retries=3, wait_time=1.0):
        """Repeatedly attempt to close all positions until successful or max retries reached"""
        positions = self.ib.positions()
        retry_count = 0

        while positions and retry_count < max_retries:
            # close all positions
            self.close_all_positions(tif='DAY')
            self.ib.sleep(wait_time)  # wait for orders to process
            
            # Check remaining positions
            positions = self.ib.positions()
            retry_count += 1
            
            if positions:
                logging.warning(f"Attempt {retry_count}/{max_retries}: Still have {len(positions)} positions open")
            else:
                logging.info("Successfully closed all positions")
                return True

        if positions:
            logging.error(f"Failed to close all positions after {max_retries} attempts. {len(positions)} positions remaining")
            return False
        
        return True

    def close_all_positions(self, tif=None):
        """Close all open positions using market orders"""
        orders_placed = 0
        
        positions = self.ib.positions()
        if not positions:
            logging.info("No positions to close")
            return 0
        
        for position in positions:
            try:
                contract = position.contract
                contract.exchange = 'SMART'
                quantity = abs(position.position)
                # If position is positive, we need to sell; if negative, we need to buy
                action = "SELL" if position.position > 0 else "BUY"
                
                # Create market order to close position
                order = Order()
                order.orderType = "MKT"
                order.action = action
                order.totalQuantity = quantity
                order.tif = tif #"DAY"
                
                trade = self.ib.placeOrder(contract, order)
                orders_placed += 1
                
                logging.info(f"Closing position for {contract.symbol}: {action} {quantity} shares at market")
                
                # Small delay between orders
                self.ib.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error closing position for {contract.symbol}: {str(e)}")
                continue
        
        logging.info(f"Placed {orders_placed} orders to close positions")
        return orders_placed

    def place_close_price_orders(self, orders_list, max_percentage=5, risk_aversion="Neutral", start_time='15:58:00 US/Eastern'):
        """Place Close Price algorithm orders
        
        Args:
            orders_list: list of dicts with keys: ticker, quantity, action
            max_percentage: max percentage of avg daily volume (0.01 to 50)
            risk_aversion: "Get Done", "Aggressive", "Neutral", or "Passive"
            start_time: optional start time in format "HH:MM:SS US/Eastern". If None, starts immediately
        """
        orders_placed = 0
        
        # If no start time provided, use current time
        if start_time is None:
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            start_time = current_time.strftime('%H:%M:%S US/Eastern')
            logging.info(f"No start time provided. Starting immediately at {start_time}")
        
        for order_info in orders_list:
            try:
                ticker = order_info['ticker']
                quantity = order_info['quantity']
                action = order_info.get('action', 'BUY')
                
                # Create contract
                contract = Stock(ticker, 'SMART', 'USD')
                
                # Create Close Price algo order
                order = Order()
                order.orderType = "MKT"
                order.action = action
                order.totalQuantity = quantity
                order.tif = "DAY"
                order.algoStrategy = "ClosePx"
                
                # Set algo parameters
                order.algoParams = []
                order.algoParams.append(TagValue("maxPctVol", str(max_percentage/100)))
                order.algoParams.append(TagValue("riskAversion", risk_aversion))
                order.algoParams.append(TagValue("startTime", start_time))
                order.algoParams.append(TagValue("forceCompletion", "1"))
                
                # Print parameters being used
                params_str = ', '.join([f"{param.tag}={param.value}" for param in order.algoParams])
                logging.info(f"Close Price Parameters: {params_str}")
                
                trade = self.ib.placeOrder(contract, order)
                orders_placed += 1
                
                self.active_trades[order.orderId] = {
                    "trade": trade,
                    "filled": False,
                    "ticker": ticker,
                    "quantity": quantity,
                    "action": action
                }
                
                logging.info(f"Placed Close Price order for {ticker}: {action} {quantity} shares")
                self.ib.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error placing Close Price order for {ticker}: {str(e)}")
                continue
        
        logging.info(f"Placed {orders_placed} Close Price orders")
        return orders_placed

    def get_portfolio_summary(self):
        """Retrieve and display detailed portfolio information using last auction closing prices"""
        positions = self.ib.positions()
        portfolio = []
        
        if not positions:
            logging.info("No positions in portfolio")
            return portfolio
            
        total_market_value = 0
        
        # Check if market is open
        current_time = datetime.now(pytz.timezone('US/Eastern'))
        market_open = (
            current_time.hour >= 9 and current_time.hour < 16 or 
            (current_time.hour == 9 and current_time.minute >= 30)
        )
        
        for position in positions:
            try:
                contract = position.contract
                
                # Request historical data for the last 2 days
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='2 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                
                if not bars or len(bars) < 2:
                    logging.error(f"Could not get enough historical data for {contract.symbol}")
                    continue
                
                # Use appropriate bar based on market status
                auction_close = bars[-2].close if market_open else bars[-1].close
                bar_date = bars[-2].date if market_open else bars[-1].date
                
                # Calculate position values
                quantity = position.position
                avg_cost = position.avgCost
                market_value = abs(quantity * auction_close)
                unrealized_pnl = (auction_close - avg_cost) * quantity
                pnl_percentage = ((auction_close / avg_cost) - 1) * 100 * (1 if quantity > 0 else -1)
                
                position_info = {
                    'ticker': contract.symbol,
                    'position': quantity,
                    'avg_cost': avg_cost,
                    'auction_close': auction_close,
                    'auction_date': bar_date,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'pnl_percentage': pnl_percentage
                }
                
                portfolio.append(position_info)
                total_market_value += market_value                
                
                self.ib.sleep(0.1)  # Small delay between requests
                
            except Exception as e:
                logging.error(f"Error processing position for {contract.symbol}: {str(e)}")
                continue
        
        logging.info(f"\nTotal Portfolio Market Value: ${total_market_value:,.2f}")
        return pd.DataFrame(portfolio)


if __name__ == "__main__":

    ib = IBConnection.get_instance()
    try:
        bot = TradingBot(ib)
        
        # Example orders list
        """
        orders = [
            {'ticker': 'META', 'quantity': 40, 'action': 'BUY'}, #AMZN
            {'ticker': 'AMD', 'quantity': 100, 'action': 'BUY'}, #NVDA
            {'ticker': 'CLNE', 'quantity': 5000, 'action': 'BUY'},#FSI
            {'ticker': 'PEIX', 'quantity': 200, 'action': 'BUY'},#SSL
            {'ticker': 'FCEL', 'quantity': 500, 'action': 'BUY'},#GEVO
        ]
        """
        orders = [
            {'ticker': 'XCH',   'quantity': 4167,   'action': 'BUY', 'times_present': 1},
            {'ticker': 'ADGM',  'quantity': 38286,  'action': 'BUY', 'times_present': 4},
            {'ticker': 'ADD',   'quantity': 3846,   'action': 'BUY', 'times_present': 1},
            {'ticker': 'ZCAR',  'quantity': 13776,  'action': 'BUY', 'times_present': 3},
            {'ticker': 'SMX',   'quantity': 1013,   'action': 'BUY', 'times_present': 1},
            {'ticker': 'GTI',   'quantity': 17477,  'action': 'BUY', 'times_present': 2},
            {'ticker': 'UAVS',  'quantity': 2857,   'action': 'BUY', 'times_present': 1},
            {'ticker': 'GALT',  'quantity': 16448,  'action': 'BUY', 'times_present': 2},
            {'ticker': 'GOEV',  'quantity': 13793,  'action': 'BUY', 'times_present': 2},
            {'ticker': 'MULN',  'quantity': 17424,  'action': 'BUY', 'times_present': 2},
            {'ticker': 'TPIC',  'quantity': 5291,   'action': 'BUY', 'times_present': 1},
        ]
        
        bot.run(orders, monitor=False)

        portfolio_summary = bot.get_portfolio_summary()

        merged_df = pd.DataFrame(orders).merge(portfolio_summary, on='ticker', how='inner').drop(columns=['unrealized_pnl', 'pnl_percentage', 'market_value', "action"])
        merged_df['diff_w_close_price'] = ((merged_df['avg_cost'] -  merged_df['auction_close']) / merged_df['auction_close'])
        
        #bot.robust_close_all_positions()
        # after market is close
        #bot.check_trades_and_positions()
        #bot.cancel_all_pending_orders()
        #bot.close_all_positions_at_open()
    finally:
        ib.disconnect()
        logging.info("Disconnected from IB")
