import pandas as pd
import numpy as np
import math
import logging
from ib_execution.utils import get_market_schedule, IBConnection, gather_nbbo_liquidity_metrics
from ib_insync import IB, Stock, MarketOrder, Order, TagValue, Forex
from datetime import datetime
import time
import pytz
import os
from pathlib import Path
import sys

class TradingBot:
    def __init__(self, ib):
        self.ib = ib
        # Create directory if it doesn't exist
        self.save_dir = os.getenv("OVERNIGHT_ROOT_PATH") + '/ib_execution/saved_daily_analysis'
        os.makedirs(self.save_dir, exist_ok=True)  

        # Setup logging for the pipeline
        root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
        log_dir = Path(root_path) / "logs"
        log_dir.mkdir(exist_ok=True)
        pipeline_log_file = log_dir / f"ib_execution_{datetime.now().strftime('%Y%m%d')}.log"

        self.pipeline_logger = logging.getLogger("ib_execution")
        self.pipeline_logger.setLevel(logging.INFO)
        self.pipeline_logger.handlers = []

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(pipeline_log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        self.pipeline_logger.addHandler(file_handler)
        self.pipeline_logger.addHandler(console_handler)

        self.pipeline_logger.info("===== Starting IB Execution =====")


    def is_market_open(self):
        """
        Check if market is open
        
        Returns:
            tuple: (is_near_close, time_to_close, is_market_open)
                - is_market_open (bool): True if market is currently open
        """
        
        # Get market status from schedule
        contract = Stock('SPY', 'SMART', 'USD')
        schedule = get_market_schedule(self.ib, contract)
        is_market_open = schedule['is_open']
      
        if not is_market_open:
            self.pipeline_logger.info(f"Market is currently {'open' if is_market_open else 'closed'}")
        
        return is_market_open

    def on_order_status(self, orderId, status, filled, remaining, avgFillPrice,
                       permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.pipeline_logger.info(f"Order Status: {status} for Order ID: {orderId}")
        self.pipeline_logger.info(f"Filled: {filled}, Remaining: {remaining}, Avg Fill Price: {avgFillPrice}")

        if status in ["Cancelled", "ApiCancelled", "Rejected"]:
            self.pipeline_logger.error(f"Order {orderId} was not successful: {status}")

    def handle_order_failure(self, trade, error):
        # Custom logic based on error type or order details
        if "insufficient funds" in str(error).lower():
            self.pipeline_logger.warning("Insufficient funds to place order. Adjusting trade size.")
            # Adjust trade size or skip
        elif "market closed" in str(error).lower():
            self.pipeline_logger.warning("Market is closed. Will try again later.")

    def place_entry_orders(self, orders_list, time_to_wait_before_cancel=-1):
        """
        close price order placement process
        orders_list: list of dicts with keys: ticker, quantity, action
        time_to_wait_before_cancel: int, minutes to wait before canceling orders. if -1, there is no cancellation
        if None, will calculate time to wait until market close + 30 seconds
        """
        # Place orders
        self.pipeline_logger.info("Starting Close Price order placement...")

        if len(orders_list) == 0:
            self.pipeline_logger.info("No orders to place. Skipping close price order placement.")
            return True
        
        success = self.place_close_price_orders(orders_list, risk_aversion="Aggressive", start_time=None) # GetDone # Aggressive # Neutral
        
        if not success:
            self.pipeline_logger.error("Failed to place close price orders")
            return False

        # Calculate time to wait if None
        if time_to_wait_before_cancel is None:
            current_time = pd.Timestamp.now(tz='America/New_York')
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Calculate minutes until market close + 30 seconds
            minutes_to_wait = (market_close - current_time).total_seconds() / 60 + 0.5
            time_to_wait_before_cancel = max(0, minutes_to_wait)
            
            self.pipeline_logger.info(f"Waiting until market close + 30 seconds ({time_to_wait_before_cancel:.1f} minutes)")
                
        # Monitor orders until they're all filled or cancelled
        if time_to_wait_before_cancel > 0:
            self.pipeline_logger.info("Monitoring orders...")
            nb_orders_cancelled, _ = self.monitor_orders(
                time_to_wait_before_cancel=time_to_wait_before_cancel
                )
            self.pipeline_logger.info(f"{len(orders_list) - nb_orders_cancelled} / {len(orders_list)} orders processed")

        # current portfolio status
        orders_df, positions_df = self.get_orders_and_positions()
        orders_list_tickers = np.array([order['ticker'] for order in orders_list])
        for ticker in orders_list_tickers[~np.isin(orders_list_tickers, positions_df['symbol'].values)]:
            self.pipeline_logger.info(f"Order {ticker} not filled")
        
        # save analysis
        df_entry_orders_analysis = self.analyze_entry_orders_execution(orders_list, save_analysis=True)

        return True
            

    def robust_cancel_all_orders(self, max_retries=3):
        """Cancel all pending orders with retries"""
        cancelled_count = 0
        retry_count = 0
        active_statuses = ['Submitted', 'PreSubmitted', 'PendingSubmit', 'PendingCancel']
        
        open_trades = self.ib.reqAllOpenOrders()
        while retry_count < max_retries:

            if not open_trades:
                self.pipeline_logger.info("No open orders to cancel.")
                return 0
            
            for trade in open_trades:
                try:
                    if trade.orderStatus.status in active_statuses:
                        if trade.order.orderId != 0:  # Skip orders with ID 0
                            self.pipeline_logger.info(f"Cancelling order {trade.order.orderId} for {trade.contract.symbol} "
                                       f"with status: {trade.orderStatus.status}")
                            self.ib.cancelOrder(trade.order)
                            cancelled_count += 1
                            self.ib.sleep(0.1)  # Small delay between cancellations
                        else:
                            self.pipeline_logger.warning(f"Skipping order with ID 0 for {trade.contract.symbol}")
                except Exception as e:
                    self.pipeline_logger.warning(f"Could not cancel order for {trade.contract.symbol}: {str(e)}")
                    continue
            
            # Check if there are still active orders
            # Wait a moment for orders to be registered
            self.ib.sleep(1)
            
            open_trades = self.ib.reqAllOpenOrders()
            if not any(trade.orderStatus.status in active_statuses for trade in open_trades):
                self.pipeline_logger.info("Successfully cancelled all active orders")
                break
            
            retry_count += 1
            if retry_count < max_retries:
                self.pipeline_logger.warning(f"Some orders still active, retrying... Attempt {retry_count + 1}/{max_retries}")
        
        self.pipeline_logger.info(f"Cancelled {cancelled_count} pending orders")
        return cancelled_count

    def get_tag_value(self, algo_params, tag_name):
        """Get value for a specific tag from algoParams
        
        Args:
            algo_params: List of TagValue objects
            tag_name: Name of the tag to find
        
        Returns:
            str: Value of the tag if found, None otherwise
        """
        for param in algo_params:
            if param.tag == tag_name:
                return param.value
        return None

    def get_orders_and_positions(self):
        """
        Check open trades and positions, returning organized DataFrames
        
        Returns:
            orders_df: DataFrame with pending orders info (symbol, status, quantity, order_type, etc.)
            positions_df: DataFrame with current positions info (symbol, quantity, avg_cost, market_value)
        
        Raises:
            ValueError: If cancelled orders appear in reqAllOpenOrders() results
        """
        # Check open trades (only returns active orders, not cancelled ones)
        open_orders = self.ib.reqAllOpenOrders()
        
        orders_data = []
        if open_orders:
            for order in open_orders:
                if order.orderStatus.status in ['Cancelled', 'ApiCancelled']:
                    raise ValueError(f"Unexpected cancelled order in reqAllOpenOrders(): {order.contract.symbol} (ID: {order.order.orderId})")
                
                orders_data.append({
                    'symbol': order.contract.symbol,
                    'status': order.orderStatus.status,
                    'quantity': order.order.totalQuantity,
                    'order_type': order.order.orderType,
                    'action': order.order.action,
                    'filled': order.orderStatus.filled,
                    'remaining': order.orderStatus.remaining,
                    'avg_fill_price': order.orderStatus.avgFillPrice,
                    'order_id': order.order.orderId,
                    'tif': order.order.tif,
                    'startTime': self.get_tag_value(order.order.algoParams, 'startTime'),
                    'riskAversion': self.get_tag_value(order.order.algoParams, 'riskAversion'),
                })
        
        orders_df = pd.DataFrame(orders_data)
        if not orders_df.empty:
            self.pipeline_logger.info(f"\nPending Orders:\n{orders_df}")
        else:
            self.pipeline_logger.info("\nNo pending orders")

        # Check positions
        positions = self.ib.positions()
        positions_data = []
        if positions:
            for position in positions:
                positions_data.append({
                    'symbol': position.contract.symbol,
                    'quantity': position.position,
                    'avg_cost': position.avgCost,
                    'market_value': abs(position.position * position.avgCost),
                    'currency': position.contract.currency,
                    'exchange': position.contract.exchange
                })
        
        positions_df = pd.DataFrame(positions_data)
        if not positions_df.empty:
            self.pipeline_logger.info(f"\nCurrent Positions:\n{positions_df}")
        else:
            self.pipeline_logger.info("\nNo positions")

        return orders_df, positions_df

    def monitor_orders(self, time_to_wait_before_cancel=5):
        """Monitor the status of all orders and cancel any remaining after specified time
        
        Args:
            time_to_wait_before_cancel: Minutes to wait before canceling remaining orders
            
        Returns:
            nb_orders_cancelled: Number of orders that were cancelled due to timeout
            final_orders_status: DataFrame containing final status of all orders
        """
        start_time = datetime.now()
        nb_orders_cancelled = 0
        
        self.pipeline_logger.info(f"Entered monitor_orders with time_to_wait_before_cancel: {time_to_wait_before_cancel} minutes")

        while True:
            # Check if timeout reached
            elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
            if elapsed_minutes >= time_to_wait_before_cancel:
                self.pipeline_logger.info(f"Timeout reached after {time_to_wait_before_cancel} minutes. Canceling remaining orders...")
                nb_orders_cancelled = self.robust_cancel_all_orders()
                break
                    
            # Get current orders status
            orders_df, _ = self.get_orders_and_positions()
            
            self.pipeline_logger.info(f"Checking if orders are pending...")
            # Log current status
            if not orders_df.empty:
                self.pipeline_logger.info(f"Orders pending: {len(orders_df)} orders")
                self.pipeline_logger.debug(f"Current order statuses:\n{orders_df[['symbol', 'status', 'filled', 'remaining']]}")
            
            # If no orders remaining, we're done
            if orders_df.empty:
                self.pipeline_logger.info("All orders completed")
                break            
            
            # Wait 10 seconds before next check
            self.ib.sleep(10)
        
        # Get final status
        final_orders_df, _ = self.get_orders_and_positions()
        
        return nb_orders_cancelled, final_orders_df

    def place_market_on_open_orders(self, position_tickers):
        """Place market orders on open"""        
        
        market_open_time = pd.Timestamp.now(tz='America/New_York').replace(hour=9, minute=30, second=0, microsecond=0)
        timeout_time = market_open_time + pd.Timedelta(seconds=30)
        
        self.pipeline_logger.info(f"Placing MOO orders and monitoring fills until {timeout_time.strftime('%H:%M:%S')} ET (30 seconds post-market open)")
        
        # Place MOO orders
        self.robust_cancel_all_orders()
        self._place_all_exit_orders(tif='OPG')
        
        # Wait until market open plus a small buffer (e.g., 30 seconds)
        while pd.Timestamp.now(tz='America/New_York') < timeout_time:
            self.ib.sleep(1)

        # Monitor executions
        timeout = 60  # 1 minute timeout
        start_time = datetime.now()

        # Then monitor for fills
        while True:
            current_time = datetime.now()
            timeout_reached = (current_time - start_time).total_seconds() > timeout
            
            # Get current session trades
            trades = self.ib.trades()
            all_opg_filled = True
            
            # Check if all OPG orders are filled
            for trade in trades:
                if (trade.order.tif == 'OPG' and 
                trade.contract.symbol in position_tickers and
                not trade.orderStatus.status == 'Filled'):
                    all_opg_filled = False
                    break
            
            if all_opg_filled or timeout_reached:
                break
            
            self.ib.sleep(1)
        
        market_on_open_analysis = []
        
        # Now collect the fill information
        trades = self.ib.trades()
        for trade in trades:
            if (trade.orderStatus.filled > 0 and 
                trade.order.tif == 'OPG' and 
                trade.contract.symbol in position_tickers):
                
                exec_time = pd.Timestamp(trade.log[-1].time).tz_convert('America/New_York')
                
                if trade.log[-1].time.date() == datetime.now().date():
                    market_on_open_analysis.append({
                        'symbol': trade.contract.symbol,
                        'exit_quantity': trade.orderStatus.filled,
                        'exit_price': trade.orderStatus.avgFillPrice,
                        'exit_time': trade.log[-1].time,
                        'status': trade.orderStatus.status,
                        'seconds_from_open': abs((exec_time - market_open_time).total_seconds()),
                        'tif': trade.order.tif,
                        'order_id': trade.order.orderId
                    })
        
        # Create analysis DataFrame
        market_on_open_analysis_df = pd.DataFrame(market_on_open_analysis).drop_duplicates(subset=['symbol', 'exit_quantity', 'exit_price'])
        if not market_on_open_analysis_df.empty:
            market_on_open_analysis_df = market_on_open_analysis_df.sort_values('seconds_from_open')

        return market_on_open_analysis_df, all_opg_filled

    def place_exit_orders(self):
        """Place exit orders"""

        positions_before = self.ib.positions()
        # Store initial position info
        initial_positions = []
        position_tickers = set()  # Add this to track valid tickers
        for pos in positions_before:
            ticker = pos.contract.symbol
            position_tickers.add(ticker)  # Add to set of valid tickers
            initial_positions.append({
                'ticker': ticker,
                'quantity': pos.position,
                'avg_entry_cost': pos.avgCost
            })

        if len(initial_positions) == 0:
            self.pipeline_logger.info("No positions detected. No exit orders placed.")
            return None

        market_on_open_analysis_df, all_opg_filled = self.place_market_on_open_orders(position_tickers)

        _, positionsdf = self.get_orders_and_positions()        

        if not all_opg_filled:
            self.pipeline_logger.warning("Not all OPG orders filled within the timeout period")
            success, market_orders_analysis_df = self.robust_exit_all_positions()
            if not success:
                self.pipeline_logger.error("Failed to close all positions on market orders when OPG orders were not filled")
                return None
        elif not positionsdf.empty:
            self.pipeline_logger.error("Something went wrong, positions were not closed but OPG orders were filled")
            return None
        else:
            market_orders_analysis_df = pd.DataFrame()
        
        # Merge MOO and market order analyses
        all_exits_df = pd.concat([
            market_on_open_analysis_df,
            market_orders_analysis_df
        ], ignore_index=True)
        
        # Check for duplicates
        duplicates = all_exits_df[all_exits_df.duplicated(subset=['ticker'], keep=False)]
        if not duplicates.empty:
            self.pipeline_logger.warning(f"Found unexpected duplicate exits for tickers:\n{duplicates}")
        
        # Merge with initial positions
        analysis = pd.DataFrame(initial_positions).merge(
            all_exits_df,
            on='symbol',
            how='outer'
        )
        
        # Calculate metrics
        if not analysis.empty:
            analysis['pnl'] = (analysis['exit_price'] - analysis['entry_price']) * analysis['exit_quantity']
            analysis['pnl_percentage'] = (analysis['exit_price'] - analysis['entry_price']) / analysis['entry_price']
            
            # Save analysis
            date_str = datetime.now().strftime('%Y_%m_%d')
            filename = f"{date_str}_exit_orders_execution_analysis.csv"
            filepath = os.path.join(self.save_dir, filename)
            analysis.to_csv(filepath)
            
            self.pipeline_logger.info("\nExit Orders Execution Analysis:")
            self.pipeline_logger.info(f"\n{analysis}")
            self.pipeline_logger.info(f"\nTotal P&L: ${analysis['pnl'].sum():.2f}")
        
        return analysis

    def robust_exit_all_positions(self, max_retries=3, wait_time=60.0):
        """Repeatedly attempt to close all positions until successful or max retries reached
        and cancel all orders. Wipes everything in the account, orders and positions.
        
        Returns:
            tuple: (success, analysis_df)
                - success (bool): Whether all positions were closed
                - analysis_df (pd.DataFrame): DataFrame containing entry/exit analysis
        """        
        positions = self.ib.positions()
        # Store initial position info
        position_tickers = set()  # Add this to track valid tickers
        for pos in positions:
            position_tickers.add(pos.contract.symbol)
        retry_count = 0

        # cancel all current orders
        self.robust_cancel_all_orders()

        if len(positions) == 0:
            self.pipeline_logger.info('No positions detected')
            return True, pd.DataFrame()

        while positions and retry_count < max_retries:
            # Get current orders status
            orders_df, positions_df = self.get_orders_and_positions()
            
            # Only place new orders for positions that don't have pending orders
            positions_needing_orders = []
            for position in positions:
                symbol = position.contract.symbol
                has_pending_order = (
                    not orders_df.empty and 
                    (orders_df['symbol'] == symbol).any()
                )
                if not has_pending_order:
                    positions_needing_orders.append(position)
            
            if positions_needing_orders:
                self.pipeline_logger.info(f"Placing new orders for {len(positions_needing_orders)} positions without pending orders")
                self._place_all_exit_orders(tif='DAY', positions_to_exit=positions_needing_orders)
            else:
                self.pipeline_logger.info("No positions without pending orders found")
            
            self.ib.sleep(wait_time)  # wait for orders to process
            positions = self.ib.positions()
            retry_count += 1
            
            if positions:
                self.pipeline_logger.warning(f"Attempt {retry_count}/{max_retries}: Still have {len(positions)} positions open")
            else:
                self.pipeline_logger.info("Successfully closed all positions")

        # Collect execution data
        trades = self.ib.trades()
        fills = []
        
        today = datetime.now().date()
        
        for trade in trades:
            # Filter for:
            # 1. Filled trades
            # 2. Today's trades
            # 3. DAY orders (from our exit attempts)
            # 4. Tickers from our original positions
            if (trade.log[-1].time.date() == today and
                trade.order.tif == 'DAY' and trade.orderStatus.filled > 0 and
                trade.contract.symbol in position_tickers):
                
                fills.append({
                    'symbol': trade.contract.symbol,
                    'exit_quantity': trade.orderStatus.filled,
                    'status': trade.orderStatus.status,
                    'exit_price': trade.orderStatus.avgFillPrice,
                    'exit_time': trade.log[-1].time,
                    'order_id': trade.order.orderId,
                    'tif': 'DAY'
                })
        
        # Create analysis DataFrame
        fills_df = pd.DataFrame(fills)
        if not fills_df.empty:
            # Keep only the last fill for each symbol (in case of multiple partial fills)
            fills_df = fills_df.sort_values('exit_time').groupby('symbol').last().reset_index()
        
        success = len(positions) == 0
        if not success:
            self.pipeline_logger.error(f"Failed to close all positions after {max_retries} attempts. {len(positions)} positions remaining")
        
        self.robust_cancel_all_orders()
        return success, fills_df

    def _place_all_exit_orders(self, tif=None, positions_to_exit=None):
        """Close all open positions using market orders on positions_to_exit if provided, otherwise close all positions"""
        orders_placed = 0
        
        positions = positions_to_exit or self.ib.positions()
        if not positions:
            self.pipeline_logger.info("No positions to close")
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
                order.tif = tif
                
                trade = self.ib.placeOrder(contract, order)
                orders_placed += 1
                
                self.pipeline_logger.info(f"Closing position for {contract.symbol}: {action} {quantity} shares at market")
                
                # Small delay between orders
                self.ib.sleep(0.1)
                
            except Exception as e:
                self.pipeline_logger.error(f"Error closing position for {contract.symbol}: {str(e)}")
                continue
        
        self.pipeline_logger.info(f"Placed {orders_placed} orders to close positions")
        return orders_placed

    def place_close_price_orders(self, orders_list, max_percentage=5, risk_aversion="Aggressive", start_time='15:58:00 US/Eastern'):
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
            self.pipeline_logger.info(f"No start time provided. Starting immediately at {start_time}")
        
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
                self.pipeline_logger.info(f"Close Price Parameters: {params_str}")
                
                trade = self.ib.placeOrder(contract, order)
                orders_placed += 1
                
                self.pipeline_logger.info(f"Placed Close Price order for {ticker}: {action} {quantity} shares")
                self.ib.sleep(0.1)
                
            except Exception as e:
                self.pipeline_logger.error(f"Error placing Close Price order for {ticker}: {str(e)}")
                continue
        
        self.pipeline_logger.info(f"Placed {orders_placed} Close Price orders")
        return orders_placed

    def get_historical_data(self, contract, market_open, quantity, avg_cost):
        """Get historical data for a ticker"""
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
            self.pipeline_logger.error(f"Could not get enough historical data for {contract.symbol}")
            return None
        
        # Use appropriate bar based on market status
        auction_close = bars[-2].close if market_open else bars[-1].close
        bar_date = bars[-2].date if market_open else bars[-1].date
        
        # Calculate position values
        market_value = abs(quantity * auction_close)
        
        position_info = {
            'ticker': contract.symbol,
            'position': quantity,
            'avg_cost': avg_cost,
            'auction_close': auction_close,
            'auction_date': bar_date,
            'market_value': market_value,
        }
        return position_info

    def analyze_entry_orders_execution(self, orders_to_place, save_analysis=False):
        """Retrieve and display detailed portfolio information using last auction closing prices"""
        positions = self.ib.positions()
        portfolio = []
        
        if not positions or len(orders_to_place) == 0:
            self.pipeline_logger.info("No positions in portfolio and orders_to_place is empty")
            return portfolio
            
        total_market_value = 0
        
        # Check if market is open
        current_time = datetime.now(pytz.timezone('US/Eastern'))
        market_open = (
            current_time.hour >= 9 and current_time.hour < 16 or 
            (current_time.hour == 9 and current_time.minute >= 30)
        )
        
        # Get set of tickers from orders
        order_tickers = {order['ticker'] for order in orders_to_place}
        position_tickers = set()
        
        for position in positions:
            try:
                position_tickers.add(position.contract.symbol)
                contract = position.contract

                position_info = self.get_historical_data(contract, market_open, position.position, position.avgCost)
                if position_info is not None:
                    portfolio.append(position_info)
                    total_market_value += position_info['market_value']
                
                self.ib.sleep(0.1)  # Small delay between requests
                
            except Exception as e:
                self.pipeline_logger.error(f"Error processing position for {contract.symbol}: {str(e)}")
                continue
        
        for ticker in order_tickers - position_tickers:
            try:
                position_info = self.get_historical_data(Stock(ticker, 'SMART', 'USD'), market_open, 0, np.nan)
                if position_info is not None:
                    portfolio.append(position_info)
                self.ib.sleep(0.1)  # Small delay between requests
            except Exception as e:
                self.pipeline_logger.error(f"Error processing position for {ticker}: {str(e)}")
                continue

        # Check for unexpected positions
        unexpected_positions = position_tickers - order_tickers
        if unexpected_positions:
            self.pipeline_logger.warning(f"Found positions for tickers not in original orders: {sorted(unexpected_positions)}")
        
        merged_df = pd.DataFrame(orders_to_place).merge(pd.DataFrame(portfolio), on='ticker', how='outer').drop(columns=['market_value', "action"])
        merged_df['diff_w_close_price'] = ((merged_df['avg_cost'] -  merged_df['auction_close']) / merged_df['auction_close'])

        if save_analysis:
            # Get the auction date from the first row (all rows should have same date)
            if not merged_df.empty and 'auction_date' in merged_df.columns:
                # Convert datetime.date to string in YYYY-MM-DD format
                
                date_str = merged_df['auction_date'].iloc[0].strftime('%Y-%m-%d').replace('-', '_')
                filename = f"{date_str}_entry_orders_execution_analysis.csv"
                filepath = os.path.join(self.save_dir, filename)
                
                # Save DataFrame
                merged_df.to_csv(filepath)
                self.pipeline_logger.info(f"Saved analysis to {filepath}")
            else:
                self.pipeline_logger.warning("Could not save analysis: DataFrame empty or missing auction_date column")
            
        return merged_df, unexpected_positions

    def get_available_cash(self):
        """Get available cash balance in USD and current positions/orders status
            This is the sum of all available cash in USD and EUR, assuming a 1. exchange rate
        Returns:
            float: Total available cash balance in USD (including EUR)
        """
        # Get account summary
        account_summary = self.ib.accountSummary()
        
        # Find available cash in USD and EUR
        usd_cash = 0
        eur_cash = 0
        
        for summary in account_summary:
            if summary.tag == 'AvailableFunds':
                if summary.currency == 'USD':
                    usd_cash = float(summary.value)
                elif summary.currency == 'EUR':
                    eur_cash = float(summary.value)
        
        # Sum that by assuming a 1. exchange rate
        total_cash = usd_cash + eur_cash
        
        return total_cash


if __name__ == "__main__":

    ib = IBConnection.get_instance(port=4002)
    try:
        bot = TradingBot(ib)
        
        # Example orders list
        orders_to_place = [
            {'ticker': 'FTAI',   'quantity': 108,   'action': 'BUY'},
            {'ticker': 'ALUR',  'quantity': 1176,  'action': 'BUY'},
            {'ticker': 'ARQQ',   'quantity': 365,   'action': 'BUY'},
            {'ticker': 'MBOT',  'quantity': 4903,  'action': 'BUY'},
            {'ticker': 'EVTL',  'quantity': 1805,  'action': 'BUY'},
            {'ticker': 'XCH',   'quantity': 4167,   'action': 'BUY'},
            {'ticker': 'ADGM',  'quantity': 9571,  'action': 'BUY'},
            {'ticker': 'ADD',   'quantity': 3846,   'action': 'BUY'},
            {'ticker': 'ZCAR',  'quantity': 4592,  'action': 'BUY'},
            {'ticker': 'SMX',   'quantity': 1013,   'action': 'BUY'},
            {'ticker': 'GTI',   'quantity': 8738,  'action': 'BUY'},
            {'ticker': 'UAVS',  'quantity': 2857,   'action': 'BUY'},
            {'ticker': 'GALT',  'quantity': 8224,  'action': 'BUY'},
            {'ticker': 'GOEV',  'quantity': 6896,  'action': 'BUY'},
            {'ticker': 'MULN',  'quantity': 8712,  'action': 'BUY'},
            {'ticker': 'TPIC',  'quantity': 5291,   'action': 'BUY'},
        ]
        
        available_cash = bot.get_available_cash()
        bot.place_entry_orders(orders_to_place, time_to_wait_before_cancel=5) # 5 minutes before market close
        bot.place_exit_orders() # before market opens
    finally:
        ib.disconnect()
        bot.pipeline_logger.info("Disconnected from IB")
