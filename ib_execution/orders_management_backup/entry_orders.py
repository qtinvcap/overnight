import logging
from pathlib import Path
import os
from datetime import datetime
import pandas as pd
import numpy as np
from ib_insync import Stock, Order, TagValue
import pytz

class EntryOrdersManager:
    def __init__(self, ib_connection):
        """
        Initialize the entry orders manager.
        
        Args:
            ib_connection: Active IBConnection instance
        """
        self.ib = ib_connection
        
        # Setup logging
        root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
        log_dir = Path(root_path) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("entry_orders")
        if not self.logger.handlers:
            log_file = log_dir / f"entry_orders_{datetime.now().strftime('%Y%m%d')}.log"
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        # Analysis storage
        self.save_dir = os.path.join(root_path, 'ib_execution/saved_daily_analysis')
        os.makedirs(self.save_dir, exist_ok=True)

    def cancel_all_orders(self):
        """
        Cancel all pending orders.
        
        Returns:
            int: Number of orders cancelled
        """
        cancelled_count = 0
        open_trades = self.ib.reqAllOpenOrders()
        
        if not open_trades:
            self.logger.info("No open orders to cancel.")
            return 0
        
        active_statuses = ['Submitted', 'PreSubmitted', 'PendingSubmit', 'PendingCancel']
        
        for trade in open_trades:
            try:
                if trade.orderStatus.status in active_statuses and trade.order.orderId != 0:
                    self.logger.info(f"Cancelling order {trade.order.orderId} for {trade.contract.symbol}")
                    self.ib.cancelOrder(trade.order)
                    cancelled_count += 1
            except Exception as e:
                self.logger.error(f"Failed to cancel order for {trade.contract.symbol}: {str(e)}")
                
        self.logger.info(f"Cancelled {cancelled_count} pending orders")
        return cancelled_count

    def get_current_positions_and_orders(self):
        """
        Get current positions and orders status.
        
        Returns:
            tuple: (orders_df, positions_df)
        """
        # Get open orders
        open_orders = self.ib.reqAllOpenOrders()
        orders_data = []
        
        for order in open_orders:
            if order.orderStatus.status in ['Cancelled', 'ApiCancelled']:
                continue
                
            orders_data.append({
                'symbol': order.contract.symbol,
                'status': order.orderStatus.status,
                'quantity': order.order.totalQuantity,
                'order_type': order.order.orderType,
                'action': order.order.action,
                'filled': order.orderStatus.filled,
                'remaining': order.orderStatus.remaining,
                'avg_fill_price': order.orderStatus.avgFillPrice,
                'order_id': order.order.orderId
            })
        
        orders_df = pd.DataFrame(orders_data)
        
        # Get positions
        positions = self.ib.positions()
        positions_data = []
        
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
        
        return orders_df, positions_df

    def place_closing_price_orders(self, orders_list, max_percentage=5, risk_aversion="Aggressive", start_time=None):
        """
        Place Close Price algorithm orders for the selected tickers.
        
        Args:
            orders_list: List of dicts with keys: ticker, quantity, action
            max_percentage: Max percentage of avg daily volume (0.01 to 50)
            risk_aversion: "Get Done", "Aggressive", "Neutral", or "Passive"
            start_time: Optional start time in format "HH:MM:SS US/Eastern"
            
        Returns:
            int: Number of orders successfully placed
        """
        orders_placed = 0
        order_tracking = {}
        
        # If no start time provided, use current time
        if start_time is None:
            current_time = datetime.now(pytz.timezone('US/Eastern'))
            start_time = current_time.strftime('15:58:00 US/Eastern')

            self.logger.info(f"No start time provided. Starting immediately at {start_time}")

        self.logger.info(f"Placing Close Price orders for {len(orders_list)} positions...")
        self.logger.info(f"Algorithm parameters: max_pct_vol={max_percentage}%, risk_aversion={risk_aversion}")
        
        for order_info in orders_list:
            try:
                ticker = order_info['ticker']
                quantity = order_info['quantity']
                action = order_info.get('action', 'BUY')
                
                # Initialize tracking for this order
                order_tracking[ticker] = {
                    'status': 'pending',
                    'error': None,
                    'order_id': None,
                    'details': {
                        'quantity': quantity,
                        'action': action,
                        'algo_params': {
                            'maxPctVol': max_percentage/100,
                            'riskAversion': risk_aversion,
                            'startTime': start_time,
                            'forceCompletion': True
                        }
                    }
                }
                
                # Create contract
                contract = Stock(ticker, 'SMART', 'USD')
                qualified_contracts = self.ib.qualifyContracts(contract)
                if not qualified_contracts:
                    error_msg = f"Could not qualify contract for {ticker}"
                    self.logger.error(error_msg)
                    order_tracking[ticker].update({'status': 'failed', 'error': error_msg})
                    continue
                
                contract = qualified_contracts[0]
                
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
                self.logger.info(f"Close Price Parameters for {ticker}: {params_str}")
                
                trade = self.ib.placeOrder(contract, order)
                orders_placed += 1
                
                # Update tracking with success
                order_tracking[ticker].update({
                    'status': 'placed',
                    'order_id': trade.order.orderId
                })
                
                self.logger.info(f"Successfully placed Close Price order for {ticker}: {action} {quantity} shares (Order ID: {trade.order.orderId})")
                self.ib.sleep(0.1)  # Small delay between orders
                
            except Exception as e:
                error_msg = f"Error placing Close Price order for {ticker}: {str(e)}"
                self.logger.error(error_msg)
                order_tracking[ticker].update({
                    'status': 'failed',
                    'error': str(e)
                })
                continue
        
        # Log summary
        successful_orders = sum(1 for info in order_tracking.values() if info['status'] == 'placed')
        failed_orders = sum(1 for info in order_tracking.values() if info['status'] == 'failed')
        
        self.logger.info(f"\nOrder Placement Summary:")
        self.logger.info(f"Total orders attempted: {len(orders_list)}")
        self.logger.info(f"Successfully placed: {successful_orders}")
        self.logger.info(f"Failed to place: {failed_orders}")
        
        # Log any failures in detail
        if failed_orders > 0:
            self.logger.error("\nFailed Orders Details:")
            for ticker, info in order_tracking.items():
                if info['status'] == 'failed':
                    self.logger.error(f"{ticker}: {info['error']}")
        
        return orders_placed, order_tracking

    def verify_order_placement(self, order_tracking):
        """
        Verify that all orders were properly placed and accepted.
        
        Args:
            order_tracking: Dict mapping ticker to order_id
            
        Returns:
            tuple: (all_accepted, placement_status)
                - all_accepted (bool): Whether all orders were accepted
                - placement_status (pd.DataFrame): Status of each order
        """
        placement_status = []
        trades = {trade.order.orderId: trade for trade in self.ib.trades()}  # Convert to dict for lookup
        
        for ticker, order_id in order_tracking.items():
            trade = trades.get(order_id)  # Use dict.get() instead of list.get()
            if trade:
                status = trade.orderStatus.status
                placement_status.append({
                    'ticker': ticker,
                    'order_id': order_id,
                    'status': status,
                    'message': trade.orderStatus.whyHeld or ''
                })
            else:
                placement_status.append({
                    'ticker': ticker,
                    'order_id': order_id,
                    'status': 'Unknown',
                    'message': 'Order not found'
                })
        
        status_df = pd.DataFrame(placement_status)
        
        # Check if all orders were accepted
        accepted_statuses = ['Submitted', 'PreSubmitted', 'Filled']
        all_accepted = all(status in accepted_statuses for status in status_df['status'])
        
        if all_accepted:
            self.logger.info("All orders successfully placed and accepted")
        else:
            self.logger.warning("Some orders were not accepted:")
            for _, row in status_df[~status_df['status'].isin(accepted_statuses)].iterrows():
                self.logger.warning(f"{row['ticker']}: {row['status']} - {row['message']}")
        
        return all_accepted, status_df

    def get_final_fill_analysis(self, original_orders):
        """
        Get final fill analysis after market close.
        
        Args:
            original_orders: List of original order requests
            
        Returns:
            pd.DataFrame: Final fill analysis
        """
        trades = self.ib.trades()
        fills = []
        
        # Process all filled trades
        for trade in trades:
            if trade.orderStatus.status == "Filled":
                fills.append({
                    'ticker': trade.contract.symbol,  # Make sure we use 'ticker' consistently
                    'filled_quantity': trade.orderStatus.filled,
                    'avg_fill_price': trade.orderStatus.avgFillPrice,
                    'fill_time': trade.log[-1].time if trade.log else None,
                    'order_id': trade.order.orderId
                })
        
        # Create DataFrames
        orders_df = pd.DataFrame(original_orders)
        fills_df = pd.DataFrame(fills)
        
        if fills_df.empty:
            self.logger.warning("No fills found")
            # Return original orders with zero fills
            orders_df['filled_quantity'] = 0
            orders_df['avg_fill_price'] = None
            orders_df['fill_time'] = None
            orders_df['fill_percentage'] = 0
            return orders_df
        
        # Merge on ticker
        analysis = orders_df.merge(
            fills_df,
            on='ticker',
            how='left'
        )
        
        # Fill NaN values for orders that weren't filled
        analysis['filled_quantity'] = analysis['filled_quantity'].fillna(0)
        analysis['avg_fill_price'] = analysis['avg_fill_price'].fillna(0)
        
        # Add fill percentage
        analysis['fill_percentage'] = (analysis['filled_quantity'] / analysis['quantity'] * 100).round(2)
        
        # Save analysis
        date_str = datetime.now().strftime('%Y_%m_%d')
        filename = f"{date_str}_entry_fills_analysis.csv"
        filepath = os.path.join(self.save_dir, filename)
        analysis.to_csv(filepath)
        
        # Log summary
        self.logger.info("\nFinal Fill Analysis:")
        self.logger.info(f"Total orders: {len(original_orders)}")
        self.logger.info(f"Fully filled orders: {len(analysis[analysis['fill_percentage'] == 100])}")
        self.logger.info(f"Partially filled orders: {len(analysis[(analysis['fill_percentage'] > 0) & (analysis['fill_percentage'] < 100)])}")
        self.logger.info(f"Unfilled orders: {len(analysis[analysis['fill_percentage'] == 0])}")
        
        # Log details for non-fully-filled orders
        non_full_fills = analysis[analysis['fill_percentage'] < 100]
        if not non_full_fills.empty:
            self.logger.info("\nOrders not fully filled:")
            for _, row in non_full_fills.iterrows():
                self.logger.info(f"{row['ticker']}: {row['fill_percentage']}% filled ({row['filled_quantity']}/{row['quantity']} shares)")
        
        return analysis