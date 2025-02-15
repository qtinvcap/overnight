import logging
from pathlib import Path
import os
from datetime import datetime
import pandas as pd
from ib_insync import Stock, Order

class ExitOrdersManager:
    def __init__(self, ib_connection):
        """
        Initialize the exit orders manager.
        
        Args:
            ib_connection: Active IBConnection instance
        """
        self.ib = ib_connection
        
        # Setup logging
        root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
        log_dir = Path(root_path) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("exit_orders")
        if not self.logger.handlers:
            log_file = log_dir / f"exit_orders_{datetime.now().strftime('%Y%m%d')}.log"
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)

        # Analysis storage
        self.save_dir = os.path.join(root_path, 'ib_execution/saved_daily_analysis')
        os.makedirs(self.save_dir, exist_ok=True)

    def get_current_positions(self):
        """
        Get current positions.
        
        Returns:
            pd.DataFrame: Current positions
        """
        positions = self.ib.positions()
        positions_data = []
        
        for position in positions:
            positions_data.append({
                'symbol': position.contract.symbol,
                'quantity': position.position,
                'avg_cost': position.avgCost,
                'market_value': abs(position.position * position.avgCost),
                'currency': position.contract.currency,
                'exchange': position.contract.exchange,
                'contract': position.contract  # Store contract for order placement
            })
        
        return pd.DataFrame(positions_data)

    def place_market_on_open_orders(self):
        """
        Place Market-On-Open orders for all current positions.
        
        Returns:
            tuple: (orders_placed, orders_data)
                - orders_placed (int): Number of orders successfully placed
                - orders_data (list): List of order details
        """
        positions_df = self.get_current_positions()
        
        if positions_df.empty:
            self.logger.info("No positions to exit")
            return 0, []
            
        orders_placed = 0
        orders_data = []
        
        for _, position in positions_df.iterrows():
            try:
                contract = position['contract']
                quantity = abs(position['quantity'])
                action = "SELL" if position['quantity'] > 0 else "BUY"
                
                order = Order()
                order.orderType = "MKT"
                order.action = action
                order.totalQuantity = quantity
                order.tif = "OPG"  # Market-On-Open
                
                self.ib.placeOrder(contract, order)
                
                orders_data.append({
                    'symbol': position['symbol'],
                    'action': action,
                    'quantity': quantity,
                    'avg_entry_cost': position['avg_cost']
                })
                
                orders_placed += 1
                self.logger.info(f"Placed MOO order for {position['symbol']}: {action} {quantity} shares")
                
            except Exception as e:
                self.logger.error(f"Failed to place MOO order for {position['symbol']}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully placed {orders_placed} MOO orders")
        return orders_placed, orders_data

    def place_market_orders(self):
        """
        Place immediate market orders for all current positions.
        
        Returns:
            tuple: (orders_placed, orders_data)
        """
        positions_df = self.get_current_positions()
        
        if positions_df.empty:
            self.logger.info("No positions to exit")
            return 0, []
            
        orders_placed = 0
        orders_data = []
        
        for _, position in positions_df.iterrows():
            try:
                contract = position['contract']
                quantity = abs(position['quantity'])
                action = "SELL" if position['quantity'] > 0 else "BUY"
                
                order = Order()
                order.orderType = "MKT"
                order.action = action
                order.totalQuantity = quantity
                order.tif = "DAY"
                
                self.ib.placeOrder(contract, order)
                
                orders_data.append({
                    'symbol': position['symbol'],
                    'action': action,
                    'quantity': quantity,
                    'avg_entry_cost': position['avg_cost']
                })
                
                orders_placed += 1
                self.logger.info(f"Placed market order for {position['symbol']}: {action} {quantity} shares")
                
            except Exception as e:
                self.logger.error(f"Failed to place market order for {position['symbol']}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully placed {orders_placed} market orders")
        return orders_placed, orders_data

    def analyze_execution(self, initial_orders):
        """
        Analyze execution of exit orders.
        
        Args:
            initial_orders: List of dicts containing initial order details
            
        Returns:
            pd.DataFrame: Analysis of execution
        """
        trades = self.ib.trades()
        executions = []
        
        # Get today's date
        today = datetime.now().date()
        
        # Process all executions from today
        for trade in trades:
            if (trade.orderStatus.status == "Filled" and 
                trade.log and 
                trade.log[-1].time.date() == today):
                
                executions.append({
                    'symbol': trade.contract.symbol,
                    'exit_quantity': trade.orderStatus.filled,
                    'exit_price': trade.orderStatus.avgFillPrice,
                    'exit_time': trade.log[-1].time,
                    'order_type': trade.order.orderType,
                    'tif': trade.order.tif
                })
        
        # Create execution analysis DataFrame
        executions_df = pd.DataFrame(executions)
        
        if not executions_df.empty:
            # Merge with initial orders
            analysis = pd.DataFrame(initial_orders).merge(
                executions_df,
                on='symbol',
                how='outer'
            )
            
            # Calculate P&L
            analysis['pnl'] = (analysis['exit_price'] - analysis['avg_entry_cost']) * analysis['exit_quantity']
            analysis['pnl_percentage'] = (analysis['exit_price'] - analysis['avg_entry_cost']) / analysis['avg_entry_cost']
            
            # Save analysis
            date_str = datetime.now().strftime('%Y_%m_%d')
            filename = f"{date_str}_exit_orders_execution_analysis.csv"
            filepath = os.path.join(self.save_dir, filename)
            analysis.to_csv(filepath)
            
            self.logger.info(f"\nExit Orders Execution Analysis saved to {filepath}")
            self.logger.info(f"Total P&L: ${analysis['pnl'].sum():.2f}")
            
            return analysis
        
        return pd.DataFrame()

    def verify_positions_closed(self):
        """
        Verify that all positions have been closed.
        
        Returns:
            tuple: (all_closed, remaining_positions)
        """
        positions_df = self.get_current_positions()
        
        if positions_df.empty:
            self.logger.info("All positions successfully closed")
            return True, pd.DataFrame()
        
        self.logger.warning(f"Found {len(positions_df)} remaining positions")
        return False, positions_df