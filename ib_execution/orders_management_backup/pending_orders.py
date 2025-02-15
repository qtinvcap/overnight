import os
import sys
from datetime import datetime
import logging
import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap wide displays
pd.set_option('display.max_rows', None)     # Show all rows

from ib_execution.orders_management.utils import IBConnectionManager

ib = IBConnectionManager.get_instance()

def setup_logger():
    logger = logging.getLogger("list_orders")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def list_pending_orders(ib, logger):
    """List all pending orders with detailed information."""
    # Force a refresh of open orders from IB
    ib.reqAllOpenOrders()
    ib.sleep(1)  # Give IB time to respond

    # Get all trades (which contain both order and contract info)
    trades = ib.trades()
    logger.debug(f"Raw trades count: {len(trades)}")
    
    if not trades:
        logger.info("No pending orders found.")
        return pd.DataFrame()
    
    orders_data = []
    
    for trade in trades:
        # Skip completed or cancelled orders
        if trade.orderStatus.status in ['Filled', 'Cancelled', 'ApiCancelled']:
            continue
            
        # Extract algo parameters if present
        algo_params = {}
        if hasattr(trade.order, 'algoParams'):
            for param in trade.order.algoParams:
                algo_params[param.tag] = param.value

        # Compile order information
        order_info = {
            'symbol': trade.contract.symbol,
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status,
            'action': trade.order.action,
            'type': trade.order.orderType,
            'quantity': trade.order.totalQuantity,
            'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining,
            'avg_price': trade.orderStatus.avgFillPrice,
            'algo': getattr(trade.order, 'algoStrategy', ''),
            'algo_params': str(algo_params) if algo_params else '',
            'why_held': trade.orderStatus.whyHeld or ''
        }
        
        orders_data.append(order_info)
    
    # Create DataFrame
    orders_df = pd.DataFrame(orders_data)
    
    if orders_df.empty:
        return orders_df
        
    # Reorder columns for better display
    columns_order = [
        'symbol', 'action', 'quantity', 'filled', 'remaining', 
        'avg_price', 'status', 'type', 'algo', 'algo_params', 
        'order_id', 'why_held'
    ]
    orders_df = orders_df[columns_order]
    
    return orders_df
def cancel_all_pending_orders(ib, logger):
    """Cancel all pending orders and return number of cancelled orders."""
    open_orders = ib.reqAllOpenOrders()
    cancelled_count = 0
    
    if not open_orders:
        logger.info("No pending orders to cancel.")
        return 0
    
    # Show current orders before cancellation
    orders_df = list_pending_orders(ib, logger)
    if not orders_df.empty:
        logger.info("\nCurrent pending orders before cancellation:")
        print("\n", orders_df.to_string(index=False), "\n")
    
    # Cancel orders
    active_statuses = ['Submitted', 'PreSubmitted', 'PendingSubmit']
    
    for trade in open_orders:
        try:
            if trade.orderStatus.status in active_statuses:
                logger.info(f"Cancelling order: {trade.contract.symbol} "
                          f"(Order ID: {trade.order.orderId}, "
                          f"Status: {trade.orderStatus.status}, "
                          f"Remaining: {trade.orderStatus.remaining} shares)")
                
                ib.cancelOrder(trade.order)
                cancelled_count += 1
                
        except Exception as e:
            logger.error(f"Error cancelling order for {trade.contract.symbol}: {str(e)}")
    
    return cancelled_count

def main():
    logger = setup_logger()
    logger.info("Starting orders management...")

    # Connect to IB with retry and proper error handling
    max_retries = 3
    ib = None
    
    for attempt in range(max_retries):
        try:
            ib = IBConnectionManager.get_instance()
            # Verify connection is active
            if not ib.isConnected():
                raise ConnectionError("Failed to establish connection with IB")
            logger.info("Successfully connected to IB")
            break
        except Exception as e:
            logger.error(f"Connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying connection...")
                continue
            else:
                logger.error("All connection attempts failed")
                return

    try:
        # Get user input for action
        print("\nChoose action:")
        print("1. List pending orders")
        print("2. Cancel all pending orders")
        print("3. Both list and cancel")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            # List pending orders
            orders_df = list_pending_orders(ib, logger)
            
            if orders_df.empty:
                logger.info("No pending orders found.")
            else:
                logger.info(f"\nFound {len(orders_df)} pending orders:")
                print("\n", orders_df.to_string(index=False), "\n")
                
                # Save to CSV
                filename = f"pending_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                orders_df.to_csv(filename, index=False)
                logger.info(f"Orders details saved to {filename}")
                
        elif choice == "2":
            # Cancel all pending orders
            cancelled_count = cancel_all_pending_orders(ib, logger)
            
            if cancelled_count > 0:
                logger.info(f"\nSuccessfully cancelled {cancelled_count} orders")
            
        elif choice == "3":
            # First list
            logger.info("\nCurrent pending orders:")
            orders_df = list_pending_orders(ib, logger)
            
            if not orders_df.empty:
                print("\n", orders_df.to_string(index=False), "\n")
                
                # Then cancel
                proceed = input("\nProceed with cancellation? (y/n): ").lower()
                if proceed == 'y':
                    cancelled_count = cancel_all_pending_orders(ib, logger)
                    if cancelled_count > 0:
                        logger.info(f"\nSuccessfully cancelled {cancelled_count} orders")
                else:
                    logger.info("Cancellation aborted")
            
        else:
            logger.error("Invalid choice")

    except Exception as e:
        logger.error(f"Error during operation: {str(e)}")
        logger.error("Exception details:", exc_info=True)
    finally:
        #if ib and ib.isConnected():
        IBConnectionManager.disconnect()
        logger.info("Disconnected from IB")

if __name__ == "__main__":
    main()