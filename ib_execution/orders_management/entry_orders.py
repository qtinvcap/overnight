from ib_execution.orders_management.utils import IBConnection
from ib_insync import Stock, Order, TagValue
from datetime import datetime
import pytz

def place_close_price_orders(ib, pipeline_logger, orders_list, max_percentage=5, risk_aversion="Aggressive", start_time='15:58:00 US/Eastern'):
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
        pipeline_logger.info(f"No start time provided. Starting immediately at {start_time}")
    
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
            pipeline_logger.info(f"Close Price Parameters: {params_str}")
            
            trade = ib.placeOrder(contract, order)
            orders_placed += 1
            
            pipeline_logger.info(f"Placed Close Price order for {ticker}: {action} {quantity} shares")
            ib.sleep(0.1)
            
        except Exception as e:
            pipeline_logger.error(f"Error placing Close Price order for {ticker}: {str(e)}")
            continue
    
    pipeline_logger.info(f"Placed {orders_placed} Close Price orders")
    return orders_placed

def place_entry_orders(ib, pipeline_logger, orders_list):
    """
    close price order placement process
    orders_list: list of dicts with keys: ticker, quantity, action
    time_to_wait_before_cancel: int, minutes to wait before canceling orders. if -1, there is no cancellation
    if None, will calculate time to wait until market close + 30 seconds
    """
    # Place orders
    pipeline_logger.info("Starting Close Price order placement...")

    if len(orders_list) == 0:
        pipeline_logger.info("No orders to place. Skipping close price order placement.")
        return True
    
    place_close_price_orders(ib, pipeline_logger, orders_list, risk_aversion="GetDone", max_percentage=10, start_time=None) # GetDone # Aggressive # Neutral
    
    return True

if __name__ == "__main__":
    from ib_execution.orders_management.utils import setup_logging

    ib = IBConnection.get_instance(port=4002)
    pipeline_logger = setup_logging()
    # Example orders list
    orders_to_place = [
        {'ticker': 'FTAI',   'quantity': 108,   'action': 'BUY'},
        {'ticker': 'ALUR',  'quantity': 1176,  'action': 'BUY'},
        {'ticker': 'ARQQ',   'quantity': 365,   'action': 'BUY'},
    ]
    
    place_entry_orders(ib, pipeline_logger, orders_to_place)