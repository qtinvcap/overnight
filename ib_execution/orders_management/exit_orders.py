from ib_execution.orders_management.utils import IBConnection
from ib_insync import Order

def place_exit_orders(ib, pipeline_logger, tif="DAY", positions_to_exit=None):
    """Close all open positions using market orders on positions_to_exit if provided, otherwise close all positions"""
    orders_placed = 0
    
    positions = positions_to_exit or ib.positions()
    if not positions:
        pipeline_logger.info("No positions to close")
        return 0
    
    for position in positions:
        try:
            pipeline_logger.info(f"Placing exit order for {position.contract.symbol}")
            contract = position.contract
            contract.exchange = 'SMART'
            quantity = abs(position.position)
            
            # Create market order to close position
            order = Order()
            order.orderType = "MKT"
            order.action = "SELL"
            order.totalQuantity = quantity
            order.tif = tif
            
            trade = ib.placeOrder(contract, order)
            orders_placed += 1
            
            pipeline_logger.info(f"Closing position for {contract.symbol}: SELL {quantity} shares at market")
            
            # Small delay between orders
            ib.sleep(0.1)
            
        except Exception as e:
            pipeline_logger.error(f"Error closing position for {contract.symbol}: {str(e)}")
            continue
    
    pipeline_logger.info(f"Placed {orders_placed} orders to close positions")
    return orders_placed

if __name__ == "__main__":
    from ib_execution.orders_management.utils import setup_logging

    ib = IBConnection.get_instance(port=4002)
    pipeline_logger = setup_logging()
    place_exit_orders(ib, pipeline_logger)