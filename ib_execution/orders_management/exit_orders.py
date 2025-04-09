from ib_execution.orders_management.utils import IBConnection
from ib_insync import Order, LimitOrder
from ib_execution.portfolio_data.retrieve_portfolio_data import get_official_closes


def place_exit_orders_fixed(ib, pipeline_logger, tif="DAY", positions_to_exit=None):
    """Close all open positions using market orders on positions_to_exit if provided, otherwise close all positions"""
    orders_placed = 0
    
    # Force a refresh of position data to get the most up-to-date positions after OPG orders
    ib.reqPositions()
    # Add a slightly longer delay to ensure the positions are fully updated
    ib.sleep(2)
    
    # Get fresh positions after the sleep
    all_positions = ib.positions()
    
    # If positions_to_exit is provided, filter all_positions
    if positions_to_exit:
        # Extract symbols from positions_to_exit for comparison
        exit_symbols = [p.contract.symbol for p in positions_to_exit]
        positions = [p for p in all_positions if p.contract.symbol in exit_symbols]
    else:
        positions = [p for p in all_positions if abs(p.position) > 0]

    if not positions:
        pipeline_logger.info(f"No positions to close with {tif} orders")
        return 0
    
    # Log the current positions for clarity
    position_details = ", ".join([f"{p.contract.symbol}: {p.position}" for p in positions])
    pipeline_logger.info(f"Current positions before placing {tif} orders: {position_details}")
    
    for position in positions:
        try:
            contract = position.contract
            symbol = contract.symbol
            contract.exchange = 'SMART'
            quantity = abs(position.position)
            
            if quantity <= 0:
                pipeline_logger.info(f"Skipping {symbol} as position is {position.position}")
                continue
                
            action = "BUY" if position.position < 0 else "SELL"
            
            pipeline_logger.info(f"Placing {tif} exit order for {symbol}: {action} {quantity} shares")
            
            # Create market order to close position
            order = Order()
            order.orderType = "MKT"
            order.action = action
            order.totalQuantity = quantity
            order.tif = tif
            
            trade = ib.placeOrder(contract, order)
            orders_placed += 1
            
            # Small delay between orders
            ib.sleep(0.2)
            
        except Exception as e:
            pipeline_logger.error(f"Error closing position for {contract.symbol}: {str(e)}")
            continue
    
    pipeline_logger.info(f"Placed {orders_placed} {tif} orders to close positions")
    return orders_placed


def count_decimals(number):
    """Return the number of decimal places in a number."""
    number_str = str(number)
    if '.' in number_str:
        return len(number_str.split('.')[1])
    else:
        return 0
    
def round_to_tick_size(price, decimal):
    """Round price to valid tick size"""
    return round(price, decimal)


def place_premarket_limit(ib, pipeline_logger):
    """Close positions using limit orders during pre-market
    
    Args:
        ib: IB connection instance
        pipeline_logger: Logger instance
        limit_prices: dict mapping symbol to limit price {symbol: price}
        tif: Time in force (default PRE for pre-market)
        positions_to_exit: Optional list of positions to exit
    """
    orders_placed = 0
    all_positions = ib.reqPositions()
    ib.sleep(1)
    positions = [p for p in all_positions if abs(p.position) > 0]
    if len(positions) == 0:
        pipeline_logger.info("No positions to close in pre-market")
        return 0
    
    official_closes = get_official_closes(ib, [p.contract.symbol for p in positions]).set_index('ticker')

    for position in positions:
        try:
            contract = position.contract
            symbol = contract.symbol
                
            contract.exchange = 'SMART'
            official_close = official_closes.loc[symbol, 'official_close']
            decimal = count_decimals(official_close)

            order = LimitOrder(
                action='SELL',
                totalQuantity=abs(position.position),
                lmtPrice=round_to_tick_size(official_close * 1.03, decimal),
                tif='GTC',
                #displaySize=int(abs(position.position) / 10),
                outsideRth=True
            )
            
            trade = ib.placeOrder(contract, order)
            orders_placed += 1
            pipeline_logger.info(f"Placed pre-market limit exit for {symbol}: SELL {abs(position.position)} @ {order.lmtPrice}")
            ib.sleep(0.2)
            
        except Exception as e:
            pipeline_logger.error(f"Error placing limit exit for {symbol}: {str(e)}")
            
    return orders_placed


if __name__ == "__main__":
    from ib_execution.orders_management.utils import setup_logging

    ib = IBConnection.get_instance(port=4002, client_id=2)
    pipeline_logger = setup_logging()
    #place_premarket_limit(ib, pipeline_logger)
    place_exit_orders_fixed(ib, pipeline_logger, tif="DAY")
