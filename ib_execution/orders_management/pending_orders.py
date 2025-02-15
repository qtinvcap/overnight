import pandas as pd
from ib_execution.orders_management.utils import IBConnection

def get_orders(ib, pipeline_logger):
    """
    Check open trades and positions, returning organized DataFrames
    
    Returns:
        orders_df: DataFrame with pending orders info (symbol, status, quantity, order_type, etc.)
        positions_df: DataFrame with current positions info (symbol, quantity, avg_cost, market_value)
    
    Raises:
        ValueError: If cancelled orders appear in reqAllOpenOrders() results
    """
    # Check open trades (only returns active orders, not cancelled ones)
    open_orders = ib.reqAllOpenOrders()
    
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
            })
    
    orders_df = pd.DataFrame(orders_data)
    if not orders_df.empty:
        pipeline_logger.info(f"\nPending Orders:\n{orders_df}")
    else:
        pipeline_logger.info("\nNo pending orders")

    return orders_df

if __name__ == "__main__":
    from ib_execution.orders_management.utils import setup_logging

    ib = IBConnection.get_instance(port=4002, client_id=10)
    pipeline_logger = setup_logging()
    get_orders(ib, pipeline_logger)
