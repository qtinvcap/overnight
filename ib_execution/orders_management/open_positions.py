import pandas as pd
from ib_execution.orders_management.utils import IBConnection

def get_orders_and_positions(ib, pipeline_logger):
    """
    Check open trades and positions, returning organized DataFrames
    
    Returns:
        orders_df: DataFrame with pending orders info (symbol, status, quantity, order_type, etc.)
        positions_df: DataFrame with current positions info (symbol, quantity, avg_cost, market_value)
    
    Raises:
        ValueError: If cancelled orders appear in reqAllOpenOrders() results
    """
    # Check positions
    positions = ib.positions()
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
        pipeline_logger.info(f"\nCurrent Positions:\n{positions_df}")
    else:
        pipeline_logger.info("\nNo positions")

    return positions_df

if __name__ == "__main__":
    from ib_execution.orders_management.utils import setup_logging

    ib = IBConnection.get_instance(port=4002, client_id=10)
    pipeline_logger = setup_logging()
    get_orders_and_positions(ib, pipeline_logger)