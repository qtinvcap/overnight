from ib_execution.orders_management.utils import IBConnection


def robust_cancel_all_orders(ib, pipeline_logger, max_retries=3):
    """Cancel all pending orders with retries"""
    cancelled_count = 0
    retry_count = 0
    active_statuses = ['Submitted', 'PreSubmitted', 'PendingSubmit', 'PendingCancel']
    
    open_trades = ib.reqAllOpenOrders()
    while retry_count < max_retries:

        if not open_trades:
            pipeline_logger.info("No open orders to cancel.")
            return 0
        
        for trade in open_trades:
            try:
                if trade.orderStatus.status in active_statuses:
                    if trade.order.orderId != 0:  # Skip orders with ID 0
                        pipeline_logger.info(f"Cancelling order {trade.order.orderId} for {trade.contract.symbol} "
                                    f"with status: {trade.orderStatus.status}")
                        ib.cancelOrder(trade.order)
                        cancelled_count += 1
                        ib.sleep(0.1)  # Small delay between cancellations
                    else:
                        pipeline_logger.warning(f"Skipping order with ID 0 for {trade.contract.symbol}")
            except Exception as e:
                pipeline_logger.warning(f"Could not cancel order for {trade.contract.symbol}: {str(e)}")
                continue
        
        # Check if there are still active orders
        # Wait a moment for orders to be registered
        ib.sleep(1)
        
        open_trades = ib.reqAllOpenOrders()
        if not any(trade.orderStatus.status in active_statuses for trade in open_trades):
            pipeline_logger.info("Successfully cancelled all active orders")
            break
        
        retry_count += 1
        if retry_count < max_retries:
            pipeline_logger.warning(f"Some orders still active, retrying... Attempt {retry_count + 1}/{max_retries}")
    
    pipeline_logger.info(f"Cancelled {cancelled_count} pending orders")
    return cancelled_count

if __name__ == "__main__":
    from ib_execution.orders_management.utils import setup_logging

    ib = IBConnection.get_instance(port=4002, client_id=2)
    pipeline_logger = setup_logging()
    robust_cancel_all_orders(ib, pipeline_logger)