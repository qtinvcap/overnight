import os
import sys
from datetime import datetime
import logging
from ib_insync import Stock
from ib_execution.orders_management.utils import IBConnectionManager
from ib_execution.orders_management.entry_orders import EntryOrdersManager

def setup_logger():
    logger = logging.getLogger("test_entry_orders")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def main():
    logger = setup_logger()
    logger.info("Starting entry orders test...")

    try:
        ib = IBConnectionManager.get_instance()
        logger.info("Successfully connected to IB")

        # Get account details first
        account = ib.managedAccounts()[0]
        logger.info(f"Using account: {account}")

        # Request account summary
        summary = ib.accountSummary(account)
        relevant_tags = ['AvailableFunds', 'BuyingPower', 'NetLiquidation']
        for detail in summary:
            if detail.tag in relevant_tags:
                logger.info(f"{detail.tag}: {detail.value} {detail.currency}")

        # Test orders
        test_orders = [
            {'ticker': 'AAPL', 'quantity': 1, 'action': 'BUY'},
            {'ticker': 'MSFT', 'quantity': 1, 'action': 'BUY'},
        ]

        # Calculate required funds
        total_required = 0
        for order in test_orders:
            contract = Stock(order['ticker'], 'SMART', 'USD')
            contract = ib.qualifyContracts(contract)[0]
            [ticker] = ib.reqTickers(contract)
            price = ticker.marketPrice()
            required = price * order['quantity']
            total_required += required
            logger.info(f"Required funds for {order['ticker']}: ${required:.2f} (Price: ${price:.2f} x {order['quantity']} shares)")

        logger.info(f"Total required funds: ${total_required:.2f}")

        # Rest of your existing code...
        entry_manager = EntryOrdersManager(ib)
        
        logger.info("Placing test orders...")
        order_tracking = entry_manager.place_closing_price_orders(
            orders_list=test_orders,
            max_percentage=5,
            risk_aversion="Aggressive",
            start_time=None)

        logger.info(f"Order tracking IDs: {order_tracking}")

        logger.info("Waiting for orders to be processed...")
        ib.sleep(2)

        #all_accepted, placement_status = entry_manager.verify_order_placement(order_tracking)
        
        #logger.info(f"Placement status:\n{placement_status}")
        
        #if not all_accepted:
        #    logger.warning("Not all orders were accepted!")
        #    # Get detailed status for each order
        #    for _, row in placement_status.iterrows():
        #        trade = ib.trades().get(row['order_id'])
        #        if trade:
        #            logger.info(f"Order {row['order_id']} ({row['ticker']}):")
        #            logger.info(f"  Status: {trade.orderStatus.status}")
        #            logger.info(f"  Why Held: {trade.orderStatus.whyHeld}")
        #            logger.info(f"  Warning: {trade.orderStatus.warning}")

    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        logger.error("Exception details:", exc_info=True)
    finally:
        ib.disconnect()
        logger.info("Disconnected from IB")

if __name__ == "__main__":
    main()