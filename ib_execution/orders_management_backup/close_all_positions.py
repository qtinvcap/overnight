

from ib_execution.orders_management.utils import IBConnectionManager
from ib_insync import Stock, MarketOrder

def close_all_positions_and_cancel_orders(ib):
    # Connect to the IB gateway or TWS
    #if not ib.isConnected():
    #ib = IBConnectionManager.get_instance()

    # Cancel all orders (both open and pending)
    open_orders = ib.openOrders()
    for order in open_orders:
        ib.cancelOrder(order)
        print(f"Canceled order {order.orderId}")

    # Closing all positions
    positions = ib.positions()
    if not positions:
        print("No open positions to close.")
    for position in positions:
        symbol = position.contract.symbol
        contract = Stock(symbol, "SMART", "USD")
        # order.tif = "IOC"
        size = position.position
        if size != 0:
            action = "SELL" if size > 0 else "BUY"
            order = MarketOrder(action, abs(size))
            trade = ib.placeOrder(contract, order)
            # trade.orderStatusEvent += on_order_status
            print(f"Placed order to close {size} of {symbol}")

def main():
    ib = IBConnectionManager.get_instance()
    close_all_positions_and_cancel_orders(ib)

if __name__ == "__main__":
    main()