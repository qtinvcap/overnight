from ib_insync import IB

def connect_to_ib():
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=56)  #


def test_connect_to_ib():
    try:
        ib = connect_to_ib()
        assert ib.isConnected() == True
        ib.disconnect()
        assert ib.isConnected() == False
        return True
    except Exception as e:
        return False
    

if __name__ == "__main__":
    test_connect_to_ib()