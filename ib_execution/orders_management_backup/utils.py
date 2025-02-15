from ib_insync import IB
import threading

class IBConnectionManager:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_connection()
        return cls._instance
    
    @classmethod
    def _create_connection(cls):
        ib = IB()
        try:
            ib.connect("127.0.0.1", 4002, clientId=56)
            return ib
        except Exception as e:
            print(f"Error connecting to IB: {str(e)}")
            return None
    
    @classmethod
    def disconnect(cls):
        if cls._instance and cls._instance.isConnected():
            cls._instance.disconnect()
            cls._instance = None