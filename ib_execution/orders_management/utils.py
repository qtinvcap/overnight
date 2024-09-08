from ib_insync import *
from datetime import datetime
import logging
import os

IB_GATEWAY_PORT = 4002
IB_TWS_PORT = 7497


class IBConnection:
    _instance = None
    _default_port = IB_GATEWAY_PORT
    
    @classmethod
    def get_instance(cls, port=None, client_id=1):
        """
        Get or create singleton IB connection
        Args:
            port: Optional port number. If not provided, uses default port
        """
        if cls._instance is None or not cls._instance.isConnected():
            cls._instance = cls._connect(port=port, client_id=client_id)
        return cls._instance
    
    @staticmethod
    def _connect(port=None, client_id=1):
        """
        Create new IB connection
        Args:
            port: Optional port number. If not provided, uses default port
        """
        ib = IB()
        port = port or IBConnection._default_port
        ib.connect('127.0.0.1', port, clientId=client_id)
        return ib
    
    @classmethod
    def set_default_port(cls, port):
        """Set the default port for future connections"""
        cls._default_port = port


from pathlib import Path
import sys

def setup_logging():
    # Setup logging for the pipeline
    root_path = os.getenv("OVERNIGHT_ROOT_PATH", os.path.expanduser("~"))
    log_dir = Path(root_path) / "logs"
    log_dir.mkdir(exist_ok=True)
    pipeline_log_file = log_dir / f"ib_execution_{datetime.now().strftime('%Y%m%d')}.log"

    pipeline_logger = logging.getLogger("ib_execution")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(pipeline_log_file, mode='a')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    pipeline_logger.addHandler(file_handler)
    pipeline_logger.addHandler(console_handler)

    pipeline_logger.info("===== Setup up Logging for IB Execution =====")
    
    return pipeline_logger

if __name__ == "__main__":
    # Test the functions
    ib = IBConnection.get_instance()
    pipeline_logger = setup_logging()