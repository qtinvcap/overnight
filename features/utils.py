import pandas as pd
import nasdaqdatalink
import numpy as np
from typing import List
import os

nasdaqdatalink.ApiConfig.api_key = os.getenv("NASDAQDATALINK_API_KEY")

ETF_TICKERS = ["SPY", "QQQ", "QQQQ", "IWM"]

MARKET_CAP_CATEGORIES = [
    "1 - Nano",
    "2 - Micro",
    "3 - Small",
    "4 - Mid",
    "5 - Large",
    "6 - Mega",
]


def get_ticker_full_tickers_list() -> List[str]:
    alltickers = nasdaqdatalink.get_table("SHARADAR/TICKERS", table="SF1", paginate=True)

    alltickers = alltickers[
        (alltickers["scalemarketcap"] == "3 - Small")
        | (alltickers["scalemarketcap"] == "2 - Micro")
        | (alltickers["scalemarketcap"] == "1 - Nano")
        | (alltickers["scalemarketcap"] == "4 - Mid")
        | (alltickers["scalemarketcap"] == "5 - Large")
        | (alltickers["scalemarketcap"] == "6 - Mega")
    ]

    ex_ticks = alltickers[alltickers["exchange"] != "OTC"]
    return list(np.unique(ex_ticks["ticker"]))
