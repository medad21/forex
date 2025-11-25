import os
import pandas as pd
from datetime import datetime, timedelta
from dukascopy import Downloader

def download_dukascopy(symbol, start_date, end_date, timeframe="1h", save_path="data/raw"):
    """
    دانلود دیتا از Dukascopy با بهترین کیفیت OHLC
    """

    os.makedirs(save_path, exist_ok=True)

    outfile = os.path.join(save_path, f"{symbol}_dukascopy.csv")

    print(f"📥 Downloading Dukascopy data for {symbol} ...")

    dl = Downloader()

    try:
        df = dl.download(
            instrument=symbol,
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d"),
            timeframe=timeframe
        )

        if df is None or df.empty:
            print(f"⚠️ No data from Dukascopy for {symbol}")
            return None

        df = df.reset_index()
        df.rename(columns={
            "index": "datetime",
            "AskOpen": "open",
            "AskHigh": "high",
            "AskLow": "low",
            "AskClose": "close",
            "Volume": "volume"
        }, inplace=True)

        df.to_csv(outfile, index=False)
        print(f"✅ Saved Dukascopy: {outfile}")

        return outfile

    except Exception as e:
        print(f"❌ Dukascopy Error: {e}")
        return None

