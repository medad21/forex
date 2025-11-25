import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from alpha_vantage.foreignexchange import ForeignExchange

# =====================
# تنظیمات API
# =====================
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =====================
# AlphaVantage Downloader
# =====================
def download_alpha_vantage(symbol, interval="60min"):
    outfile = os.path.join(RAW_DIR, f"{symbol}_av.csv")

    fx = ForeignExchange(key=API_KEY_ALPHA, output_format='pandas')

    print(f"📥 Downloading AlphaVantage data for {symbol} ...")
    df, _ = fx.get_currency_exchange_intraday(
        from_symbol=symbol[:3],
        to_symbol=symbol[3:],
        interval=interval
    )

    df.reset_index(inplace=True)
    df.rename(columns={
        'date': 'datetime',
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close'
    }, inplace=True)
    df['volume'] = 0

    # ادغام با CSV قبلی (اگر وجود دارد)
    if os.path.exists(outfile):
        df_old = pd.read_csv(outfile)
        df = pd.concat([df_old, df], ignore_index=True).drop_duplicates(subset=['datetime']).reset_index(drop=True)

    df.to_csv(outfile, index=False)
    print(f"✅ Saved AlphaVantage: {outfile}")
    return outfile

# =====================
# Dukascopy Downloader
# =====================
def download_dukascopy(symbol, timeframe='1h', start_date=None, end_date=None):
    outfile = os.path.join(RAW_DIR, f"{symbol}_duk.csv")

    # لینک API Dukascopy CSV (مثال: https://www.dukascopy.com/swiss/english/marketwatch/historical/) - CSV دستی یا wget)
    # برای MVP، از CSV نمونه استفاده می‌کنیم یا دانلود دستی

    print(f"📥 Dukascopy download placeholder for {symbol} ({timeframe})")

    # اگر CSV موجود باشد فقط نمایش بده
    if os.path.exists(outfile):
        print(f"✅ Dukascopy CSV exists: {outfile}")
    else:
        print(f"⚠️ Dukascopy CSV not found. Please download manually and place in {RAW_DIR}")
    return outfile

# =====================
# تابع اصلی مدیریت داده
# =====================
def update_data(symbols):
    for symbol in symbols:
        download_alpha_vantage(symbol)
        download_dukascopy(symbol)

# =====================
# اجرای مستقل
# =====================
if __name__ == '__main__':
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'GC', 'BTC']
    update_data(symbols)
