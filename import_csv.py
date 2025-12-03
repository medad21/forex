import pandas as pd
import os
import database # فرض بر این است که database.py است

# ⚠️ این دیکشنری را مطابق با فایل‌های CSV که دارید، تنظیم کنید.
CSV_FILES = {
    "EURUSD": "EURUSD_data.csv",
    "XAUUSD": "XAUUSD_data.csv",
    # اگر فایل‌های دیگری دارید اینجا اضافه کنید:
    # "GBPUSD": "GBPUSD_data.csv",
    # "BTCUSD": "BTCUSD_data.csv",
}
INTERVAL = "1h"

def import_csv_to_db():
    print("🚀 Starting CSV import to local database...")
    database.init_db() # مطمئن شوید دیتابیس راه‌اندازی شده است

    for symbol, filename in CSV_FILES.items():
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}. Skipping {symbol}.")
            continue

        try:
            # خواندن فایل
            # ⚠️ جداکننده (sep=';') را مطابق با فایل‌های CSV شما تنظیم کردم.
            df = pd.read_csv(filename, sep=';')
            df.columns = [c.lower() for c in df.columns]
            
            # پاکسازی و آماده‌سازی
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            else:
                 # اگر ستون datetime نبود، از index استفاده کن
                df = df.reset_index()
                df.rename(columns={'index': 'datetime'}, inplace=True)
            
            # تبدیل به عدد
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # ذخیره در دیتابیس
            database.save_candles(df.dropna(), symbol, INTERVAL)
            print(f"✅ Successfully imported {len(df.dropna())} rows for {symbol}.")

        except Exception as e:
            print(f"❌ Error importing {symbol}: {e}")

if __name__ == "__main__":
    import_csv_to_db()
