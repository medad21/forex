import os
import pandas as pd
from sqlalchemy import create_engine, text
raw_url = os.environ.get("DATABASE_URL")
# خط تست را اضافه کنید:
if raw_url:
    print("✅ DATABASE_URL is read. Length:", len(raw_url))
else:
    print("❌ DATABASE_URL is None or empty in the Python environment.")

# ... ادامه کد شما
# 1. دریافت و تمیزسازی آدرس دیتابیس
raw_url = os.environ.get("DATABASE_URL")

# اگر متغیر نبود یا خالی بود، از SQLite استفاده کن (جلوگیری از کرش)
if not raw_url or raw_url.strip() == "":
    print("⚠️ DATABASE_URL is empty or missing. Falling back to local SQLite.")
    DB_URL = "sqlite:///market_data.db"
else:
    # اصلاح باگ نسخه جدید SQLAlchemy برای آدرس‌های Postgres
    DB_URL = raw_url.replace("postgres://", "postgresql://", 1)

# 2. ساخت موتور اتصال
try:
    engine = create_engine(DB_URL)
except Exception as e:
    print(f"❌ Error creating DB engine: {e}")
    # فال‌بک نهایی
    engine = create_engine("sqlite:///market_data.db")

def init_db():
    """جدول کندل‌ها را اگر وجود نداشته باشد می‌سازد"""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS candles (
        symbol VARCHAR(20),
        interval VARCHAR(10),
        datetime TIMESTAMP,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume FLOAT,
        PRIMARY KEY (symbol, interval, datetime)
    );
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_query))
            conn.commit()
        print("✅ Database table initialized successfully.")
    except Exception as e:
        print(f"❌ Database Initialization Error: {e}")

def save_candles(df, symbol, interval):
    """ذخیره کندل‌ها در دیتابیس"""
    if df.empty:
        return

    df = df.copy()
    df['symbol'] = symbol
    df['interval'] = interval
    
    if 'datetime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={'index': 'datetime', 'Date': 'datetime'}, inplace=True)
    
    try:
        # تنظیم متد ذخیره‌سازی بر اساس نوع دیتابیس
        method = 'multi' if 'postgres' in DB_URL else None
        df.to_sql('candles', engine, if_exists='append', index=False, chunksize=500, method=method)
    except Exception as e:
        # نادیده گرفتن خطای تکراری بودن دیتا
        if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower() or "UNIQUE constraint" in str(e).lower():
            pass 
        else:
            print(f"⚠️ DB Save Error: {e}")

def get_all_candles(symbol, interval):
    """خواندن تمام کندل‌های ذخیره شده"""
    try:
        query = text("SELECT * FROM candles WHERE symbol=:sym AND interval=:inv ORDER BY datetime ASC")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sym": symbol, "inv": interval})
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"❌ DB Read Error: {e}")
        return pd.DataFrame()

# اجرای اولیه
init_db()
