import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import urlparse # 🛑 ماژول حیاتی برای پارس کردن ایمن URL

# -----------------------------------------------------
# 1. تعریف آدرس اتصال با 3 اولویت و تفکیک اجزا
# -----------------------------------------------------

DB_URL = None
raw_url = None

# --- اولویت 1: استفاده از DATABASE_PUBLIC_URL یا DATABASE_URL (و پارس قوی) ---
# اولویت را به آدرس عمومی می دهیم، زیرا در Railway کمتر دچار مشکل می شود
database_url_candidate = os.environ.get("DATABASE_PUBLIC_URL")
if not database_url_candidate:
    database_url_candidate = os.environ.get("DATABASE_URL")

if database_url_candidate:
    try:
        # 🛑 تفکیک اجباری URL با urlparse برای استخراج اجزای تمیز
        url_parts = urlparse(database_url_candidate)
        
        # 🛑 ساخت URL تمیز با پروتکل صحیح (postgresql://) از اجزای جدا شده
        # url_parts.path شامل نام دیتابیس است (مانند /railway)
        raw_url = "postgresql://{user}:{password}@{host}:{port}{path}".format(
            user=url_parts.username,
            password=url_parts.password,
            host=url_parts.hostname,
            port=url_parts.port,
            path=url_parts.path 
        )
        print("✅ PostgreSQL URL components parsed and reconstructed (Railway URL).")

    except Exception as e:
        print(f"❌ Failed to parse Railway DATABASE_URL: {e}. Falling back...")
        raw_url = None 

# --- اولویت 2: بازسازی از متغیرهای PG* (اگر پارس Railway شکست خورده باشد) ---
if not raw_url:
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    database = os.environ.get("PGDATABASE")

    if user and password and host and port and database:
        raw_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        print("✅ PostgreSQL URL constructed from PG* variables (Method 2).")


# --- تصمیم گیری نهایی ---
if raw_url:
    DB_URL = raw_url
    # فقط بخشی از URL را برای لاگ نمایش می دهیم تا اطلاعات حساس لو نرود
    print(f"🔗 Attempting connection to PostgreSQL with final URL: {DB_URL[:40]}...") 
else:
    # ⚠️ فال‌بک نهایی به SQLite
    print("❌ All PostgreSQL variables are missing. Falling back to local SQLite.")
    DB_URL = "sqlite:///market_data.db"
    print("⚠️ Local SQLite used. Data will be lost on server restart!")


# -----------------------------------------------------
# 2. ساخت موتور اتصال
# -----------------------------------------------------
try:
    engine = create_engine(DB_URL)
except Exception as e:
    print(f"❌ Error creating DB engine: {e}. Falling back to SQLite...")
    engine = create_engine("sqlite:///market_data.db")


# -----------------------------------------------------
# 3. توابع دیتابیس
# -----------------------------------------------------

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
        if 'sqlite' not in engine.url.drivername:
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
        # تنظیم متد ذخیره‌سازی بر اساس نوع دیتابیس (برای سرعت بالاتر در Postgres)
        method = 'multi' if 'postgresql' in engine.url.drivername else None
        
        df.to_sql('candles', engine, if_exists='append', index=False, chunksize=500, method=method)

    except Exception as e:
        # نادیده گرفتن خطای تکراری بودن دیتا (برای PostgreSQL و SQLite)
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
        if 'sqlite' not in engine.url.drivername:
            print(f"❌ DB Read Error: {e}")
        return pd.DataFrame()

# اجرای اولیه برای ساخت جدول
init_db()
