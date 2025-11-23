import os
import pandas as pd
from sqlalchemy import create_engine, text

# -----------------------------------------------------
# 1. تعریف آدرس اتصال با اولویت PostgreSQL (بازسازی از اجزا)
# -----------------------------------------------------

# تلاش برای خواندن متغیرهای PG* که در Railway تضمین شده‌اند (PGUSER, PGPASSWORD, PGHOST, PGPORT, PGDATABASE)
user = os.environ.get("PGUSER")
password = os.environ.get("PGPASSWORD")
host = os.environ.get("PGHOST")
port = os.environ.get("PGPORT")
database = os.environ.get("PGDATABASE")

DB_URL = None

if user and password and host and port and database:
    # 💡 ساخت آدرس اتصال PostgreSQL از اجزای PG*
    # استفاده از postgresql:// به جای postgres:// برای سازگاری کامل با SQLAlchemy
    raw_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    # اگر Railway از پروتکل قدیمی استفاده کند، آن را اصلاح می‌کنیم
    DB_URL = raw_url.replace("postgres://", "postgresql://", 1)
    
    print("✅ PostgreSQL Connection URL successfully constructed from PG* variables.")
    print("⚠️ Please ensure the 'psycopg2-binary' package is installed in your requirements.txt.")
else:
    # ⚠️ فال‌بک نهایی به SQLite
    print("❌ Cannot find PGUSER/PGPASSWORD/etc. Falling back to local SQLite.")
    DB_URL = "sqlite:///market_data.db"
    print("⚠️ Local SQLite used. Data will be lost on server restart!")


# -----------------------------------------------------
# 2. ساخت موتور اتصال
# -----------------------------------------------------
try:
    engine = create_engine(DB_URL)
except Exception as e:
    print(f"❌ Error creating DB engine: {e}. Falling back to SQLite...")
    # فال‌بک نهایی در صورت کرش موتور
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
        # کوئری ایمن با پارامترها
        query = text("SELECT * FROM candles WHERE symbol=:sym AND interval=:inv ORDER BY datetime ASC")
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sym": symbol, "inv": interval})
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"❌ DB Read Error: {e}")
        return pd.DataFrame()

# اجرای اولیه برای ساخت جدول
init_db()
