import os
import pandas as pd
from sqlalchemy import create_engine, text

# دریافت آدرس دیتابیس از متغیرهای محیطی Railway
# اگر متغیر نبود، به عنوان جایگزین از فایل لوکال استفاده می‌کند (برای تست روی سیستم خودتان)
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///market_data.db")

# اصلاح باگ رایج در SQLAlchemy (تبدیل postgres:// به postgresql://)
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

# ساخت موتور اتصال
engine = create_engine(DB_URL)

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

    # آماده‌سازی دیتا
    df = df.copy()
    df['symbol'] = symbol
    df['interval'] = interval
    
    # استانداردسازی نام ستون‌ها برای دیتابیس
    if 'datetime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={'index': 'datetime', 'Date': 'datetime'}, inplace=True)
    
    try:
        # استفاده از متد قدرتمند to_sql پانداس
        # method='multi' برای سرعت بالاتر در Postgres
        df.to_sql('candles', engine, if_exists='append', index=False, chunksize=1000, method='multi')
    except Exception as e:
        # خطای تکراری بودن کلید (IntegrityError) در اینجا طبیعی است و نادیده گرفته می‌شود
        # چون ممکن است کندل‌های قدیمی دوباره فچ شده باشند
        if "unique constraint" in str(e).lower() or "duplicate key" in str(e).lower():
            pass 
        else:
            print(f"⚠️ DB Save Error: {e}")

def get_all_candles(symbol, interval):
    """خواندن تمام کندل‌های ذخیره شده برای یک نماد"""
    try:
        query = text("SELECT * FROM candles WHERE symbol=:sym AND interval=:inv ORDER BY datetime ASC")
        
        # استفاده از کانکشن برای خواندن امن
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"sym": symbol, "inv": interval})
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
        return df
    except Exception as e:
        print(f"❌ DB Read Error: {e}")
        return pd.DataFrame()

# اجرای اولیه هنگام ایمپورت شدن فایل
init_db()
