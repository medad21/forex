import sqlite3
import pandas as pd
import os

DB_NAME = "market_data.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT,
            interval TEXT,
            datetime TIMESTAMP,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, interval, datetime)
        )
    ''')
    conn.commit()
    conn.close()

def save_candles(df, symbol, interval):
    if df.empty:
        return
    
    # اطمینان از فرمت صحیح دیتافریم
    df = df.copy()
    df['symbol'] = symbol
    df['interval'] = interval
    
    # تبدیل تاریخ به فرمت استاندارد اگر لازم باشد
    if 'datetime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={'index': 'datetime', 'Date': 'datetime'}, inplace=True)
    
    conn = get_connection()
    try:
        # استفاده از متد to_sql پانداز
        # if_exists='append' یعنی دیتاهای جدید را اضافه کن
        df.to_sql('candles', conn, if_exists='append', index=False, method='multi', chunksize=500)
    except sqlite3.IntegrityError:
        # اگر دیتا تکراری بود (به خاطر کلید اصلی)، نادیده بگیر (یا می‌توان آپدیت کرد)
        pass
    except Exception as e:
        print(f"DB Save Error: {e}")
    finally:
        conn.close()

def get_all_candles(symbol, interval):
    conn = get_connection()
    try:
        query = f"SELECT * FROM candles WHERE symbol='{symbol}' AND interval='{interval}' ORDER BY datetime ASC"
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        print(f"DB Read Error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# اجرای اولیه برای ساخت جدول
init_db()
