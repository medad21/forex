import os
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# Railway automatically provides the DATABASE_URL environment variable
DATABASE_URL = os.environ.get("DATABASE_URL") 
engine = None

if DATABASE_URL:
    try:
        # Create the connection engine
        engine = create_engine(DATABASE_URL)
        print("✅ Database connection engine created successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
else:
    print("❌ DATABASE_URL not found. Running without persistence.")

# ----------------------------------------------------------------------
# Core Database Functions
# ----------------------------------------------------------------------

def save_candles(df: pd.DataFrame, symbol: str, interval: str):
    """Saves DataFrame candles to the database, ensuring no duplicates based on datetime."""
    if engine is None:
        return
    
    # Create a unique table name (e.g., btc_usd_1h)
    table_name = f"{symbol.replace('/', '_').lower()}_{interval.lower()}"
    
    try:
        # Save data. We use 'append' because we only save new data.
        df.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"✅ {len(df)} new candles saved to table {table_name}.")
    except Exception as e:
        # This might catch errors if data already exists or table structure issue
        print(f"❌ Error saving data to DB: {e}")

def get_all_candles(symbol: str, interval: str) -> pd.DataFrame:
    """Reads ALL stored candles for a given symbol/interval from the database."""
    if engine is None:
        return pd.DataFrame()
        
    table_name = f"{symbol.replace('/', '_').lower()}_{interval.lower()}"
    
    try:
        # Read all data from the table
        df = pd.read_sql_table(table_name, engine)
        # Convert datetime column to proper format and sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        print(f"✅ Loaded {len(df)} candles from DB for {symbol}.")
        return df
    except Exception as e:
        # Returns empty DataFrame if the table doesn't exist yet
        return pd.DataFrame()
