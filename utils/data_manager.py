# utils/data_manager.py
import os
import pandas as pd
from utils.dukascopy import download_dukascopy
from utils.alpha_vantage import download_alpha_vantage

DATA_PATH = "data/forex"

def ensure_dirs():
    os.makedirs(DATA_PATH, exist_ok=True)

def save_combined(symbol, df_new):
    ensure_dirs()
    file_path = f"{DATA_PATH}/{symbol}.csv"

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_all = pd.concat([df_old, df_new]).drop_duplicates()
    else:
        df_all = df_new

    df_all = df_all.sort_values(by="timestamp" if "timestamp" in df_all else df_all.columns[0])
    df_all.to_csv(file_path, index=False)

    return df_all

def update_all_sources(symbol="EURUSD"):
    print("Downloading from Dukascopy…")
    df1 = download_dukascopy(symbol)

    print("Downloading from AlphaVantage…")
    df2 = download_alpha_vantage(symbol)

    print("Combining and saving…")
    df_all = save_combined(symbol, pd.concat([df1, df2]))

    print("DONE. Total rows:", len(df_all))
    return df_all
