import os
import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange


# API KEY مستقیم داخل فایل
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"


def download_alpha_vantage(symbol, save_path="data/raw"):
    """
    دانلود داده تایم‌فریم 1h از AlphaVantage
    خروجی به صورت CSV ذخیره می‌شود
    """

    # ایجاد پوشه اگر وجود نداشت
    os.makedirs(save_path, exist_ok=True)

    outfile = os.path.join(save_path, f"{symbol}_av.csv")

    if not API_KEY_ALPHA:
        print("⚠️ No AlphaVantage API key. Skipping AV download.")
        return None

    print(f"📥 Downloading AlphaVantage data for {symbol} ...")

    try:
        # آبجکت API
        fx = ForeignExchange(key=API_KEY_ALPHA, output_format='pandas')

        # گرفتن دیتای دقیقه‌ای یا ساعتی
        df, meta = fx.get_currency_exchange_intraday(
            from_symbol=symbol[:3],   # EUR
            to_symbol=symbol[3:],     # USD
            interval="60min"          # 1h
        )

        # تبدیل شاخص زمانی به ستون
        df.reset_index(inplace=True)

        # تغییر نام ستون‌ها برای هماهنگی با مدل‌ها
        df.rename(columns={
            "date": "datetime",
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        }, inplace=True)

        # AlphaVantage حجم ندارد → صفر می‌گذاریم
        df["volume"] = 0

        # ذخیره CSV
        df.to_csv(outfile, index=False)
        print(f"✅ Saved AlphaVantage CSV: {outfile}")

        return outfile

    except Exception as e:
        print(f"❌ AlphaVantage Error: {e}")
        return None
