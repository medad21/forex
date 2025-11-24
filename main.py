# ... (بقیه کدهای main.py بدون تغییر)

# ---------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------

# ... (تابع get_candles بدون تغییر)

def process_data(df):
    """ ✅ نسخه تقویت شده: اضافه کردن MFI، STOCH و SuperTrend برای افزایش دقت """
    if df is None or df.empty: return pd.DataFrame()
    
    try:
        # 1. تبدیل اولیه
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        if len(df) < 60: return df # حداقل 60 کندل برای STOCH و اندیکاتورهای جدید

        # 2. محاسبه اندیکاتورهای قبلی
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=6, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.donchian(lower_length=20, upper_length=20, append=True)
        
        # 3. ✅ اضافه کردن اندیکاتورهای جدید برای دقت بیشتر
        
        # 3.1. Stochastics (نوسانگر)
        df.ta.stoch(k=14, d=3, append=True)
        
        # 3.2. Money Flow Index (حجم)
        df.ta.mfi(length=14, append=True)
        
        # 3.3. SuperTrend (ترند قوی)
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
        
        # تطبیق نام ستون‌ها
        if 'ATRr_14' in df.columns: df['ATR_14'] = df['ATRr_14']
        if 'ADX_14' not in df.columns and 'ADX' in df.columns: df['ADX_14'] = df['ADX']
        if 'STOCHk_14_3_3' in df.columns: df['STOCH_K'] = df['STOCHk_14_3_3']
        if 'STOCHd_14_3_3' in df.columns: df['STOCH_D'] = df['STOCHd_14_3_3']
        if 'SUPERT_10_3.0' in df.columns: df['SUPERT'] = df['SUPERT_10_3.0']
        if 'SUPERTd_10_3.0' in df.columns: df['SUPERT_D'] = df['SUPERTd_10_3.0']

        # اصلاح داده‌های NaN
        df = df.fillna(method='ffill') 
        df = df.fillna(method='bfill') 
        df = df.fillna(0)

        # تعریف ستون‌های مشتق شده (بدون تغییر)
        df['DCL'] = df.get('DCL_20_20', df['low'])
        df['DCU'] = df.get('DCU_20_20', df['high'])

        df['Returns'] = df['close'].pct_change().fillna(0)
        
        df['Volatility'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], 0)
        
        df['EMA_Diff_Fast'] = np.where(df['close'] != 0, (df.get('EMA_20', df['close']) - df.get('EMA_50', df['close'])) / df['close'], 0)
        df['EMA_Diff_Slow'] = np.where(df['close'] != 0, (df.get('EMA_50', df['close']) - df.get('EMA_100', df['close'])) / df['close'], 0)
        
        df['Hour'] = df['datetime'].dt.hour
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        df['HV_20'] = df['Returns'].rolling(20).std().fillna(0)
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        print(f"⚠️ Error in process_data: {e}")
        traceback.print_exc()
        return df

def get_ml_prediction(df):
    report = {"ensemble_score": 0, "message": "AI: غیرفعال", "individual_results": {}, "ml_score_final": 0}
    
    if not GLOBAL_MODELS_LOADED or len(df) < 5: 
        return 0, report

    try:
        # ✅ لیست ویژگی‌های جدید (شامل MFI, STOCH_K و SUPERT_D)
        feature_cols = [
            'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
            'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
            'MFI_14', 'STOCH_K', 'SUPERT_D' # 👈 ویژگی‌های جدید
        ]
        
        # ... (بقیه تابع get_ml_prediction بدون تغییر)
        # ... (در اینجا باید مدل‌های AI را مجددا آموزش دهید تا این 3 ویژگی جدید را یاد بگیرند)
        
# ... (بقیه کدهای main.py بدون تغییر)
