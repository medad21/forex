def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    td_symbol = SYMBOL_MAP.get(symbol_key, symbol_key)

    if td is None:
        print("❌ TD API not initialized.")
        return pd.DataFrame()

    print(f"⏳ (TD) Downloading {td_symbol}...")

    # در هر درخواست 4500 کندل (حد مجاز)
    CHUNK_LIMIT = 4500  
    HOURS_PER_DAY = 24
    total_candles = days * HOURS_PER_DAY

    # تعداد چانک‌های لازم
    num_chunks = math.ceil(total_candles / CHUNK_LIMIT)

    all_frames = []
    end_time = None  # اجازه می‌دهد از آخر به عقب برویم

    for i in range(num_chunks):
        try:
            ts = td.time_series(
                symbol=td_symbol,
                interval=interval,
                outputsize=CHUNK_LIMIT,
                timezone="Exchange",
                end_date=end_time
            ).as_json()

            # بررسی سالم بودن خروجی
            if isinstance(ts, dict):
                print(f"⚠️ API Error: {ts}")
                break

            if not isinstance(ts, list) or len(ts) == 0:
                print(f"⚠️ Empty Chunk ({i+1}/{num_chunks})")
                break

            df = pd.DataFrame(ts)
            df = df.rename(columns={c: c.lower() for c in df.columns})

            # تبدیل نوع داده
            for col in ['open','high','low','close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
            df['datetime'] = pd.to_datetime(df['datetime'])

            df = df.sort_values('datetime').reset_index(drop=True)
            all_frames.append(df)

            # حرکت به عقب: جدیدترین datetime این chunk را می‌گیریم
            end_time = df['datetime'].min().strftime("%Y-%m-%d %H:%M:%S")

            # کمی مکث برای رد شدن از Rate Limit
            time.sleep(1.2)

        except Exception as e:
            print(f"❌ Chunk Error: {e}")
            break

    if not all_frames:
        print("❌ No chunks downloaded.")
        return pd.DataFrame()

    full_df = pd.concat(all_frames).drop_duplicates(subset=['datetime'])
    full_df = full_df.sort_values('datetime').reset_index(drop=True)

    return full_df[['datetime','open','high','low','close','volume']]
