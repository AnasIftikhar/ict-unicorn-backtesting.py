import requests
import zipfile
import io
import time
import threading
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytz

# ============================================================
#  CONFIGURATION
# ============================================================

SYMBOL        = "ETHUSDT.P"       # Binance symbol — use BTCUSDT for spot, BTCUSDT.P for perpetual
INTERVAL      = "3m"            # Candle interval: 1m, 3m, 5m, 15m, 1h, 4h, 1d
DAYS_BACK     = 365             # How many days of history to download
TIMEZONE      = "Asia/Karachi"  # Output timestamp timezone
OUTPUT_FILE   = rf"Data\{SYMBOL}-{INTERVAL}-{DAYS_BACK}.csv"  # Save path

IS_PERP       = SYMBOL.endswith(".P")
CLEAN_SYMBOL  = SYMBOL.replace(".P", "")

# Download settings
TIMEOUT       = 300             # Seconds before timeout (increased for large 1m files)
RETRIES       = 3               # Retry attempts on timeout
RETRY_DELAY   = 5               # Seconds between retries

# Binance Vision base URL (auto-selected based on symbol)
VISION_URL    = "https://data.binance.vision/data/futures/um/monthly/klines" if IS_PERP else "https://data.binance.vision/data/spot/monthly/klines"

# Binance API (auto-selected based on symbol)
API_URL       = "https://fapi.binance.com/fapi/v1/klines" if IS_PERP else "https://api.binance.com/api/v3/klines"
API_LIMIT     = 1500            # Max candles per API call



# ============================================================

# Thread-safe printing lock
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    """Thread-safe print — prevents garbled output from parallel downloads."""
    with _print_lock:
        print(*args, **kwargs)


def format_size(bytes_total):
    """Human-readable file size."""
    if bytes_total >= 1_000_000:
        return f"{bytes_total / 1_000_000:.1f} MB"
    elif bytes_total >= 1_000:
        return f"{bytes_total / 1_000:.1f} KB"
    return f"{bytes_total} B"


def download_month_stream(url, label="", retries=RETRIES, timeout=TIMEOUT):
    """
    Stream download with live progress indicator.
    Uses thread-safe printing with month label prefix.
    """
    for attempt in range(retries):
        try:
            # (connect timeout, read timeout) — read timeout resets per chunk
            r = requests.get(url, stream=True, timeout=(10, 30))

            if r.status_code != 200:
                return r  # Let caller handle non-200

            # Get total size if available
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            chunks = []
            start_time = time.time()

            for chunk in r.iter_content(chunk_size=1048576):  # 1MB chunks
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)

                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0

                    # Live progress with speed
                    if total:
                        pct = downloaded / total * 100
                        tprint(f"  {label}  {format_size(downloaded)} / {format_size(total)} ({pct:.0f}%)  🚀 {format_size(speed)}/s")
                    else:
                        tprint(f"  {label}  {format_size(downloaded)} downloaded  🚀 {format_size(speed)}/s")

            # Reconstruct response content from streamed chunks
            r._content = b"".join(chunks)
            return r

        except requests.exceptions.Timeout:
            tprint(f"  {label} ⏳ Timeout (attempt {attempt+1}/{retries}), retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        except requests.exceptions.ChunkedEncodingError:
            tprint(f"  {label} ⚠️  Connection dropped (attempt {attempt+1}/{retries}), retrying...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            tprint(f"  {label} ❌ Error: {e}")
            return None

    tprint(f"  {label} ❌ Failed after all retries")
    return None


def parse_vision_csv(z):
    """Parse CSV from zip, handling files with or without header row."""
    raw = z.open(z.namelist()[0])
    first_line = raw.readline().decode('utf-8').strip()
    raw.seek(0)

    has_header = not first_line.split(',')[0].strip().lstrip('-').isdigit()

    df = pd.read_csv(raw, header=0 if has_header else None)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_volume', 'trades',
                  'taker_buy_base', 'taker_buy_quote', 'ignore']
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def download_via_api(symbol, interval, start_ms, end_ms):
    """Fallback: fetch candles from Binance Futures API for current month."""
    all_candles = []
    current = start_ms
    batch = 0

    while current < end_ms:
        try:
            params = {
                "symbol":    symbol,
                "interval":  interval,
                "startTime": current,
                "limit":     API_LIMIT,
            }
            r = requests.get(API_URL, params=params, timeout=30)
            data = r.json()

            if not data or isinstance(data, dict):
                break

            filtered = [c for c in data if c[0] <= end_ms]
            all_candles.extend(filtered)
            batch += 1
            print(f"\r  ⬇️  {len(all_candles):,} candles fetched (batch {batch})...",
                  end="", flush=True)

            current = data[-1][0] + 1
            if len(data) < API_LIMIT:
                break

        except Exception as e:
            print(f"\n  ❌ API error: {e}")
            break

    print()  # New line

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def download_from_vision(symbol=SYMBOL, interval=INTERVAL, days_back=DAYS_BACK):
    pkt = pytz.timezone(TIMEZONE)
    now = datetime.now(pkt)
    start_date = now - timedelta(days=days_back)

    print("=" * 60)
    print(f"  Symbol   : {symbol}  ({interval})")
    print(f"  From     : {start_date.strftime('%Y-%m-%d %H:%M')} PKT")
    print(f"  To       : {now.strftime('%Y-%m-%d %H:%M')} PKT")
    print(f"  Days     : {days_back}")
    print("=" * 60)

    # Build list of (year, month) needed
    months_to_download = []
    current = start_date.replace(day=1)
    while current <= now:
        months_to_download.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    print(f"\nMonths needed: {months_to_download}\n")

    # Separate current month (API) from completed months (Vision)
    completed_months = [(y, m) for y, m in months_to_download
                        if not (y == now.year and m == now.month)]
    current_month    = [(y, m) for y, m in months_to_download
                        if y == now.year and m == now.month]

    workers = max(1, len(completed_months))  # One worker per month — all parallel
    print(f"🚀 Downloading {len(completed_months)} months with {workers} parallel workers...\n")

    results = {}  # {(year, month): DataFrame or None}

    def fetch_month(year, month):
        """Worker function — downloads one month and returns parsed DataFrame."""
        label = f"📦 {year}-{str(month).zfill(2)}"
        url = f"{VISION_URL}/{CLEAN_SYMBOL}/{interval}/{CLEAN_SYMBOL}-{interval}-{year}-{str(month).zfill(2)}.zip"
        tprint(f"{label}  ⬇️  Starting...")

        r = download_month_stream(url, label=label)

        if r is None:
            tprint(f"{label}  ❌ Failed\n")
            return year, month, None
        if r.status_code == 404:
            tprint(f"{label}  ⚠️  Not on Vision — trying API...\n")
            month_start = datetime(year, month, 1, tzinfo=pytz.UTC)
            month_end   = datetime(year + 1, 1, 1, tzinfo=pytz.UTC) if month == 12 else datetime(year, month + 1, 1, tzinfo=pytz.UTC)
            start_ms    = int(month_start.timestamp() * 1000)
            end_ms      = int(month_end.timestamp() * 1000)
            df = download_via_api(CLEAN_SYMBOL, interval, start_ms, end_ms)
            if df is not None:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
                tprint(f"{label}  ✅ {len(df):,} candles (via API)\n")
                return year, month, df
            tprint(f"{label}  ❌ API fallback also failed\n")
            return year, month, None
        if r.status_code != 200:
            tprint(f"{label}  ❌ HTTP {r.status_code}\n")
            return year, month, None

        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = parse_vision_csv(z)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
            tprint(f"{label}  ✅ {len(df):,} candles\n")
            return year, month, df
        except Exception as e:
            tprint(f"{label}  ❌ Parse error: {e}\n")
            return year, month, None

    # ── Parallel download of completed months ───────────────
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_month, y, m): (y, m)
                   for y, m in completed_months}
        for future in as_completed(futures):
            year, month, df = future.result()
            if df is not None:
                results[(year, month)] = df

    # ── API fallback for current month (sequential, single call) ─
    for year, month in current_month:
        tprint(f"📡 {year}-{str(month).zfill(2)} (current month — using API)")
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_ms = int(month_start.astimezone(pytz.UTC).timestamp() * 1000)
        end_ms   = int(now.astimezone(pytz.UTC).timestamp() * 1000)
        df = download_via_api(CLEAN_SYMBOL, interval, start_ms, end_ms)
        if df is not None:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
            results[(year, month)] = df
            tprint(f"  ✅ {len(df):,} candles\n")
        else:
            tprint(f"  ⚠️  No data returned\n")

    # Reconstruct frames in chronological order
    frames = [results[key] for key in sorted(results.keys()) if key in results]

    if not frames:
        print("❌ No data downloaded!")
        return None

    print("🔀 Merging all months...")
    final = pd.concat(frames).drop_duplicates(subset='timestamp').sort_values('timestamp')

    start_naive = start_date.replace(tzinfo=None)
    now_naive   = now.replace(tzinfo=None)
    final = final[(final['timestamp'] >= start_naive) & (final['timestamp'] <= now_naive)]
    final = final.reset_index(drop=True)

    return final


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    df = download_from_vision()

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'=' * 60}")
        print(f"✅ Saved to   : {OUTPUT_FILE}")
        print(f"Total candles : {len(df):,}")
        print(f"From          : {df['timestamp'].min()}")
        print(f"To            : {df['timestamp'].max()}")
        print(f"{'=' * 60}")
        print(df.tail())