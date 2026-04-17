"""
RunPod Serverless Handler — ICT Unicorn Strategy Optimizer
==========================================================
Receives job parameters via RunPod API, downloads fresh market data from
Binance Vision, runs full parameter optimization across all CPU cores,
then uploads the results CSV to Google Drive and returns a summary.

Environment variables required in RunPod:
  GOOGLE_SERVICE_ACCOUNT_JSON  — Base64-encoded Google service account JSON key
                                  (only needed if you want Drive upload)

Optional env vars:
  RUNPOD_MAX_CORES  — Override CPU count (defaults to all available cores)
"""

import runpod
import json
import os
import base64
import importlib.util
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy as np
import pandas as pd


# ── Import strategy ──
from unicorn import UnicornStrategy  # noqa: F401  (needed inside optimize.py workers)

# ── Import optimizer functions ──
# optimize.py __main__ block is guarded so importing is safe
from optimize import (
    build_combinations,
    run_single_backtest,
    init_worker,
)

# ── Import Binance-Vision downloader ──
# Filename contains a hyphen so we use importlib instead of a plain import
_bv_spec = importlib.util.spec_from_file_location(
    "binance_vision",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Binance-Vision.py"),
)
_bv_mod = importlib.util.module_from_spec(_bv_spec)
_bv_spec.loader.exec_module(_bv_mod)
download_from_vision = _bv_mod.download_from_vision


# =========================================================================
# GOOGLE DRIVE UPLOAD
# =========================================================================

def upload_to_google_drive(csv_bytes: bytes, filename: str, folder_id: str = None):
    """
    Upload a CSV file to Google Drive using a service account.
    Credentials are read from the GOOGLE_SERVICE_ACCOUNT_JSON env var
    (base64-encoded JSON key file).
    Returns the webViewLink on success, None on failure or if not configured.
    """
    sa_json_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json_b64:
        print("GOOGLE_SERVICE_ACCOUNT_JSON not set — skipping Drive upload.")
        return None

    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaInMemoryUpload

        creds_dict = json.loads(base64.b64decode(sa_json_b64).decode())
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        service = build("drive", "v3", credentials=creds)

        file_metadata = {"name": filename}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        media = MediaInMemoryUpload(csv_bytes, mimetype="text/csv")
        uploaded = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
        ).execute()

        link = uploaded.get("webViewLink")
        print(f"Uploaded to Google Drive: {link}")
        return link

    except Exception as e:
        print(f"Google Drive upload failed: {e}")
        return None


# =========================================================================
# MAIN HANDLER
# =========================================================================

def handler(job):
    """
    RunPod serverless entry point.

    Expected job["input"] keys (all optional — defaults match optimize.py config):
      symbol                    str   e.g. "ETHUSDT.P"
      interval                  str   e.g. "3m"
      days_back                 int   e.g. 365
      metric_mode               str   "basic" or "advanced"
      initial_balance           float e.g. 100000
      drive_folder_id           str   Google Drive folder ID (or null)
      output_filename           str   Base name for the output CSV

      fvg_sensitivity_values    list  e.g. ["Extreme","High","Normal","Low"]
      swing_length_min          int   e.g. 3
      swing_length_max          int   e.g. 10
      swing_length_step         int   e.g. 1
      require_retracement_values list e.g. [false]
      tpsl_methods              list  e.g. ["Unicorn","Dynamic"]
      use_1to1rr_values         list  e.g. [true]
      risk_amount_values        list  e.g. ["Highest","High","Normal","Low","Lowest"]
      tp_percent_min/max/step   float Fixed-TP range (only for tpslMethod="Fixed")
      sl_percent_min/max/step   float Fixed-SL range (only for tpslMethod="Fixed")
    """
    job_input = job.get("input", {})

    # ── Parse job input ──────────────────────────────────────────────────────
    symbol          = job_input.get("symbol",          "ETHUSDT.P")
    interval        = job_input.get("interval",        "3m")
    days_back       = int(job_input.get("days_back",   365))
    metric_mode     = job_input.get("metric_mode",     "basic")
    initial_balance = float(job_input.get("initial_balance", 100_000))
    drive_folder_id = job_input.get("drive_folder_id", None)
    output_filename = job_input.get("output_filename", "optimization_results")

    fvg_sensitivity_values     = job_input.get("fvg_sensitivity_values",     ["Extreme", "High", "Normal", "Low"])
    swing_length_min           = int(job_input.get("swing_length_min",       3))
    swing_length_max           = int(job_input.get("swing_length_max",       10))
    swing_length_step          = int(job_input.get("swing_length_step",      1))
    require_retracement_values = job_input.get("require_retracement_values", [False])
    tpsl_methods               = job_input.get("tpsl_methods",               ["Unicorn", "Dynamic"])
    use_1to1rr_values          = job_input.get("use_1to1rr_values",          [True])
    risk_amount_values         = job_input.get("risk_amount_values",         ["Highest", "High", "Normal", "Low", "Lowest"])
    tp_percent_min             = float(job_input.get("tp_percent_min",       0.1))
    tp_percent_max             = float(job_input.get("tp_percent_max",       1.0))
    tp_percent_step            = float(job_input.get("tp_percent_step",      0.1))
    sl_percent_min             = float(job_input.get("sl_percent_min",       0.1))
    sl_percent_max             = float(job_input.get("sl_percent_max",       1.0))
    sl_percent_step            = float(job_input.get("sl_percent_step",      0.1))

    # ── 1. Download market data from Binance Vision ──────────────────────────
    print(f"Downloading {symbol} {interval} data ({days_back} days)...")
    df = download_from_vision(symbol=symbol, interval=interval, days_back=days_back)
    if df is None:
        return {"error": "Failed to download market data from Binance Vision"}

    # Rename lowercase → backtesting.py standard column names
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    print(f"Data loaded: {len(df):,} bars  |  {df.index[0].date()} → {df.index[-1].date()}")

    # ── 2. Build parameter combinations ──────────────────────────────────────
    swing_length_values = list(range(swing_length_min, swing_length_max + 1, swing_length_step))
    tp_percent_values   = [round(v, 4) for v in np.arange(tp_percent_min,  tp_percent_max  + 1e-9, tp_percent_step)]
    sl_percent_values   = [round(v, 4) for v in np.arange(sl_percent_min,  sl_percent_max  + 1e-9, sl_percent_step)]

    worker_args, total_combinations = build_combinations(
        fvg_sensitivity_values,
        swing_length_values,
        require_retracement_values,
        tpsl_methods,
        use_1to1rr_values,
        risk_amount_values,
        tp_percent_values,
        sl_percent_values,
        set(),  # No completed combinations — always fresh run on RunPod
    )
    print(f"Total combinations: {total_combinations:,}  |  To run: {len(worker_args):,}")

    # ── 3. Run optimization ───────────────────────────────────────────────────
    num_cores = int(os.environ.get("RUNPOD_MAX_CORES", cpu_count()))
    print(f"Running on {num_cores} CPU cores...")
    start_time = datetime.now()

    results_list = []
    with Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(df, metric_mode, initial_balance),
    ) as pool:
        for i, result in enumerate(pool.imap_unordered(run_single_backtest, worker_args), 1):
            if result is not None:
                results_list.append(result)
            if i % 50 == 0 or i == len(worker_args):
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                pct  = i / len(worker_args) * 100
                print(f"  [{i:,}/{len(worker_args):,}] {pct:.1f}%  |  {rate:.1f} combos/s")

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"Done in {elapsed_total:.0f}s  |  Valid results: {len(results_list):,}")

    if not results_list:
        return {"error": "No valid results — all combinations had fewer than 20 trades"}

    # ── 4. Process results ────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_list)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_cols] = results_df[numeric_cols].fillna(0)
    results_df = results_df.sort_values("Sharpe Ratio", ascending=False)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    # ── 5. Upload to Google Drive ─────────────────────────────────────────────
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{output_filename}_{symbol}_{interval}_{timestamp_str}.csv"
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")

    drive_url = upload_to_google_drive(csv_bytes, filename, drive_folder_id)

    # ── 6. Build response ─────────────────────────────────────────────────────
    best  = results_df.iloc[0].to_dict()
    top10 = results_df.head(10).to_dict("records")

    response = {
        "status":             "complete",
        "total_combinations": total_combinations,
        "valid_results":      len(results_df),
        "elapsed_seconds":    round(elapsed_total, 1),
        "output_filename":    filename,
        "drive_url":          drive_url,
        "top10":              top10,
        "best_params": {
            "tpslMethod":          best.get("tpslMethod"),
            "fvgSensitivity":      best.get("fvgSensitivity"),
            "swingLength":         int(best.get("swingLength", 0)),
            "requireRetracement":  bool(best.get("requireRetracement", False)),
            "use1to1RR":           bool(best.get("use1to1RR", True)),
            "riskAmount":          best.get("riskAmount"),
            "tpPercent":           best.get("tpPercent"),
            "slPercent":           best.get("slPercent"),
        },
        "best_metrics": {
            "Return [%]":        round(float(best.get("Return [%]",        0)), 4),
            "Sharpe Ratio":      round(float(best.get("Sharpe Ratio",      0)), 4),
            "Max. Drawdown [%]": round(float(best.get("Max. Drawdown [%]", 0)), 4),
            "# Trades":          int(best.get("# Trades", 0)),
            "Win Rate [%]":      round(float(best.get("Win Rate [%]",      0)), 4),
            "Profit Factor":     round(float(best.get("Profit Factor",     0)), 4),
        },
    }

    # Fallback: include base64-encoded CSV if Drive upload was not configured
    if not drive_url:
        response["csv_base64"] = base64.b64encode(csv_bytes).decode("utf-8")
        response["csv_note"] = (
            "Google Drive not configured. "
            "Decode csv_base64 with base64 to get the CSV file."
        )

    return response


# ── RunPod entry point ──
runpod.serverless.start({"handler": handler})
