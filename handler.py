"""
RunPod Serverless Handler — ICT Unicorn Strategy Optimizer
==========================================================
Generator handler — streams real-time progress back to trigger_job.py
exactly like optimize.py does locally.
"""

import runpod
import json
import os
import base64
import importlib.util
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Suppress tqdm progress bars from FractionalBacktest.run in all worker processes.
# Must be set in the main process before Pool fork so workers inherit it.
os.environ['TQDM_DISABLE'] = '1'

import numpy as np
import pandas as pd

from unicorn import UnicornStrategy  # noqa: F401
from optimize import build_combinations, run_single_backtest, init_worker

# ── Import Binance-Vision downloader (hyphen in filename requires importlib) ──
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
    sa_json_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_json_b64:
        return None
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaInMemoryUpload

        creds_dict = json.loads(base64.b64decode(sa_json_b64).decode())
        creds = Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds)
        file_metadata = {"name": filename}
        if folder_id:
            file_metadata["parents"] = [folder_id]
        media = MediaInMemoryUpload(csv_bytes, mimetype="text/csv")
        uploaded = service.files().create(
            body=file_metadata, media_body=media, fields="id, webViewLink"
        ).execute()
        return uploaded.get("webViewLink")
    except Exception as e:
        print(f"Google Drive upload failed: {e}")
        return None


# =========================================================================
# GENERATOR HANDLER — streams progress like optimize.py
# =========================================================================

def handler(job):
    """
    Generator handler — yields progress updates during the run so
    trigger_job.py can display them in real time in the CMD window.
    Final yield is the full result dict.
    """
    job_input = job.get("input", {})

    # ── Parse parameters ─────────────────────────────────────────────────
    symbol          = job_input.get("symbol",          "ETHUSDT.P")
    interval        = job_input.get("interval",        "3m")
    # NOTE: use `job_input.get(key) or default` instead of `job_input.get(key, default)`
    # so that an explicit null value in the JSON payload falls back to the default too.
    # dict.get(key, default) only uses default when the key is ABSENT; if the key is
    # present with a null/None value, it returns None — causing float(None) TypeError.
    days_back       = int(job_input.get("days_back")       or 365)
    metric_mode     = job_input.get("metric_mode")         or "basic"
    initial_balance = float(job_input.get("initial_balance") or 100_000)
    drive_folder_id = job_input.get("drive_folder_id")     or None
    output_filename = job_input.get("output_filename")     or "optimization_results"

    fvg_sensitivity_values     = job_input.get("fvg_sensitivity_values")     or ["Extreme", "High", "Normal", "Low"]
    swing_length_min           = int(job_input.get("swing_length_min")       or 3)
    swing_length_max           = int(job_input.get("swing_length_max")       or 10)
    swing_length_step          = int(job_input.get("swing_length_step")      or 1)
    require_retracement_values = job_input.get("require_retracement_values") or [False]
    tpsl_methods               = job_input.get("tpsl_methods")               or ["Unicorn", "Dynamic"]
    use_1to1rr_values          = job_input.get("use_1to1rr_values")          or [True]
    risk_amount_values         = job_input.get("risk_amount_values")         or ["Highest", "High", "Normal", "Low", "Lowest"]
    tp_percent_min             = float(job_input.get("tp_percent_min")       or 0.1)
    tp_percent_max             = float(job_input.get("tp_percent_max")       or 1.0)
    tp_percent_step            = float(job_input.get("tp_percent_step")      or 0.1)
    sl_percent_min             = float(job_input.get("sl_percent_min")       or 0.1)
    sl_percent_max             = float(job_input.get("sl_percent_max")       or 1.0)
    sl_percent_step            = float(job_input.get("sl_percent_step")      or 0.1)

    # ── 1. Download data ──────────────────────────────────────────────────
    yield {"stage": "DATA", "msg": f"Downloading {symbol} {interval} ({days_back} days) from Binance Vision..."}

    df = download_from_vision(symbol=symbol, interval=interval, days_back=days_back)
    if df is None:
        yield {"error": "Failed to download market data from Binance Vision"}
        return

    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    yield {"stage": "DATA", "msg": f"Data loaded: {len(df):,} bars  |  {df.index[0].date()} -> {df.index[-1].date()}"}

    # ── 2. Build combinations ─────────────────────────────────────────────
    swing_length_values = list(range(swing_length_min, swing_length_max + 1, swing_length_step))
    tp_percent_values   = [round(v, 4) for v in np.arange(tp_percent_min,  tp_percent_max  + 1e-9, tp_percent_step)]
    sl_percent_values   = [round(v, 4) for v in np.arange(sl_percent_min,  sl_percent_max  + 1e-9, sl_percent_step)]

    worker_args, total_combinations = build_combinations(
        fvg_sensitivity_values, swing_length_values, require_retracement_values,
        tpsl_methods, use_1to1rr_values, risk_amount_values,
        tp_percent_values, sl_percent_values, set(),
    )

    num_cores = int(os.environ.get("RUNPOD_MAX_CORES", cpu_count()))

    yield {
        "stage": "SETUP",
        "msg": (
            f"{'='*60}\n"
            f"  Total combinations : {total_combinations:,}\n"
            f"  CPU cores          : {num_cores}\n"
            f"  Metric mode        : {metric_mode}\n"
            f"  Bars loaded        : {len(df):,}\n"
            f"  Methods            : {tpsl_methods}\n"
            f"  fvgSensitivity     : {fvg_sensitivity_values}\n"
            f"  swingLength        : {swing_length_min}-{swing_length_max} step {swing_length_step}\n"
            f"  riskAmount         : {risk_amount_values}\n"
            f"{'='*60}"
        )
    }

    # ── 3. Run optimization ───────────────────────────────────────────────
    yield {"stage": "OPT", "msg": f"Starting optimization... ({total_combinations:,} combinations)"}

    results_list = []
    start_time   = datetime.now()
    processed    = 0

    with Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(df, metric_mode, initial_balance),
    ) as pool:
        for result in pool.imap_unordered(run_single_backtest, worker_args):
            processed += 1
            if result is not None and 'Error' not in result:
                results_list.append(result)

            # Stream progress every 10 combinations — mirrors optimize.py display
            if processed % 10 == 0 or processed == len(worker_args):
                elapsed  = (datetime.now() - start_time).total_seconds()
                rate     = processed / elapsed if elapsed > 0 else 0
                remaining_est = (len(worker_args) - processed) / rate if rate > 0 else 0
                pct      = processed / len(worker_args) * 100
                eta_h    = int(remaining_est // 3600)
                eta_m    = int((remaining_est % 3600) // 60)
                eta_s    = int(remaining_est % 60)

                msg = (
                    f"  [{processed:>5,}/{len(worker_args):,}] "
                    f"{pct:5.1f}%  |  "
                    f"{rate:5.1f} combos/s  |  "
                    f"Valid: {len(results_list):,}  |  "
                    f"ETA: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s"
                )
                print(msg, flush=True)
                yield {"stage": "OPT", "msg": msg}

    elapsed_total = (datetime.now() - start_time).total_seconds()
    h = int(elapsed_total // 3600)
    m = int((elapsed_total % 3600) // 60)
    s = int(elapsed_total % 60)
    yield {"stage": "OPT", "msg": f"Optimization done in {h:02d}h {m:02d}m {s:02d}s  |  Valid results: {len(results_list):,}"}

    if not results_list:
        yield {"error": (
            f"No valid results — all {total_combinations} combinations had fewer than 20 trades. "
            f"Bars loaded: {len(df):,}. "
            f"Try a higher timeframe (5m/15m), more days_back, or a different symbol."
        )}
        return

    # ── 4. Process & export ───────────────────────────────────────────────
    yield {"stage": "EXPORT", "msg": "Processing results and exporting CSV..."}

    results_df   = pd.DataFrame(results_list)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    # fillna(0) handles NaN; replace handles inf/-inf (e.g. Profit Factor = inf
    # when there are no losing trades) — both would break JSON serialisation.
    results_df[numeric_cols] = (
        results_df[numeric_cols]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
    )
    # Nullable metrics (Sortino, Calmar, etc.) store Python None, which makes pandas
    # infer object dtype — select_dtypes misses them and fillna above skips them.
    # Force-coerce them to numeric so they don't propagate as None into best_metrics.
    nullable_metric_cols = [
        'Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]',
        'Win Rate [%]', 'Profit Factor', 'Avg. Trade [%]', 'Exposure Time [%]',
        'Sortino Ratio', 'Calmar Ratio', 'Expectancy [%]',
        'Best Trade [%]', 'Worst Trade [%]',
        'Max. Absolute DD [%]', 'Avg Win / Avg Loss',
        'Max Win Streak', 'Max Loss Streak',
        'Long Win Rate [%]', 'Long PnL [%]',
        'Short Win Rate [%]', 'Short PnL [%]',
    ]
    for col in nullable_metric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

    results_df   = results_df.sort_values("Sharpe Ratio", ascending=False)
    results_df.insert(0, "Rank", range(1, len(results_df) + 1))

    filename      = f"{symbol}-{interval}-{days_back}.csv"
    csv_bytes     = results_df.to_csv(index=False).encode("utf-8")

    drive_url = upload_to_google_drive(csv_bytes, filename, drive_folder_id)

    # ── 5. Build final result ─────────────────────────────────────────────
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
            "tpslMethod":         best.get("tpslMethod"),
            "fvgSensitivity":     best.get("fvgSensitivity"),
            "swingLength":        int(best.get("swingLength", 0)),
            "requireRetracement": bool(best.get("requireRetracement", False)),
            "use1to1RR":          bool(best.get("use1to1RR", True)),
            "riskAmount":         best.get("riskAmount"),
            "tpPercent":          best.get("tpPercent"),
            "slPercent":          best.get("slPercent"),
        },
        "best_metrics": {
            "Return [%]":        round(float(best.get("Return [%]")        or 0), 4),
            "Sharpe Ratio":      round(float(best.get("Sharpe Ratio")      or 0), 4),
            "Max. Drawdown [%]": round(float(best.get("Max. Drawdown [%]") or 0), 4),
            "# Trades":          int(  best.get("# Trades")                or 0),
            "Win Rate [%]":      round(float(best.get("Win Rate [%]")      or 0), 4),
            "Profit Factor":     round(float(best.get("Profit Factor")     or 0), 4),
        },
    }

    if not drive_url:
        response["csv_base64"] = base64.b64encode(csv_bytes).decode("utf-8")
        response["csv_note"]   = "Decode csv_base64 to get the CSV file."

    yield response


runpod.serverless.start({"handler": handler})
