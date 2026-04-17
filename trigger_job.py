"""
RunPod Job Trigger — ICT Unicorn Optimizer
==========================================
Sends an optimization job to your RunPod Serverless endpoint,
polls until complete, then saves the CSV result locally.

Setup:
  1. Set env vars:
       RUNPOD_API_KEY     = your RunPod API key
       RUNPOD_ENDPOINT_ID = your serverless endpoint ID
  2. Optionally edit DEFAULT_PAYLOAD below or pass a --config JSON file.

Usage:
  python trigger_job.py
  python trigger_job.py --config my_params.json
  python trigger_job.py --api-key rp_xxx --endpoint abc123
"""

import argparse
import base64
import json
import os
import sys
import time

import requests

# ── Credentials (override via env vars or CLI args) ──
RUNPOD_API_KEY     = os.environ.get("RUNPOD_API_KEY",     "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "bu8gmsydjm6dre")

# ── Default optimization parameters — matches optimize.py defaults ──
DEFAULT_PAYLOAD = {
    # Data
    "symbol":    "BTCUSDT.P",
    "interval":  "5m",
    "days_back": 365,

    # Metrics
    "metric_mode":     "advanced",   # "basic" or "advanced"
    "initial_balance": 1000,

    # Output
    "output_filename": "optimization_results",
    "drive_folder_id": None,      # Set to your Google Drive folder ID to upload there

    # Parameter ranges
    "fvg_sensitivity_values":     ["Extreme", "High", "Normal", "Low"],
    "swing_length_min":           3,
    "swing_length_max":           10,
    "swing_length_step":          1,
    "require_retracement_values": [False],
    "tpsl_methods":               ["Unicorn", "Dynamic"],
    "use_1to1rr_values":          [True],
    "risk_amount_values":         ["Highest", "High", "Normal", "Low", "Lowest"],

    # Fixed TP/SL range (only used when "Fixed" is in tpsl_methods)
    "tp_percent_min":  0.1,
    "tp_percent_max":  1.0,
    "tp_percent_step": 0.1,
    "sl_percent_min":  0.1,
    "sl_percent_max":  1.0,
    "sl_percent_step": 0.1,
}


# =========================================================================
# RUNPOD API HELPERS
# =========================================================================

def _headers():
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }


def submit_job(payload: dict) -> str:
    """Submit a job to the RunPod endpoint. Returns the job ID."""
    url  = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
    resp = requests.post(url, headers=_headers(), json={"input": payload}, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    return job_id


def poll_job(job_id: str, poll_interval: int = 5) -> dict | None:
    """
    Poll job status and show live elapsed timer while running.
    Prints any stream items (DATA/SETUP messages) if RunPod delivers them.
    Returns the final result dict on completion, None on failure.
    """
    import sys
    stream_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/stream/{job_id}"
    status_url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    print(f"\nJob {job_id} submitted. Waiting for results...\n")
    print("=" * 60)

    seen_count   = 0
    final_result = None
    start_time   = time.time()
    last_status  = None

    while True:
        resp = requests.get(stream_url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data   = resp.json()
        status = data.get("status")

        # Print any new stream items (e.g. DATA/SETUP messages from handler)
        stream_items = data.get("stream", [])
        new_items = stream_items[seen_count:]
        if new_items:
            sys.stdout.write("\r" + " " * 60 + "\r")  # clear timer line
            for item in new_items:
                output = item.get("output", {})
                if "error" in output:
                    print(f"\nERROR: {output['error']}")
                elif "msg" in output:
                    print(output["msg"])
                elif "status" in output and output.get("status") == "complete":
                    final_result = output
        seen_count = len(stream_items)

        if status == "COMPLETED":
            sys.stdout.write("\r" + " " * 60 + "\r")
            if final_result is None:
                # One extra stream fetch — the final yield may lag behind the
                # COMPLETED status update by one polling cycle.
                resp2      = requests.get(stream_url, headers=_headers(), timeout=30)
                data2      = resp2.json()
                remaining  = data2.get("stream", [])[seen_count:]
                for item in remaining:
                    output = item.get("output", {})
                    if isinstance(output, dict) and output.get("status") == "complete":
                        final_result = output
                        break

            if final_result is None:
                # Final fallback: status endpoint.
                # For generator handlers RunPod may return output as a list of
                # all yielded items — find the one with status == "complete".
                r2     = requests.get(status_url, headers=_headers(), timeout=30)
                output = r2.json().get("output")
                if isinstance(output, list):
                    for item in reversed(output):
                        if isinstance(item, dict) and item.get("status") == "complete":
                            final_result = item
                            break
                elif isinstance(output, dict):
                    final_result = output
            return final_result or {}

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.stdout.write("\r" + " " * 60 + "\r")
            print(f"\nJob ended with status: {status}")
            return None

        # Live elapsed timer
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        status_label = status or "PENDING"
        if status_label != last_status:
            last_status = status_label
        sys.stdout.write(f"\r  [{status_label}]  Elapsed: {h:02d}h {m:02d}m {s:02d}s  |  Polling every {poll_interval}s...")
        sys.stdout.flush()

        time.sleep(poll_interval)


# =========================================================================
# MAIN
# =========================================================================

def main():
    global RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID

    parser = argparse.ArgumentParser(description="Trigger ICT Unicorn optimization on RunPod")
    parser.add_argument("--config",   help="Path to JSON file with job parameters")
    parser.add_argument("--api-key",  help="RunPod API key (overrides env var)")
    parser.add_argument("--endpoint", help="RunPod endpoint ID (overrides env var)")
    parser.add_argument("--poll-interval", type=int, default=15,
                        help="Seconds between status polls (default: 15)")
    args = parser.parse_args()

    if args.api_key:
        RUNPOD_API_KEY = args.api_key
    if args.endpoint:
        RUNPOD_ENDPOINT_ID = args.endpoint

    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set. Export it or pass --api-key.")
        sys.exit(1)
    if not RUNPOD_ENDPOINT_ID:
        print("ERROR: RUNPOD_ENDPOINT_ID not set. Export it or pass --endpoint.")
        sys.exit(1)

    # Build payload
    payload = DEFAULT_PAYLOAD.copy()
    if args.config:
        with open(args.config) as f:
            payload.update(json.load(f))

    # Print job summary
    print("=" * 60)
    print("ICT UNICORN — RunPod Serverless Job")
    print("=" * 60)
    print(f"  Symbol       : {payload['symbol']} {payload['interval']}")
    print(f"  Days back    : {payload['days_back']}")
    print(f"  Methods      : {payload['tpsl_methods']}")
    print(f"  Metric mode  : {payload['metric_mode']}")
    print(f"  Endpoint     : {RUNPOD_ENDPOINT_ID}")
    print(f"  Drive folder : {payload.get('drive_folder_id') or 'Not configured'}")
    print("=" * 60)

    # Submit
    job_id = submit_job(payload)
    print(f"\nJob submitted: {job_id}")

    # Poll
    result = poll_job(job_id, poll_interval=args.poll_interval)
    if result is None:
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"  Status          : {result.get('status')}")
    print(f"  Total combos    : {result.get('total_combinations')}")
    print(f"  Valid results   : {result.get('valid_results')}")
    print(f"  Elapsed         : {result.get('elapsed_seconds')}s")
    print(f"  Output file     : {result.get('output_filename')}")

    if result.get("drive_url"):
        print(f"\n  Google Drive URL: {result['drive_url']}")

    # Save CSV locally if Drive was not used
    if result.get("csv_base64"):
        out_path = result.get("output_filename", "results.csv")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(result["csv_base64"]))
        print(f"\n  CSV saved locally: {out_path}")

    # Best params
    print("\n  Best Parameters:")
    for k, v in result.get("best_params", {}).items():
        print(f"    {k:<26}: {v}")

    print("\n  Best Metrics:")
    for k, v in result.get("best_metrics", {}).items():
        print(f"    {k:<26}: {v}")

    print("\n" + "=" * 60)
    print("To reproduce in unicorn.py:")
    bp = result.get("best_params", {})
    print(f"  UnicornStrategy.tpslMethod         = '{bp.get('tpslMethod')}'")
    print(f"  UnicornStrategy.fvgSensitivity     = '{bp.get('fvgSensitivity')}'")
    print(f"  UnicornStrategy.swingLength        = {bp.get('swingLength')}")
    print(f"  UnicornStrategy.requireRetracement = {bp.get('requireRetracement')}")
    print(f"  UnicornStrategy.use1to1RR          = {bp.get('use1to1RR')}")
    print(f"  UnicornStrategy.riskAmount         = '{bp.get('riskAmount')}'")
    print(f"  UnicornStrategy.tpPercent          = {bp.get('tpPercent')}")
    print(f"  UnicornStrategy.slPercent          = {bp.get('slPercent')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
