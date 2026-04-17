"""
ICT UNICORN STRATEGY OPTIMIZER — Full Parameter Optimization
Fully matched with unicorn.py parameters and logic.
Features: Checkpoint/Resume, Parallel Processing, Two-Tier Metrics.

NOTE ON COMBINATION GENERATION:
  This optimizer separates combinations by tpslMethod to avoid useless
  cross-products (e.g., riskAmount does nothing when tpslMethod="Fixed").
  Three pools are generated independently, then merged into one master list.
"""

from backtesting.lib import FractionalBacktest as Backtest
import pandas as pd
import numpy as np
from unicorn import UnicornStrategy
import warnings
import os
import json
from datetime import datetime
import tempfile
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


# ===== ⚙️ CONFIGURATION SECTION - EDIT THESE SETTINGS =====

# ----- DATA SOURCE CONFIGURATION -----
USE_DEFAULT_CSV  = True
DEFAULT_CSV_FILE = 'Data/ETHUSDT.P-3m-365.csv'

# ----- PARAMETER CONFIGURATION -----
USE_DEFAULT_PARAMETERS = True

# === FVG Sensitivity (string enum — maps to atr multiplier inside strategy) ===
# Options: "Extreme", "High", "Normal", "Low"
FVG_SENSITIVITY_VALUES = ["Extreme", "High", "Normal", "Low"]

# === Swing Length ===
SWINGLENGTH_MIN  = 3
SWINGLENGTH_MAX  = 10
SWINGLENGTH_STEP = 1

# === Require Retracement (True/False) ===
REQUIRE_RETRACEMENT_VALUES = [False]

# === TP/SL Method ===
# Options: "Unicorn", "Dynamic", "Fixed"
# The optimizer generates separate combination pools per method — see below.
TPSL_METHODS = ["Unicorn", "Dynamic"]

# === use1to1RR — applied to Unicorn and Dynamic methods only ===
USE_1TO1RR_VALUES = [True]

# === riskAmount — applied to Unicorn and Dynamic methods only ===
# Options: "Highest", "High", "Normal", "Low", "Lowest"
RISK_AMOUNT_VALUES = ["Highest", "High", "Normal", "Low", "Lowest"]

# === Fixed TP % range — only used when tpslMethod = "Fixed" ===
TP_PERCENT_MIN  = 0.1
TP_PERCENT_MAX  = 1.0
TP_PERCENT_STEP = 0.1

# === Fixed SL % range — only used when tpslMethod = "Fixed" ===
SL_PERCENT_MIN  = 0.1
SL_PERCENT_MAX  = 1.0
SL_PERCENT_STEP = 0.1

# ----- METRIC CONFIGURATION -----
USE_BASIC_METRICS = True   # True = BASIC (fast), False = ADVANCED (comprehensive)

# ----- PERFORMANCE CONFIGURATION -----
USE_ALL_CPU_CORES = True
MANUAL_CPU_CORES  = 4

# ----- CHECKPOINT CONFIGURATION -----
CHECKPOINT_ENABLED            = True
CHECKPOINT_INTERVAL_BASIC     = 100
CHECKPOINT_INTERVAL_ADVANCED  = 10

# ----- PROGRESS DISPLAY CONFIGURATION -----
PROGRESS_DISPLAY_INTERVAL = 10

# ===== END OF CONFIGURATION SECTION =====


# ===== CHECKPOINT FILES =====
CHECKPOINT_FILE     = 'unicorn_optimize_checkpoint.csv'
CHECKPOINT_METADATA = 'unicorn_optimize_checkpoint_meta.json'


# ===== GLOBAL VARIABLES =====
global_df              = None
global_metric_mode     = 'advanced'
global_initial_balance = 100000


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def init_worker(df_data, metric_mode, initial_balance):
    """Initialize each worker process with shared read-only globals."""
    global global_df, global_metric_mode, global_initial_balance
    global_df              = df_data
    global_metric_mode     = metric_mode
    global_initial_balance = initial_balance
    os.environ['TQDM_DISABLE'] = '1'


def calculate_absolute_drawdown(equity_df, initial_balance):
    """Drawdown from initial balance (not rolling peak — absolute floor view)."""
    if equity_df is None or len(equity_df) == 0:
        return None
    try:
        min_equity  = equity_df['Equity'].min()
        absolute_dd = ((initial_balance - min_equity) / initial_balance) * 100
        return -abs(absolute_dd)
    except Exception:
        return None


def calculate_streak_metrics(trades_df):
    """Compute maximum consecutive winning and losing streaks."""
    if trades_df is None or len(trades_df) == 0:
        return None, None
    returns          = trades_df['ReturnPct'] if 'ReturnPct' in trades_df.columns else trades_df['PnL']
    is_win           = returns > 0
    max_win_streak   = max_loss_streak = 0
    curr_win_streak  = curr_loss_streak = 0
    for win in is_win:
        if win:
            curr_win_streak  += 1
            curr_loss_streak  = 0
            max_win_streak    = max(max_win_streak,  curr_win_streak)
        else:
            curr_loss_streak += 1
            curr_win_streak   = 0
            max_loss_streak   = max(max_loss_streak, curr_loss_streak)
    return max_win_streak, max_loss_streak


def calculate_avg_win_loss_ratio(trades_df):
    """Average winning trade / absolute average losing trade."""
    if trades_df is None or len(trades_df) == 0:
        return None
    returns  = trades_df['ReturnPct'] if 'ReturnPct' in trades_df.columns else trades_df['PnL']
    winning  = returns[returns > 0]
    losing   = returns[returns < 0]
    if len(winning) == 0 or len(losing) == 0:
        return None
    avg_loss = abs(losing.mean())
    return winning.mean() / avg_loss if avg_loss != 0 else None


def calculate_winning_losing_counts(trades_df):
    """Raw count of winning and losing trades."""
    if trades_df is None or len(trades_df) == 0:
        return 0, 0
    returns = trades_df['ReturnPct'] if 'ReturnPct' in trades_df.columns else trades_df['PnL']
    return len(returns[returns > 0]), len(returns[returns < 0])


def calculate_direction_metrics(trades_df):
    """Long vs Short win rate (%) and total PnL."""
    if trades_df is None or len(trades_df) == 0:
        return None, None, None, None
    long_trades  = trades_df[trades_df['Size'] > 0]
    short_trades = trades_df[trades_df['Size'] < 0]
    long_wr = long_pnl = short_wr = short_pnl = None
    if len(long_trades) > 0:
        long_wr  = (long_trades['PnL']  > 0).sum() / len(long_trades)  * 100
        long_pnl = long_trades['PnL'].sum()
    if len(short_trades) > 0:
        short_wr  = (short_trades['PnL'] > 0).sum() / len(short_trades) * 100
        short_pnl = short_trades['PnL'].sum()
    return long_wr, long_pnl, short_wr, short_pnl


# =========================================================================
# WORKER FUNCTION
# =========================================================================

def run_single_backtest(args):
    """
    Worker — executes one parameter combination and returns a result dict.
    Parameter dict keys must match UnicornStrategy class attribute names exactly.
    """
    combination_num, params = args
    try:
        global global_df, global_metric_mode, global_initial_balance

        bt_worker = Backtest(
            global_df,
            UnicornStrategy,
            cash=global_initial_balance,
            commission=0.0002,
            exclusive_orders=True,
            trade_on_close=True,
            margin=0.00001
        )
        stats = bt_worker.run(**params)

        # Hard gate — discard statistically unreliable results before building dict.
        # Saves memory, speeds up results processing, and keeps CSV clean.
        if stats['# Trades'] < 20:
            return None

        result = {'Combination': combination_num, **params}

        # ── BASIC METRICS (always collected) ──
        result.update({
            'Return [%]':          stats['Return [%]'],
            'Sharpe Ratio':        stats['Sharpe Ratio'],
            'Sortino Ratio':       stats.get('Sortino Ratio',  None),
            'Calmar Ratio':        stats.get('Calmar Ratio',   None),
            'Max. Drawdown [%]':   stats['Max. Drawdown [%]'],
            '# Trades':            stats['# Trades'],
            'Win Rate [%]':        stats['Win Rate [%]'],
            'Profit Factor':       stats['Profit Factor'],
            'Expectancy [%]':      stats.get('Expectancy [%]', None),
            'Avg. Trade [%]':      stats['Avg. Trade [%]'],
            'Avg. Trade Duration': str(stats['Avg. Trade Duration']),
            'Exposure Time [%]':   stats['Exposure Time [%]'],
        })

        # ── ADVANCED METRICS (optional) ──
        if global_metric_mode == 'advanced':
            trades_df = stats.get('_trades',       None)
            equity_df = stats.get('_equity_curve', None)

            max_win_streak, max_loss_streak = calculate_streak_metrics(trades_df)
            avg_win_loss                    = calculate_avg_win_loss_ratio(trades_df)
            winning_cnt, losing_cnt         = calculate_winning_losing_counts(trades_df)
            long_wr, long_pnl, short_wr, short_pnl = calculate_direction_metrics(trades_df)
            absolute_dd                     = calculate_absolute_drawdown(equity_df, global_initial_balance)

            result.update({
                'Max. Absolute DD [%]': absolute_dd,
                'Avg Win / Avg Loss':   avg_win_loss,
                'Max Loss Streak':      max_loss_streak,
                'Max Win Streak':       max_win_streak,
                'Winning Trades':       winning_cnt,
                'Losing Trades':        losing_cnt,
                'Best Trade [%]':       stats.get('Best Trade [%]',  None),
                'Worst Trade [%]':      stats.get('Worst Trade [%]', None),
                'Max. Trade Duration':  str(stats['Max. Trade Duration']),
                'Long Win Rate [%]':    long_wr,
                'Long Total PnL':       long_pnl,
                'Short Win Rate [%]':   short_wr,
                'Short Total PnL':      short_pnl,
            })

        return result

    except Exception as e:
        return {
            'Combination':   combination_num,
            **params,
            'Return [%]':    None,
            'Sharpe Ratio':  None,
            'Error':         str(e),
        }


# =========================================================================
# CHECKPOINT FUNCTIONS
# =========================================================================

def load_checkpoint():
    """Load prior run from checkpoint files if they exist and user confirms."""
    if os.path.exists(CHECKPOINT_FILE) and os.path.exists(CHECKPOINT_METADATA):
        if os.path.getsize(CHECKPOINT_METADATA) == 0:
            return None, set(), None

        print("\n" + "=" * 60)
        print("📂 CHECKPOINT FOUND!")
        print("=" * 60)

        with open(CHECKPOINT_METADATA, 'r') as f:
            metadata = json.load(f)

        results_df = pd.read_csv(CHECKPOINT_FILE)
        completed  = set(results_df['Combination'].tolist())

        print(f"Previously completed : {len(completed)} combinations")
        print(f"Started at           : {metadata['start_time']}")
        print(f"Last saved           : {metadata['last_save_time']}")

        response = input("\nResume from checkpoint? (y/n): ").strip().lower()
        if response == 'y':
            return results_df, completed, metadata

    return None, set(), None


def save_checkpoint(results_df, metadata):
    """Atomic checkpoint save — write to temp file then rename to avoid corruption."""
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, dir=".", suffix=".tmp") as tf:
            results_df.to_csv(tf.name, index=False)
            temp_csv = tf.name
        os.replace(temp_csv, CHECKPOINT_FILE)

        metadata['last_save_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with tempfile.NamedTemporaryFile('w', delete=False, dir=".", suffix=".tmp") as tf:
            json.dump(metadata, tf, indent=2)
            temp_json = tf.name
        os.replace(temp_json, CHECKPOINT_METADATA)

        print(f"💾 Checkpoint saved: {len(results_df)} combinations")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")


# =========================================================================
# COMBINATION GENERATOR
# =========================================================================

def build_combinations(
    fvg_sensitivity_values,
    swing_length_values,
    require_retracement_values,
    tpsl_methods,
    use_1to1rr_values,
    risk_amount_values,
    tp_percent_values,
    sl_percent_values,
    completed_combinations,
):
    """
    Generate all valid parameter dicts, separated by tpslMethod pool to avoid
    useless cross-products.

    Pool A — "Unicorn" and "Dynamic":
        Iterate: fvgSensitivity × swingLength × requireRetracement
                 × use1to1RR × riskAmount

    Pool B — "Fixed":
        Iterate: fvgSensitivity × swingLength × requireRetracement
                 × tpPercent × slPercent
        (riskAmount and use1to1RR are irrelevant for Fixed — kept at defaults)

    Returns: (worker_args, total_combinations)
    """
    worker_args = []
    combo_num   = 0

    # ── Pool A: "Unicorn" and "Dynamic" methods ──
    methods_ab = [m for m in tpsl_methods if m in ("Unicorn", "Dynamic")]
    for method in methods_ab:
        for fvg_sens in fvg_sensitivity_values:
            for swing in swing_length_values:
                for req_ret in require_retracement_values:
                    for use_rr in use_1to1rr_values:
                        for risk_amt in risk_amount_values:
                            combo_num += 1
                            if combo_num not in completed_combinations:
                                params = {
                                    'tpslMethod':          method,
                                    'fvgSensitivity':      fvg_sens,
                                    'swingLength':         swing,
                                    'requireRetracement':  req_ret,
                                    'use1to1RR':           use_rr,
                                    'riskAmount':          risk_amt,
                                    # Fixed-mode params kept at strategy defaults
                                    'tpPercent':           0.3,
                                    'slPercent':           0.4,
                                }
                                worker_args.append((combo_num, params))

    # ── Pool B: "Fixed" method ──
    if "Fixed" in tpsl_methods:
        for fvg_sens in fvg_sensitivity_values:
            for swing in swing_length_values:
                for req_ret in require_retracement_values:
                    for tp_pct in tp_percent_values:
                        for sl_pct in sl_percent_values:
                            combo_num += 1
                            if combo_num not in completed_combinations:
                                params = {
                                    'tpslMethod':          'Fixed',
                                    'fvgSensitivity':      fvg_sens,
                                    'swingLength':         swing,
                                    'requireRetracement':  req_ret,
                                    # Irrelevant for Fixed but must be present
                                    'use1to1RR':           True,
                                    'riskAmount':          'Normal',
                                    'tpPercent':           round(tp_pct, 4),
                                    'slPercent':           round(sl_pct, 4),
                                }
                                worker_args.append((combo_num, params))

    total = combo_num
    return worker_args, total


# =========================================================================
# MAIN BLOCK — INTERACTIVE CLI
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ICT UNICORN — FULL PARAMETER OPTIMIZATION")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # STEP 1: DATA SOURCE
    # ──────────────────────────────────────────────────────────
    print("\n📁 DATA SOURCE CONFIGURATION")
    print("-" * 40)

    if USE_DEFAULT_CSV:
        print(f"[1] Default CSV: {DEFAULT_CSV_FILE}")
        print("[2] Enter custom CSV path")
        data_choice = input("\nSelect option (1/2) [default=1]: ").strip() or "1"
        if data_choice == "2":
            csv_file = input("Enter path to CSV file: ").strip()
        else:
            csv_file = DEFAULT_CSV_FILE
    else:
        csv_file = input("Enter path to CSV file: ").strip()

    print(f"\nLoading data from: {csv_file}")
    df = pd.read_csv(csv_file)

    df.rename(columns={
        'timestamp': 'timestamp',
        'open':      'Open',
        'high':      'High',
        'low':       'Low',
        'close':     'Close',
        'volume':    'Volume'
    }, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    print(f"✅ Data loaded: {len(df):,} bars  |  "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    # ──────────────────────────────────────────────────────────
    # STEP 2: METRIC DEPTH
    # ──────────────────────────────────────────────────────────
    print("\n📊 METRIC CONFIGURATION")
    print("-" * 40)
    print("[1] BASIC  — Core metrics only (faster)")
    print("[2] ADVANCED — Full metrics including streaks, direction breakdown, abs drawdown")

    if USE_BASIC_METRICS:
        metric_choice = input("\nSelect option (1/2) [default=1]: ").strip() or "1"
    else:
        metric_choice = input("\nSelect option (1/2) [default=2]: ").strip() or "2"

    metric_mode = 'basic' if metric_choice == "1" else 'advanced'
    print(f"✅ Metric mode: {metric_mode.upper()}")

    # ──────────────────────────────────────────────────────────
    # CHECK CHECKPOINT
    # ──────────────────────────────────────────────────────────
    existing_results, completed_combinations, checkpoint_metadata = load_checkpoint()

    # ──────────────────────────────────────────────────────────
    # STEP 3: PARAMETER CONFIGURATION
    # ──────────────────────────────────────────────────────────
    print("\n⚙️  PARAMETER CONFIGURATION")
    print("-" * 40)
    print("[1] Use default parameter ranges (from configuration section)")
    print("[2] Enter custom parameter ranges")

    param_choice = input("\nSelect option (1/2) [default=1]: ").strip() or "1"

    if param_choice == "1":
        # ── Build from CONFIG constants ──
        fvg_sensitivity_values    = FVG_SENSITIVITY_VALUES
        swing_length_values       = list(range(SWINGLENGTH_MIN, SWINGLENGTH_MAX + 1, SWINGLENGTH_STEP))
        require_retracement_values = REQUIRE_RETRACEMENT_VALUES
        tpsl_methods              = TPSL_METHODS
        use_1to1rr_values         = USE_1TO1RR_VALUES
        risk_amount_values        = RISK_AMOUNT_VALUES
        tp_percent_values         = [
            round(v, 4) for v in np.arange(TP_PERCENT_MIN, TP_PERCENT_MAX + 1e-9, TP_PERCENT_STEP)
        ]
        sl_percent_values         = [
            round(v, 4) for v in np.arange(SL_PERCENT_MIN, SL_PERCENT_MAX + 1e-9, SL_PERCENT_STEP)
        ]

    else:
        # ── Custom parameter entry ──
        print("\n--- FVG Sensitivity (comma-separated from: Extreme, High, Normal, Low) ---")
        raw = input(f"[default: {','.join(FVG_SENSITIVITY_VALUES)}]: ").strip()
        fvg_sensitivity_values = [s.strip() for s in raw.split(',')] if raw else FVG_SENSITIVITY_VALUES

        print("\n--- Swing Length ---")
        sl_min  = int(input(f"  Min  [default: {SWINGLENGTH_MIN}]: ").strip()  or SWINGLENGTH_MIN)
        sl_max  = int(input(f"  Max  [default: {SWINGLENGTH_MAX}]: ").strip()  or SWINGLENGTH_MAX)
        sl_step = int(input(f"  Step [default: {SWINGLENGTH_STEP}]: ").strip() or SWINGLENGTH_STEP)
        swing_length_values = list(range(sl_min, sl_max + 1, sl_step))

        print("\n--- Require Retracement (comma-separated True/False) ---")
        raw = input("[default: False,True]: ").strip()
        if raw:
            require_retracement_values = [s.strip().lower() == 'true' for s in raw.split(',')]
        else:
            require_retracement_values = REQUIRE_RETRACEMENT_VALUES

        print("\n--- TP/SL Methods (comma-separated from: Unicorn, Dynamic, Fixed) ---")
        raw = input(f"[default: {','.join(TPSL_METHODS)}]: ").strip()
        tpsl_methods = [s.strip() for s in raw.split(',')] if raw else TPSL_METHODS

        print("\n--- use1to1RR — Unicorn/Dynamic only (comma-separated True/False) ---")
        raw = input("[default: True,False]: ").strip()
        if raw:
            use_1to1rr_values = [s.strip().lower() == 'true' for s in raw.split(',')]
        else:
            use_1to1rr_values = USE_1TO1RR_VALUES

        print("\n--- riskAmount — Unicorn/Dynamic only (comma-separated) ---")
        raw = input(f"[default: {','.join(RISK_AMOUNT_VALUES)}]: ").strip()
        risk_amount_values = [s.strip() for s in raw.split(',')] if raw else RISK_AMOUNT_VALUES

        print("\n--- Fixed TP % (only used when tpslMethod='Fixed') ---")
        tp_min  = float(input(f"  Min  [default: {TP_PERCENT_MIN}]: ").strip()  or TP_PERCENT_MIN)
        tp_max  = float(input(f"  Max  [default: {TP_PERCENT_MAX}]: ").strip()  or TP_PERCENT_MAX)
        tp_step = float(input(f"  Step [default: {TP_PERCENT_STEP}]: ").strip() or TP_PERCENT_STEP)
        tp_percent_values = [round(v, 4) for v in np.arange(tp_min, tp_max + 1e-9, tp_step)]

        print("\n--- Fixed SL % (only used when tpslMethod='Fixed') ---")
        sl_min_f  = float(input(f"  Min  [default: {SL_PERCENT_MIN}]: ").strip()  or SL_PERCENT_MIN)
        sl_max_f  = float(input(f"  Max  [default: {SL_PERCENT_MAX}]: ").strip()  or SL_PERCENT_MAX)
        sl_step_f = float(input(f"  Step [default: {SL_PERCENT_STEP}]: ").strip() or SL_PERCENT_STEP)
        sl_percent_values = [round(v, 4) for v in np.arange(sl_min_f, sl_max_f + 1e-9, sl_step_f)]

    # ──────────────────────────────────────────────────────────
    # BUILD COMBINATIONS
    # ──────────────────────────────────────────────────────────
    worker_args, total_combinations = build_combinations(
        fvg_sensitivity_values,
        swing_length_values,
        require_retracement_values,
        tpsl_methods,
        use_1to1rr_values,
        risk_amount_values,
        tp_percent_values,
        sl_percent_values,
        completed_combinations,
    )

    remaining = len(worker_args)

    # ──────────────────────────────────────────────────────────
    # PARAMETER BREAKDOWN
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 PARAMETER BREAKDOWN")
    print("=" * 60)
    print(f"  fvgSensitivity      : {fvg_sensitivity_values}")
    print(f"  swingLength         : {swing_length_values}")
    print(f"  requireRetracement  : {require_retracement_values}")
    print(f"  tpslMethod          : {tpsl_methods}")
    print(f"  use1to1RR           : {use_1to1rr_values}  (Unicorn/Dynamic only)")
    print(f"  riskAmount          : {risk_amount_values}  (Unicorn/Dynamic only)")
    print(f"  tpPercent range     : {tp_percent_values[:3]}...{tp_percent_values[-1]}  (Fixed only)")
    print(f"  slPercent range     : {sl_percent_values[:3]}...{sl_percent_values[-1]}  (Fixed only)")
    print("-" * 60)

    # Pool size breakdown
    methods_ab = [m for m in tpsl_methods if m in ("Unicorn", "Dynamic")]
    pool_ab = (
        len(methods_ab) *
        len(fvg_sensitivity_values) *
        len(swing_length_values) *
        len(require_retracement_values) *
        len(use_1to1rr_values) *
        len(risk_amount_values)
    )
    pool_fixed = 0
    if "Fixed" in tpsl_methods:
        pool_fixed = (
            len(fvg_sensitivity_values) *
            len(swing_length_values) *
            len(require_retracement_values) *
            len(tp_percent_values) *
            len(sl_percent_values)
        )

    print(f"  Pool A (Unicorn+Dynamic) : {pool_ab:,} combinations")
    print(f"  Pool B (Fixed)           : {pool_fixed:,} combinations")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL combinations       : {total_combinations:,}")
    print(f"  Already completed        : {len(completed_combinations):,}")
    print(f"  Remaining to run         : {remaining:,}")

    if remaining == 0:
        print("\n✅ All combinations already completed! Loading checkpoint results...")
        all_results_df = existing_results
    else:
        # ──────────────────────────────────────────────────────
        # CPU CONFIGURATION
        # ──────────────────────────────────────────────────────
        available_cores = cpu_count()
        if USE_ALL_CPU_CORES:
            num_cores = available_cores
        else:
            num_cores = min(MANUAL_CPU_CORES, available_cores)

        print(f"\n💻 Using {num_cores} CPU cores (of {available_cores} available)")

        # ──────────────────────────────────────────────────────
        # CHECKPOINT METADATA INIT
        # ──────────────────────────────────────────────────────
        if checkpoint_metadata is None:
            checkpoint_metadata = {
                'start_time':       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'last_save_time':   None,
                'total_combinations': total_combinations,
                'metric_mode':      metric_mode,
                'csv_file':         csv_file,
            }

        checkpoint_interval = (
            CHECKPOINT_INTERVAL_BASIC if metric_mode == 'basic'
            else CHECKPOINT_INTERVAL_ADVANCED
        )

        # ──────────────────────────────────────────────────────
        # RUN OPTIMIZATION
        # ──────────────────────────────────────────────────────
        print(f"\n🚀 Starting optimization... ({remaining:,} combinations remaining)")
        print("=" * 60)

        results_list = []
        if existing_results is not None:
            results_list = existing_results.to_dict('records')

        completed_count   = len(completed_combinations)
        processed_in_run  = 0
        start_time        = datetime.now()

        try:
            with Pool(
                processes=num_cores,
                initializer=init_worker,
                initargs=(df, metric_mode, 100000)
            ) as pool:
                for result in pool.imap_unordered(run_single_backtest, worker_args):
                    completed_count  += 1
                    processed_in_run += 1
                    results_list.append(result)

                    # Progress display
                    if processed_in_run % PROGRESS_DISPLAY_INTERVAL == 0:
                        elapsed   = (datetime.now() - start_time).total_seconds()
                        rate      = processed_in_run / elapsed if elapsed > 0 else 0
                        remaining_est = (remaining - processed_in_run) / rate if rate > 0 else 0
                        pct       = (completed_count / total_combinations) * 100

                        eta_h = int(remaining_est // 3600)
                        eta_m = int((remaining_est % 3600) // 60)
                        eta_s = int(remaining_est % 60)
                        print(
                            f"  [{completed_count:>6,}/{total_combinations:,}] "
                            f"{pct:5.1f}%  |  "
                            f"{rate:5.1f} combos/s  |  "
                            f"ETA: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s"
                        )

                    # Checkpoint save
                    if (CHECKPOINT_ENABLED and
                            processed_in_run % checkpoint_interval == 0):
                        temp_df = pd.DataFrame(results_list)
                        save_checkpoint(temp_df, checkpoint_metadata)

        except KeyboardInterrupt:
            print("\n\n⚠️  Optimization interrupted by user.")
            print("Saving checkpoint before exit...")
            if results_list:
                temp_df = pd.DataFrame(results_list)
                save_checkpoint(temp_df, checkpoint_metadata)
                print(f"✅ Checkpoint saved with {len(results_list)} results.")
            print("Re-run the script and choose 'y' to resume.")
            exit(0)

        all_results_df = pd.DataFrame(results_list)

        total_elapsed = (datetime.now() - start_time).total_seconds()
        total_h = int(total_elapsed // 3600)
        total_m = int((total_elapsed % 3600) // 60)
        total_s = int(total_elapsed % 60)
        print(f"\n⏱️  Optimization completed in: {total_h:02d}h {total_m:02d}m {total_s:02d}s")

    # ──────────────────────────────────────────────────────────
    # PROCESS RESULTS
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📈 PROCESSING RESULTS")
    print("=" * 60)

    # Only fill numeric columns with 0 where appropriate — don't touch strings/booleans
    numeric_cols = all_results_df.select_dtypes(include=[np.number]).columns
    all_results_df[numeric_cols] = all_results_df[numeric_cols].fillna(0)

    # Sort by Sharpe Ratio (descending)
    all_results_df = all_results_df.sort_values('Sharpe Ratio', ascending=False)
    all_results_df.insert(0, 'Rank', range(1, len(all_results_df) + 1))

    # Export ALL results
    data_stem = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f'{data_stem}.csv'
    all_results_df.to_csv(output_file, index=False)
    print(f"✅ All results saved to: {output_file}")
    print(f"   Total rows exported : {len(all_results_df):,}")

    # ──────────────────────────────────────────────────────────
    # DISPLAY TOP 10
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🏆 TOP 10 COMBINATIONS (sorted by Sharpe Ratio)")
    print("=" * 60)

    base_cols = [
        'Rank', 'Combination',
        'tpslMethod', 'fvgSensitivity', 'swingLength', 'requireRetracement',
        'use1to1RR', 'riskAmount', 'tpPercent', 'slPercent',
        'Return [%]', 'Sharpe Ratio', 'Win Rate [%]', 'Profit Factor', '# Trades',
    ]
    display_cols = [c for c in base_cols if c in all_results_df.columns]

    top10 = all_results_df.head(10)
    with pd.option_context('display.max_columns', None, 'display.width', 160,
                           'display.float_format', '{:.4f}'.format):
        print(top10[display_cols].to_string(index=False))

    # ──────────────────────────────────────────────────────────
    # DISPLAY BEST COMBINATION DETAILS
    # ──────────────────────────────────────────────────────────
    if len(all_results_df) > 0:
        best = all_results_df.iloc[0]
        print("\n" + "=" * 60)
        print("🥇 BEST COMBINATION DETAILS")
        print("=" * 60)
        print(f"  Rank              : {best['Rank']}")
        print(f"  Combination #     : {best['Combination']}")
        print(f"  tpslMethod        : {best.get('tpslMethod', 'N/A')}")
        print(f"  fvgSensitivity    : {best.get('fvgSensitivity', 'N/A')}")
        print(f"  swingLength       : {best.get('swingLength', 'N/A')}")
        print(f"  requireRetracement: {best.get('requireRetracement', 'N/A')}")
        print(f"  use1to1RR         : {best.get('use1to1RR', 'N/A')}")
        print(f"  riskAmount        : {best.get('riskAmount', 'N/A')}")
        print(f"  tpPercent         : {best.get('tpPercent', 'N/A')}")
        print(f"  slPercent         : {best.get('slPercent', 'N/A')}")
        print("-" * 40)
        print(f"  Return [%]        : {best.get('Return [%]', 0):.4f}")
        print(f"  Sharpe Ratio      : {best.get('Sharpe Ratio', 0):.4f}")
        print(f"  Sortino Ratio     : {best.get('Sortino Ratio', 0):.4f}")
        print(f"  Max. Drawdown [%] : {best.get('Max. Drawdown [%]', 0):.4f}")
        print(f"  # Trades          : {best.get('# Trades', 0):.0f}")
        print(f"  Win Rate [%]      : {best.get('Win Rate [%]', 0):.4f}")
        print(f"  Profit Factor     : {best.get('Profit Factor', 0):.4f}")
        print(f"  Avg. Trade [%]    : {best.get('Avg. Trade [%]', 0):.4f}")
        print(f"  Exposure Time [%] : {best.get('Exposure Time [%]', 0):.4f}")
        if metric_mode == 'advanced':
            print(f"  Abs. Drawdown [%] : {best.get('Max. Absolute DD [%]', 0):.4f}")
            print(f"  Avg Win/Avg Loss  : {best.get('Avg Win / Avg Loss', 0):.4f}")
            print(f"  Max Loss Streak   : {best.get('Max Loss Streak', 0):.0f}")
            print(f"  Long Win Rate [%] : {best.get('Long Win Rate [%]', 0):.4f}")
            print(f"  Short Win Rate [%]: {best.get('Short Win Rate [%]', 0):.4f}")

        print("\n📋 To reproduce best result in unicorn.py:")
        print(f"   UnicornStrategy.tpslMethod         = '{best.get('tpslMethod', 'N/A')}'")
        print(f"   UnicornStrategy.fvgSensitivity     = '{best.get('fvgSensitivity', 'N/A')}'")
        print(f"   UnicornStrategy.swingLength        = {int(best.get('swingLength', 10))}")
        print(f"   UnicornStrategy.requireRetracement = {bool(best.get('requireRetracement', False))}")
        print(f"   UnicornStrategy.use1to1RR          = {bool(best.get('use1to1RR', True))}")
        print(f"   UnicornStrategy.riskAmount         = '{best.get('riskAmount', 'Normal')}'")
        print(f"   UnicornStrategy.tpPercent          = {best.get('tpPercent', 0.3)}")
        print(f"   UnicornStrategy.slPercent          = {best.get('slPercent', 0.4)}")

    # ──────────────────────────────────────────────────────────
    # CLEAN UP CHECKPOINT FILES
    # ──────────────────────────────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"\n🗑️  Checkpoint file removed: {CHECKPOINT_FILE}")
    if os.path.exists(CHECKPOINT_METADATA):
        os.remove(CHECKPOINT_METADATA)
        print(f"🗑️  Checkpoint metadata removed: {CHECKPOINT_METADATA}")

    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE")
    print(f"   Results file: {output_file}")
    print("=" * 60)