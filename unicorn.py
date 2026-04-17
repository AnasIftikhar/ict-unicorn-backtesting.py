"""
ICT Unicorn Strategy — Flux Charts Implementation
ATR uses pandas_ta with mamode='rma' (Wilder smoothing, exact TradingView match).
Entry: Market order at bar close after BB-FVG overlap is confirmed.
Three TP/SL modes: Unicorn (lowest/highest pivot), Dynamic (ATR-based), Fixed (percent).
"""

from backtesting import Strategy
from backtesting.lib import FractionalBacktest
import pandas as pd
import numpy as np
import pandas_ta as ta
from math import floor

# ===== BACKTEST CONFIGURATION (Global Settings) =====
CSV_FILE        = 'Data/ETHUSDT.P-3m-365.csv'  # Path to your OHLCV CSV file
STARTING_CASH   = 5000       # Starting account balance
LEVERAGE_MARGIN = 0.00001    # Extremely high leverage to bypass margin checks
TRADE_ON_CLOSE  = True       # True = fills on bar close (matches process_orders_on_close=true)
COMMISSION_FEE  = 0.0002        # 0.0 = 0% commission fees


class UnicornStrategy(Strategy):
    """
    ICT Unicorn — detects overlap between a Breaker Block (invalidated Order Block)
    and a Fair Value Gap of opposite direction, then enters a market order on the
    next confirmed bar close.

    State machine (matches Pine exactly):
      "Waiting For FVG-BB Overlap"
        → "FVG-BB Overlap"
          → "Enter Position"  (immediate if no retracement required)
          → "Require Retracement" → "Enter Position"
            → "Entry Taken"
    """

    # ===== PARAMETERS — mirrors Pine Settings UI exactly =====

    # --- General Configuration ---
    # Pine: input.string("Normal", "FVG Detection Sensitivity", options=["Extreme","High","Normal","Low"])
    fvgSensitivity       = "Low"   # Options: "Extreme", "High", "Normal", "Low"

    # Pine: input.int(10, 'Swing Length')
    swingLength          = 3         # Pivot lookback for OB detection

    # Pine: input.bool(false, "Require Retracement")
    requireRetracement   = False      # Wait for price to reach FVG before entry

    # --- TP / SL ---
    # Pine: input.string("Unicorn", "TP / SL Method", options=["Unicorn","Dynamic","Fixed"])
    tpslMethod           = "Dynamic"  # Options: "Unicorn", "Dynamic", "Fixed"

    # Pine: input.string("Normal", "Dynamic Risk", options=["Highest","High","Normal","Low","Lowest"])
    riskAmount           = "High"   # Options: "Highest", "High", "Normal", "Low", "Lowest"

    # Pine: input.float(0.3, "Fixed Take Profit %")  — only used when tpslMethod = "Fixed"
    tpPercent            = 0.3        # Fixed take-profit %

    # Pine: input.float(0.4, "Fixed Stop Loss %")   — only used when tpslMethod = "Fixed"
    slPercent            = 0.4        # Fixed stop-loss %

    # Pine: input.bool(false, "Force 1:1 Risk:Reward")
    use1to1RR            = True       # Override Unicorn/Dynamic RR to exactly 1:1

    # =========================================================================
    def init(self):
        """Initialize indicators and all state variables."""

        # --- Resolve sensitivity multiplier from string ---
        # Pine: fvgSensitivity = fvgSensitivityText == "Extreme" ? 6 : "High" ? 2 : "Normal" ? 1.5 : 1
        _sens_map = {"Extreme": 6.0, "High": 2.0, "Normal": 1.5, "Low": 1.0}
        self._fvg_sensitivity = _sens_map.get(self.fvgSensitivity, 1.5)

        # --- Resolve slATRMult from string ---
        # Pine: slATRMult = riskAmount == "Highest" ? 9.5 : "High" ? 6 : "Normal" ? 5 : "Low" ? 4 : "Lowest" ? 1.5
        _risk_map = {"Highest": 9.5, "High": 6.0, "Normal": 5.0, "Low": 4.0, "Lowest": 1.5}
        self._sl_atr_mult = _risk_map.get(self.riskAmount, 5.0)

        # --- Resolve RR multipliers ---
        # Pine: UnicornRR = use1to1RR ? 1.0 : 0.57  |  DynamicRR = use1to1RR ? 1.0 : 0.86
        # Pine hardcoded constants: UnicornRR=0.57, DynamicRR=0.86
        self._unicorn_rr = 1.0 if self.use1to1RR else 0.57
        self._dynamic_rr = 1.0 if self.use1to1RR else 0.86

        # --- Pine hardcoded constants (not in settings UI) ---
        _atr_len         = 10    # Pine: const int atrLen = 10
        _atr_len_unicorn = 50    # Pine: const int atrLenUnicorn = 50
        _unicorn_tpsl_len = 100  # Pine: const int unicornTPSLLength = 100
        _unicorn_sl_off  = 4.75  # Pine: const float dbgUnicornSLOffset = 4.75
        _max_atr_mult    = 3.5   # Pine: const float maxATRMult = 3.5
        _entry_wait_bars = 1     # Pine: const int entryWaitBars = 1
        _risk_per_trade  = 1.0   # Pine: default_qty_value = 100 (1% equity risk per trade)

        # Store for use in next()
        self._unicorn_sl_offset  = _unicorn_sl_off
        self._max_atr_mult       = _max_atr_mult
        self._entry_wait_bars    = _entry_wait_bars
        self._risk_per_trade     = _risk_per_trade

        # ===== INDICATOR CALCULATIONS =====

        # ATR(10) — used for FVG size filter and OB size cap
        # Pine: atr = ta.atr(atrLen)   [atrLen = 10, hardcoded]
        self.atr_fvg = self.I(
            self._calc_atr, self.data.df, _atr_len, name='ATR_FVG'
        )

        # ATR(50) — used for Unicorn and Dynamic TP/SL sizing
        # Pine: atrUnicorn = ta.atr(atrLenUnicorn)   [atrLenUnicorn = 50, hardcoded]
        self.atr_unicorn = self.I(
            self._calc_atr, self.data.df, _atr_len_unicorn, name='ATR_Unicorn'
        )

        # Rolling Highest High(100) — Pine: highUnicornTPSL = ta.highest(unicornTPSLLength)
        self.roll_high = self.I(
            self._calc_rolling_high, self.data.df, _unicorn_tpsl_len, name='RollHigh'
        )

        # Rolling Lowest Low(100) — Pine: lowUnicornTPSL = ta.lowest(unicornTPSLLength)
        self.roll_low = self.I(
            self._calc_rolling_low, self.data.df, _unicorn_tpsl_len, name='RollLow'
        )

        # ===== STATE VARIABLES =====

        # --- FVG list ---
        # Each entry: {'top': float, 'bottom': float, 'is_bull': bool,
        #              'start_bar': int, 'end_bar': int|None}
        self._fvg_list = []

        # --- OB lists (Bull and Bear separately, mirrors Pine's bullishOrderBlocksList) ---
        # Each entry: {'top': float, 'bottom': float, 'ob_type': 'Bull'|'Bear',
        #              'start_bar': int, 'is_breaker': bool, 'break_bar': int|None}
        self._bull_ob_list = []
        self._bear_ob_list = []

        # --- Swing tracking (from findOBSwings) ---
        self._swing_type      = 0     # 0 = last swing was Top, 1 = last swing was Bottom
        self._top_bar         = None  # bar index of last confirmed swing top
        self._top_y           = None  # high value of last confirmed swing top
        self._top_crossed     = False # True once close has broken above this top
        self._btm_bar         = None  # bar index of last confirmed swing bottom
        self._btm_y           = None  # low value of last confirmed swing bottom
        self._btm_crossed     = False # True once close has broken below this bottom

        # --- Trade state machine ---
        self._state                = "Waiting For FVG-BB Overlap"
        self._overlap_direction    = None   # 'Bull' or 'Bear'
        self._retrace_to           = None   # price level for retracement check
        self._enter_position_bar   = None   # bar index when "Enter Position" state started
        self._in_trade             = False

        # TP/SL for hybrid exit
        self._sl_target = None
        self._tp_target = None

    # =========================================================================
    # INDICATOR CALCULATION METHODS
    # =========================================================================

    def _calc_atr(self, df, period):
        """
        ATR using pandas_ta RMA mode — exact TradingView Wilder smoothing match.
        Pine: ta.atr(period)
        """
        atr = ta.atr(
            high=df['High'], low=df['Low'], close=df['Close'],
            length=period, mamode='rma'
        )
        return atr.fillna(0).values.copy()

    def _calc_rolling_high(self, df, period):
        """
        Rolling highest high over `period` bars.
        Pine: ta.highest(unicornTPSLLength)   [of high series]
        """
        return df['High'].rolling(period, min_periods=1).max().fillna(df['High']).values.copy()

    def _calc_rolling_low(self, df, period):
        """
        Rolling lowest low over `period` bars.
        Pine: ta.lowest(unicornTPSLLength)    [of low series]
        """
        return df['Low'].rolling(period, min_periods=1).min().fillna(df['Low']).values.copy()

    # =========================================================================
    # ZONE MANAGEMENT METHODS
    # =========================================================================

    def _detect_fvgs(self, idx):
        """
        Detect a new FVG on the current (just-confirmed) bar.

        Pine (barstate.isconfirmed block):
            bullFVG = low > high[2] and close[1] > high[2] and bullCondition
            bearFVG = high < low[2] and close[1] < low[2] and bearCondition
            FVGSizeEnough = FVGSize * fvgSensitivity > atr

        Pine index → Python index mapping:
            Pine [0] (current bar)  = Python [-1]
            Pine [1] (1 bar ago)    = Python [-2]
            Pine [2] (2 bars ago)   = Python [-3]
        """
        if idx < 2:
            return

        atr = self.atr_fvg[-1]
        if atr <= 0:
            return

        # Current bar and 2 bars back
        h0 = self.data.High[-1];   l0 = self.data.Low[-1]
        o0 = self.data.Open[-1];   c0 = self.data.Close[-1]
        h1 = self.data.High[-2];   l1 = self.data.Low[-2]
        o1 = self.data.Open[-2];   c1 = self.data.Close[-2]
        h2 = self.data.High[-3];   l2 = self.data.Low[-3]
        o2 = self.data.Open[-3];   c2 = self.data.Close[-3]

        # Pine: barSizeSum (body sizes)
        fs0 = abs(c0 - o0)
        fs1 = abs(c1 - o1)
        fs2 = abs(c2 - o2)
        bar_size_sum = fs0 + fs1 + fs2

        # Pine: maxCODiff = max(|close[2] - open[1]|, |close[1] - open[0]|)
        max_co_diff = max(abs(c2 - o1), abs(c1 - o0))

        # Pine: fvgBars == "Same Type" → all 3 bars must be same candle direction
        all_bear = (o0 > c0) and (o1 > c1) and (o2 > c2)
        all_bull = (o0 <= c0) and (o1 <= c1) and (o2 <= c2)
        if not (all_bear or all_bull):
            return

        # Pine: bullCondition / bearCondition (Average Range filter method)
        # (barSizeSum * fvgSensitivity > atr / 1.5) and (maxCODiff <= atr)
        condition = (
            (bar_size_sum * self._fvg_sensitivity > atr / 1.5) and
            (max_co_diff <= atr)
        )
        if not condition:
            return

        # Pine: bullFVG = low > high[2] and close[1] > high[2]
        # Python: l0 > h2 and c1 > h2
        if l0 > h2 and c1 > h2:
            fvg_size = abs(l0 - h2)
            # Pine: FVGSizeEnough = FVGSize * fvgSensitivity > atr
            if fvg_size * self._fvg_sensitivity > atr:
                # Pine: createFVGInfo(low, high[2], true, ...) → max=low, min=high[2]
                # top = low of current bar, bottom = high of 2-bars-ago
                self._fvg_list.append({
                    'top':       l0,
                    'bottom':    h2,
                    'is_bull':   True,
                    'start_bar': idx,
                    'end_bar':   None,
                })
                # Keep last 20 FVGs — Pine: showLastXFVGs = 20
                while len(self._fvg_list) > 20:
                    self._fvg_list.pop(0)

        # Pine: bearFVG = high < low[2] and close[1] < low[2]
        # Python: h0 < l2 and c1 < l2
        elif h0 < l2 and c1 < l2:
            fvg_size = abs(h0 - l2)
            if fvg_size * self._fvg_sensitivity > atr:
                # Pine: createFVGInfo(low[2], high, false, ...) → max=low[2], min=high
                # top = low of 2-bars-ago, bottom = high of current bar
                self._fvg_list.append({
                    'top':       l2,
                    'bottom':    h0,
                    'is_bull':   False,
                    'start_bar': idx,
                    'end_bar':   None,
                })
                while len(self._fvg_list) > 20:
                    self._fvg_list.pop(0)

    def _update_fvgs(self, idx):
        """
        Invalidate FVGs when price fully closes through them.
        Pine: fvgEndMethod = "Close" (default)
          Bull FVG ends when close < fvg.min (bottom)
          Bear FVG ends when close > fvg.max (top)
        """
        close = self.data.Close[-1]
        for fvg in self._fvg_list:
            if fvg['end_bar'] is not None:
                continue
            if fvg['is_bull'] and close < fvg['bottom']:
                fvg['end_bar'] = idx
            elif not fvg['is_bull'] and close > fvg['top']:
                fvg['end_bar'] = idx

    def _find_ob_swings(self, idx):
        """
        Pine findOBSwings(swingLength):
          upper = ta.highest(swingLength)    -- highest of High over last swingLength bars
          lower = ta.lowest(swingLength)     -- lowest of Low over last swingLength bars
          swingType = high[swingLength] > upper ? 0 : low[swingLength] < lower ? 1 : swingType

        When swingType flips to 0 → new confirmed swing TOP at bar (idx - swingLength).
        When swingType flips to 1 → new confirmed swing BOTTOM at bar (idx - swingLength).
        """
        n = self.swingLength
        if idx < n + 1:
            return

        # ta.highest(n) / ta.lowest(n) = max/min of the last n bars (current bar included)
        # Python: self.data.High[-n:]
        upper = float(np.max(self.data.High[-n:]))
        lower = float(np.min(self.data.Low[-n:]))

        # Pine high[n] / low[n] = value n bars ago
        # Python: self.data.High[-(n+1)]
        high_n = float(self.data.High[-(n + 1)])
        low_n  = float(self.data.Low[-(n + 1)])

        prev_type = self._swing_type

        if high_n > upper:
            self._swing_type = 0
        elif low_n < lower:
            self._swing_type = 1
        # else: unchanged (Pine: swingType stays)

        # New swing TOP detected
        if self._swing_type == 0 and prev_type != 0:
            self._top_bar     = idx - n
            self._top_y       = high_n
            self._top_crossed = False

        # New swing BOTTOM detected
        if self._swing_type == 1 and prev_type != 1:
            self._btm_bar     = idx - n
            self._btm_y       = low_n
            self._btm_crossed = False

    def _update_order_blocks(self, idx):
        """
        Form new OBs when close breaks through a swing extreme.
        Transition OBs → Breaker Blocks when price re-enters the OB.
        Remove invalidated BBs.

        Pine OB formation (bullish):
          When close > top.y and not top.crossed:
            Find the candle with the lowest 'low' between current bar and swing top.
            That candle becomes the OB (top = its high, bottom = its low).

        Pine OB → BB transition:
          obEndMethod = "Close" (default)
          Bull OB becomes BB when min(open, close) < OB.bottom
          Bear OB becomes BB when max(open, close) > OB.top

        Pine BB invalidation:
          bbEndMethod = "Close" (default)
          Bull BB removed when close > BB.top
          Bear BB removed when close < BB.bottom
        """
        close  = float(self.data.Close[-1])
        open_  = float(self.data.Open[-1])
        atr    = float(self.atr_fvg[-1])

        # ── Bullish OB formation (close breaks above swing top) ──
        if self._top_bar is not None and not self._top_crossed:
            if close > self._top_y:
                self._top_crossed = True
                bars_since_top = idx - self._top_bar  # how many bars back is the swing top

                if bars_since_top > 0:
                    # Pine loop: for i = 1 to (bar_index - top.x) - 1
                    # Looks back from 1 bar ago to bars_since_top-1 bars ago
                    # Finds the candle with the minimum 'low' in that range
                    # OB = that candle (top = its high, bottom = its low)
                    best_low  = None
                    best_high = None
                    for i in range(1, bars_since_top):
                        if i + 1 > len(self.data):
                            break
                        # i bars ago = Python index -(i+1)
                        l_i = float(self.data.Low[-(i + 1)])
                        h_i = float(self.data.High[-(i + 1)])
                        if best_low is None or l_i < best_low:
                            best_low  = l_i
                            best_high = h_i

                    if best_low is None:
                        best_low  = float(self.data.Low[-2])
                        best_high = float(self.data.High[-2])

                    ob_size = abs(best_high - best_low)
                    # Pine: if obSize <= atr * maxATRMult
                    if 0 < ob_size <= atr * self._max_atr_mult:
                        self._bull_ob_list.insert(0, {
                            'top':        best_high,
                            'bottom':     best_low,
                            'ob_type':    'Bull',
                            'start_bar':  idx,
                            'is_breaker': False,
                            'break_bar':  None,
                        })
                        if len(self._bull_ob_list) > 40:
                            self._bull_ob_list.pop()

        # ── Bearish OB formation (close breaks below swing bottom) ──
        if self._btm_bar is not None and not self._btm_crossed:
            if close < self._btm_y:
                self._btm_crossed = True
                bars_since_btm = idx - self._btm_bar

                if bars_since_btm > 0:
                    # Pine loop: for i = 1 to (bar_index - btm.x) - 1
                    # Finds the candle with the maximum 'high' in that range
                    best_high = None
                    best_low  = None
                    for i in range(1, bars_since_btm):
                        if i + 1 > len(self.data):
                            break
                        h_i = float(self.data.High[-(i + 1)])
                        l_i = float(self.data.Low[-(i + 1)])
                        if best_high is None or h_i > best_high:
                            best_high = h_i
                            best_low  = l_i

                    if best_high is None:
                        best_high = float(self.data.High[-2])
                        best_low  = float(self.data.Low[-2])

                    ob_size = abs(best_high - best_low)
                    if 0 < ob_size <= atr * self._max_atr_mult:
                        self._bear_ob_list.insert(0, {
                            'top':        best_high,
                            'bottom':     best_low,
                            'ob_type':    'Bear',
                            'start_bar':  idx,
                            'is_breaker': False,
                            'break_bar':  None,
                        })
                        if len(self._bear_ob_list) > 40:
                            self._bear_ob_list.pop()

        # ── OB → Breaker Block transitions ──
        # Pine obEndMethod = "Close" → condition uses min/max(open, close)

        to_remove = []
        for i, ob in enumerate(self._bull_ob_list):
            if not ob['is_breaker']:
                # Bull OB becomes BB when min(open, close) < bottom
                if min(open_, close) < ob['bottom']:
                    ob['is_breaker'] = True
                    ob['break_bar']  = idx
            else:
                # BB invalidated when close > top (bbEndMethod = "Close")
                if close > ob['top']:
                    to_remove.append(i)
        for i in reversed(to_remove):
            self._bull_ob_list.pop(i)

        to_remove = []
        for i, ob in enumerate(self._bear_ob_list):
            if not ob['is_breaker']:
                # Bear OB becomes BB when max(open, close) > top
                if max(open_, close) > ob['top']:
                    ob['is_breaker'] = True
                    ob['break_bar']  = idx
            else:
                # BB invalidated when close < bottom
                if close < ob['bottom']:
                    to_remove.append(i)
        for i in reversed(to_remove):
            self._bear_ob_list.pop(i)

    def _check_bb_fvg_overlap(self, idx):
        """
        Scan all active Breaker Blocks against all active FVGs for overlap.

        Pine condition:
          (curOB.obType == "Bear" and curFVG.isBull) → Long signal
          (curOB.obType == "Bull" and not curFVG.isBull) → Short signal
          doBBFVGTouch(curOB.info, curFVG.info) → vertical price overlap > 0%
          time == curFVG.info.startTime or time == curOB.info.breakTime
            → only fires on the bar the FVG was created OR the OB just became a BB

        Returns: (direction, retrace_to, fvg, ob) or (None, None, None, None)
        """
        all_obs = self._bull_ob_list + self._bear_ob_list

        for ob in all_obs:
            if not ob['is_breaker']:
                continue

            for fvg in self._fvg_list:
                if fvg['end_bar'] is not None:
                    continue  # FVG already invalidated

                # ── Direction gate: opposite types required ──
                # Pine: (obType=="Bear" and isBull) or (obType=="Bull" and not isBull)
                if ob['ob_type'] == 'Bear' and fvg['is_bull']:
                    direction = 'Bull'   # Long entry
                elif ob['ob_type'] == 'Bull' and not fvg['is_bull']:
                    direction = 'Bear'   # Short entry
                else:
                    continue

                # ── Price overlap gate ──
                # Pine: doBBFVGTouch checks intersectionArea > overlapThresholdPercentage (=0)
                # Simplified: vertical ranges must intersect
                overlap_top    = min(ob['top'],    fvg['top'])
                overlap_bottom = max(ob['bottom'], fvg['bottom'])
                if overlap_top <= overlap_bottom:
                    continue  # No vertical overlap

                # ── Timing gate (CRITICAL) ──
                # Pine: time == curFVG.info.startTime or time == curOB.info.breakTime
                # Python: idx == fvg['start_bar'] or idx == ob['break_bar']
                timing_ok = (idx == fvg['start_bar']) or (idx == ob['break_bar'])
                if not timing_ok:
                    continue

                # Retracement target = FVG top for bull entry, FVG bottom for bear entry
                # Pine: retraceTo = curFVG.info.isBull ? curFVG.info.max : curFVG.info.min
                retrace_to = fvg['top'] if fvg['is_bull'] else fvg['bottom']

                return direction, retrace_to, fvg, ob

        return None, None, None, None

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def _calculate_position_size(self, entry_price, sl_price):
        """
        Position size from % equity risk.
        Risk amount auto-compounds: higher equity → larger risk amount → same % risk.
        """
        if sl_price is None or entry_price <= 0:
            return 1
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            return 1
        risk_amount   = self.equity * (self._risk_per_trade / 100.0)
        ideal         = risk_amount / sl_distance
        max_contracts = self.equity * 20 * 0.95 / entry_price
        return int(max(1, round(min(ideal, max_contracts))))

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def next(self):
        """
        Strategy execution — exact Pine state machine order:
        1. Hybrid Exit Check
        2. Update zones (FVG detect, FVG invalidate, OB swing, OB/BB update)
        3. State Machine (Waiting → Overlap → Enter → Entry Taken)
        4. Reset on flat
        """
        idx = len(self.data) - 1

        # Warmup guard — need enough bars for all indicators
        # Pine hardcoded: unicornTPSLLength=100, atrLenUnicorn=50
        min_bars = max(
            self.swingLength * 2 + 2,
            101,   # unicornTPSLLength + 1
            52,    # atrLenUnicorn + 2
            5
        )
        if idx < min_bars:
            return

        # ===== 1. HYBRID EXIT — catches TP/SL misses from backtesting.py =====
        if self.position.size != 0 and self._sl_target and self._tp_target:
            h = self.data.High[-1]
            l = self.data.Low[-1]
            if self.position.is_long:
                if h >= self._tp_target or l <= self._sl_target:
                    self.position.close()
            elif self.position.is_short:
                if l <= self._tp_target or h >= self._sl_target:
                    self.position.close()

        # ===== 2. UPDATE ZONES =====
        self._detect_fvgs(idx)
        self._update_fvgs(idx)
        self._find_ob_swings(idx)
        self._update_order_blocks(idx)

        # ===== 3. RESET WHEN FLAT =====
        if self.position.size == 0 and self._in_trade:
            self._in_trade   = False
            self._sl_target  = None
            self._tp_target  = None
            self._state      = "Waiting For FVG-BB Overlap"

        # ===== 4. STATE MACHINE =====

        # ── "Waiting For FVG-BB Overlap" ──
        # Pine: createNewUnicorn only when na(lastUnicorn.exitPrice) is False
        #       i.e., no active trade — matches position.size == 0
        if self._state == "Waiting For FVG-BB Overlap" and self.position.size == 0:
            direction, retrace_to, fvg, ob = self._check_bb_fvg_overlap(idx)
            if direction is not None:
                self._overlap_direction = direction
                self._retrace_to        = retrace_to
                self._state             = "FVG-BB Overlap"
                # Immediately advance within same bar (Pine does this in one confirmed bar call)
                if not self.requireRetracement:
                    # Pine: if not dbgRequireRetracement → state := "Enter Position"
                    self._enter_position_bar = idx
                    self._state = "Enter Position"
                else:
                    self._state = "Require Retracement"

        # ── "Require Retracement" ──
        # Pine: if overlapDirection == "Bull" and low < retraceTo → enter
        #       if overlapDirection == "Bear" and high > retraceTo → enter
        elif self._state == "Require Retracement":
            if self._overlap_direction == 'Bull' and self.data.Low[-1] < self._retrace_to:
                self._enter_position_bar = idx
                self._state = "Enter Position"
            elif self._overlap_direction == 'Bear' and self.data.High[-1] > self._retrace_to:
                self._enter_position_bar = idx
                self._state = "Enter Position"

        # ── "Enter Position" ──
        # Pine: if bar_index >= enterPositionBar + entryWaitBars → enter at close
        if self._state == "Enter Position":
            if self._enter_position_bar is not None and idx >= self._enter_position_bar + self._entry_wait_bars:
                entry_price = float(self.data.Close[-1])
                atr_u       = float(self.atr_unicorn[-1])
                sl, tp      = None, None

                if self.tpslMethod == "Fixed":
                    # Fixed % method
                    # Pine: slTarget = entryPrice * (1 ± slPercent / 100)
                    #       tpTarget = entryPrice * (1 ± tpPercent / 100)
                    if self._overlap_direction == 'Bull':
                        sl = entry_price * (1 - self.slPercent / 100.0)
                        tp = entry_price * (1 + self.tpPercent / 100.0)
                    else:
                        sl = entry_price * (1 + self.slPercent / 100.0)
                        tp = entry_price * (1 - self.tpPercent / 100.0)

                elif self.tpslMethod == "Dynamic":
                    # Dynamic ATR method
                    # Pine: slTarget = entryPrice ± atrUnicorn * slATRMult
                    #       tpTarget = entryPrice ± |entry - sl| * DynamicRR
                    if self._overlap_direction == 'Bull':
                        sl = entry_price - atr_u * self._sl_atr_mult
                        tp = entry_price + abs(entry_price - sl) * self._dynamic_rr
                    else:
                        sl = entry_price + atr_u * self._sl_atr_mult
                        tp = entry_price - abs(entry_price - sl) * self._dynamic_rr

                else:
                    # Unicorn method (default, tpslMethod == "Unicorn")
                    # Pine: slTarget = lowUnicornTPSL - atrUnicorn * dbgUnicornSLOffset  (long)
                    #                = highUnicornTPSL + atrUnicorn * dbgUnicornSLOffset (short)
                    #       tpTarget = entryPrice ± |entry - sl| * UnicornRR
                    if self._overlap_direction == 'Bull':
                        sl = float(self.roll_low[-1])  - atr_u * self._unicorn_sl_offset
                        tp = entry_price + abs(entry_price - sl) * self._unicorn_rr
                    else:
                        sl = float(self.roll_high[-1]) + atr_u * self._unicorn_sl_offset
                        tp = entry_price - abs(entry_price - sl) * self._unicorn_rr

                if sl is None or tp is None or sl <= 0 or tp <= 0:
                    # Bad levels — abort signal, wait for next
                    self._state = "Waiting For FVG-BB Overlap"
                    return

                self._sl_target = sl
                self._tp_target = tp

                size = self._calculate_position_size(entry_price, sl)

                # Pine: strategy.close_all() before each entry
                if self.position.size != 0:
                    self.position.close()

                if self._overlap_direction == 'Bull':
                    # Pine: strategy.entry("Long", strategy.long)
                    #       strategy.exit("Long Exit", limit=tp, stop=sl)
                    self.buy(size=size, tp=tp, sl=sl)
                else:
                    # Pine: strategy.entry("Short", strategy.short)
                    #       strategy.exit("Short Exit", limit=tp, stop=sl)
                    self.sell(size=size, tp=tp, sl=sl)

                self._in_trade = True
                self._state    = "Entry Taken"

        # ── "Entry Taken" ──
        # Waiting for position to close (handled by hybrid exit + backtesting.py TP/SL)
        # Pine: monitors for TP/SL hit — handled above and by the framework
        elif self._state == "Entry Taken":
            if self.position.size == 0:
                self._in_trade  = False
                self._sl_target = None
                self._tp_target = None
                self._state     = "Waiting For FVG-BB Overlap"


# ===== EXAMPLE USAGE =====
if __name__ == '__main__':
    import sys

    # ── Load OHLCV data ──
    csv_file = CSV_FILE
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    df = pd.read_csv(csv_file)

    # Rename to backtesting.py standard column names
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

    # ── Initialize backtest ──
    bt = FractionalBacktest(
        df,
        UnicornStrategy,
        cash=STARTING_CASH,
        commission=COMMISSION_FEE,
        exclusive_orders=True,
        trade_on_close=TRADE_ON_CLOSE,
        margin=LEVERAGE_MARGIN
    )

    print("=" * 60)
    print("ICT UNICORN — BACKTEST")
    print("=" * 60)
    print(f"Data file : {csv_file}")
    print(f"Cash      : ${STARTING_CASH:,}")
    print(f"TP/SL mode: {UnicornStrategy.tpslMethod}")
    print("=" * 60)

    stats = bt.run()
    print(stats)

    # ── Export trades ──
    trades = stats['_trades']
    trades.to_csv('unicorn_trade_results.csv', index=False)
    print("\n>>> Success! Trade data saved to: unicorn_trade_results.csv")

    # ── Export equity curve ──
    equity_curve = stats['_equity_curve'][['Equity', 'DrawdownPct']]
    equity_curve.index.name = 'timestamp'
    equity_curve.to_csv('unicorn_equity_curve.csv')
    print(">>> Success! Equity curve saved to: unicorn_equity_curve.csv")

    # Uncomment to visualize:
    # bt.plot()