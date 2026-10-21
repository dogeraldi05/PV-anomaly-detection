"""PV anomaly detection with weather-normalised baselines.

Read top to bottom, in the order the notebook uses it:
    1. Data       load Kaggle CSVs or generate a synthetic plant, TT_II topology
    2. Baseline   expected AC power per inverter from irradiance and temperature
    3. Detection  plant events (80% rule) and one function per O&M rule
    4. Reporting  ranking, step report, evaluation against labels
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

INTERVAL_H = 0.25        # 15-min data
DAYLIGHT_IRR = 0.10      # kW/m2; below this, ratios are dominated by noise
TRAIN_DAYS = 14          # history window that defines "normal"

PARENT_SHARE = 0.80      # if this share of children drop together, blame the parent
DROP_RATIO = 0.50        # an interval counts as "dropped" below half the expected output
OUTAGE_MIN_INTERVALS = 4 # 1 h
PI_THRESHOLD = 0.93      # new fault: below 93% of own best days and of the fleet
CHRONIC_THRESHOLD = 0.90 # chronic: below 90% of the best peers
STEP_MIN_DAYS = 3        # step report: days on each side
STEP_MIN_PCT = 3.0       # step report: smallest step, % of the level before


# --------------------------------------------------------------------------- 1. Data

# Plant 1 uses "15-05-2020 00:00"; Plant 2 uses "2020-05-15 00:00:00".
_DATE_FORMATS = ("%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse DATE_TIME trying each known format; fail loudly if none fits."""
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    raise ValueError(f"Unrecognised DATE_TIME format, sample: {values.iloc[0]!r}")


def load_kaggle(plant: int, data_dir: str | Path = "data") -> pd.DataFrame:
    """Generation + weather CSVs of one Kaggle plant, merged and named TT_II."""
    d = Path(data_dir)
    gen = pd.read_csv(d / f"Plant_{plant}_Generation_Data.csv")
    wx = pd.read_csv(d / f"Plant_{plant}_Weather_Sensor_Data.csv")
    return assign_topology(merge(gen, wx))


def merge(gen: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """One row per inverter per timestamp; a missing reading becomes `no_data`.

    Loggers drop rows when an inverter stops reporting. Rebuilding the full
    timestamp x inverter grid makes the gap visible, without pretending it is
    zero output.
    """
    gen = gen.assign(timestamp=parse_timestamps(gen["DATE_TIME"])).rename(columns={
        "SOURCE_KEY": "inverter_id", "AC_POWER": "ac_power_kw", "DC_POWER": "dc_power_kw",
    })[["timestamp", "inverter_id", "ac_power_kw", "dc_power_kw"]]
    wx = wx.assign(timestamp=parse_timestamps(wx["DATE_TIME"])).rename(columns={
        "IRRADIATION": "irradiation_kw_m2", "MODULE_TEMPERATURE": "module_temp_c",
    })[["timestamp", "irradiation_kw_m2", "module_temp_c"]]
    wx = wx.groupby("timestamp", as_index=False).mean()  # one sensor; guard duplicates

    grid = pd.MultiIndex.from_product(
        [wx["timestamp"], gen["inverter_id"].unique()], names=["timestamp", "inverter_id"]
    ).to_frame(index=False)
    df = grid.merge(gen, on=["timestamp", "inverter_id"], how="left").merge(wx, on="timestamp")
    df["no_data"] = df["ac_power_kw"].isna()
    return df.sort_values(["inverter_id", "timestamp"]).reset_index(drop=True)


def assign_topology(df: pd.DataFrame) -> pd.DataFrame:
    """Rename inverters to the TT_II asset code (transformer_inverter).

    Field convention: TT_II_SS = transformer (01-99), inverter (01-99),
    string (01-99). It says where every string hangs, so a fault can be
    attributed to the right level. Kaggle has no topology: every inverter goes
    on transformer 01, numbered by its original SOURCE_KEY (kept in source_key).
    """
    codes = {k: f"01_{i:02d}" for i, k in enumerate(sorted(df["inverter_id"].unique()), start=1)}
    return df.rename(columns={"inverter_id": "source_key"}).assign(
        inverter_id=lambda d: d["source_key"].map(codes))


def data_quality(df: pd.DataFrame) -> dict:
    """Checks worth reading before trusting any model output."""
    day = df["irradiation_kw_m2"] > DAYLIGHT_IRR
    dc_ac = (df.loc[day, "dc_power_kw"] / df.loc[day, "ac_power_kw"].where(lambda s: s > 0)).median()
    return {
        "inverters": df["inverter_id"].nunique(),
        "days": df["timestamp"].dt.normalize().nunique(),
        "missing_readings_pct": round(100 * df["no_data"].mean(), 2),
        "zero_ac_in_daylight_pct": round(100 * (df.loc[day, "ac_power_kw"] <= 0).mean(), 2),
        # ~1.02 is physical; Kaggle Plant 1 shows ~10x, so the model uses AC.
        "median_dc_ac_ratio": round(float(dc_ac), 2),
    }


def synthetic_plant(seed: int = 42, string_loss: float = 0.05, n_inverters: int = 22,
                    days: int = 34, capacity_kw: float = 1400.0):
    """Synthetic plant in the Kaggle schema, with labelled faults.

    Public PV data has no fault labels, so detection quality can only be
    measured on injected faults. New faults start after the 14-day history;
    the chronic inverter is low from day one. A 2 h grid trip on day 23 hits
    the whole plant and is not a label (no inverter is at fault).
    Returns (merged frame, labels: inverter_id, date, fault_type).
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-05-15", periods=days * 96, freq="15min")
    hours = (ts.hour + ts.minute / 60).to_numpy()
    day_idx = (ts.normalize() - ts[0]).days.to_numpy()

    # Bell-shaped irradiance with day-level and intra-day clouds.
    clear = np.clip(np.sin(np.pi * (hours - 6) / 12.5), 0, None) ** 1.3
    irr = clear * rng.uniform(0.55, 1.0, days)[day_idx] * np.clip(1 - rng.gamma(0.6, 0.12, len(ts)), 0.2, 1)
    ambient = 24 + 8 * np.clip(np.sin(np.pi * (hours - 8) / 14), 0, None) + rng.normal(0, 0.7, len(ts))
    module = ambient + 28 * irr + rng.normal(0, 1.0, len(ts))
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    wx = pd.DataFrame({"DATE_TIME": stamp, "IRRADIATION": irr, "MODULE_TEMPERATURE": module})

    faults = {  # inverter: (type, first day, last day)
        "01_04": ("outage", 20, 21),          # trips mid-morning
        "01_08": ("string_loss", 18, days - 1),  # one string of ~20 opens
        "01_13": ("soiling", 16, days - 1),   # -1.2% per day
        "01_17": ("intermittent", 25, 27),    # contactor chatter
        "01_10": ("chronic", 0, days - 1),    # 15% low from day one
        "01_20": ("no_data", 29, 29),         # logger silent 09:00-15:00
    }
    rows, labels = [], []
    for i in range(1, n_inverters + 1):
        inv = f"01_{i:02d}"
        size = rng.uniform(0.95, 1.05)  # nameplate / orientation spread
        # One weather sensor, but clouds hit each block at slightly different moments.
        local = irr * np.clip(rng.normal(1, 0.06, len(ts)), 0.6, 1.3)
        ac = capacity_kw * size * local * (1 - 0.004 * (module - 25)) * 0.97
        ac *= rng.normal(1, 0.02, len(ts)) * rng.normal(1, 0.012, days)[day_idx]
        mult = np.ones(len(ts))
        if inv in faults:
            kind, d0, d1 = faults[inv]
            w = (day_idx >= d0) & (day_idx <= d1)
            if kind == "outage":
                mult[w & (hours >= 10)] = 0.0
            elif kind == "string_loss":
                mult[w] *= 1 - string_loss
            elif kind == "soiling":
                mult[w] *= 1 - 0.012 * (day_idx[w] - d0 + 1)
            elif kind == "intermittent":
                mult[w & (rng.random(len(ts)) < 0.25)] = 0.0
            elif kind == "chronic":
                mult[w] *= 0.85
            elif kind == "no_data":
                mult[w & (hours >= 9) & (hours < 15)] = np.nan
            labels += [(inv, ts[0] + pd.Timedelta(days=d), kind) for d in range(d0, d1 + 1)]
        mult[(day_idx == 23) & (hours >= 11) & (hours < 13)] = 0.0  # grid trip
        ac = np.clip(ac * mult, 0, 0.80 * capacity_kw * size)       # clipping at AC rating
        keep = ~np.isnan(ac)
        rows.append(pd.DataFrame({"DATE_TIME": stamp[keep], "SOURCE_KEY": inv,
                                  "AC_POWER": ac[keep], "DC_POWER": ac[keep] / 0.977}))
    labels = pd.DataFrame(labels, columns=["inverter_id", "date", "fault_type"])
    return assign_topology(merge(pd.concat(rows), wx)), labels


# --------------------------------------------------------------------------- 2. Baseline

def _features(df: pd.DataFrame) -> np.ndarray:
    """PV power equation P = k*G*(1 + gamma*(T_mod - 25)), expanded, plus G^2."""
    g, t = df["irradiation_kw_m2"].to_numpy(), df["module_temp_c"].to_numpy()
    return np.column_stack([g, g * (t - 25.0), g ** 2])


def fit_baseline(df: pd.DataFrame, train_days: int = TRAIN_DAYS) -> pd.DataFrame:
    """Learn each inverter's expected power on the history window; score every row.

    Robust (Huber) regression, so a few bad points in the history do not bend
    the baseline. Zero-output intervals are events, not the shape of the curve,
    so they stay out of the fit. Adds expected_ac_kw, performance_index, is_train.
    """
    train_end = df["timestamp"].min().normalize() + pd.Timedelta(days=train_days)
    out = []
    for inv, g in df.groupby("inverter_id"):
        hist = g[(g["timestamp"] < train_end) & (g["irradiation_kw_m2"] > DAYLIGHT_IRR)
                 & (g["ac_power_kw"] > 0)]
        scale = max(hist["ac_power_kw"].max(), 1.0)  # numerical stability for Huber
        model = HuberRegressor(max_iter=500).fit(_features(hist), hist["ac_power_kw"] / scale)
        expected = np.clip(model.predict(_features(g)) * scale, 0, None)
        expected[g["irradiation_kw_m2"].to_numpy() <= 0] = 0.0
        out.append(g.assign(
            expected_ac_kw=expected,
            performance_index=np.where(g["irradiation_kw_m2"] > DAYLIGHT_IRR,
                                       g["ac_power_kw"] / np.where(expected > 0, expected, np.nan), np.nan),
            is_train=g["timestamp"] < train_end,
        ))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- 3. Detection

def plant_events(scored: pd.DataFrame) -> pd.DataFrame:
    """Mark daylight intervals where the plant, not an inverter, is at fault.

    Attribution follows the plant tree (plant > transformer > inverter): when
    >= 80% of the inverters drop at the same moment (zero, under half the
    expected output, or silent), the cause is upstream (grid, transformer,
    curtailment, logger). Reported once, and kept out of every inverter rule,
    so one grid trip does not become 22 alarms.
    """
    sunny = scored["irradiation_kw_m2"] > DAYLIGHT_IRR
    dropped = scored["no_data"] | (scored["ac_power_kw"] < DROP_RATIO * scored["expected_ac_kw"])
    share = (dropped & sunny).groupby(scored["timestamp"]).transform("mean")
    return scored.assign(plant_event=sunny & (share >= PARENT_SHARE))


def daily_table(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per inverter-day: daylight only, plant events left out."""
    day = scored[(scored["irradiation_kw_m2"] > DAYLIGHT_IRR) & ~scored["plant_event"]].copy()
    day["date"] = day["timestamp"].dt.normalize()
    day["zero"] = day["ac_power_kw"] <= 0
    day["expected_measured"] = day["expected_ac_kw"].where(~day["no_data"])  # gaps are not losses
    day["loss_kwh"] = (day["expected_ac_kw"] - day["ac_power_kw"]).clip(lower=0) * INTERVAL_H

    d = day.groupby(["inverter_id", "date"]).agg(
        energy_kwh=("ac_power_kw", lambda s: s.sum() * INTERVAL_H),
        expected_kwh=("expected_measured", lambda s: s.sum() * INTERVAL_H),
        zero_intervals=("zero", "sum"),
        no_data_intervals=("no_data", "sum"),
        loss_kwh=("loss_kwh", "sum"),
        is_train=("is_train", "all"),
    ).reset_index()
    d["pi_daily"] = d["energy_kwh"] / d["expected_kwh"].where(d["expected_kwh"] > 0)
    # Same-day comparison with the fleet removes plant-wide effects (clouds, curtailment).
    d["pi_vs_fleet"] = d["pi_daily"] / d.groupby("date")["pi_daily"].transform("median")

    # Clear-sky day: daily insolation within 80% of the sunniest days (P95).
    insolation = day.drop_duplicates("timestamp").groupby("date")["irradiation_kw_m2"].sum() * INTERVAL_H
    d["clear_sky"] = d["date"].map(insolation >= 0.8 * insolation.quantile(0.95))

    # Own best days: P90 of daily PI on clear-sky history days.
    best = d[d["is_train"] & d["clear_sky"]].groupby("inverter_id")["pi_daily"].quantile(0.9)
    d["ref_self"] = d["inverter_id"].map(best).fillna(1.0)

    # Best peers: P90 of the fleet's daily energy. Days with gaps or zeros get no
    # ratio: those hours belong to the outage / no-data rules.
    complete = d["energy_kwh"].where((d["no_data_intervals"] == 0) & (d["zero_intervals"] == 0))
    by_day = complete.groupby(d["date"])
    d["fleet_ratio"] = complete / by_day.transform(lambda s: s.quantile(0.9))
    d["fleet_gap_kwh"] = (by_day.transform("median") - complete).clip(lower=0)
    return d


def rule_outage(d: pd.DataFrame) -> pd.Series:
    """At least 1 h of zero output with the sun up (trip, open breaker)."""
    return d["zero_intervals"] >= OUTAGE_MIN_INTERVALS


def rule_no_data(d: pd.DataFrame) -> pd.Series:
    """At least 1 h without a reading: a comms job for O&M, not an electrical one."""
    return d["no_data_intervals"] >= OUTAGE_MIN_INTERVALS


def rule_new_fault(d: pd.DataFrame) -> pd.Series:
    """Below 93% of the inverter's own best days AND below 93% of the fleet that day.

    The own-best-day reference catches string loss, soiling, a failed MPPT. The
    fleet condition keeps plant-wide effects from flagging everyone.
    """
    return (d["pi_daily"] / d["ref_self"] < PI_THRESHOLD) & (d["pi_vs_fleet"] < PI_THRESHOLD)


def rule_chronic(d: pd.DataFrame) -> pd.Series:
    """Always well below the best peers (typical clear-sky ratio < 90% of fleet P90).

    Its own baseline learnt the low level as normal; only the fleet can show it.
    Healthy nameplate spread stays within ~5-8% of the best peers.
    """
    typical = d["fleet_ratio"].where(d["clear_sky"]).groupby(d["inverter_id"]).transform("median")
    return (typical < CHRONIC_THRESHOLD) & (d["fleet_ratio"] < CHRONIC_THRESHOLD)


# Checked in order; the first rule that fires gives the day its reason.
RULES = {
    "zero output in daylight": rule_outage,
    "no data": rule_no_data,
    "new fault vs own best days": rule_new_fault,
    "chronic vs fleet": rule_chronic,
}


def detect(scored: pd.DataFrame) -> pd.DataFrame:
    """Daily table with one flag column per rule, a reason and the loss per day."""
    d = daily_table(scored)
    flags = pd.DataFrame({name: rule(d) for name, rule in RULES.items()})
    d["anomaly"] = flags.any(axis=1)
    d["reason"] = np.select([flags[c] for c in flags], list(flags), default="")
    # A chronic inverter matches its own baseline: its loss is the gap to a typical peer.
    chronic = d["reason"] == "chronic vs fleet"
    d["loss_kwh"] = d["loss_kwh"].where(~chronic, d["fleet_gap_kwh"])
    d["chronic"] = flags["chronic vs fleet"]
    return d


# --------------------------------------------------------------------------- 4. Reporting

def step_change(series: pd.Series) -> dict | None:
    """Date a lasting step down in a daily series (information, not an alarm).

    One open string is a step of ~1/number-of-strings (~5%), close to the daily
    threshold. Looking at all days at once makes it visible: the split that best
    explains the series as two flat levels (least absolute deviation), reported
    when the level after is >= 3% lower, with >= 3 days on each side.
    """
    s = series.dropna()
    best = None
    for k in range(STEP_MIN_DAYS, len(s) - STEP_MIN_DAYS + 1):
        a, b = s.iloc[:k], s.iloc[k:]
        cost = (a - a.median()).abs().sum() + (b - b.median()).abs().sum()
        if best is None or cost < best[0]:
            best = (cost, k)
    if best is None:
        return None
    k = best[1]
    pct = round(100 * (s.iloc[k:].median() / s.iloc[:k].median() - 1), 1)
    return {"step_date": s.index[k], "step_pct": pct} if pct <= -STEP_MIN_PCT else None


def ranking(d: pd.DataFrame) -> pd.DataFrame:
    """What an O&M team reads first: where the energy is being lost (test window)."""
    test = d[~d["is_train"]].assign(flagged_loss=lambda t: t["loss_kwh"].where(t["anomaly"], 0.0))
    r = test.groupby("inverter_id").agg(
        flagged_days=("anomaly", "sum"),
        loss_kwh=("flagged_loss", "sum"),
        pi_median=("pi_daily", "median"),
        chronic=("chronic", "any"),
    ).round({"loss_kwh": 0, "pi_median": 3})
    clean = d[d["zero_intervals"] < OUTAGE_MIN_INTERVALS]
    steps = {inv: step_change(g.set_index("date")["pi_vs_fleet"]) or {}
             for inv, g in clean.groupby("inverter_id")}
    r["step_date"] = [steps.get(i, {}).get("step_date") for i in r.index]
    r["step_pct"] = [steps.get(i, {}).get("step_pct") for i in r.index]
    return r.sort_values("loss_kwh", ascending=False)


def plant_event_summary(scored: pd.DataFrame) -> dict:
    """Plant-level events, reported once instead of as one alarm per inverter."""
    ev = scored[scored["plant_event"]]
    lost = ((ev["expected_ac_kw"] - ev["ac_power_kw"]).clip(lower=0) * INTERVAL_H).sum()
    return {"hours": ev["timestamp"].nunique() * INTERVAL_H,
            "days": ev["timestamp"].dt.normalize().nunique(),
            "loss_kwh": round(float(lost))}


def evaluate(d: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Precision / recall at inverter-day level, test window only."""
    m = d.loc[~d["is_train"], ["inverter_id", "date", "anomaly"]].merge(
        labels, on=["inverter_id", "date"], how="left")
    label = m["fault_type"].notna()
    tp, fp, fn = int((m["anomaly"] & label).sum()), int((m["anomaly"] & ~label).sum()), int((~m["anomaly"] & label).sum())
    return {"precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2),
            "tp": tp, "fp": fp, "fn": fn,
            "recall_by_fault": m[label].groupby("fault_type")["anomaly"].mean().round(2).to_dict()}


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline -> plant events -> rules. Returns (scored intervals, daily table)."""
    scored = plant_events(fit_baseline(df))
    return scored, detect(scored)
