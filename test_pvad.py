"""Each test reads as an O&M rule: given this situation, what should be reported?

Rules are tested on small hand-built tables; the last tests run the whole
method on the synthetic plant with labelled faults.
"""
import numpy as np
import pandas as pd
import pytest

import pvad


# --------------------------------------------------------------------------- data

def test_both_kaggle_date_formats_parse_to_the_same_time():
    a = pvad.parse_timestamps(pd.Series(["15-05-2020 00:15"]))
    b = pvad.parse_timestamps(pd.Series(["2020-05-15 00:15:00"]))
    assert a.iloc[0] == b.iloc[0] == pd.Timestamp("2020-05-15 00:15")


def test_unknown_date_format_fails_loudly():
    with pytest.raises(ValueError):
        pvad.parse_timestamps(pd.Series(["05/15/2020 00:15"]))


def test_missing_rows_come_back_as_no_data_not_zero():
    gen = pd.DataFrame({"DATE_TIME": ["2020-05-15 12:00:00"] * 2 + ["2020-05-15 12:15:00"],
                        "SOURCE_KEY": ["A", "B", "A"], "AC_POWER": 500.0, "DC_POWER": 510.0})
    wx = pd.DataFrame({"DATE_TIME": ["2020-05-15 12:00:00", "2020-05-15 12:15:00"],
                       "IRRADIATION": 0.8, "MODULE_TEMPERATURE": 45.0})
    df = pvad.merge(gen, wx)
    assert len(df) == 4  # B at 12:15 is back
    assert df["no_data"].sum() == 1
    assert df.loc[df["no_data"], "ac_power_kw"].isna().all()


def test_inverters_are_named_tt_ii_and_keep_their_source_key():
    df = pvad.assign_topology(pd.DataFrame({"inverter_id": ["zKey", "aKey", "mKey"]}))
    assert dict(zip(df["source_key"], df["inverter_id"])) == {
        "aKey": "01_01", "mKey": "01_02", "zKey": "01_03"}


# --------------------------------------------------------------------------- plant events

def _interval(down: int, total: int = 10) -> pd.DataFrame:
    """One sunny 15-min interval: `down` inverters at zero, the rest healthy."""
    return pd.DataFrame({
        "timestamp": pd.Timestamp("2020-05-20 12:00"),
        "irradiation_kw_m2": 0.9, "expected_ac_kw": 1000.0, "no_data": False,
        "ac_power_kw": [0.0] * down + [1000.0] * (total - down),
    })


def test_when_80pct_of_inverters_drop_together_the_plant_is_at_fault():
    assert pvad.plant_events(_interval(down=8))["plant_event"].all()


def test_below_80pct_the_drop_stays_with_each_inverter():
    assert not pvad.plant_events(_interval(down=7))["plant_event"].any()


def test_a_whole_plant_logger_outage_is_a_plant_event():
    df = _interval(down=0).assign(ac_power_kw=np.nan, no_data=True)
    assert pvad.plant_events(df)["plant_event"].all()


# --------------------------------------------------------------------------- rules

def _day(n: int = 2, **cols) -> pd.DataFrame:
    """Daily table for `n` inverters on one day, healthy unless overridden."""
    base = {"inverter_id": [f"01_{i:02d}" for i in range(1, n + 1)],
            "date": pd.Timestamp("2020-05-20"), "zero_intervals": 0, "no_data_intervals": 0,
            "pi_daily": 1.0, "pi_vs_fleet": 1.0, "ref_self": 1.0,
            "fleet_ratio": 0.97, "clear_sky": True}
    return pd.DataFrame({**base, **cols})


def test_one_hour_of_zero_output_in_sun_is_an_outage():
    assert pvad.rule_outage(_day(zero_intervals=[4, 3])).tolist() == [True, False]


def test_one_hour_without_data_is_no_data_not_outage():
    d = _day(no_data_intervals=[4, 0])
    assert pvad.rule_no_data(d).tolist() == [True, False]
    assert not pvad.rule_outage(d).any()


def test_new_fault_needs_loss_vs_own_best_days_and_vs_the_fleet():
    d = _day(n=3,  # string loss | plant-wide cloudy day | normal day, a bit below best
             pi_daily=[0.90, 0.88, 0.97], ref_self=[1.03, 1.00, 1.03],
             pi_vs_fleet=[0.90, 1.00, 0.96])
    assert pvad.rule_new_fault(d).tolist() == [True, False, False]


def test_chronic_is_an_inverter_always_below_its_best_peers():
    # 01_01 has been ~15% under its peers from day one: its own baseline learnt
    # that, only the fleet shows it. 01_02 has one bad day: not chronic.
    dates = pd.date_range("2020-05-15", periods=4)
    d = pd.concat([
        _day(n=1, date=dates, inverter_id="01_01", fleet_ratio=[0.84, 0.86, 0.85, 0.93]),
        _day(n=1, date=dates, inverter_id="01_02", fleet_ratio=[0.97, 0.98, 0.85, 0.97]),
    ], ignore_index=True)
    assert pvad.rule_chronic(d).tolist() == [True, True, True, False] + [False] * 4


# --------------------------------------------------------------------------- step report

def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-05-15", periods=len(values)))


def test_step_report_dates_and_sizes_a_lasting_drop():
    step = pvad.step_change(_series([1.00, 1.01, 0.99, 1.00, 1.00, 0.95, 0.96, 0.95, 0.94, 0.95]))
    assert step == {"step_date": pd.Timestamp("2020-05-20"), "step_pct": -5.0}


def test_no_step_for_noise_small_drops_or_too_few_days():
    assert pvad.step_change(_series([1.00, 0.98, 1.01, 0.99, 1.02, 1.00, 0.99])) is None
    assert pvad.step_change(_series([1.00, 1.00, 1.00, 0.98, 0.98, 0.98])) is None
    assert pvad.step_change(_series([1.00, 1.00, 0.90, 0.90, 0.90])) is None


# --------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def synthetic():
    df, labels = pvad.synthetic_plant(seed=7)
    scored, daily = pvad.run(df)
    return scored, daily, labels


def test_every_alarm_fault_is_caught_without_false_alarms(synthetic):
    _, daily, labels = synthetic
    res = pvad.evaluate(daily, labels)
    for kind in ("outage", "intermittent", "no_data", "chronic"):
        assert res["recall_by_fault"][kind] == 1.0
    assert res["precision"] >= 0.9


def test_one_open_string_is_dated_by_the_step_report(synthetic):
    # 5% loss injected on day 18 (2020-06-02): below the daily alarm, still dated.
    _, daily, _ = synthetic
    row = pvad.ranking(daily).loc["01_08"]
    assert abs((row["step_date"] - pd.Timestamp("2020-06-02")).days) <= 1
    assert -7 <= row["step_pct"] <= -3


def test_grid_trip_is_one_plant_event_not_22_outages(synthetic):
    scored, daily, _ = synthetic
    assert pvad.plant_event_summary(scored)["hours"] == pytest.approx(2.0, abs=0.5)
    assert daily.loc[daily["date"] == "2020-06-07", "zero_intervals"].sum() == 0


def test_ranking_puts_faulty_inverters_on_top(synthetic):
    _, daily, labels = synthetic
    assert set(pvad.ranking(daily).index[:3]) <= set(labels["inverter_id"])
