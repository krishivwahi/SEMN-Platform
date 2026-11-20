"""
tests/test_factors.py
============================
Phase 2: exposures, factor regression and risk estimates, checked against
simulated data whose true parameters are known.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from semn.factors import (
    asset_covariance,
    bias_statistic,
    diagnostics,
    factor_covariance,
    fit_factor_returns,
    ledoit_wolf,
    next_returns,
    specific_variance,
    standardize,
    style_exposures,
)
from semn.factors.exposures import MOMENTUM_SKIP, MOMENTUM_WINDOW, REVERSAL_WINDOW

DAYS = [date(2020, 1, 1) + timedelta(days=i) for i in range(400)]


def _returns(series: dict[str, np.ndarray]) -> pl.DataFrame:
    return pl.concat([
        pl.DataFrame({"date": DAYS[: len(r)], "security_id": sid, "ret": r}) for sid, r in series.items()
    ])


# ---------------------------------------------------------------------------
# Exposures
# ---------------------------------------------------------------------------

class TestExposures:
    def test_beta_recovers_true_beta(self):
        rng = np.random.default_rng(0)
        mkt = rng.normal(0, 0.01, 400)
        rets = _returns({"HI": 1.5 * mkt + rng.normal(0, 0.002, 400), "LO": 0.5 * mkt + rng.normal(0, 0.002, 400)})
        x = style_exposures(rets, pl.DataFrame({"date": DAYS, "ret": mkt})).filter(pl.col("date") == DAYS[-1])
        assert dict(x.select("security_id", "beta").rows()) == pytest.approx({"HI": 1.5, "LO": 0.5}, abs=0.05)

    def test_momentum_and_reversal_windows(self):
        """Log return 0.01 per day for 300 days, then 0.02: windows must pick exactly the right days."""
        lr = np.r_[np.full(300, 0.01), np.full(100, 0.02)]
        x = style_exposures(_returns({"A": np.expm1(lr)}), pl.DataFrame({"date": DAYS, "ret": np.zeros(400)}))
        last = x.filter(pl.col("date") == DAYS[-1]).row(0, named=True)
        assert last["reversal"] == pytest.approx(REVERSAL_WINDOW * 0.02)
        assert last["momentum"] == pytest.approx(152 * 0.01 + 79 * 0.02)  # days 148..378 straddle the change at 300
        t = 320  # window t-251..t-21 = days 69..299: all 0.01
        assert x.filter(pl.col("date") == DAYS[t])["momentum"].item() == pytest.approx((MOMENTUM_WINDOW - MOMENTUM_SKIP) * 0.01)

    def test_no_look_ahead(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 400)
        mkt = pl.DataFrame({"date": DAYS, "ret": rng.normal(0, 0.01, 400)})
        changed = r.copy()
        changed[301:] += 0.05
        a = style_exposures(_returns({"A": r}), mkt).filter(pl.col("date") <= DAYS[300])
        b = style_exposures(_returns({"A": changed}), mkt).filter(pl.col("date") <= DAYS[300])
        assert a.equals(b)

    def test_short_history_gives_null_not_a_guess(self):
        x = style_exposures(_returns({"NEW": np.full(30, 0.01)}), pl.DataFrame({"date": DAYS, "ret": np.full(400, 0.01)}))
        assert x.filter(pl.col("date") == DAYS[29])[["beta", "momentum", "volatility"]].null_count().row(0) == (1, 1, 1)

    def test_size_from_market_cap(self):
        rets = _returns({"A": np.zeros(5)})
        cap = pl.DataFrame({"date": DAYS[:5], "security_id": "A", "market_cap": [1e9] * 5})
        assert style_exposures(rets, pl.DataFrame({"date": DAYS, "ret": np.zeros(400)}), cap)["size"][0] == pytest.approx(np.log(1e9))


class TestStandardize:
    def _df(self):
        rng = np.random.default_rng(2)
        n = 200
        return pl.DataFrame({
            "date": [DAYS[0]] * n,
            "security_id": [f"S{i}" for i in range(n)],
            "x": pl.Series([*rng.normal(0, 1, n - 2), 50.0, None], dtype=pl.Float64),
            "market_cap": rng.uniform(1e9, 1e11, n),
        })

    def test_cap_weighted_mean_zero_unit_std(self):
        z = standardize(self._df(), ["x"], weight="market_cap")
        priced = z.filter(pl.col("security_id") != "S199")
        assert (priced["x"] * priced["market_cap"]).sum() / priced["market_cap"].sum() == pytest.approx(0, abs=1e-10)
        assert priced["x"].std() == pytest.approx(1, abs=0.02)

    def test_outlier_clipped_and_missing_set_to_zero(self):
        z = standardize(self._df(), ["x"])
        assert z.filter(pl.col("security_id") == "S198")["x"].item() < 4  # the 50σ outlier is winsorized
        assert z.filter(pl.col("security_id") == "S199")["x"].item() == 0.0


# ---------------------------------------------------------------------------
# Factor regression on a simulated factor model
# ---------------------------------------------------------------------------

def _simulate(n=300, t=120, noise=0.01, seed=3):
    """Returns with known factor returns: r = 0.001 + X·f + ε."""
    rng = np.random.default_rng(seed)
    styles = ["s1", "s2", "s3"]
    true_f = rng.normal(0, 0.01, (t, 3))
    rows = []
    for d in range(t):
        X = rng.normal(0, 1, (n, 3))
        r = 0.001 + X @ true_f[d] + rng.normal(0, noise, n)
        rows.append(pl.DataFrame({
            "date": DAYS[d], "security_id": [f"S{i}" for i in range(n)],
            **{s: X[:, k] for k, s in enumerate(styles)},
            "ret_next": r, "market_cap": rng.uniform(1e9, 1e11, n),
            "sector": rng.choice(["Tech", "Energy", "Health"], n),
        }))
    return pl.concat(rows), styles, true_f


class TestFactorRegression:
    def test_noise_free_recovers_factor_returns_exactly(self):
        panel, styles, true_f = _simulate(noise=0.0)
        f, resid = fit_factor_returns(panel, styles)
        assert f.select(styles).to_numpy() == pytest.approx(true_f, abs=1e-12)
        assert f["market"].to_numpy() == pytest.approx(0.001, abs=1e-12)
        assert resid["resid"].abs().max() < 1e-12

    def test_residuals_weighted_orthogonal_to_exposures(self):
        panel, styles, _ = _simulate()
        _, resid = fit_factor_returns(panel, styles, weight="market_cap", sector="sector")
        day = panel.filter(pl.col("date") == DAYS[5]).join(resid, on=["date", "security_id"])
        w = np.sqrt(day["market_cap"].to_numpy())
        X = np.hstack([pd_dummies(day["sector"]), day.select(styles).to_numpy()])
        assert X.T @ (w * day["resid"].to_numpy()) == pytest.approx(np.zeros(X.shape[1]), abs=1e-8)

    def test_noisy_estimates_track_truth(self):
        panel, styles, true_f = _simulate()
        f, _ = fit_factor_returns(panel, styles, weight="market_cap")
        for k, s in enumerate(styles):
            assert np.corrcoef(f[s].to_numpy(), true_f[:, k])[0, 1] > 0.95

    def test_sector_columns_replace_intercept(self):
        panel, styles, _ = _simulate(t=3)
        f, _ = fit_factor_returns(panel, styles, sector="sector")
        assert f.columns == ["date", "sector:Energy", "sector:Health", "sector:Tech", *styles]

    def test_next_returns_aligns_on_exposure_date(self):
        rets = _returns({"A": np.array([0.01, 0.02, 0.03])})
        assert next_returns(rets)["ret_next"].to_list() == [0.02, 0.03, None]


def pd_dummies(s: pl.Series) -> np.ndarray:
    cats = sorted(s.unique().to_list())
    return (s.to_numpy()[:, None] == np.array(cats)[None, :]).astype(float)


# ---------------------------------------------------------------------------
# Risk estimates
# ---------------------------------------------------------------------------

class TestCovariance:
    def test_ledoit_wolf_matches_sklearn(self):
        sk = pytest.importorskip("sklearn.covariance")
        R = np.random.default_rng(4).normal(0, 0.01, (60, 40))  # fewer days than assets: sample cov is singular
        ours, shrink = ledoit_wolf(R)
        ref, ref_shrink = sk.ledoit_wolf(R)
        assert shrink == pytest.approx(ref_shrink) and ours == pytest.approx(ref)
        assert diagnostics(np.cov(R.T, bias=True))["condition_number"] > diagnostics(ours)["condition_number"]

    def test_model_covariance_is_psd_and_calibrated(self):
        """Fit on the first 250 days, forecast the next 100: bias statistic ≈ 1."""
        rng = np.random.default_rng(5)
        n, t, styles = 150, 350, ["s1", "s2"]
        X = rng.normal(0, 1, (n, 2))  # fixed exposures
        f_vol, spec_vol = np.array([0.01, 0.006]), rng.uniform(0.01, 0.03, n)
        rows = []
        for d in range(t):
            r = 0.0005 + X @ (rng.normal(0, 1, 2) * f_vol) + rng.normal(0, 1, n) * spec_vol
            rows.append(pl.DataFrame({"date": DAYS[d], "security_id": [f"S{i:03d}" for i in range(n)],
                                      "s1": X[:, 0], "s2": X[:, 1], "ret_next": r}))
        panel = pl.concat(rows)
        train = panel.filter(pl.col("date") < DAYS[250])
        fr, resid = fit_factor_returns(train, styles)
        names, F = factor_covariance(fr, half_life=500)
        D = specific_variance(resid, half_life=500)["specific_var"].to_numpy()
        S = asset_covariance(np.hstack([np.ones((n, 1)), X]), F, D)
        assert diagnostics(S)["psd"] and diagnostics(S)["symmetric"]

        test = panel.filter(pl.col("date") >= DAYS[250]).sort("date", "security_id")
        realized = test["ret_next"].to_numpy().reshape(-1, n) - test.group_by("date", maintain_order=True).agg(pl.col("ret_next").mean())["ret_next"].to_numpy()[:, None]
        # Forecast vol of each stock's return relative to the cross-sectional mean (removes the market term).
        M = np.eye(n) - 1 / n
        vol = np.sqrt(np.diag(M @ S @ M))
        assert bias_statistic(realized, vol[None, :]) == pytest.approx(1.0, abs=0.1)
        assert bias_statistic(realized, vol[None, :] / 2) == pytest.approx(2.0, abs=0.2)  # halved forecast is flagged

    def test_specific_variance_needs_history_and_weights_recent_dates(self):
        days = DAYS[:40]
        resid = pl.concat([
            pl.DataFrame({"date": days, "security_id": "A", "resid": [0.01] * 20 + [0.03] * 20}),
            pl.DataFrame({"date": days[:5], "security_id": "B", "resid": [0.01] * 5}),
        ])
        sv = specific_variance(resid, half_life=5)
        assert sv["security_id"].to_list() == ["A"]                  # B has too little history
        assert 0.01 ** 2 < sv["specific_var"].item() < 0.03 ** 2
        assert sv["specific_var"].item() > (0.01 ** 2 + 0.03 ** 2) / 2  # recent (larger) residuals dominate
