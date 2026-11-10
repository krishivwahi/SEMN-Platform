# SEMN Engineering Roadmap — v2

> **Core principle:** SEMN is not a trading app. It is a research and execution system for
> **cross-sectional, market-neutral** equity strategy evaluation.
>
> **Daily question:** *"Am I improving strategy validity, or just adding features?"*

| | |
|---|---|
| **Version** | v2 (2026-11-10) — supersedes the original 7-phase PDF roadmap |
| **Timeline** | ~19 weeks (4.5 months) for the core platform; stretch tracks are extra |
| **Current status** | Phase 1 built and tested except the vendor adapter, which waits on decision D1. See [Phase 1 status](#phase-1-status) |

---

## What changed from v1

1. **Data source is now an explicit Phase 1 decision.** Every later phase (fundamentals, GICS history, tick data) depends on it.
2. **End-to-end thin slice before C++.** A Python reference backtester (needed anyway for parity tests) ships with the optimizer; the C++20 core moves later as a performance/parity phase.
3. **Hawkes order-flow and Level-2 replay moved to stretch tracks.** Daily OHLCV cannot support them; they need tick/L2 data.
4. **Risk parameters reconciled.** Gross exposure is the *output* of volatility targeting (with a cap); drawdown thresholds are volatility-relative and have a re-entry rule; factor-exposure limits apply to standardized exposures.
5. **Targets recalibrated.** The "net Sharpe > 2.0" target is replaced with process-based acceptance criteria (DSR, CPCV distribution). ICIR now states its period.
6. **Experiment tracking starts in Phase 1**, not Phase 6 — data snapshots are hashed from day one.
7. **Universe scoped to S&P 500** (S&P 1500 optional if the vendor supports it). Russell 1000 history is dropped unless affordable data is found.
8. **Securities keyed by a permanent `security_id`**, never by ticker.

---

## Final architecture

```
Point-in-Time Data ─► Factor Risk Model ─► Cross-Sectional Alpha ─► Portfolio Optimizer
        │                                                                  │
        │                                                                  ▼
        │                                        Execution Engine (Python reference ⇄ C++20 core)
        │                                                                  │
        └──────────────► Experiment Ledger ◄──── Validation (CPCV/DSR/Capacity) ◄┘
                                │
                                ▼
                       PDF Tear-Sheet Dossier
```

Python orchestrates research; pybind11 (or nanobind) connects Python to the C++20 execution core.

## Architectural transition matrix

| Dimension | Legacy build | SEMN target |
|---|---|---|
| Universe | Single asset (SPY or BTC-USD) | Point-in-time S&P 500 constituents (S&P 1500 optional) keyed by `security_id` |
| Exposure | Directional long/flat | Dollar-neutral (Σwᵢ = 0) and beta-neutral (Σwᵢβᵢ = 0) |
| Risk | Static sizing; post-hoc drawdown | Barra-style factor neutralization; Ledoit–Wolf / RMT covariance; vol-relative drawdown controls |
| Execution | $1 commission; static 5 bps slippage | Almgren–Chriss impact, participation caps, borrow & financing |
| Alpha | Directional triple-barrier XGBoost | LightGBM LambdaMART ranking on residual returns (Hawkes = stretch) |
| Validation | Rolling walk-forward; embargo | CPCV, Deflated Sharpe Ratio, AUM capacity curves |
| Systems | Python, CSV, Spring Boot, React | Python + Polars/Parquet; C++20 + pybind11; ClickHouse optional; automated PDF dossier |

---

## Open decisions

| ID | Decision | Options | Blocks | Status |
|---|---|---|---|---|
| **D1** | Market & reference data vendor | CRSP + Compustat via WRDS (gold standard, needs institutional access) · Norgate (delisted names + historical index membership) · Sharadar via Nasdaq Data Link (delisted names, PIT fundamentals with filing dates) · Free sources (Wikipedia/GitHub constituent lists — noisy, no delisting returns) | Phase 1 wk 1, Phase 2 fundamentals & GICS | **Pending** |
| D2 | Storage backend | Parquet + Polars (default) · ClickHouse (only if query patterns justify it) | Phase 1 | Parquet first |
| D3 | Tick / L2 data for stretch tracks | Databento, Polygon, or skip | Stretch S1, S2 | Deferred |

The vendor chosen in D1 must provide, at minimum: raw unadjusted OHLCV, split and dividend events,
delisting dates **and delisting returns/prices**, historical ticker changes, historical index
membership with effective dates, and (for Phase 2) point-in-time fundamentals and GICS history.

---

## Milestones at a glance

| # | Phase | Weeks | Cumulative | Milestone |
|---|---|---|---|---|
| 1 | Point-in-time data & universe | 3 | 3 | Reproducible universe snapshots + DQ report |
| 2 | Factor risk model & covariance | 3 | 6 | Factor panel, residual returns, covariance diagnostics |
| 3 | Cross-sectional alpha | 3 | 9 | Signal-quality report |
| 4 | Optimizer + Python reference backtester | 3 | 12 | **End-to-end thin slice: data → alpha → weights → net-of-cost P&L** |
| 5 | Validation suite (CPCV, DSR, capacity) | 2 | 14 | Validation report + experiment ledger |
| 6 | C++20 execution core | 3 | 17 | Python/C++ parity on deterministic fixtures |
| 7 | Tear-sheet dossier & deterministic replay | 2 | 19 | Audit-ready PDF + replay harness |
| S1–S3 | Stretch tracks | — | — | Hawkes, Level-2, live feed gate |

---

## Phase 1 — Point-in-Time Data Architecture & Universe

**Timeline:** 3 weeks · **Goal:** a survivorship-bias-free, auditable, reproducible multi-asset data layer.

### Scope

- **Security master.** A permanent internal `security_id` per security, with a dated ticker history
  (`security_id, ticker, valid_from, valid_to`) and GICS sector/industry history. All joins use `security_id`.
  Tickers are display labels only — they change (FB → META) and are reused by unrelated companies.
- **Trading calendar.** NYSE sessions via `exchange_calendars`; daily data keyed by **session date**, not UTC midnight.
  Rebalance schedule derived from the calendar.
- **Raw ingestion.** Unadjusted OHLCV, split factors, dividends, delistings, symbol changes → Parquet
  (partitioned by year), written by a vendor adapter in `semn/data/sources/`.
- **Corporate actions & returns.**
  - Total-return series from raw prices + splits + dividends. Returns are computed from raw data and
    action factors; adjusted price series are derived views, never inputs to filters.
  - **Delisting returns:** apply the vendor's delisting return (or a documented fallback, e.g. −30% for
    performance-related delistings when unknown) on the delisting date.
- **Liquidity inputs computed in-house, PIT.** `ADV₂₀ = rolling_mean(raw_close × raw_volume, 20)` using data up to and
  including session t only; minimum observation count required.
- **Universe filters at rebalance t:** PIT index membership (`entry_date ≤ t < exit_date`), `ADV₂₀ ≥ $25M`,
  **raw** close `≥ $5`, last observation no older than `max_staleness` sessions (default 5).
  Filters applied to **raw, unadjusted** prices and volumes.
- **Storage.** `DataStore` protocol with a Parquet/Polars backend. The existing `ClickHouseLoader` becomes an
  optional backend behind the same protocol.
- **Snapshots & manifests.** Each universe build writes a Parquet snapshot plus a JSON manifest
  (input file hashes, vendor/version, filter parameters, code revision, content hash).
- **Data-quality report.** Coverage by date, missing bars, stale-price exclusions, suspicious returns
  (|r| > 50% without a corporate action), zero/negative prices, split-day volume anomalies, membership gaps.

### Target layout

```
semn/data/
├── sources/<vendor>.py      # vendor adapter → raw Parquet
├── security_master.py       # security_id, ticker history, GICS history
├── calendar.py              # NYSE sessions, rebalance dates
├── corporate_actions.py     # split/dividend factors, total returns, delisting returns
├── store.py                 # DataStore protocol; ParquetStore (ClickHouseStore optional)
├── universe.py              # PIT universe (vectorized)
├── quality.py               # data-quality report
└── snapshot.py              # versioned snapshots + manifest
```

### Week plan

| Week | Work |
|---|---|
| 1 | Settle **D1**. Security master, calendar, vendor adapter, raw Parquet ingestion. |
| 2 | Corporate actions, total-return series, delisting returns, in-house ADV₂₀. |
| 3 | Vectorized universe panel (`join_asof` with staleness tolerance), snapshots + manifests, DQ report, invariant tests. |

### Verification invariants (tests)

- [x] No security appears before its inclusion date or on/after its exit date — checked across **every** rebalance date in the panel (fixture test; property-based version still to add).
- [x] A reused ticker never merges two securities.
- [x] The $5 price floor and ADV use raw prices (fixture: a stock with a later 10:1 split must pass the floor before the split).
- [x] ADV does not spike on a split date.
- [x] Total return across a split day equals the economic return (split-neutral).
- [x] Delisting return is applied exactly once, on the delisting date.
- [x] Stale observations older than `max_staleness` are excluded and reported.
- [x] Building the same snapshot twice yields the same content hash.
- [x] Panel build for N ≥ 500 × T ≥ 2,500 completes in seconds; single cross-section retrieval is sub-second (measured 1.4 s / 0.14 s at 500 × 2,768 sessions; a benchmark, not a CI test).

### Deliverable

Reproducible point-in-time universe snapshots with manifests, a total-return panel, and a data-quality report.

### Phase 1 status

Everything except the vendor adapter is built and tested; the adapter waits on **D1**.

| Item | Status |
|---|---|
| Vendor adapter (`sources/<vendor>.py`) → raw Parquet | ⏸ blocked on **D1** |
| Security master (`security_master.py`): `security_id`, dated ticker + GICS history | ✅ overlapping intervals rejected; unmapped vendor rows reported |
| Trading calendar (`calendar.py`): NYSE sessions, rebalance dates | ✅ `exchange_calendars` XNYS |
| Corporate actions (`corporate_actions.py`): total returns, delisting returns | ✅ split/dividend factors from raw prices; delisting return compounded once (−30% fallback for performance delistings) |
| In-house ADV₂₀ (raw close × raw volume, min 20 obs) | ✅ |
| Universe (`universe.py`): PIT membership, staleness in sessions, vectorized | ✅ NaN prices treated as missing, duplicate rows rejected, stale market cap treated as missing |
| `DataStore` protocol + `ParquetStore` (`store.py`) | ✅ partitioned by year, write-then-swap; `ClickHouseStore` not built (optional) |
| Snapshots + manifests (`snapshot.py`) | ✅ content-addressed; input hashes, params, vendor, code revision; hash verified on load |
| Data-quality report (`quality.py`) | ✅ coverage, missing bars, duplicates, off-session rows, bad prices, suspicious returns, split-day volume, membership gaps, stale exclusions |
| ClickHouse tick/bar loader | ✅ optional backend (`[clickhouse]` extra); bounds read as New York time and sent as UTC |

**Fixes to existing code**

- [x] Re-key `UniverseBuilder` on `security_id`.
- [x] Document and enforce that `close` / volume inputs are **raw**, unadjusted (ADV is now derived from the same raw panel).
- [x] Add `max_staleness` (in sessions); report exclusions.
- [x] Vectorize `build_panel` (single cross-join + `join_asof` pass instead of a per-date loop).
- [x] Align `LiquidityFilter` defaults with this roadmap (`adv_floor=25e6`, `price_floor=5.0`, no default market-cap floor).
- [x] `ClickHouseLoader`: bind symbols as `{symbols:Array(String)}` instead of string-escaping.
- [x] Key daily bars by session date; resample ticks using exchange-local (New York) dates. Tick bars still include pre/post-market trades (regular-hours filter is a follow-up).
- [x] Add `pyproject.toml` with optional extras (`[clickhouse]`, `[report]`, `[hawkes]`, `[dev]`); drop `tick` from core requirements.

---

## Phase 2 — Factor Risk Model & Covariance Denoising

**Timeline:** 3 weeks · **Goal:** separate systematic exposure from residual returns; stable risk estimates.

- **Exposures** Xₜ ∈ ℝᴺˣᴷ, point-in-time: market beta (β vs SPY, 252-day), size (log market cap),
  value (book-to-market / earnings yield, using **filing dates**), momentum (12–1 month), short-term reversal
  (1 month), realized volatility (60 day), GICS sector dummies (PIT history).
- **Standardization:** winsorize, then cross-sectional z-score (cap-weighted mean 0, unit std) at each date.
  All factor-neutrality limits downstream refer to these standardized exposures.
- **Residualization:** Rₜ = Xₜfₜ + εₜ via cross-sectional WLS (weights √market cap). Store fₜ and residuals εₜ.
  Alpha models train on residual returns.
- **Covariance:** Σ = X·F·Xᵀ + D (factor covariance F with EWMA, specific variance D). Compare against
  Ledoit–Wolf shrinkage of the sample covariance; RMT (Marchenko–Pastur clipping) optional, validated
  against the shrinkage baseline.
- **Invariants:** Σ symmetric and PSD (λ_min ≥ −1e-10); report condition number before/after denoising;
  residuals uncorrelated with exposures by construction; out-of-sample risk forecast bias statistic ≈ 1.

**Deliverable:** factor exposure panel, factor and residual returns, covariance diagnostics.

---

## Phase 3 — Cross-Sectional Alpha Engine

**Timeline:** 3 weeks · **Goal:** rank future residual returns across the universe.

- **Models:** linear baseline (ridge on ranked features) → LightGBM LambdaMART / CatBoost ranking.
  Horizons T+1 … T+5 on residual returns.
- **Features:** daily-data features only (price/volume, reversal, momentum variants, volatility, liquidity,
  fundamentals from Phase 2). Each date: percentile-rank or z-score, then orthogonalize against sector and beta.
- **Labels:** forward residual returns; purge overlapping labels when training.
- **Signal metrics:** Rank IC per date; per-period ICIR = mean(IC)/std(IC); **annualized** ICIR =
  per-period ICIR × √(periods per year). Report IC decay by horizon, turnover, and stability by sector and regime.
  *Research targets (hypotheses, not guarantees):* mean Rank IC ≥ 0.03 at the 5-day horizon using
  non-overlapping samples; annualized ICIR ≥ 1.0.
- **Invariants:** shuffled-label test (IC ≈ 0), feature ablations, label-overlap checks, gross signal quality
  reported separately from net-of-cost results.

**Deliverable:** reproducible ranking experiments and a signal-quality report.

---

## Phase 4 — Portfolio Optimizer & Python Reference Backtester

**Timeline:** 3 weeks · **Goal:** turn residual alpha into constrained dollar-neutral weights and simulate them with realistic friction. **This phase completes the end-to-end thin slice.**

### Optimizer

maximize  wᵀα − (λ_risk/2)·wᵀΣw − λ_cost·Σᵢ Impact(Δwᵢ)  (CVXPY + OSQP/Clarabel)

Constraints (on unit-gross weights):

- Dollar neutral Σwᵢ = 0; beta neutral |Σwᵢβᵢ| ≤ 0.01
- Gross Σ|wᵢ| = 1 (scaled afterwards by volatility targeting)
- Style-factor neutrality |Σᵢ wᵢ·zᵢₖ| ≤ 0.05 on standardized exposures; sector net |Σᵢ∈s wᵢ| ≤ 0.02
- Position bounds −0.02 ≤ wᵢ ≤ 0.02 (at unit gross)
- Liquidity: traded shares ≤ 5% of ADV₂₀; hard-to-borrow names cannot be shorted

### Exposure scaling and risk controls

- **Volatility targeting:** gross Gₜ = min(G_max, σ_target / σ̂ₜ), σ_target = 6–8% annualized, G_max configurable (default 2.0).
  Gross is the *output* of this rule, not a fixed constant.
- **Drawdown controls (vol-relative):** with σ_m = σ_target/√12,
  soft cut at drawdown ≥ 1.5σ_m (≈3% at 7% vol) → halve gross;
  hard stop at ≥ 2.0σ_m (≈4%) → unwind per execution policy.
  **Re-entry:** after a hard stop, stay flat for a cooldown (default 20 sessions), re-enter at 50% gross,
  ramp to full over 20 sessions. Report how often each control fires and its cost.

### Python reference backtester (`semn/backtest/`)

- Signals at close t fill no earlier than open t+1.
- Cost model (`semn/costs/`): commissions, half-spread, Almgren–Chriss impact
  ΔP = sgn(Q)·[γσ(|Q|/ADV)^0.5 + ησ(|Q|/ADV)], parameters calibrated, not assumed.
- Financing: short borrow drag Σ_shorts |wᵢ|·bᵢ/360 (hard-to-borrow availability, fee tiers);
  cash financing at SOFR ± spread; day-count convention documented.
- Target weights kept distinct from filled positions; per-order and per-day cost attribution.

### Invariants

Feasible/infeasible problems, neutrality after scaling, bound and budget edges, stale prices, missing or
borrow-constrained names, partial fills, gap days; equity = cash + marked positions after every event;
net performance degrades monotonically as participation rises from 1% to 15% of ADV in controlled tests.

**Deliverable:** optimizer, reference backtester, cost model, constraint and drawdown test suite, first net-of-cost results.

---

## Phase 5 — Validation Suite: CPCV, DSR & Capacity

**Timeline:** 2 weeks · **Goal:** estimate robustness under leakage, selection bias and impact.

- **CPCV:** N temporal groups, test combinations C(N,k), purge overlapping labels, embargo after test sets.
  Report the distribution of out-of-sample Sharpe paths, not a single path.
- **Deflated Sharpe Ratio:** corrects for number of trials K, skewness and kurtosis. Validate against a
  reference implementation (Bailey & López de Prado). K comes from the experiment ledger, not memory.
- **Holdout:** a final untouched test period, used once.
- **Capacity:** AUM $10M → $500M under explicit ADV, participation, spread, impact, borrow and turnover assumptions;
  report net return, Sharpe, turnover, drawdown and cost drag by AUM; define the ceiling from stated criteria.
- **Experiment ledger:** extends the Phase 1 manifests — data snapshot hash, feature set, model params,
  code revision, seeds, costs, windows, outputs. Every run is logged; no cherry-picking.

**Acceptance criteria (replace the v1 Sharpe target):**
DSR > 0.95; median CPCV path Sharpe > 0 with a stated share of positive paths; results survive realistic costs at
the chosen AUM; residual exposures within limits. A net Sharpe of roughly 0.5–1.5 on this universe and data
with these checks passing is a credible outcome.

**Deliverable:** validation report, experiment ledger, capacity curve.

---

## Phase 6 — C++20 Execution Core

**Timeline:** 3 weeks · **Goal:** a fast, deterministic execution and accounting core with proven parity.

- Portfolio valuation, position/cash accounting, order matching, impact and financing in C++20 (`src/core/`).
- Exposed to Python via pybind11 or nanobind; built with CMake + scikit-build-core.
- **Parity:** fills, positions, cash and P&L match the Python reference engine on deterministic fixtures
  (bit-exact where possible, documented tolerance otherwise).
- Benchmarks: full-history backtest runtime vs Python reference.

**Deliverable:** tested C++ core, parity suite, benchmark report.

---

## Phase 7 — Tear-Sheet Dossier & Deterministic Replay

**Timeline:** 2 weeks · **Goal:** audit-ready evidence and live-like simulation.

- **PDF dossier** (matplotlib + WeasyPrint or LaTeX): rolling net Sharpe/Sortino, drawdown, factor attribution,
  exposures over time, turnover, cost drag (bps), capacity curve, CPCV distribution, DSR, experiment ledger summary.
- **Deterministic replay:** stream historical daily events through the engine as if live; log every decision and fill;
  identical output across runs.

**Deliverable:** auto-generated PDF dossier and replay harness.

---

## Stretch tracks (outside core timeline)

| Track | Prerequisite | Scope |
|---|---|---|
| **S1 — Hawkes order-flow** | Timestamped trade data (D3) | Bivariate Hawkes on buy/sell events; branching ratio Γ as a liquidity-stress signal for execution aggressiveness. Keep only if it adds stable out-of-sample value. |
| **S2 — Level-2 order-book harness** | Historical depth data (D3) | Queue position, cancellations, fill probabilities for limit orders. |
| **S3 — Live feed gate** | Phases 1–7 complete; monitoring in place | WebSocket market data into the replay harness. **Live order placement remains out of scope.** |

---

## Repository layout (target)

```
Live-Algorithmic-Trading-Platform/
├── pyproject.toml              # extras: [clickhouse], [report], [hawkes], [dev]
├── CMakeLists.txt
├── ROADMAP.md
├── src/core/                   # C++20: engine.cpp, impact_model.cpp, bindings.cpp
├── semn/
│   ├── data/                   # Phase 1 (see layout above)
│   ├── factors/                # risk_model.py, covariance.py
│   ├── alpha/                  # ranking_model.py, features.py (hawkes_microstructure.py = S1)
│   ├── optimizer/              # portfolio_optimizer.py, risk_controls.py
│   ├── costs/                  # impact.py, financing.py
│   ├── backtest/               # reference_engine.py
│   ├── evaluation/             # cpcv.py, dsr.py, capacity.py, ledger.py
│   └── reporting/              # tear_sheet.py
└── tests/
```

---

## Interview talking points

- **Why the backtest is realistic:** next-bar fills, calibrated impact, borrow and financing, raw-price universe filters, delisting returns, no lookahead.
- **Why the ML is valid:** point-in-time features with filing dates, residual-return labels, purge/embargo, CPCV, DSR with an honest trial count.
- **Why the strategy matters:** factor and beta neutrality, capacity analysis, regime and sector breakdowns, net-of-cost risk-adjusted metrics.

## Common failure points

- Starting with UI, live trading, or C++ before an end-to-end Python pipeline works.
- Keying data on tickers; filtering on adjusted prices; ignoring delisting returns.
- Adding Hawkes/L2 features without suitable data.
- Treating a Sharpe target or drawdown stop as guaranteed.
- Selecting models on the final holdout period.

*Educational research project. References to industry practice are design inspiration only.*
