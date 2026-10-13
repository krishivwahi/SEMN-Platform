# SEMN — Systematic Equity Market-Neutral Platform

[![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](#)
[![Language: Python 3.11+](https://img.shields.io/badge/Language-Python%203.11%2B-blue.svg)](#)
[![Strategy: Market-Neutral](https://img.shields.io/badge/Strategy-Statistical%20Arbitrage-green.svg)](#)
[![Risk: Millennium-Compliant](https://img.shields.io/badge/Risk-Millennium%20Guardrails-red.svg)](#)

A high-performance quantitative research and execution platform built for **cross-sectional statistical arbitrage** in a multi-manager institutional setting.

Designed to meet the systems engineering and risk management requirements of multi-manager platforms (Millennium, Citadel, Point72-style pod constraints):

- **Dollar-neutral & beta-neutral** portfolios across S&P 500 / Russell 1000
- **Barra/Axioma-style** factor risk model with Ledoit-Wolf + RMT covariance denoising
- **Almgren-Chriss** market impact, volume participation curves, short-borrow financing matrix
- **LambdaMART / LightGBM** cross-sectional ranking on factor-residualised returns
- **Hawkes process** order-flow microstructure for liquidity shock detection
- **CPCV + Deflated Sharpe Ratio** validation with parametric AUM capacity curves
- **C++20 execution core** exposed to Python via pybind11
- **Automated institutional tear-sheet** PDF dossier (factor attribution, capacity, drawdown)

---

## Repository Layout

```
semn-platform/
├── CMakeLists.txt                        # C++20 build config (pybind11 bindings)
├── requirements.txt                      # Python dependencies
├── src/core/                             # High-performance C++20 execution core
│   ├── engine.cpp                        # Vectorized order-matching & portfolio accounting
│   ├── impact_model.cpp                  # Almgren-Chriss impact & borrow fee computation
│   └── bindings.cpp                      # pybind11 Python interfaces
├── platform/
│   ├── data/
│   │   ├── clickhouse_loader.py          # Columnar tick & bar pipeline (Phase 1)
│   │   └── universe.py                   # Point-in-time constituents & liquidity filters (Phase 1)
│   ├── factors/
│   │   ├── risk_model.py                 # Cross-sectional factor regression — Barra/Axioma (Phase 3)
│   │   └── covariance.py                 # Ledoit-Wolf shrinkage & RMT eigenvalue clipping (Phase 3)
│   ├── alpha/
│   │   ├── ranking_model.py              # LightGBM / CatBoost pairwise ranking (Phase 4)
│   │   └── hawkes_microstructure.py      # Bivariate Hawkes point process (Phase 4)
│   ├── optimizer/
│   │   └── portfolio_optimizer.py        # Markowitz QP with factor constraints (Phase 5)
│   ├── evaluation/
│   │   ├── cpcv.py                       # Combinatorial Purged Cross-Validation (Phase 6)
│   │   └── capacity.py                   # AUM capacity curves & Deflated Sharpe (Phase 6)
│   └── reporting/
│       └── tear_sheet.py                 # Institutional PDF performance dossier (Phase 7)
└── tests/
    ├── test_cpp_engine.py                # C++ engine invariant & accounting parity
    ├── test_factor_neutrality.py         # Beta orthogonalisation verification
    └── test_optimization.py              # Convex constraint & budget boundary validation
```

---

## Build Phases

| Phase | Scope | Duration |
|-------|-------|----------|
| **1** | High-performance data architecture & point-in-time universe | 3 weeks |
| **2** | C++20 execution engine & institutional friction modeling | 3 weeks |
| **3** | Factor risk model & covariance matrix denoising | 2–3 weeks |
| **4** | Cross-sectional alpha signal engine & Hawkes microstructure | 3 weeks |
| **5** | Convex portfolio optimization & Millennium risk guardrails | 2–3 weeks |
| **6** | Institutional validation suite (CPCV & capacity analysis) | 2 weeks |
| **7** | Simulation gateway & institutional tear-sheet dossier | 2 weeks |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Net Sharpe (24-month OOS) | > 2.0 |
| Sortino Ratio | > 3.0 |
| Max Drawdown | < 4.0% |
| Mean Rank IC | ≥ 0.04 |
| IC Information Ratio | ≥ 0.75 |
| Market Beta | < 0.05 |
| Annualized Volatility | 6%–8% |
