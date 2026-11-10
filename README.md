# SEMN — Systematic Equity Research Platform

[![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-blue.svg)](#)
[![Language: Python 3.11+](https://img.shields.io/badge/Language-Python%203.11%2B-blue.svg)](#)

A high-performance quantitative research and execution platform for systematic equity strategies.

---

## Repository Layout

```
semn-platform/
├── CMakeLists.txt          # C++20 build config
├── pyproject.toml          # Python package + extras: [clickhouse], [report], [hawkes], [dev]
├── ROADMAP.md              # Engineering roadmap (v2)
├── src/core/               # High-performance C++20 execution core
├── semn/
│   ├── data/               # Data pipeline & universe construction
│   ├── factors/            # Risk modelling
│   ├── alpha/              # Signal generation
│   ├── optimizer/          # Portfolio construction
│   ├── evaluation/         # Validation framework
│   └── reporting/          # Performance reporting
└── tests/
```

---

## Stack

- **Python 3.11+** — research, modelling, data pipeline
- **C++20** — execution core, exposed to Python via pybind11
- **ClickHouse** (optional) — columnar market data storage
- **Polars / PyArrow** — high-performance dataframe operations

## Install

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[clickhouse]"     # optional ClickHouse backend
```
