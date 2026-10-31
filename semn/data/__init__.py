"""
semn/data/__init__.py
==========================
Public API surface for the SEMN platform data layer (Phase 1).
"""

from semn.data.clickhouse_loader import (
    BAR_SCHEMA,
    TICK_SCHEMA,
    ClickHouseClientProtocol,
    ClickHouseLoader,
)
from semn.data.universe import (
    UNIVERSE_SCHEMA,
    LiquidityFilter,
    UniverseBuilder,
)

__all__ = [
    # Loader
    "ClickHouseLoader",
    "ClickHouseClientProtocol",
    "TICK_SCHEMA",
    "BAR_SCHEMA",
    # Universe
    "UniverseBuilder",
    "LiquidityFilter",
    "UNIVERSE_SCHEMA",
]
