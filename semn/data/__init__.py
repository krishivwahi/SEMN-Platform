"""
semn/data/__init__.py
==========================
Public API surface for the SEMN platform data layer (Phase 1).
"""

from semn.data import calendar
from semn.data.clickhouse_loader import (
    BAR_SCHEMA,
    TICK_SCHEMA,
    ClickHouseClientProtocol,
    ClickHouseLoader,
)
from semn.data.corporate_actions import total_returns
from semn.data.quality import quality_report
from semn.data.security_master import SecurityMaster
from semn.data.snapshot import content_hash, load_snapshot, write_snapshot
from semn.data.store import DataStore, ParquetStore
from semn.data.universe import (
    UNIVERSE_SCHEMA,
    LiquidityFilter,
    UniverseBuilder,
)

__all__ = [
    "calendar",
    # Loader (optional ClickHouse backend)
    "ClickHouseLoader",
    "ClickHouseClientProtocol",
    "TICK_SCHEMA",
    "BAR_SCHEMA",
    # Reference data & returns
    "SecurityMaster",
    "total_returns",
    # Storage & snapshots
    "DataStore",
    "ParquetStore",
    "content_hash",
    "write_snapshot",
    "load_snapshot",
    # Universe & quality
    "UniverseBuilder",
    "LiquidityFilter",
    "UNIVERSE_SCHEMA",
    "quality_report",
]
