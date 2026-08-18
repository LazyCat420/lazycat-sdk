"""Contract test verifying that lazycat-sdk has no PostgreSQL dependencies or references."""
import os
import sys
import pytest


def test_lazycat_sdk_has_no_postgres_references():
    """Verify that importing lazycat-sdk requires no PostgreSQL environment or drivers."""
    # Ensure DATABASE_URL is not set
    os.environ.pop("DATABASE_URL", None)

    import lazycat
    assert lazycat is not None

    # Check imported driver modules
    pg_drivers = {"psycopg", "psycopg2", "asyncpg", "pg8000", "sqlalchemy.dialects.postgresql"}
    loaded_drivers = [mod for mod in sys.modules if any(mod == drv or mod.startswith(drv + ".") for drv in pg_drivers)]
    assert len(loaded_drivers) == 0, f"Found PostgreSQL drivers loaded: {loaded_drivers}"


def test_lazycat_submodules_import_cleanly_without_db():
    """Verify that core lazycat components import without database configuration."""
    from lazycat import agent, cache, config, llm, sse
    assert agent is not None
    assert cache is not None
    assert config is not None
    assert llm is not None
    assert sse is not None
