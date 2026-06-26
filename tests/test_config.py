import os
import pytest
from lazycat.config import Config, get_env_or_raise

def test_config_raises_on_missing_env(monkeypatch):
    # Clear all environment variables that Config expects
    for key in ["PRISM_URL", "VAULT_SERVICE_URL", "LAZY_TOOL_SERVICE_PORT", "PRISM_SERVICE_PORT"]:
        monkeypatch.delenv(key, raising=False)
        
    with pytest.raises(EnvironmentError) as exc_info:
        config = Config()
        _ = config.PRISM_URL
        
    assert "Missing required environment variable" in str(exc_info.value)

def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("PRISM_URL", "http://prism")
    monkeypatch.setenv("VAULT_SERVICE_URL", "http://vault")
    monkeypatch.setenv("LAZY_TOOL_SERVICE_PORT", "8000")
    monkeypatch.setenv("PRISM_SERVICE_PORT", "8001")
    
    config = Config()
    assert config.PRISM_URL == "http://prism"
    assert config.LAZY_TOOL_SERVICE_PORT == "8000"
