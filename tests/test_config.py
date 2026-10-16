import os
import pytest
from lazycat.config import Config, get_env_or_raise

def test_config_raises_on_missing_env(monkeypatch):
    # Clear all environment variables that Config expects
    for key in ["LAZY_AGENT_URL", "VAULT_SERVICE_URL", "LAZY_AGENT_SERVICE_PORT"]:
        monkeypatch.delenv(key, raising=False)
        
    config = Config()
    assert config.LAZY_AGENT_URL == "http://10.0.0.16:5591"
    assert config.VAULT_SERVICE_URL == "http://10.0.0.16:5599"
    assert config.LAZY_AGENT_SERVICE_PORT == "5591"

def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("LAZY_AGENT_URL", "http://agent")
    monkeypatch.setenv("VAULT_SERVICE_URL", "http://vault")
    monkeypatch.setenv("LAZY_AGENT_SERVICE_PORT", "8000")
    
    config = Config()
    assert config.LAZY_AGENT_URL == "http://agent"
    assert config.LAZY_AGENT_SERVICE_PORT == "8000"
