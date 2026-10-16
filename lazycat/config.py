import os

def get_env_or_raise(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val

class Config:
    @property
    def LAZY_AGENT_URL(self):
        # lazy-agent-service native gateway.
        return os.getenv("LAZY_AGENT_URL", "http://10.0.0.16:5591")

    @property
    def VAULT_SERVICE_URL(self):
        return os.getenv("VAULT_SERVICE_URL", "http://10.0.0.16:5599")

    @property
    def LAZY_AGENT_SERVICE_PORT(self):
        return os.getenv("LAZY_AGENT_SERVICE_PORT", "5591")

    @property
    def PROJECT_NAME(self) -> str:
        return os.getenv("PROJECT_NAME", "lazycat-sdk-app")

# Global config instance, loaded on import
config = Config()
