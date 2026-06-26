import os

def get_env_or_raise(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val

class Config:
    @property
    def PRISM_URL(self):
        return get_env_or_raise("PRISM_URL")

    @property
    def VAULT_SERVICE_URL(self):
        return get_env_or_raise("VAULT_SERVICE_URL")

    @property
    def LAZY_TOOL_SERVICE_PORT(self):
        return get_env_or_raise("LAZY_TOOL_SERVICE_PORT")

    @property
    def PRISM_SERVICE_PORT(self):
        return get_env_or_raise("PRISM_SERVICE_PORT")

# Global config instance, loaded on import
config = Config()
