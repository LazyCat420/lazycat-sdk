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

    @property
    def PRISM_ENABLED(self) -> bool:
        return os.getenv("PRISM_ENABLED", "True").lower() == "true"

    @property
    def JETSON_VLLM_URL(self) -> str:
        return os.getenv("JETSON_VLLM_URL", "http://10.0.0.30:8000")

    @property
    def DGX_SPARK_VLLM_URL(self) -> str:
        return os.getenv("DGX_SPARK_VLLM_URL", "http://10.0.0.141:8000")

    @property
    def PROJECT_NAME(self) -> str:
        return os.getenv("PROJECT_NAME", "lazycat-sdk-app")

# Global config instance, loaded on import
config = Config()
