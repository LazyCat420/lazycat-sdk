import pytest
import asyncio
from lazycat.models import RunRequest, AgentProfile
from lazycat.client import RunClient, RunClientError, StreamChunk

@pytest.mark.asyncio
async def test_run_request_model():
    req = RunRequest(
        profile=AgentProfile(
            name="test-agent",
            system_prompt="You are a helpful assistant.",
            model="qwen-coder"
        ),
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert req.profile.name == "test-agent"
    assert req.max_iterations == 15
    
    data = req.model_dump(exclude_none=True)
    assert "bench_task" not in data
    assert data["profile"]["provider"] == "vllm"

# In a real environment we'd use respx or httpx-mock to mock the endpoints.
# For now, we just ensure the client instantiates correctly.
def test_client_init():
    client = RunClient(base_url="http://test")
    assert client.base_url == "http://test"
    assert client.project == "lazycat-sdk-app"
