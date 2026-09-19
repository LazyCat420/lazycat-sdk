# Migrating to the LazyCat SDK V1 Client

With the release of the central `lazy-agent-service`, the `lazycat-sdk` transitions from running its own internal LLM and tool-call loops (`AgentHarness`) to being a thin, reliable client for the shared runtime.

## What's Changed

- **No more local agent loops.** The generic execution logic has moved to the `lazy-agent-service`.
- **Typed Models.** We now provide Pydantic models in `lazycat.models` matching the new v1 run contract (`RunRequest`, `RunResult`, `StreamEvent`, etc.).
- **Reliable Streaming.** The new `RunClient` (`lazycat.client.RunClient`) automatically reconnects and handles sequence decoding for Server-Sent Events (SSE).
- **Transport Attribution.** Request idempotency and transport configurations are now handled in the single `RunClient`.

## Backwards Compatibility

For existing consumers (e.g. `trading-service`, `HTML-Notes`), the legacy `AgentHarness` and `BaseAgent` interfaces remain available and backwards compatible!
They internally construct a `RunRequest` and proxy the execution to the shared runtime. You do not need to rewrite your agent logic immediately.

## Native V1 Integration Example

For new applications, or when migrating, you can directly construct the `RunRequest` and use the `RunClient`:

```python
import asyncio
from lazycat.models import RunRequest, AgentProfile
from lazycat.client import RunClient

async def run_agent():
    client = RunClient(base_url="http://lazy-agent-service:3000")
    
    req = RunRequest(
        profile=AgentProfile(
            name="analyst-agent",
            system_prompt="Analyze the data...",
            model="qwen-coder"
        ),
        messages=[{"role": "user", "content": "Start task."}]
    )
    
    # Start the run
    run_id = await client.create_run(req)
    
    # Stream events reliably (auto-reconnects on drop)
    async for event in client.stream_run(run_id):
        if event.type == "chunk":
            print(event.content, end="", flush=True)
        elif event.type == "done":
            print("\\n\\nFinal Output:", event.result.final_output)

if __name__ == "__main__":
    asyncio.run(run_agent())
```
