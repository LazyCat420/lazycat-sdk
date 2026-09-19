import json
import httpx
import pytest
from lazycat.client import RuntimeClient, RunEventDecodeError
from lazycat.models import CreateRunRequest


class Fragments(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        for byte in self.content:
            yield bytes([byte])


@pytest.mark.asyncio
@pytest.mark.parametrize('separator', ['\n', '\r\n'])
async def test_fragmented_unicode_heartbeat_duplicate_and_terminal(separator):
    def event(identity, kind, data):
        return 'data: ' + json.dumps(dict(id=identity, run_id='run-1', type=kind,
            timestamp='2026-09-19T00:00:00Z', data=data), ensure_ascii=False) + separator * 2
    delta = event('evt-1', 'message.delta', {'delta': 'News ☀'})
    stream = (': heartbeat' + separator * 2 + delta + delta + event('evt-2', 'run.completed', {})).encode()
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=Fragments(stream)))) as http:
        client = RuntimeClient(base_url='http://runtime.example', client=http)
        events = [e async for e in client.stream_run(CreateRunRequest(profile_id='test', input='news'))]
    assert [e.id for e in events] == ['evt-1', 'evt-2']
    assert events[0].data['delta'] == 'News ☀'


@pytest.mark.asyncio
@pytest.mark.parametrize('payload', [b'', b': heartbeat\r\n\r\n', b'data: [DONE]\n\n'])
async def test_premature_eof_is_explicit(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=payload))) as http:
        with pytest.raises(RunEventDecodeError):
            _ = [e async for e in RuntimeClient(client=http).stream_run(CreateRunRequest(profile_id='test', input='news'))]
