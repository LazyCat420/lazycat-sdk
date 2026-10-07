"""grounded_research's news and web sources go through lazy-agent-service.

They used to call the `ddgs` library, which scrapes DuckDuckGo and, on a blocked
IP, tries about nine engines per call. The free search engines bot-block this
network's one public IP, so every automated search made that worse for every
project. news_search (keyed news APIs) and web_search (Exa's keyless index, one
cache and rate limit for the network) replace them.
"""
import ast
import asyncio
import pathlib

import importlib

# The module, not the function: lazycat/__init__.py re-exports the
# grounded_research FUNCTION under the module's own name.
gr = importlib.import_module("lazycat.grounded_research")


class _Client:
    """Stands in for httpx.AsyncClient; records each POST."""
    calls = []
    reply = {}
    error = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **k):
        _Client.calls.append((url, json))
        if _Client.error:
            raise _Client.error

        class _Resp:
            def json(self_inner):
                return _Client.reply
        return _Resp()


def _use(monkeypatch, reply=None, error=None):
    _Client.calls, _Client.reply, _Client.error = [], reply or {}, error
    monkeypatch.setattr(gr.httpx, "AsyncClient", _Client)


def test_news_comes_from_news_search(monkeypatch):
    _use(monkeypatch, {"items": [{"title": "Fed holds rates", "url": "https://example.com/fed",
                                  "snippet": "s", "date": "Sat, 05 Sep 2026 22:10:00 GMT"}]})
    out = asyncio.run(gr._shared_news("fed rates", 8))
    assert _Client.calls == [(f"{gr.DEFAULT_SEARCH_SERVICE_URL}/execute/news_search", {"topic": "fed rates", "limit": 8})]
    assert out[0].title == "Fed holds rates" and out[0].source == "news_search"
    assert out[0].published_at is not None and out[0].published_at.year == 2026


def test_web_comes_from_the_shared_web_search(monkeypatch):
    _use(monkeypatch, {"status": "ok", "results": [{"title": "A", "url": "https://example.com/a",
                                                    "snippet": "s", "published": "2026-10-06T11:56:11.244Z"}]})
    out = asyncio.run(gr._shared_web("hiking sandals", 8))
    assert _Client.calls == [(f"{gr.DEFAULT_SEARCH_SERVICE_URL}/execute/web_search", {"query": "hiking sandals", "limit": 8})]
    assert out[0].url == "https://example.com/a" and out[0].source == "web_search"
    assert out[0].published_at.isoformat().startswith("2026-10-06T11:56:11")


def test_a_refused_or_failed_search_contributes_nothing(monkeypatch):
    _use(monkeypatch, {"status": "rate_limited", "results": [], "error": "resume in 40 s"})
    assert asyncio.run(gr._shared_web("q", 8)) == []
    _use(monkeypatch, error=OSError("lazy-agent-service down"))
    assert asyncio.run(gr._shared_web("q", 8)) == []
    assert asyncio.run(gr._shared_news("q", 8)) == []


def test_the_module_no_longer_imports_ddgs():
    """A guard on the code itself: no `ddgs` or DuckDuckGo import, anywhere in it."""
    tree = ast.parse(pathlib.Path(gr.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("ddgs" in name or "duckduckgo" in name for name in imported), imported
