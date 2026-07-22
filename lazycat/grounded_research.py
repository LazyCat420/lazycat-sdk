"""Grounded research — reliable, fast, retrieval-augmented web research.

The universal port of trading-service/app/services/web_search.py (the logic behind
the trading AI-strategy chat's web search). It is the RELIABLE counterpart to the
agentic `research()`: instead of letting an LLM decide to call flaky read/scrape
tools in a loop, it does the retrieval itself in deterministic Python —

  1. fan out over news sources concurrently (DuckDuckGo News + Text, keyless;
     Finnhub if FINNHUB_API_KEY is set),
  2. dedupe (Jaccard title similarity) + optional freshness gate,
  3. scrape the top-N article bodies via the scraper HTTP API (:8001/scrape/batch,
     the consolidated scraper that actually works — NOT the broken agent tool),
  4. format a numbered [1][2][3] citation context,
  5. run ONE direct vLLM synthesis pass into a {title, overview, answer, sources}
     brief (skippable — pass synthesize=False to get the grounded context back).

Why this exists: on this stack the agent's read_url/read_web_page fail on hard news
URLs and scrape_url (the bridge) is broken, so agentic research is slow/shallow.
This path uses the working scraper directly and one LLM call — reliable and fast.

    from lazycat.grounded_research import grounded_research
    brief = await grounded_research("what's moving the US stock market today",
                                    domain="finance")
    # -> {"title": ..., "overview": ..., "answer": <markdown>, "sources": [...]}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from .logging import get_logger

logger = get_logger(__name__)

# The consolidated scraper (trading-service/app/scraper) — served at :8001. Used
# via HTTP so this shared helper needs no scraper code of its own.
DEFAULT_SCRAPER_URL = os.getenv("SCRAPER_SERVICE_URL", "http://10.0.0.16:8001")
# Gold Spark (fast) for the single synthesis pass. Direct completion, no agent loop.
DEFAULT_VLLM_URL = os.getenv("GROUNDED_VLLM_URL", os.getenv("VLLM_URL", "http://10.0.0.141:8000"))

_MIN_SEARCH_INTERVAL = 1.5  # ddgs rate-limit guard
_last_search = 0.0

_STOP = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "is",
         "are", "was", "by", "with", "from", "as", "how", "why", "what", "s"}


@dataclass
class Article:
    title: str = ""
    url: str = ""
    snippet: str = ""
    full_text: str = ""
    source: str = "web"
    published_at: Optional[datetime] = None


def _parse_dt(s: str) -> Optional[datetime]:
    """Best-effort ISO/RFC timestamp parse → aware UTC datetime."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ── Sources ────────────────────────────────────────────────────────────────

async def _ddg_news(query: str, max_results: int) -> list[Article]:
    global _last_search
    wait = _MIN_SEARCH_INTERVAL - (time.monotonic() - _last_search)
    if wait > 0:
        await asyncio.sleep(wait)
    try:
        from ddgs import DDGS
        def _do():
            with DDGS() as d:
                return list(d.news(query, max_results=max_results, timelimit="w"))
        raw = await asyncio.to_thread(_do)
        _last_search = time.monotonic()
    except Exception as e:
        logger.warning("ddg news failed for %r: %s", query[:50], e)
        return []
    out = []
    for r in raw:
        out.append(Article(
            title=r.get("title", ""), url=r.get("url", r.get("href", "")),
            snippet=r.get("body", r.get("excerpt", "")), source="ddg_news",
            published_at=_parse_dt(r.get("date", ""))))
    return out


async def _ddg_text(query: str, max_results: int) -> list[Article]:
    global _last_search
    wait = _MIN_SEARCH_INTERVAL - (time.monotonic() - _last_search)
    if wait > 0:
        await asyncio.sleep(wait)
    try:
        from ddgs import DDGS
        def _do():
            with DDGS() as d:
                return list(d.text(query, max_results=max_results, timelimit="w"))
        raw = await asyncio.to_thread(_do)
        _last_search = time.monotonic()
    except Exception as e:
        logger.warning("ddg text failed for %r: %s", query[:50], e)
        return []
    out = []
    for r in raw:
        out.append(Article(
            title=r.get("title", ""), url=r.get("href", r.get("link", "")),
            snippet=r.get("body", r.get("snippet", "")), source="ddg_text"))
    return out


async def _finnhub(query: str, ticker: Optional[str], max_results: int) -> list[Article]:
    """Optional — only runs when FINNHUB_API_KEY is set and the client is installed."""
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return []
    try:
        import finnhub  # type: ignore
        client = finnhub.Client(api_key=key)
        def _do():
            if ticker:
                today = datetime.now(timezone.utc).date()
                frm = (today - timedelta(days=3)).isoformat()
                return client.company_news(ticker.upper(), _from=frm, to=today.isoformat())
            return client.general_news("general")
        raw = await asyncio.to_thread(_do)
    except Exception as e:
        logger.warning("finnhub failed for %r: %s", query[:50], e)
        return []
    qwords = set(query.lower().split()) - _STOP
    out = []
    for r in (raw or [])[: max_results * 3]:
        headline = r.get("headline", "")
        text = f"{headline} {r.get('summary', '')}".lower()
        if qwords and not ticker and not any(w in text for w in qwords):
            continue  # relevance filter for general market news
        ts = r.get("datetime")
        pub = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        out.append(Article(title=headline, url=r.get("url", ""),
                           snippet=r.get("summary", ""), source="finnhub", published_at=pub))
        if len(out) >= max_results:
            break
    return out


# ── Aggregation ──────────────────────────────────────────────────────────────

def _dedup(articles: list[Article]) -> list[Article]:
    seen: list[set[str]] = []
    unique: list[Article] = []
    for a in articles:
        words = set(a.title.lower().split()) - _STOP
        if not words:
            unique.append(a)
            continue
        dup = False
        for s in seen:
            if s and len(words & s) / len(words | s) > 0.6:
                dup = True
                break
        if not dup:
            seen.append(words)
            unique.append(a)
    return unique


def _apply_freshness(articles: list[Article], max_age_minutes: Optional[int],
                     discard_undated: bool) -> list[Article]:
    """Optional freshness gate. Unlike trading's strict trading-grade gate, research
    defaults to KEEPING undated articles (discard_undated=False) — a good source
    without a parseable timestamp is still worth citing in a brief."""
    if not max_age_minutes:
        kept = list(articles)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
        kept = []
        for a in articles:
            if a.published_at is None:
                if not discard_undated:
                    kept.append(a)
            elif a.published_at >= cutoff:
                kept.append(a)
    kept.sort(key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
              reverse=True)
    return kept


async def _scrape_top(articles: list[Article], top_n: int, scraper_url: str,
                      max_chars: int) -> None:
    """Fill full_text for the top-N via the scraper's /scrape/batch (engine=auto)."""
    targets = [a for a in articles[:top_n] if a.url and not a.full_text]
    if not targets:
        return
    jobs = [{"url": a.url, "engine": "auto", "options": {"max_chars": max_chars}}
            for a in targets]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=8.0)) as client:
            resp = await client.post(f"{scraper_url.rstrip('/')}/scrape/batch",
                                     json={"jobs": jobs, "max_concurrency": min(len(jobs), 5)})
            if resp.status_code != 200:
                logger.warning("scrape/batch %s: %s", resp.status_code, resp.text[:200])
                return
            results = resp.json().get("results") or []
    except Exception as e:
        logger.warning("scrape/batch failed: %s", e)
        return
    by_url = {r.get("url"): r for r in results}
    got = 0
    for a in targets:
        r = by_url.get(a.url)
        if r and r.get("success") and r.get("content"):
            a.full_text = str(r["content"])[:max_chars]
            got += 1
    logger.info("grounded_research: scraped %d/%d top articles", got, len(targets))


def format_for_context(articles: list[Article], max_chars: int = 6000) -> str:
    """Numbered [1][2][3] citation block for prompt injection (mirrors web_search.py)."""
    if not articles:
        return ""
    parts = ["── WEB SEARCH RESULTS ──",
             "Cite these sources using [1], [2], [3] in your answer.\n"]
    total = 0
    now = datetime.now(timezone.utc)
    for i, a in enumerate(articles, 1):
        text = (a.full_text or a.snippet or "")[:2000]
        age = ""
        if a.published_at:
            d = (now - a.published_at).total_seconds()
            age = f" ({int(d/60)}min ago)" if d < 3600 else (
                f" ({int(d/3600)}h ago)" if d < 86400 else f" ({int(d/86400)}d ago)")
        block = f"[{i}] {a.title}{age}\n{a.url}\n{text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)


# ── Synthesis (one direct vLLM completion — no agent loop) ───────────────────

async def _discover_model(client: httpx.AsyncClient, vllm_url: str) -> Optional[str]:
    try:
        r = await client.get(f"{vllm_url}/v1/models", timeout=8.0)
        if r.status_code == 200:
            data = r.json().get("data") or []
            if data:
                return data[0]["id"]
    except Exception as e:
        logger.warning("model discovery failed: %s", e)
    return None


async def _synthesize(query: str, context: str, sources: list[dict],
                      schema: Optional[dict], domain: Optional[str],
                      vllm_url: str, model: Optional[str], timeout: float) -> Optional[dict]:
    contract = json.dumps(schema, indent=2) if schema else (
        '{"title": "<short headline>", "overview": "<one-sentence bottom line>", '
        '"answer": "<the findings in GitHub-flavored Markdown with [1][2] citations>"}')
    dom = f" You are researching {domain}." if domain else ""
    prompt = (
        f"You are a research analyst.{dom} Using ONLY the web search results below, "
        f'answer this request: "{query}"\n\n'
        "Return ONLY a JSON object (no prose, no code fence) with EXACTLY these keys "
        "(each value describes what to put there):\n" + contract + "\n\n"
        "Ground every claim in the sources and cite them inline with [1], [2], etc. "
        "Be concrete: name the actual tickers, companies, figures, and dates the "
        "sources give — never invent numbers not present. If the sources are thin, "
        "say so briefly rather than padding.\n\n" + context)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
            if not model:
                model = await _discover_model(client, vllm_url)
            if not model:
                return None
            resp = await client.post(f"{vllm_url}/v1/chat/completions", json={
                "model": model, "temperature": 0.3, "max_tokens": 1600,
                "messages": [{"role": "user", "content": prompt}]})
            text = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("grounded synthesis failed: %s", e)
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    data.setdefault("sources", sources)
    if not data.get("sources"):
        data["sources"] = sources
    return data


# ── Public entry point ───────────────────────────────────────────────────────

async def grounded_research(
    query: str,
    *,
    ticker: Optional[str] = None,
    domain: Optional[str] = None,
    schema: Optional[dict] = None,
    synthesize: bool = True,
    max_articles: int = 6,
    scrape_top_n: int = 3,
    max_age_minutes: Optional[int] = 2880,
    discard_undated: bool = False,
    scrape_max_chars: int = 4000,
    scraper_url: Optional[str] = None,
    vllm_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 60.0,
) -> Any:
    """Retrieval-augmented research. Fast + reliable (no agent loop).

    Returns (synthesize=True) the {title, overview, answer, sources} dict, or None
    if nothing was found. With synthesize=False returns
    {context, sources, articles} so the caller can run its own LLM pass.
    """
    scraper_url = scraper_url or DEFAULT_SCRAPER_URL
    vllm_url = vllm_url or DEFAULT_VLLM_URL

    # 1. fan out over sources concurrently
    layers = await asyncio.gather(
        _ddg_news(query, 8),
        _ddg_text(query, 8),
        _finnhub(query, ticker, 8),
        return_exceptions=True,
    )
    articles: list[Article] = []
    for r in layers:
        if isinstance(r, list):
            articles.extend(r)
        elif isinstance(r, Exception):
            logger.warning("grounded_research source error: %s", r)
    articles = [a for a in articles if a.title and a.url]
    if not articles:
        logger.warning("grounded_research: no articles for %r", query[:60])
        return None

    # 2. dedupe + freshness + cap
    articles = _dedup(articles)
    articles = _apply_freshness(articles, max_age_minutes, discard_undated)[:max_articles]

    # 3. scrape top-N full text
    await _scrape_top(articles, scrape_top_n, scraper_url, scrape_max_chars)

    # 4. build citation context + source list
    def _host(u: str) -> str:
        m = re.match(r"https?://(?:www\.)?([^/]+)", u or "")
        return m.group(1) if m else ""
    context = format_for_context(articles)
    sources = [{"title": a.title[:140], "url": a.url,
                # publisher = the site host (real publisher unknown); `source` keeps
                # the retrieval layer (ddg_news/finnhub) for debugging.
                "publisher": _host(a.url), "source": a.source,
                "published": a.published_at.isoformat() if a.published_at else ""}
               for a in articles]

    if not synthesize:
        return {"context": context, "sources": sources, "articles": articles}

    # 5. one direct vLLM synthesis pass
    brief = await _synthesize(query, context, sources, schema, domain,
                              vllm_url, model, timeout)
    if not brief:
        logger.warning("grounded_research: synthesis empty for %r", query[:60])
        return None
    logger.info("grounded_research(%r): %d articles -> brief with %d sources",
                query[:50], len(articles), len(brief.get("sources") or []))
    return brief
