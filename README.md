# lazycat-sdk

## grounded_research: where results come from (2026-10-06)

`lazycat.grounded_research` gathers sources from three places:

- **News:** lazy-agent-service `news_search` (keyed news APIs), `POST {LAZY_TOOL_SERVICE_URL}/execute/news_search`.
- **Web:** lazy-agent-service `web_search` (Exa's keyless index), `POST {LAZY_TOOL_SERVICE_URL}/execute/web_search`. It has one cache and one rate limit for every project on the network.
- **Finnhub,** when a ticker is given.

`LAZY_TOOL_SERVICE_URL` defaults to `http://10.0.0.16:5591`.

It used to call the `ddgs` library for news and web results. `ddgs` scrapes DuckDuckGo, and on a blocked IP tries about nine engines for every call. The free search engines bot-block this network's one public IP (lazy-agent-service `documentation/chapters/04-shared-web-search.md`), so every one of those calls made the block worse for every project. `ddgs` is no longer a dependency.

A search that is refused (`rate_limited`, `busy`) or cannot be reached contributes no articles; it never raises. `tests/test_grounded_research_sources.py` covers:

- both request URLs and bodies;
- the mapping to `Article`, including RFC 2822 news dates;
- refused and failed searches;
- a check on the module's imports that it does not import `ddgs`.
