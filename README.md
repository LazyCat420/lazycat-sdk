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


## `/execute` API key (2026-10-11)

lazy-tool-service now enforces `EXECUTE_API_KEY` on `POST /execute/*`. This repo sends it as
`x-api-key` when the env var is set (it comes from the deployed `.env`); without it the shared
search answers 401. Rollout record, live checks and how to rotate the key:
lazy-agent-service `documentation/chapters/11-security-hardening-plan-2026-10-11.md` (Item 3).
