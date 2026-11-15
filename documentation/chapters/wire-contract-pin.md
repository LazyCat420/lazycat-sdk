---
part: Features
status: shipped
updated: 2026-11-14
---

# The pinned wire contract: resync it whenever lazy-agent-service regenerates

`RuntimeClient(verify_contract=True)` fetches `GET /v1/contracts/wire-schema` once and refuses to run when the served `digest` differs from `WIRE_CONTRACT_SHA256` in `lazycat/contract.py`. That file is generated from `contracts/runtime-wire-v1.json` by `scripts/generate_runtime_models.py`.

**Status 2026-11-14.**
- The pin is `sha256-7eda8541…`, the same digest the NAS lazy-agent-service serves (resynced in `5572734`; previous: `24a772dc…` from the 2026-10-24 resync).
- Trading-client's AI Chat (through trading-service's `sdk_adapter.py`, which sets `verify_contract=True`) failed every turn with "Response incomplete: Runtime wire contract digest mismatch" until this resync.
- Deployed 2026-10-24 18:12-18:15 PDT: the SDK folder was copied to `/volume1/docker/lazycat-sdk`, then `trading-client` and `trading-service` were restarted (`deploy.sh --restart-only`, no image rebuild; both containers bind-mount that folder read-only). A chat turn through `/api/v1/chat/stream` then returned `run.admitted`, tokens and `run.completed`.
- Any other container that mounts the SDK and sets `verify_contract=True` stays on the old module until its next restart.

**Root cause.** lazy-agent-service commit `32320f8` (2026-10-19) regenerated `contracts/generated/runtime-wire-v1.json` without the decision schemas (digest `68d61908…` to `24a772dc…`). The SDK still pinned `68d61908…` from its 2026-10-16 resync (`f82ad1c`). The service was redeployed with the new schema, so the check failed on every run.

**How to resync.** Copy the service's `contracts/generated/runtime-wire-v1.json` over `contracts/runtime-wire-v1.json`, run `scripts/generate_runtime_models.py` (needs `datamodel-codegen`), and commit all three generated files. The decision models in `lazycat/models.py` are hand-written and unaffected; only the generated `DecisionRequest`/`DecisionResult` models disappear.

**Resolved tickets.**
- AI Chat "Runtime wire contract digest mismatch": SDK pin `68d61908…` vs served `24a772dc…`; fixed by resyncing the schema in `29e08b5` (on `main`).
- **Recurrence 2026-11-14** (open risk below, exactly as predicted): lazy-agent-service `ab4c953` (wave4c route pruning) regenerated the schema — digest `24a772dc…` to `7eda8541…` (two datetime patterns now require seconds) — and redeployed at 18:52 PST; every AI Chat turn failed with `STREAM_FAILED` / "Runtime wire contract digest mismatch". Fixed by resyncing in SDK `5572734` (`main`), tar-synced to `/volume1/docker/lazycat-sdk`, then `deploy.sh --restart-only` on trading-client and trading-service. Smoke: chat turn returned `run.admitted`, tokens, `run.completed` in 6.8s.

**Open risk.** Nothing links the two repos: the next time lazy-agent-service regenerates `runtime-wire-v1.json` and redeploys, every `verify_contract=True` client fails the same way until the SDK is resynced and the containers restarted. Resync the SDK in the same landing batch as any schema regeneration.
