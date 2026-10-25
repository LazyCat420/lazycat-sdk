---
part: Features
status: shipped
updated: 2026-10-24
---

# The pinned wire contract: resync it whenever lazy-agent-service regenerates

`RuntimeClient(verify_contract=True)` fetches `GET /v1/contracts/wire-schema` once and refuses to run when the served `digest` differs from `WIRE_CONTRACT_SHA256` in `lazycat/contract.py`. That file is generated from `contracts/runtime-wire-v1.json` by `scripts/generate_runtime_models.py`.

**Status 2026-10-24.**
- The pin is `sha256-24a772dc…`, the same digest the NAS lazy-agent-service serves.
- Trading-client's AI Chat (through trading-service's `sdk_adapter.py`, which sets `verify_contract=True`) failed every turn with "Response incomplete: Runtime wire contract digest mismatch" until this resync.

**Root cause.** lazy-agent-service commit `32320f8` (2026-10-19) regenerated `contracts/generated/runtime-wire-v1.json` without the decision schemas (digest `68d61908…` to `24a772dc…`). The SDK still pinned `68d61908…` from its 2026-10-16 resync (`f82ad1c`). The service was redeployed with the new schema, so the check failed on every run.

**How to resync.** Copy the service's `contracts/generated/runtime-wire-v1.json` over `contracts/runtime-wire-v1.json`, run `scripts/generate_runtime_models.py` (needs `datamodel-codegen`), and commit all three generated files. The decision models in `lazycat/models.py` are hand-written and unaffected; only the generated `DecisionRequest`/`DecisionResult` models disappear.

**Resolved tickets.**
- AI Chat "Runtime wire contract digest mismatch": SDK pin `68d61908…` vs served `24a772dc…`; fixed by resyncing the schema (commit recorded below at landing).
