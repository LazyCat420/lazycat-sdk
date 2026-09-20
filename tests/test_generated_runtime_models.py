import json
from pathlib import Path

from lazycat.generated_runtime_models import RunEvent, RunResult
from lazycat.contract import WIRE_CONTRACT_SHA256, WIRE_CONTRACT_VERSION


def test_generated_models_validate_canonical_wire_fixtures():
    document = json.loads((Path(__file__).parents[1] / "contracts" / "runtime-wire-v1.json").read_text())
    assert document["contract_version"] == "runtime-wire.v1.0.0"
    RunEvent.model_validate({
        "id": "evt-1", "run_id": "run-1", "type": "run.started",
        "timestamp": "2026-09-19T10:00:00Z", "data": {"status": "running"},
    })
    result = RunResult.model_validate({
        "run_id": "run-1", "status": "waiting_for_approval", "messages": [],
    })
    assert result.status.value == "waiting_for_approval"


def test_generated_contract_identity_matches_bundled_metadata():
    document = json.loads((Path(__file__).parents[1] / "contracts" / "runtime-wire-v1.json").read_text())
    assert WIRE_CONTRACT_VERSION == document["contract_version"]
    assert WIRE_CONTRACT_SHA256 == document["digest"]
