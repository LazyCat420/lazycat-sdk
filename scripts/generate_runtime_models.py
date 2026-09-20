#!/usr/bin/env python3
"""Reproducibly generate Pydantic wire models from the runtime JSON Schema."""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
schema = ROOT / "contracts" / "runtime-wire-v1.json"
output = ROOT / "lazycat" / "generated_runtime_models.py"
generator = shutil.which("datamodel-codegen")
if generator is None:
    raise SystemExit("datamodel-codegen is required; install the SDK dev dependencies first")

output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", suffix=".json") as source:
    json.dump(json.loads(schema.read_text())["schema"], source)
    source.flush()
    subprocess.run([
        generator,
        "--input", source.name,
        "--input-file-type", "jsonschema",
        "--output", str(output),
        "--output-model-type", "pydantic_v2.BaseModel",
        "--use-standard-collections",
        "--use-schema-description",
        "--disable-timestamp",
    ], check=True)
    generated = output.read_text()
    generated = re.sub(r"#   filename:.*", "#   filename: runtime-wire-v1.json", generated, count=1)
    output.write_text(generated)
print(f"generated {output.relative_to(ROOT)} from {schema.relative_to(ROOT)}")
