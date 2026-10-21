from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM_PROMPT = """You are the MetaAgent for a causal streaming time-series adapter.
Choose exactly one action: WAIT, REVISE_ADAPTER, or REVISE_META.
Use only evidence present in the report and revision history. A proposal must identify a
concrete editable object and an expected observable effect. Return a JSON object with keys:
action, evidence, diagnosis, change, expected_effect, tool_calls.
The evaluator, data-access timing, loss, and gate formula are fixed and cannot be edited.
tool_calls is a JSON list; each item has tool_name and tool_input.
Available editor tools:
- str_replace: tool_input {path, old_str, new_str}; old_str must match exactly once.
- write_file: tool_input {path, content}; replaces the whole file.
- set_config: tool_input {path, key, value}; sets one key in a JSON config file,
  or rewrites one "key: value" text line for non-JSON files.
Return WAIT with an empty tool_calls list when there is no concrete new evidence for a change.
"""

DEFAULT_REVIEW_INSTRUCTIONS = """Editable surfaces (paths are relative to an isolated candidate copy, never the
running version):

For REVISE_ADAPTER (files under the adapter surface, shown in report.adapter_files):
- config.json: JSON object with the adapter's numeric configuration. Prefer set_config
  for single-key numeric changes.
- adapter.py: the adapter implementation. It must keep a top-level factory
  build_adapter(version, config, dimension) returning an AdapterAgent whose predict()
  returns a finite H x D correction.

For REVISE_META (files under your own instruction surface, shown in report.meta_files):
- system_prompt.md: your role and output contract.
- review_instructions.md: your diagnostic and proposal procedure.
Revise these only when past revision outcomes show a defect in how you diagnose,
propose, or use evidence. The output JSON schema and the tool contract must remain valid.

Decision principles (initial working procedure, itself revisable via REVISE_META):
- WAIT: no concrete new evidence, or more feedback is needed.
- REVISE_ADAPTER: a specific forecast problem with a candidate change targeting it.
- REVISE_META: past revision outcomes reveal a defect in your own diagnosis,
  proposal, or evidence-use procedure.
"""


def compose_system_prompt(system_md: str, instructions_md: str) -> str:
    return system_md.rstrip() + "\n\n" + instructions_md.rstrip() + "\n"


def seed_meta_source(root: Path) -> None:
    """Create the editable MetaAgent surface: system_prompt.md + review_instructions.md."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "system_prompt.md").write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
    (root / "review_instructions.md").write_text(DEFAULT_REVIEW_INSTRUCTIONS, encoding="utf-8")


def load_meta_source(root: Path) -> str:
    system_md = (root / "system_prompt.md").read_text(encoding="utf-8")
    instructions_md = (root / "review_instructions.md").read_text(encoding="utf-8")
    return compose_system_prompt(system_md, instructions_md)


def build_review_prompt(report: dict[str, Any], history: list[dict[str, Any]]) -> str:
    return json.dumps({"report": report, "revision_history": history}, ensure_ascii=False, indent=2)


def build_repair_prompt(feedback: str) -> str:
    return (
        "The previous proposal could not be applied by the local executor. "
        "Feedback from the executor:\n"
        f"{feedback}\n"
        "Return a corrected complete proposal as one JSON object with the same schema "
        "(action, evidence, diagnosis, change, expected_effect, tool_calls). "
        'Return {"action": "WAIT", "tool_calls": []} to abandon the change.'
    )
