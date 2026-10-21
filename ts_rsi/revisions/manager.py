from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from ts_rsi.meta.tools import CandidateEditor
from ts_rsi.types import RevisionProposal, RevisionRecord, ToolExecutionResult


class RevisionManager:
    """Creates isolated candidate copies and records every proposed edit."""

    def __init__(self, workspace: Path, revision_log: object) -> None:
        self.workspace = workspace
        self.candidates_root = workspace / "candidates"
        self.candidates_root.mkdir(parents=True, exist_ok=True)
        self.revision_log = revision_log

    def create_adapter_candidate(
        self,
        source_root: Path,
        parent_adapter_version: str,
        parent_meta_version: str,
        proposal: RevisionProposal,
    ) -> tuple[str, CandidateEditor, RevisionRecord, tuple[ToolExecutionResult, ...]]:
        revision_id = f"revision-{uuid4().hex[:12]}"
        candidate_root = self.candidates_root / revision_id
        candidate_root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source_root, candidate_root, dirs_exist_ok=True)
        editor = CandidateEditor(candidate_root)
        results = tuple(
            editor.apply(
                tool_name=str(tool_call["tool_name"]),
                tool_input=dict(tool_call["tool_input"]),
            )
            for tool_call in proposal.tool_calls
        )
        record = RevisionRecord(
            revision_id=revision_id,
            parent_adapter_version=parent_adapter_version,
            parent_meta_version=parent_meta_version,
            action=proposal.action,
            observed_problem="; ".join(proposal.evidence),
            diagnosis=proposal.diagnosis,
            proposed_change=proposal.change,
            expected_effect=proposal.expected_effect,
            status="candidate_created",
            metadata={"candidate_root": str(candidate_root), "tool_results": [asdict(item) for item in results]},
        )
        self.revision_log.append(record)
        return revision_id, editor, record, results

    def save_candidate_manifest(self, revision_id: str, manifest: dict[str, object]) -> Path:
        path = self.candidates_root / revision_id / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def discard_candidate(self, revision_id: str) -> None:
        candidate = self.candidates_root / revision_id
        if candidate.exists():
            shutil.rmtree(candidate)
