from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ts_rsi.adapters.materialize import materialize_adapter
from ts_rsi.evaluation.adapter_trial import CandidateTrial
from ts_rsi.evaluation.reports import build_review_report
from ts_rsi.gating.discounted_ridge import DiscountedRidgeGate
from ts_rsi.meta.agent import MetaAgent
from ts_rsi.meta.api_log import ApiCallLogger
from ts_rsi.revisions.manager import RevisionManager
from ts_rsi.revisions.records import RevisionLog
from ts_rsi.streaming.versioned import RunningVersion
from ts_rsi.types import ForecastRecord, RevisionProposal, RevisionRecord, ToolExecutionResult


class MetaReviewOrchestrator:
    """Drives periodic MetaAgent reviews, candidate trials, and acceptance decisions.

    The stream runner calls on_step() with each batch of matured records. Reviews only
    happen when no candidate trial is in progress; API failures degrade to WAIT so the
    forecasting loop is never blocked.
    """

    def __init__(
        self,
        *,
        meta_agent: MetaAgent,
        revision_manager: RevisionManager,
        revision_log: RevisionLog,
        api_logger: ApiCallLogger,
        versions: list[RunningVersion],
        adapter_root: Path,
        gate_factory: Callable[[], DiscountedRidgeGate],
        scale: np.ndarray,
        dimension: int,
        horizon: int,
        review_batch_records: int,
        warmup_records: int,
        evaluation_records: int,
        max_repairs_per_session: int = 1,
    ) -> None:
        if not versions:
            raise ValueError("At least the official running version is required")
        self.meta_agent = meta_agent
        self.revision_manager = revision_manager
        self.revision_log = revision_log
        self.api_logger = api_logger
        self.versions = versions
        self.adapter_root = adapter_root
        self.gate_factory = gate_factory
        self.scale = scale
        self.dimension = dimension
        self.horizon = horizon
        self.review_batch_records = review_batch_records
        self.warmup_records = warmup_records
        self.evaluation_records = evaluation_records
        self.max_repairs_per_session = max_repairs_per_session

        self._official = versions[0]
        self._pending: list[ForecastRecord] = []
        self._trial: CandidateTrial | None = None
        self._trial_candidate: RunningVersion | None = None
        self._trial_record: RevisionRecord | None = None
        self._trial_root: Path | None = None

        self.review_count = 0
        self.trials_started = 0
        self.trials_accepted = 0
        self.trials_rejected = 0
        self.edit_failures = 0
        self.deferred_meta_revisions = 0

    def on_step(self, matured: tuple[ForecastRecord, ...]) -> None:
        if not matured:
            return
        if self._trial is not None:
            self._collect_trial_pairs(matured)
            if self._trial is not None and self._trial.ready():
                self._finalize_trial()
        official_records = [
            record for record in matured if record.adapter_version == self._official.version
        ]
        self._pending.extend(official_records)
        if self._trial is None and len(self._pending) >= self.review_batch_records:
            records = self._pending[-self.review_batch_records :]
            self._pending = []
            self._run_review(records)

    def summary(self) -> dict[str, int]:
        return {
            "reviews": self.review_count,
            "trials_started": self.trials_started,
            "trials_accepted": self.trials_accepted,
            "trials_rejected": self.trials_rejected,
            "edit_failures": self.edit_failures,
            "deferred_meta_revisions": self.deferred_meta_revisions,
        }

    def _run_review(self, records: list[ForecastRecord]) -> None:
        self.review_count += 1
        session_id = f"review-{self.review_count:04d}"
        begin = getattr(self.meta_agent, "begin_session", None)
        if callable(begin):
            begin()
        report = build_review_report(
            f"report-{self.review_count:04d}",
            records,
            self.scale,
            self._official.adapter.describe(),
            self.revision_log.recent(10),
        )
        report_dict = report.to_dict()
        report_dict["adapter_files"] = self._read_files(self.adapter_root)
        report_dict["editable_contract"] = (
            "adapter.py must keep build_adapter(version, config, dimension) returning an "
            "AdapterAgent whose predict() returns a finite H x D correction; config.json "
            "holds the numeric configuration."
        )
        proposal = self._call_meta(
            session_id,
            "regular_review",
            lambda: self.meta_agent.review(report_dict, self.revision_log.recent(10)),
        )
        if proposal is None:
            return
        if proposal.action == "REVISE_ADAPTER":
            self._start_adapter_trial(session_id, proposal)
        elif proposal.action == "REVISE_META":
            self.deferred_meta_revisions += 1
            self.revision_log.append(
                RevisionRecord(
                    revision_id=f"meta-{session_id}",
                    parent_adapter_version=self._official.version,
                    parent_meta_version=self.meta_agent.version,
                    action=proposal.action,
                    observed_problem="; ".join(proposal.evidence),
                    diagnosis=proposal.diagnosis,
                    proposed_change=proposal.change,
                    expected_effect=proposal.expected_effect,
                    status="deferred",
                    metadata={"reason": "REVISE_META execution is not wired in this minimal loop"},
                )
            )

    def _start_adapter_trial(self, session_id: str, proposal: RevisionProposal) -> None:
        revision_id, editor, record, results = self.revision_manager.create_adapter_candidate(
            source_root=self.adapter_root,
            parent_adapter_version=self._official.version,
            parent_meta_version=self.meta_agent.version,
            proposal=proposal,
        )
        candidate_root = Path(str(record.metadata["candidate_root"]))
        candidate_adapter = None
        failure_messages: list[str] = []
        repairs = 0
        while True:
            failed_tools = [result for result in results if not result.ok]
            failure_messages = [result.message for result in failed_tools]
            if not failed_tools:
                try:
                    candidate_adapter = materialize_adapter(
                        candidate_root,
                        version=f"adapter-cand-{revision_id[-8:]}",
                        dimension=self.dimension,
                        smoke_horizon=min(self.horizon, 8),
                    )
                    break
                except Exception as exc:
                    failure_messages = [f"materialize: {type(exc).__name__}: {exc}"]
            if repairs >= self.max_repairs_per_session or not hasattr(self.meta_agent, "repair"):
                self.edit_failures += 1
                record.status = "edit_failed"
                record.accepted = False
                record.actual_effect = "; ".join(failure_messages)
                self.revision_log.append(record)
                self.revision_manager.discard_candidate(revision_id)
                return
            repairs += 1
            feedback = json.dumps(
                {
                    "failed_tool_results": [asdict(item) for item in failed_tools],
                    "materialize_error": failure_messages if not failed_tools else None,
                    "candidate_files": self._read_files(candidate_root),
                },
                ensure_ascii=False,
                indent=2,
            )
            followup = self._call_meta(
                session_id, "followup", lambda: self.meta_agent.repair(feedback)  # type: ignore[union-attr]
            )
            if followup is None or followup.action != "REVISE_ADAPTER" or not followup.tool_calls:
                results = (ToolExecutionResult("repair", False, "MetaAgent abandoned the repair"),)
                continue
            results = tuple(
                editor.apply(str(tool_call["tool_name"]), dict(tool_call["tool_input"]))
                for tool_call in followup.tool_calls
            )

        record.status = "trial_started"
        self.revision_log.append(record)
        candidate = RunningVersion(
            adapter=candidate_adapter,
            gate=self.gate_factory(),
            meta_version=self.meta_agent.version,
        )
        self.versions.append(candidate)
        self._trial = CandidateTrial(
            candidate_id=revision_id,
            baseline_version=self._official.version,
            candidate_version=candidate.version,
            warmup_records=self.warmup_records,
            evaluation_records=self.evaluation_records,
            records=[],
        )
        self._trial_candidate = candidate
        self._trial_record = record
        self._trial_root = candidate_root
        self.trials_started += 1

    def _collect_trial_pairs(self, matured: tuple[ForecastRecord, ...]) -> None:
        assert self._trial is not None
        baseline = {
            record.origin_index: record
            for record in matured
            if record.adapter_version == self._trial.baseline_version
        }
        candidate = {
            record.origin_index: record
            for record in matured
            if record.adapter_version == self._trial.candidate_version
        }
        for origin, base_record in baseline.items():
            candidate_record = candidate.get(origin)
            if candidate_record is not None:
                self._trial.add_pair(base_record, candidate_record)

    def _finalize_trial(self) -> None:
        assert self._trial is not None and self._trial_candidate is not None
        assert self._trial_record is not None and self._trial_root is not None
        summary = self._trial.summarize(self.scale)
        record = self._trial_record
        record.actual_effect = json.dumps(asdict(summary), ensure_ascii=False)
        record.accepted = summary.accepted
        record.status = "accepted" if summary.accepted else "rejected"
        self.revision_log.append(record)
        if summary.accepted:
            self.versions[:] = [v for v in self.versions if v is not self._official]
            self._official = self._trial_candidate
            self.adapter_root = self._trial_root
            self._pending = []
            self.trials_accepted += 1
        else:
            self.versions[:] = [v for v in self.versions if v is not self._trial_candidate]
            self.revision_manager.discard_candidate(record.revision_id)
            self.trials_rejected += 1
        self._trial = None
        self._trial_candidate = None
        self._trial_record = None
        self._trial_root = None

    def _call_meta(self, session_id: str, category: str, invoke: Callable[[], RevisionProposal]) -> RevisionProposal | None:
        is_api = hasattr(self.meta_agent, "client")
        start = time.perf_counter()
        response = None
        error: str | None = None
        try:
            proposal = invoke()
            response = getattr(self.meta_agent, "last_response", None)
            return proposal
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            if is_api:
                self.api_logger.log(
                    session_id=session_id,
                    category=category,
                    model=str(getattr(getattr(self.meta_agent, "client", None), "model", "unknown")),
                    response=response,
                    latency_seconds=time.perf_counter() - start,
                    error=error,
                )

    @staticmethod
    def _read_files(root: Path) -> dict[str, str]:
        files: dict[str, str] = {}
        for name in ("config.json", "adapter.py"):
            path = root / name
            if path.exists():
                files[name] = path.read_text(encoding="utf-8")
        return files
