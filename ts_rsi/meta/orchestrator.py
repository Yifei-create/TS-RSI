from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ts_rsi.adapters.materialize import materialize_adapter
from ts_rsi.evaluation.adapter_trial import CandidateTrial
from ts_rsi.evaluation.reports import build_review_report
from ts_rsi.gating.discounted_ridge import DiscountedRidgeGate
from ts_rsi.meta.agent import ApiMetaAgent, MetaAgent
from ts_rsi.meta.api_log import ApiCallLogger
from ts_rsi.meta.prompts import load_meta_source
from ts_rsi.revisions.manager import RevisionManager
from ts_rsi.revisions.records import RevisionLog
from ts_rsi.streaming.versioned import RunningVersion
from ts_rsi.types import (
    EvaluationSummary,
    ForecastRecord,
    RevisionProposal,
    RevisionRecord,
    ToolExecutionResult,
)

ADAPTER_FILES = ("config.json", "adapter.py")
META_FILES = ("system_prompt.md", "review_instructions.md")


@dataclass
class _AdapterTrialState:
    trial: CandidateTrial
    candidate: RunningVersion
    record: RevisionRecord
    candidate_root: Path
    side: str  # "single", "old", or "new"


@dataclass
class _MetaTrialState:
    record: RevisionRecord
    candidate_root: Path
    new_agent: ApiMetaAgent


class MetaReviewOrchestrator:
    """Drives MetaAgent reviews, adapter candidate trials, and meta self-revision trials.

    Reviews only happen while no trial is in progress. API failures degrade to WAIT so
    the forecasting loop is never blocked. REVISE_META is settled by the fixed
    old/new proposal comparison from the specification: both meta versions get one
    proposal on the same report, their adapter candidates are evaluated on the same
    subsequent data, and the new meta version is accepted iff U_new > U_old.
    """

    def __init__(
        self,
        *,
        meta_agent: ApiMetaAgent,
        meta_root: Path,
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
        self.meta_root = meta_root
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
        self._trials: list[_AdapterTrialState] = []
        self._meta_trial: _MetaTrialState | None = None

        self.review_count = 0
        self.trials_started = 0
        self.trials_accepted = 0
        self.trials_rejected = 0
        self.edit_failures = 0
        self.meta_trials = 0
        self.meta_accepted = 0

    def on_step(self, matured: tuple[ForecastRecord, ...]) -> None:
        if not matured:
            return
        if self._trials:
            self._collect_trial_pairs(matured)
            if self._trials and all(state.trial.ready() for state in self._trials):
                self._finalize_trials()
        official_records = [
            record for record in matured if record.adapter_version == self._official.version
        ]
        self._pending.extend(official_records)
        if not self._trials and len(self._pending) >= self.review_batch_records:
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
            "meta_trials": self.meta_trials,
            "meta_accepted": self.meta_accepted,
        }

    # ------------------------------------------------------------------ reviews

    def _run_review(self, records: list[ForecastRecord]) -> None:
        self.review_count += 1
        session_id = f"review-{self.review_count:04d}"
        self.meta_agent.begin_session()
        report = build_review_report(
            f"report-{self.review_count:04d}",
            records,
            self.scale,
            self._official.adapter.describe(),
            self.revision_log.recent(10),
        )
        report_dict = report.to_dict()
        report_dict["adapter_files"] = self._read_files(self.adapter_root, ADAPTER_FILES)
        report_dict["meta_files"] = self._read_files(self.meta_root, META_FILES)
        report_dict["editable_contract"] = (
            "adapter.py must keep build_adapter(version, config, dimension) returning an "
            "AdapterAgent whose predict() returns a finite H x D correction; config.json "
            "holds the numeric configuration; system_prompt.md and review_instructions.md "
            "define the MetaAgent itself and may only be edited via REVISE_META."
        )
        proposal = self._call_meta(
            self.meta_agent,
            session_id,
            "regular_review",
            lambda: self.meta_agent.review(report_dict, self.revision_log.recent(10)),
        )
        if proposal is None or proposal.action == "WAIT":
            return
        if proposal.action == "REVISE_ADAPTER":
            state = self._build_adapter_candidate(
                proposal, self.adapter_root, session_id, side="single"
            )
            if state is not None:
                self._trials = [state]
                self.trials_started += 1
        elif proposal.action == "REVISE_META":
            self._run_meta_trial(session_id, proposal, report_dict)

    # ------------------------------------------------------- adapter candidates

    def _build_adapter_candidate(
        self,
        proposal: RevisionProposal,
        source_root: Path,
        session_id: str,
        side: str,
        agent: ApiMetaAgent | None = None,
    ) -> _AdapterTrialState | None:
        """Copy the adapter surface, apply the proposal's edits, materialize, and
        register the candidate version. Returns None when editing fails for good."""
        agent = agent or self.meta_agent
        revision_id, editor, record, results = self.revision_manager.create_adapter_candidate(
            source_root=source_root,
            parent_adapter_version=self._official.version,
            parent_meta_version=agent.version,
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
            if repairs >= self.max_repairs_per_session:
                self.edit_failures += 1
                record.status = "edit_failed"
                record.accepted = False
                record.actual_effect = "; ".join(failure_messages)
                self.revision_log.append(record)
                self.revision_manager.discard_candidate(revision_id)
                return None
            repairs += 1
            feedback = json.dumps(
                {
                    "failed_tool_results": [asdict(item) for item in failed_tools],
                    "materialize_error": failure_messages if not failed_tools else None,
                    "candidate_files": self._read_files(candidate_root, ADAPTER_FILES),
                },
                ensure_ascii=False,
                indent=2,
            )
            followup = self._call_meta(
                agent, session_id, "followup", lambda: agent.repair(feedback)
            )
            if followup is None or followup.action not in {"REVISE_ADAPTER", "REVISE_META"} or not followup.tool_calls:
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
            meta_version=agent.version,
        )
        self.versions.append(candidate)
        state = _AdapterTrialState(
            trial=CandidateTrial(
                candidate_id=revision_id,
                baseline_version=self._official.version,
                candidate_version=candidate.version,
                warmup_records=self.warmup_records,
                evaluation_records=self.evaluation_records,
                records=[],
            ),
            candidate=candidate,
            record=record,
            candidate_root=candidate_root,
            side=side,
        )
        return state

    # ------------------------------------------------------------ meta revision

    def _run_meta_trial(
        self, session_id: str, proposal: RevisionProposal, report_dict: dict
    ) -> None:
        """Section 7 of the specification: old and new MetaAgent each get exactly one
        proposal on the same report; the resulting adapter candidates are evaluated on
        the same subsequent data."""
        revision_id, editor, record, results = self.revision_manager.create_adapter_candidate(
            source_root=self.meta_root,
            parent_adapter_version=self._official.version,
            parent_meta_version=self.meta_agent.version,
            proposal=proposal,
        )
        candidate_root = Path(str(record.metadata["candidate_root"]))
        failed = [result for result in results if not result.ok]
        if failed:
            feedback = json.dumps(
                {
                    "failed_tool_results": [asdict(item) for item in failed],
                    "candidate_files": self._read_files(candidate_root, META_FILES),
                },
                ensure_ascii=False,
                indent=2,
            )
            followup = self._call_meta(
                self.meta_agent, session_id, "followup", lambda: self.meta_agent.repair(feedback)
            )
            if followup is not None and followup.tool_calls:
                results = tuple(
                    editor.apply(str(tool_call["tool_name"]), dict(tool_call["tool_input"]))
                    for tool_call in followup.tool_calls
                )
                failed = [result for result in results if not result.ok]
        if failed:
            record.status = "edit_failed"
            record.accepted = False
            record.actual_effect = "; ".join(item.message for item in failed)
            self.revision_log.append(record)
            self.revision_manager.discard_candidate(revision_id)
            self.edit_failures += 1
            return

        try:
            new_prompt = load_meta_source(candidate_root)
            if not new_prompt.strip():
                raise ValueError("empty meta prompt")
            new_agent = ApiMetaAgent(self.meta_agent.client, new_prompt, self.meta_agent.max_calls)
        except Exception as exc:
            record.status = "edit_failed"
            record.accepted = False
            record.actual_effect = f"meta materialize: {type(exc).__name__}: {exc}"
            self.revision_log.append(record)
            self.revision_manager.discard_candidate(revision_id)
            self.edit_failures += 1
            return

        history = self.revision_log.recent(10)
        trial_states: list[_AdapterTrialState] = []
        for side, agent in (("old", self.meta_agent), ("new", new_agent)):
            agent.begin_session()
            side_proposal = self._call_meta(
                agent,
                session_id,
                "meta_trial_proposal",
                lambda agent=agent: agent.review(report_dict, history),
            )
            if side_proposal is None or side_proposal.action != "REVISE_ADAPTER":
                continue
            state = self._build_adapter_candidate(
                side_proposal, self.adapter_root, session_id, side=side, agent=agent
            )
            if state is not None:
                trial_states.append(state)

        self.meta_trials += 1
        if not trial_states:
            record.status = "meta_trial_vacuous"
            record.accepted = False
            record.actual_effect = "Neither old nor new MetaAgent produced a runnable candidate"
            self.revision_log.append(record)
            self.revision_manager.discard_candidate(revision_id)
            return

        record.status = "meta_trial_started"
        record.metadata["new_meta_version"] = new_agent.version
        self.revision_log.append(record)
        self._trials = trial_states
        self._meta_trial = _MetaTrialState(record=record, candidate_root=candidate_root, new_agent=new_agent)
        self.trials_started += len(trial_states)

    # ------------------------------------------------------------------ trials

    def _collect_trial_pairs(self, matured: tuple[ForecastRecord, ...]) -> None:
        by_version_origin: dict[tuple[str, int], ForecastRecord] = {}
        for record in matured:
            by_version_origin[(record.adapter_version, record.origin_index)] = record
        for state in self._trials:
            for record in matured:
                if record.adapter_version != state.trial.baseline_version:
                    continue
                candidate_record = by_version_origin.get(
                    (state.trial.candidate_version, record.origin_index)
                )
                if candidate_record is not None:
                    state.trial.add_pair(record, candidate_record)

    def _finalize_trials(self) -> None:
        summaries: list[tuple[_AdapterTrialState, EvaluationSummary]] = [
            (state, state.trial.summarize(self.scale)) for state in self._trials
        ]
        baseline_loss = summaries[0][1].baseline_loss

        # Adapter outcome: promote the best candidate when it beats the official loss.
        best_state, best_summary = min(summaries, key=lambda item: item[1].candidate_loss)
        promote = best_summary.candidate_loss < baseline_loss
        for state, summary in summaries:
            accepted = promote and state is best_state
            record = state.record
            record.actual_effect = json.dumps(asdict(summary), ensure_ascii=False)
            record.accepted = accepted
            record.status = "accepted" if accepted else "rejected"
            self.revision_log.append(record)
            if accepted:
                self.versions[:] = [v for v in self.versions if v is not self._official]
                self._official = state.candidate
                self.adapter_root = state.candidate_root
                self._pending = []
                self.trials_accepted += 1
            else:
                self.versions[:] = [v for v in self.versions if v is not state.candidate]
                self.revision_manager.discard_candidate(state.record.revision_id)
                self.trials_rejected += 1

        # Meta outcome: U = L0 - min(L0, L_candidate) per side; accept new iff strictly better.
        if self._meta_trial is not None:
            meta = self._meta_trial
            utilities: dict[str, float] = {"old": 0.0, "new": 0.0}
            for state, summary in summaries:
                utilities[state.side] = baseline_loss - min(baseline_loss, summary.candidate_loss)
            accept_new = utilities["new"] > utilities["old"]
            meta.record.actual_effect = json.dumps(
                {
                    "baseline_loss": baseline_loss,
                    "utility_old": utilities["old"],
                    "utility_new": utilities["new"],
                    "new_meta_version": meta.new_agent.version,
                },
                ensure_ascii=False,
            )
            meta.record.accepted = accept_new
            meta.record.status = "accepted" if accept_new else "rejected"
            self.revision_log.append(meta.record)
            if accept_new:
                self.meta_agent = meta.new_agent
                self.meta_root = meta.candidate_root
                self.meta_accepted += 1
            else:
                self.revision_manager.discard_candidate(meta.record.revision_id)
            self._meta_trial = None

        self._trials = []

    # ------------------------------------------------------------------ helpers

    def _call_meta(
        self,
        agent: ApiMetaAgent,
        session_id: str,
        category: str,
        invoke: Callable[[], RevisionProposal],
    ) -> RevisionProposal | None:
        start = time.perf_counter()
        response = None
        error: str | None = None
        try:
            proposal = invoke()
            response = agent.last_response
            return proposal
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self.api_logger.log(
                session_id=session_id,
                category=category,
                model=str(getattr(agent.client, "model", "unknown")),
                response=response,
                latency_seconds=time.perf_counter() - start,
                error=error,
            )

    @staticmethod
    def _read_files(root: Path, names: tuple[str, ...]) -> dict[str, str]:
        files: dict[str, str] = {}
        for name in names:
            path = root / name
            if path.exists():
                files[name] = path.read_text(encoding="utf-8")
        return files
