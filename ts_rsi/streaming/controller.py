from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ts_rsi.evaluation.reports import ReviewReport, build_review_report
from ts_rsi.meta.agent import MetaAgent
from ts_rsi.revisions.records import RevisionLog
from ts_rsi.types import AdapterDescription, RevisionProposal, RevisionRecord


class ReviewController:
    """Schedules review opportunities; it never changes the fixed stream protocol."""

    def __init__(
        self,
        meta_agent: MetaAgent,
        revision_log: RevisionLog,
        review_batch_records: int,
    ) -> None:
        if review_batch_records <= 0:
            raise ValueError("review_batch_records must be positive")
        self.meta_agent = meta_agent
        self.revision_log = revision_log
        self.review_batch_records = review_batch_records
        self._since_review = 0
        self._report_index = 0
        self._trial_active = False

    def observe_matured(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._since_review += count

    def review_due(self) -> bool:
        return self._since_review >= self.review_batch_records and not self._trial_active

    def build_and_review(
        self,
        records: list[Any],
        scale: Any,
        adapter: AdapterDescription,
    ) -> tuple[ReviewReport, RevisionProposal]:
        if not self.review_due():
            raise RuntimeError("Review is not due or a candidate trial is active")
        self._report_index += 1
        report = build_review_report(
            report_id=f"report-{self._report_index:06d}",
            records=records,
            scale=scale,
            adapter=adapter,
            revisions=self.revision_log.records,
        )
        proposal = self.meta_agent.review(report.to_dict(), self.revision_log.recent())
        self._since_review = 0
        return report, proposal

    def register_proposal(
        self,
        proposal: RevisionProposal,
        adapter_version: str,
    ) -> RevisionRecord:
        record = RevisionRecord(
            revision_id=f"proposal-{self._report_index:06d}",
            parent_adapter_version=adapter_version,
            parent_meta_version=self.meta_agent.version,
            action=proposal.action,
            observed_problem="; ".join(proposal.evidence),
            diagnosis=proposal.diagnosis,
            proposed_change=proposal.change,
            expected_effect=proposal.expected_effect,
            status="waiting" if proposal.action == "WAIT" else "candidate_pending",
        )
        self.revision_log.append(record)
        return record

    def set_trial_active(self, active: bool) -> None:
        self._trial_active = active

    def state(self) -> dict[str, Any]:
        return {
            "meta_version": self.meta_agent.version,
            "since_review": self._since_review,
            "trial_active": self._trial_active,
            "revision_count": len(self.revision_log.records),
        }
