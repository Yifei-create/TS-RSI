from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaProposalUtility:
    baseline_loss: float
    candidate_loss: float

    @property
    def utility(self) -> float:
        return self.baseline_loss - min(self.baseline_loss, self.candidate_loss)


@dataclass(frozen=True)
class MetaTrialSummary:
    old: MetaProposalUtility
    new: MetaProposalUtility
    accept_new_meta: bool


def compare_meta_proposals(
    baseline_loss: float,
    old_candidate_loss: float,
    new_candidate_loss: float,
) -> MetaTrialSummary:
    old = MetaProposalUtility(baseline_loss, old_candidate_loss)
    new = MetaProposalUtility(baseline_loss, new_candidate_loss)
    return MetaTrialSummary(old=old, new=new, accept_new_meta=new.utility > old.utility)
