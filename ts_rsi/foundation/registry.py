from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationModelSpec:
    name: str
    checkpoint: str
    package: str
    source: str
    multivariate: bool


FOUNDATION_MODELS: dict[str, FoundationModelSpec] = {
    "chronos-2": FoundationModelSpec(
        name="chronos-2",
        checkpoint="amazon/chronos-2",
        package="chronos-forecasting>=2.1",
        source="https://github.com/amazon-science/chronos-forecasting",
        multivariate=True,
    ),
    "moirai-2": FoundationModelSpec(
        name="moirai-2",
        checkpoint="Salesforce/moirai-2.0-R-small",
        package="uni2ts",
        source="https://github.com/SalesforceAIResearch/uni2ts",
        multivariate=True,
    ),
    "timesfm-2.5": FoundationModelSpec(
        name="timesfm-2.5",
        checkpoint="google/timesfm-2.5-200m-pytorch",
        package="timesfm",
        source="https://github.com/google-research/timesfm",
        multivariate=False,
    ),
}


def resolve_foundation_model(name: str) -> FoundationModelSpec:
    key = name.strip().lower()
    if key not in FOUNDATION_MODELS:
        raise ValueError(f"Unsupported frozen foundation model {name!r}: {sorted(FOUNDATION_MODELS)}")
    return FOUNDATION_MODELS[key]
