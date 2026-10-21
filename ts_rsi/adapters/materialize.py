from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ts_rsi.adapters.base import AdapterAgent
from ts_rsi.types import ForecastContext

CANONICAL_ADAPTER_SOURCE = Path(__file__).resolve().parent / "mean_residual.py"

_BUILD_ADAPTER_SNIPPET = '''


def build_adapter(version, config, dimension):
    """Fixed construction contract used by the candidate trial harness."""
    return MeanResidualAdapter(
        version=version,
        history_window=int(config["history_window"]),
        cache_size=int(config["cache_size"]),
        dimension=dimension,
    )
'''


def seed_adapter_source(root: Path, configuration: dict[str, Any]) -> None:
    """Create the editable adapter surface: config.json plus adapter.py."""
    root.mkdir(parents=True, exist_ok=True)
    source = CANONICAL_ADAPTER_SOURCE.read_text(encoding="utf-8")
    (root / "adapter.py").write_text(source + _BUILD_ADAPTER_SNIPPET, encoding="utf-8")
    (root / "config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _load_candidate_module(adapter_path: Path) -> ModuleType:
    digest = hashlib.sha1(str(adapter_path).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(f"ts_rsi_candidate_{digest}", adapter_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load candidate module from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_adapter(
    candidate_root: Path,
    version: str,
    dimension: int,
    smoke_horizon: int = 4,
) -> AdapterAgent:
    """Instantiate an adapter from an (edited) adapter directory and run smoke checks."""
    config_path = candidate_root / "config.json"
    adapter_path = candidate_root / "adapter.py"
    if not config_path.exists() or not adapter_path.exists():
        raise ValueError("Candidate directory must contain config.json and adapter.py")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    module = _load_candidate_module(adapter_path)
    build = getattr(module, "build_adapter", None)
    if not callable(build):
        raise ValueError("adapter.py must define build_adapter(version, config, dimension)")
    adapter = build(version=version, config=config, dimension=dimension)
    if not isinstance(adapter, AdapterAgent):
        raise ValueError("build_adapter must return an AdapterAgent instance")

    context = ForecastContext(
        origin_index=7,
        origin_time=datetime(2000, 1, 1),
        values=np.zeros((8, dimension), dtype=np.float64),
        timestamps=tuple(datetime(2000, 1, 1) for _ in range(8)),
    )
    correction = np.asarray(
        adapter.predict(context, np.zeros((smoke_horizon, dimension), dtype=np.float64)),
        dtype=np.float64,
    )
    if correction.shape != (smoke_horizon, dimension):
        raise ValueError(
            f"Candidate predict returned {correction.shape}; expected {(smoke_horizon, dimension)}"
        )
    if not np.isfinite(correction).all():
        raise ValueError("Candidate predict returned NaN or infinite values")
    return adapter
