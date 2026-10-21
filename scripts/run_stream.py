from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ts_rsi.adapters.materialize import materialize_adapter, seed_adapter_source
from ts_rsi.config import load_config
from ts_rsi.data.loader import fit_standardization, load_csv
from ts_rsi.evaluation.metrics import loss_pair
from ts_rsi.foundation.base import CallableForecaster
from ts_rsi.foundation.frozen import load_frozen_foundation_model
from ts_rsi.gating.discounted_ridge import DiscountedRidgeGate
from ts_rsi.meta.agent import ApiMetaAgent, MetaAgent, WaitingMetaAgent
from ts_rsi.meta.api_client import OpenAICompatibleClient
from ts_rsi.meta.api_log import ApiCallLogger, ApiPrices
from ts_rsi.meta.orchestrator import MetaReviewOrchestrator
from ts_rsi.revisions.manager import RevisionManager
from ts_rsi.revisions.records import RevisionLog
from ts_rsi.streaming.buffer import CompletedRecordBuffer
from ts_rsi.streaming.pending import PendingForecasts
from ts_rsi.streaming.runner import CausalStreamRunner
from ts_rsi.streaming.versioned import RunningVersion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causal stream with an optional API-backed MetaAgent."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--foundation",
        choices=["frozen", "naive"],
        default="frozen",
        help="naive uses a repeat-last-value stand-in; no checkpoint download needed.",
    )
    parser.add_argument(
        "--max-origins",
        type=int,
        default=None,
        help="Truncate the stream to N forecast origins for development runs.",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Force WaitingMetaAgent (fixed adapter, zero API calls) as the control arm.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override foundation device, e.g. cuda:0; defaults to config value.",
    )
    parser.add_argument("--workspace", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Run tag used for the results file name; defaults to <foundation>-<dataset>.",
    )
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_meta_agent(config, force_waiting: bool = False) -> MetaAgent:
    api = config.api
    if not force_waiting and api.model and os.environ.get(api.api_key_env, ""):
        client = OpenAICompatibleClient.from_environment(
            api.base_url, api.model, api.api_key_env, api.timeout_seconds
        )
        return ApiMetaAgent("meta-v0", client, max_calls=api.max_calls_per_session)
    return WaitingMetaAgent()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    config = load_config(args.config)
    data_path = args.data or config.dataset.path
    data = load_csv(data_path, config.dataset.name, config.dataset.timestamp_column)
    values = data.values
    timestamps = tuple(item.to_pydatetime() for item in data.timestamps)
    if args.max_origins is not None:
        keep = config.stream.lookback - 1 + args.max_origins + config.stream.horizon
        values = values[:keep]
        timestamps = timestamps[:keep]
    train_end = int(len(values) * config.dataset.train_ratio)
    scale = fit_standardization(values[:train_end]).scale
    dimension = values.shape[1]

    adapter_root = args.workspace / "adapter_src"
    seed_adapter_source(
        adapter_root,
        configuration={
            "type": "mean_residual",
            "history_window": config.stream.adapter_history_window,
            "cache_size": config.stream.adapter_cache_size,
        },
    )
    adapter = materialize_adapter(adapter_root, version="adapter-v0", dimension=dimension)

    def gate_factory() -> DiscountedRidgeGate:
        return DiscountedRidgeGate(
            dimension=dimension,
            half_life=config.stream.gate_half_life,
            ridge_lambda=config.stream.gate_lambda,
            scale=scale,
        )

    meta_agent = build_meta_agent(config, force_waiting=args.no_meta)
    versions = [RunningVersion(adapter=adapter, gate=gate_factory(), meta_version=meta_agent.version)]

    if args.foundation == "naive":
        foundation = CallableForecaster(lambda context_values, horizon: np.tile(context_values[-1], (horizon, 1)))
    else:
        foundation = load_frozen_foundation_model(
            name=config.foundation.name,
            checkpoint=config.foundation.checkpoint,
            horizon=config.stream.horizon,
            context_length=config.stream.lookback,
            dimension=dimension,
            device=args.device or config.foundation.device,
        )

    runner = CausalStreamRunner(
        values=values,
        timestamps=timestamps,
        lookback=config.stream.lookback,
        horizon=config.stream.horizon,
        foundation=foundation,
        versions=versions,
        completed=CompletedRecordBuffer(config.stream.adapter_cache_size),
        pending=PendingForecasts(),
    )
    revision_log = RevisionLog(args.workspace / "revisions.jsonl")
    revision_manager = RevisionManager(args.workspace / "revision_workspace", revision_log)
    run_tag = args.tag or f"{args.foundation}-{config.dataset.name}"
    api_logger = ApiCallLogger(
        Path("logs") / f"api_calls-{run_tag}.jsonl",
        ApiPrices(
            input_per_mtok=config.api.input_price_per_mtok,
            output_per_mtok=config.api.output_price_per_mtok,
        ),
    )
    orchestrator = MetaReviewOrchestrator(
        meta_agent=meta_agent,
        revision_manager=revision_manager,
        revision_log=revision_log,
        api_logger=api_logger,
        versions=versions,
        adapter_root=adapter_root,
        gate_factory=gate_factory,
        scale=scale,
        dimension=dimension,
        horizon=config.stream.horizon,
        review_batch_records=config.stream.review_batch_records,
        warmup_records=config.stream.candidate_warmup_records,
        evaluation_records=config.stream.candidate_evaluation_records,
    )

    matured_count = 0
    prediction_count = 0
    results_path = args.workspace / "results" / f"{run_tag}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    loss_totals: dict[str, list[float]] = {}
    with results_path.open("w", encoding="utf-8") as results_file:
        for step in runner.run():
            orchestrator.on_step(step.matured)
            matured_count += len(step.matured)
            prediction_count += len(step.predictions)
            for record in step.matured:
                pair = loss_pair(record, scale)
                results_file.write(
                    json.dumps(
                        {
                            "forecast_id": record.forecast_id,
                            "origin_index": record.origin_index,
                            "adapter_version": record.adapter_version,
                            "meta_version": record.meta_version,
                            "loss_base": pair.base,
                            "loss_final": pair.final,
                            "gate": record.gate.tolist(),
                        }
                    )
                    + "\n"
                )
                totals = loss_totals.setdefault(record.adapter_version, [0.0, 0.0, 0.0])
                totals[0] += pair.base
                totals[1] += pair.final
                totals[2] += 1
    print(f"stream done: {prediction_count} forecasts, {matured_count} matured records")
    print(f"meta agent: {meta_agent.version} ({type(meta_agent).__name__})")
    print(f"orchestrator: {orchestrator.summary()}")
    if api_logger.call_count:
        print(f"api calls logged: {api_logger.call_count} -> {api_logger.path}")
    print(f"results -> {results_path}")
    for version, (base_sum, final_sum, count) in sorted(loss_totals.items()):
        if count:
            improvement = (base_sum - final_sum) / base_sum * 100 if base_sum else 0.0
            print(
                f"  {version}: n={int(count)} base_mse={base_sum / count:.4f} "
                f"final_mse={final_sum / count:.4f} improvement={improvement:+.2f}%"
            )


if __name__ == "__main__":
    main()
