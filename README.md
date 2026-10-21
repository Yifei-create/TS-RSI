# TS-RSI

TS-RSI is a causal streaming framework for adapting a frozen, black-box time-series foundation model through two local logical roles. The default frozen backbone is **Chronos-2**; the same interface also supports **Moirai-2** and **TimesFM 2.5**.

- `AdapterAgent` executes a local residual correction program and updates numeric parameters from fully matured forecasts.
- `MetaAgent` reviews compact reports and proposes versioned adapter or meta-process edits through an API. A deterministic local editor applies edits only to an isolated candidate.

The foundation model is exposed through `BlackBoxForecaster.predict(context, horizon)`. No adapter code can access gradients, hidden states, or weights. The first adapter is a mean residual correction over the latest completed records. The gate is the fixed per-variable exponentially discounted ridge solution from the project specification.

## Layout

```text
ts_rsi/
  foundation/   frozen Chronos-2, Moirai-2, and TimesFM 2.5 wrappers
  adapters/     executable correction programs
  gating/       fixed discounted ridge gate
  streaming/    causal pending queue, buffers, and runner
  evaluation/   fixed metrics, reports, and candidate trials
  meta/         API client, prompts, agent schema, editor tools
  revisions/    isolated candidate manager and append-only log
  data/         catalog, loader, chronological split, windows
configs/experiment.yaml
scripts/run_stream.py
data/raw/
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The repository contains the eight benchmark files under `data/raw/`. The ETT files are from the authors' [ETDataset](https://github.com/zhouhaoyi/ETDataset). Weather, Electricity, Traffic, and Exchange are from the THUML [Time-Series-Library dataset repository](https://huggingface.co/datasets/thuml/Time-Series-Library), released under CC BY 4.0. See `THIRD_PARTY_NOTICES.md`. Dataset download code is intentionally not part of the experiment package.

## Frozen foundation model configuration

The foundation model is loaded once from a local checkpoint and used in inference mode throughout the stream. Adapter and MetaAgent never receive gradients, hidden states, or weights. Set `foundation.name` to `chronos-2`, `moirai-2`, or `timesfm-2.5`, and set `foundation.checkpoint` to the downloaded local checkpoint directory.

```yaml
foundation:
  name: chronos-2
  checkpoint: checkpoints/chronos-2
  device: cpu
```

The ORCA reference uses the same local checkpoint pattern. Install only the backend selected for an experiment; the three supported packages are `chronos-forecasting`, `uni2ts`, and `timesfm`.

## MetaAgent API configuration

The time-series model itself does not use an API. Only MetaAgent review and revision proposals use an OpenAI-compatible API. The default `configs/experiment.yaml` points at DeepSeek (`https://api.deepseek.com/v1`, model `deepseek-chat`); any OpenAI-compatible endpoint works by changing `api.base_url`, `api.model`, and `api.api_key_env`.

1. Create `.env` from `.env.example` and paste your key (the variable name must match `api.api_key_env`):

   ```bash
   cp .env.example .env   # then edit OPENAI_API_KEY=sk-...
   ```

2. Run the stream. `scripts/run_stream.py` loads `.env` automatically; without a key it falls back to `WaitingMetaAgent` (always WAIT, zero API calls).

   ```bash
   # development run: naive foundation stand-in, truncated stream, MetaAgent reviews live
   python scripts/run_stream.py --foundation naive --max-origins 800

   # full run with the frozen foundation model (requires the local checkpoint)
   python scripts/run_stream.py
   ```

Each review session is limited to `api.max_calls_per_session` calls (initial decision plus one repair follow-up). Every API call is appended to `logs/api_calls.jsonl` with category (`regular_review` / `followup`), token usage, latency, and an optional cost estimate from `api.input_price_per_mtok` / `api.output_price_per_mtok`. Tool executions are counted separately from API calls. Revision proposals and trial outcomes are appended to `artifacts/revisions.jsonl`; candidate copies live under `artifacts/revision_workspace/candidates/`.

The API response must be JSON with `action` in `{WAIT, REVISE_ADAPTER, REVISE_META}` and optional deterministic `tool_calls`. The editor is restricted to a candidate directory and rejects path traversal and ambiguous text replacements. A candidate adapter is materialized from its edited `config.json` + `adapter.py` (which must keep the `build_adapter(version, config, dimension)` factory), runs with its own fresh gate through a fixed warmup plus evaluation window, and is accepted only when its standardized loss beats the running version on records created after the candidate. `REVISE_META` decisions are currently logged as `deferred`; the old/new meta-proposal comparison from the specification is not wired yet.

## Causal protocol

For a forecast issued at origin `i`, all `H` targets become available only at `i + H`. The runner first releases mature records, evaluates them, updates only the matching version's adapter and gate, and then emits a new forecast. Candidate comparisons use records created after the candidate and a fixed warmup plus evaluation length. Acceptance is `baseline_loss - candidate_loss > 0`; ties retain the current version.

The specification in `时序RSI方案_API明确版.md` is the source of truth for the experiment protocol and starting values. This repository intentionally contains the framework and data preparation; it does not start experiments or make API calls during setup.
