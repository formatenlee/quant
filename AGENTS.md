# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
Two tightly-coupled Python products (China A-share quant research), all under `src/`:
- **Data pipeline** (`quant_cursor`): AKShare-based ETL — build a tradable universe, download OHLCV, export/dump to Microsoft Qlib `.bin`. Entry: `python -m quant_cursor <cmd>` (`universe`, `download`, `download-minute`, `download-derivatives`, `repair-fields`, `list`, plus `qlib …` / `pipeline …` routed subcommands). See `src/cli.py`.
- **ML models** (`bpc`, `bpc_v3`, `bpc_v4`): PyTorch models that consume Qlib data. `bpc_v4` (current) fuses a frozen Kronos tokenizer with BPC features. Run docs: `src/bpc_v4/README.md`.

### Environment (already provisioned by the update script)
- Python virtualenv at `/workspace/.venv` (Python 3.12, CPU-only — no GPU). Activate with `source /workspace/.venv/bin/activate`.
- **Package layout gotcha:** `src/` *is* the `quant_cursor` package (its `config.py` sets `PROJECT_ROOT=/workspace`, so `data_dir=/workspace/data`). It is made importable via a symlink `…/site-packages/quant_cursor -> /workspace/src` that the update script (re)creates. Don't `pip install -e .` — there is no `pyproject.toml`/`setup.py`.
- To work on the ML packages (`bpc`/`bpc_v3`/`bpc_v4`) as top-level modules, run from `/workspace/src` (e.g. `cd src && python -m bpc_v4.train`).

### Running / non-obvious caveats
- `--config` is a **top-level** CLI option and must come *before* the subcommand: `python -m quant_cursor --config foo.yaml universe`.
- **AKShare network:** sources are Chinese servers. CSI (`stock_zh_index_hist_csindex`) and Sina index lists work reliably from the cloud VM; the EastMoney endpoints (`stock_zh_index_spot_em`, the `em_index_categories` fetch) are flaky here (intermittent `RemoteDisconnected`) and can trip the downloader's circuit breaker, which sleeps ~900s. For quick checks, use a config with `em_index_categories: []` / `include_stocks: false` and a low `ban_cooldown_seconds`. Numeric non-SZ indices (e.g. `000300`, `000905`) download via the reliable CSI path.
- Full `universe` build with `include_stocks: true` is slow: it walks every Shenwan L2 industry's constituents under rate-limiting.

### ML stack reality
- `torch` here is the CPU wheel. `bpc_v4` config/train default `device="cuda"` but fall back to CPU via `torch.cuda.is_available()`.
- Running `python -m bpc_v4.train` end-to-end needs a populated Qlib `cn_data` dataset (`~/.qlib/qlib_data/cn_data`, NOT in the repo) plus the custom Kronos HF model `NeoQuasar/Kronos-Tokenizer-base`. Without that data it cannot train E2E; the `BPCV4Model` itself trains fine on synthetic tensors.
- **Pre-existing breakage (not env-related):** `src/bpc_v4/quick_test.py` imports symbols that don't exist (`BPCv4Config`/`BPCv4Dataset`/`collate_fn`/`BPCv4Model`); the real entry point is `bpc_v4/train.py` with `BPCV4Model`. `bpc_v3.train` fails to import because it pulls from the stale duplicate tree `src/quant_cursor/bpc_v3/`. The nested `src/quant_cursor/` is a stale partial copy — the active code is the top level of `src/`.

### Remote training server sync
The user runs training on a remote GPU server and wants every code change pushed there immediately.
- **Path mapping:** local `/workspace/src/`  ↔  remote `/home/user/pdl/mylab/quant/quant_cursor/` (the remote package dir is named `quant_cursor`, the local one is `src`). The remote project root `/home/user/pdl/mylab/quant/` also holds `config.yaml`, `data/`, `checkpoints/`, `requirements*.txt` that are NOT in this git repo — do not overwrite them.
- **How:** use the git-ignored helper `/workspace/.sync_to_server.sh` (rsync over ssh, code-only, no `--delete`). Run it after editing any file under `src/`. `./.sync_to_server.sh --dry-run` previews changes.
- **Credentials:** SSH `user@183.232.132.248:31407`. The script reads `SYNC_SSH_PASSWORD` if set; otherwise a fallback is baked into the (un-committed) script. For durability across agent sessions, store the password as a Cursor secret named `SYNC_SSH_PASSWORD` and recreate `.sync_to_server.sh` if it's missing (it is git-ignored so it won't appear in a fresh checkout).
- The remote `requirements.txt` / `requirements-qlib.txt` / `requirements-train.txt` mirror the deps installed locally (akshare/pandas/pyarrow/pyyaml, pyqlib, torch).

### Lint / test / build
- No lint config, no test framework, no build system are present in the repo. Use `python -m compileall src` as a syntax/build sanity check.
