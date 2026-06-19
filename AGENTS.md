# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
Two tightly-coupled Python products (China A-share quant research), all under `src/`:
- **Data pipeline** (`quant_cursor`): AKShare-based ETL — build a tradable universe, download OHLCV, export/dump to Microsoft Qlib `.bin`. Entry: `python -m quant_cursor <cmd>` (`universe`, `download`, `download-minute`, `download-derivatives`, `repair-fields`, `list`, plus `qlib …` / `pipeline …` routed subcommands). See `src/cli.py`.
- **ML models** (`bpc`, `bpc_v3`, `bpc_v4`): PyTorch models that consume Qlib data. `bpc_v4` (current) fuses a frozen Kronos tokenizer with BPC features. Run docs: `src/bpc_v4/README.md`.

### Environment (already provisioned by the update script)
- Python virtualenv at `/workspace/.venv` (Python 3.12, CPU-only — no GPU). Activate with `source /workspace/.venv/bin/activate`.
- **Package layout:** 仓库 `src/` 即 `quant_cursor` Python 包；部署到服务器时整目录同步为 `quant/quant_cursor/`。不要在 `src/` 下再建嵌套 `quant_cursor/`。服务器上在 `quant/` 根目录执行 `python -m quant_cursor.bpc_v3.train` / `quant_cursor.bpc_v4.train`（`PYTHONPATH=quant`）。

### Running / non-obvious caveats
- `--config` is a **top-level** CLI option and must come *before* the subcommand: `python -m quant_cursor --config foo.yaml universe`.
- **AKShare network:** sources are Chinese servers. CSI (`stock_zh_index_hist_csindex`) and Sina index lists work reliably from the cloud VM; the EastMoney endpoints (`stock_zh_index_spot_em`, the `em_index_categories` fetch) are flaky here (intermittent `RemoteDisconnected`) and can trip the downloader's circuit breaker, which sleeps ~900s. For quick checks, use a config with `em_index_categories: []` / `include_stocks: false` and a low `ban_cooldown_seconds`. Numeric non-SZ indices (e.g. `000300`, `000905`) download via the reliable CSI path.
- Full `universe` build with `include_stocks: true` is slow: it walks every Shenwan L2 industry's constituents under rate-limiting.

### ML stack reality
- `torch` here is the CPU wheel. `bpc_v4` config/train default `device="cuda"` but fall back to CPU via `torch.cuda.is_available()`.
- Running `python -m bpc_v4.train` end-to-end needs a populated Qlib `cn_data` dataset (`~/.qlib/qlib_data/cn_data`, NOT in the repo) plus the custom Kronos HF model `NeoQuasar/Kronos-Tokenizer-base`. Without that data it cannot train E2E; the `BPCV4Model` itself trains fine on synthetic tensors.
- **Pre-existing breakage (not env-related):** `src/bpc_v4/quick_test.py` imports symbols that don't exist; the real entry is `quant_cursor.bpc_v4.train` → `src/bpc_v4/train.py`.

### Lint / test / build
- No lint config, no test framework, no build system are present in the repo. Use `python -m compileall src` as a syntax/build sanity check.
