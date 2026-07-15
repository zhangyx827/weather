# STCast 2024 Contract

`data/processed/stcast_saudi_2024/` is currently an hourly storage contract for 2024.

Confirmed local state:

- Pressure files cover `2024-01-01T00:00:00` through `2024-12-31T23:00:00`.
- Surface files cover `2024-01-01T00:00:00` through `2024-12-31T23:00:00`.
- Each day contains `24` hourly timestamps.
- Supplemental pressure files under `era5_pressure_levels_2024_missing/` are level-completion inputs, not fallback runs.

This means the repository should not describe `stcast_saudi_2024` as a `6h` dataset. The directory layout is hourly even if some downstream consumers only sample a `6h` subset.

## Stats Contract

`data/processed/stcast_saudi_2024_stats/` is not a full-year hourly summary.

Current local stats files:

- `mean_std.json`
- `mean_std_single.json`

Their current `count` is `1220`, which matches the `6h` window from `2024-01-01T00:00:00` through `2024-10-31T18:00:00`.

Operational rule:

- Treat `stcast_saudi_2024/` as the storage and audit contract.
- Treat `stcast_saudi_2024_stats/` as a separate training-window artifact whose cadence and date range must be stated explicitly.
- Do not infer full-year hourly completeness from `count = 1220`.

## Audit Commands

Audit the actual hourly dataset:

```bash
python3 scripts/audit_saudi_stcast_dataset.py \
  --root-dir data/processed/stcast_saudi_2024 \
  --year 2024 \
  --cadence-hours 1 \
  --compact \
  --output-json logs/stcast_saudi_2024_audit_1h.json
```

Validate the current `6h` stats window against the same hourly dataset:

```bash
python3 scripts/audit_saudi_stcast_dataset.py \
  --root-dir data/processed/stcast_saudi_2024 \
  --year 2024 \
  --cadence-hours 6 \
  --stats-dir data/processed/stcast_saudi_2024_stats \
  --stats-start 2024-01-01T00:00:00 \
  --stats-end 2024-10-31T18:00:00 \
  --stats-step-hours 6 \
  --compact \
  --output-json logs/stcast_saudi_2024_audit_6h_stats.json
```

Expected interpretation:

- The `1h` audit should pass.
- The `6h` audit against the hourly directory should fail on extra hourly timestamps.
- The same `6h` audit can still report valid stats counts for the stated Jan 1 to Oct 31 sampling window.

## Conversion and Stats Generation

`scripts/convert_saudi_era5_to_stcast_npy.py` writes every timestamp present in the monthly surface file. With the current 2024 ERA5 inputs, that produces hourly STCast output.

`scripts/build_saudi_stcast_stats.py` computes stats for the explicit `[train_start, train_end]` window and `--step-hours` cadence you pass in. The stats files therefore inherit that sampling contract; they do not define the storage cadence of the raw STCast directory.
