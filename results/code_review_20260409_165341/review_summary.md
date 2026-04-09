# Code Review Summary

## High-priority findings

1. The old pipeline mixed up "highest local effect" and "best paper-ready result". This caused the saved champion to drift away from the thesis standard. The fix is the new dual-champion flow: `best_attack_raw.json` and `best_attack_paper.json`.
2. Candidate identity originally ignored `headroom_error_mix`, `global_shift_fraction`, and `tail_focus_multiplier`. That could merge different experiments into one row during dedupe and summary building. The fix is to include these fields in candidate keys and tables.
3. Recheck used to follow only the paper-oriented ranking. That could miss the true peak local-effect candidate. The fix is the union recheck pool: top paper candidates plus top raw candidates.

## Why the previous strongest result missed the paper line

- The previous raw peak already had strong local effect, but its legacy ASR stayed at about 1.48%, slightly below the 1.50% minimum line.
- The main problem was not attack strength. It was contract alignment: local effect was strong, but the broader effect seen by the legacy metric was just not wide enough.

## What each optimization direction tried to fix

- `spread_recovery`: widen the effective impact a little so the legacy ASR can move above the paper minimum without destroying local strength.
- `selection_balance`: avoid overly extreme poisoned samples and improve the chance of passing the paper line with cleaner directional behavior.
- `loss_rebalance`: keep the same attack family, but retune how strongly training focuses on the target nodes, tail horizons, and target shift size.

## Direction comparison

- `loss_rebalance`: paper local ASR `7.31%`, legacy ASR `1.74%`, clean MAE drift `3.63%`, direction match `71.11%`, target attainment `-0.0006`, minimum line `passed`.
- `selection_balance`: paper local ASR `6.99%`, legacy ASR `1.71%`, clean MAE drift `3.85%`, direction match `68.59%`, target attainment `0.0195`, minimum line `passed`.
- `spread_recovery`: paper local ASR `6.85%`, legacy ASR `1.65%`, clean MAE drift `3.97%`, direction match `73.36%`, target attainment `0.0017`, minimum line `passed`.

## Current recommendation

- Keep `loss_rebalance` as the current paper-main direction. It gives the best paper champion among the three tested routes.