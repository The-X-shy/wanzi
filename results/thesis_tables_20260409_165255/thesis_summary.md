# Thesis Experiment Summary

- METR-LA frozen baseline: best MAE `0.3651`, spread `2.94%`.
- Raw champion: `directional_headroom` + `dual_focus` + `directional_focus` gives legacy mean ASR `1.74%`, local raw-space ASR `7.31%`, clean MAE drift `3.63%`, and local target attainment `-0.0006`.
- Paper champion: `directional_headroom` + `dual_focus` + `directional_focus` gives legacy mean ASR `1.74%`, local raw-space ASR `7.31%`, clean MAE drift `3.63%`, and local target attainment `-0.0006`.
- Raw champion and paper champion are the same candidate.
- Strategy comparison: `error` has the highest mean local ASR `6.03%`.
- Window-family tradeoff: mean local ASR leader is `hybrid` at `6.03%`, while peak local ASR leader is `hybrid` at `7.08%`.
- Simple smoothing ASR effect: `1.42% -> 1.45%`.
- Cross-dataset replay: `2` candidates replayed on the secondary dataset.
- Cross-dataset best local raw-space ASR: `11.76%` with clean MAE drift `2.09%`.
- Minimum paper bar: `met`; strong paper bar: `not met`.
- Previous mainline local-ASR bar (`5.61%`): `beaten`.
- Stop rule: `triggered`; extra follow-up search recommendation: `yes`.
