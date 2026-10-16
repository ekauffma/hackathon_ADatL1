# Leaderboard

Every submission is evaluated on the test samples of `config/baseline.yml`: zero-bias run 396102 against the
four benchmark signals. Ranked by **mean AUC**; ties go to the model with fewer BOPs. ★ marks the
*Pareto front*: models that no other model beats on both mean AUC and BOPs.
See [How to submit](README.md#how-to-submit) to add your team.

| # | team | mean AUC | lowest AUC | highest AUC | reconstruction error | BOPs | submitted |
|---|---|---|---|---|---|---|---|
| 1 | [signal_hunters](submissions/signal_hunters) ★ | 0.9335 | 0.8994 | 0.9773 | 0.7518 | 7,838 | [`16e43ef`](https://github.com/ekauffma/hackathon_ADatL1/commit/16e43efbc898c31de661a318e952adf1385680d8) 2026-10-16 06:43 by [tm1307](https://github.com/tm1307) |
| 2 | [CTRL_FREAKS](submissions/CTRL_FREAKS) | 0.8495 | 0.7620 | 0.9435 | 0.5301 | 20,394 | [`14178d1`](https://github.com/ekauffma/hackathon_ADatL1/commit/14178d102d3323d4c896c21474c61bbae79de3ff) 2026-10-16 08:42 by [JainYash26](https://github.com/JainYash26) |
| 3 | [baseline-dense-qae](submissions/baseline-dense-qae) | 0.8112 | 0.7339 | 0.9123 | 0.4934 | 470,811 | [`d297f57`](https://github.com/ekauffma/hackathon_ADatL1/commit/d297f57a9ac2adc3d709d12a2aed9d6ab0f16b12) 2026-10-16 03:37 by [ekauffma](https://github.com/ekauffma) |
| 4 | [baseline-dense-ae](submissions/baseline-dense-ae) | 0.8084 | 0.7357 | 0.9102 | 0.4867 | 2,575,131 | [`d297f57`](https://github.com/ekauffma/hackathon_ADatL1/commit/d297f57a9ac2adc3d709d12a2aed9d6ab0f16b12) 2026-10-16 03:37 by [ekauffma](https://github.com/ekauffma) |

*Last updated 2026-10-16 08:43 UTC.*
