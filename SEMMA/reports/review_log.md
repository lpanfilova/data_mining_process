# SEMMA methodological review

Critical self-review, not an independent expert endorsement. Framework:
[SAS Introduction to SEMMA](https://documentation.sas.com/doc/en/emref/15.3/n061bzurmej4j3n1jnj8bbjjm1a2.htm).
The framework supplies the five phases; the judgments below concern this project.
Original families, parameters, split and mean-validation-MAE selection rule are
retained. The previously inspected test period is a reused benchmark. No new
feature search, tuning, or post-test model substitution is performed.

## Round 1 — original implementation

| Phase and relevant subtasks | Critique | Revision |
| --- | --- | --- |
| Sample: acquire, define population, partition | Full-table sampling is sensible, but excluded warm-up hours and seasonal coverage lack an explicit accounting artifact; a review cannot renew test independence. | Added sample accounting and full-day checks; documented warm-up use and the reused benchmark. |
| Explore: describe, relationships, anomalies | Most relationships concern same-day weather, which is not actually a predictor. Seasonal/closure changes are acknowledged without a compact development-period audit. | Added lag-feature associations and monthly counts/zeros/closure summaries; restricted these to development data and made their descriptive role explicit. |
| Modify: select, transform, clean, construct | Lagging relies on ordered consecutive rows without enforcing that precondition. Missing rental history is silently imputed, contradicting the stated availability requirement. | Reject unordered/gapped history and missing/fractional rental lags. Retain weather-only median imputation and the original causal feature set. |
| Model: techniques, fitting, comparison, selection | MAE selection is legitimate, but Ridge fits squared loss; the small average validation gain can be overread as stable superiority. | Clarified objective mismatch, exported paired fold MAE differences and labeled Ridge a development-selected candidate, not an operational champion. |
| Assess: errors, reliability, usefulness, scoring | Error slices omit the comparator, and an archived model may be mistaken for a deployment recommendation after its primary-metric failure. | Added paired slice errors/counts and an explicit assessment artifact withholding Ridge promotion; retain its model artifact to reproduce the original choice. |

Round 1 execution passed all seven code cells and the existing verifier. The
model/primary results were unchanged: validation selected Ridge (mean MAE 332.63
versus 348.14); test MAE 330.76 versus persistence 301.37. Paired fold MAE differences
were +27.23, -15.75, -58.02. Test slice differences were +10.60 in October and
+45.67 in November, -201.43 in 103 closed hours and +48.55 in 1,241 operating hours.
Thus closures are difficult for both methods but do not explain Ridge's overall
relative failure; Ridge's gain on closed hours offsets part of its operating-hour loss.

## Round 2 — review of the executed revision

| Phase | Actual follow-up critique | Revision |
| --- | --- | --- |
| Sample | The 24-hour validation fit gap differs from the final fit's zero-hour gap, and chronology alone does not document all day-boundary/availability assumptions. | Documented the conservative CV/final-fit difference without changing it after seeing test results; added full-day partition checks and explicit protocol metadata. |
| Explore | The original emphasis on closures can misattribute the relative MAE shortfall: round 1 shows Ridge is better on closures but worse on operating hours and in both months. | Added exposure-weighted contributions and signed biases to paired slices; revised the explanation to reflect operating-hour losses, not a closure-only story. |
| Modify | Stricter lag validation still cannot prove that independently supplied lag columns reflect available history. | Added a small as-of-day input builder that refuses future history and reconstructs all 24 inputs from earlier observations; compare against bulk-generated features and test missing/gapped history. |
| Model | The mean validation winner wins only two of three periods; a small average gain is insufficient evidence of stable superiority. | Added per-fold and aggregate MAE skill versus persistence and explicitly retained the original candidate/selection rule without significance, equivalence or operational-champion claims. |
| Assess | The verifier recomputes only selected metrics and trusts several saved reports; the weekly interval is too weak to establish equality or prospective value. | Independently recompute comparator metrics, paired slices, daily/block summaries and decision consistency; label the eight-week interval exploratory and retain a no-promotion conclusion. |

## Final verification and assessment

Round 2 and a final replay passed all seven code cells without cell errors or
stderr. All 19 compared report/figure/example/prediction files reproduced
byte-for-byte. Verification independently recalculated both models' test metrics,
validation summaries, daily errors, paired slice contributions, and the seeded
weekly-block interval. It also checked dataset/code/model hashes, exact lag
alignment, future-target perturbation, nine chronological validation fits,
training-only transform statistics, saved-model reload, a full 24-hour as-of
scoring batch, reordered inputs, and missing-weather inference. Nine malformed
input cases, three invalid lag histories, and two invalid as-of histories were
rejected. README/report links passed. No model parameters or original forecasts
were changed by these corrections.

Ridge's validation MAE skill was +4.46%, with two of three favorable folds. On the
reused benchmark it had MAE **330.76 versus 301.37** (+29.39 rentals/hour, +9.75%)
and RMSE **439.15 versus 516.96**. The operating-hour MAE loss contributes +44.83
to the aggregate difference; the closed-hour gain contributes -15.44. Ridge's
operating-hour signed error averages -236.22 rentals/hour, consistent with
underprediction in that slice; it is not a causal explanation. Do not optimize
against this already-inspected period.

The exploratory eight-week resampling interval for the MAE difference is
**[-42.74, 98.30]**. Neither this interval nor the lower RMSE establishes MAE
superiority, equivalence, non-inferiority, or service-cost benefit. Keep Ridge as
the archived validation-selected candidate; **promotion over persistence is not
supported**. Keeping persistence as reference is not proof that it is an adequate
operational forecasting policy either.

| Phase | Final assessment and limit |
| --- | --- |
| Sample | Complete hourly history and chronological accounting; one annual cycle and a reused final season cannot establish broad generalization. CV's extra fit gap differs transparently from final fitting. |
| Explore | Relevant lag/regime summaries and paired diagnostics; all associations and post-test explanations remain descriptive. |
| Modify | Causal feature construction and shared validated scoring recipe; real reporting latency and externally supplied calendar/history still need verification. |
| Model | Original MAE selection correctly reproduced; squared-loss Ridge and a small unstable validation gain are not evidence of primary-metric success. |
| Assess | Primary failure, secondary gains and uncertainty are explicit; the saved model is reproducible, not an endorsed replacement for the baseline. |

Remaining constraints: only eight test weeks, possible dependence across weekly
blocks, no new independent test period, no operational loss target, and no
verified inventory/wait-time benefit. Direct versions are pinned but a fresh
installation and full transitive lock are not verified. `pip check` retains the
pre-existing `langgraph-sdk 0.4.8`/`websockets 17.2` conflict (SDK requires
`>=14,<17`); environment packages were not changed and project execution passes.
Next work requires new dated observations and agreed operational criteria before
any new feature/threshold/model comparison can support prospective acceptance.
