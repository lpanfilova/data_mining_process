# CRISP-DM methodological review

Scope: a critical self-review, not independent expert endorsement. Criteria:
[CRISP-DM 1.0 reference model, sections II.1–II.6](https://www.kde.cs.uni-kassel.de/lehre/ws2012-13/kdd/files/CRISPWP-0800.pdf).
The source establishes the task framework; criticisms and remedies below are
project-specific judgments. A missing business prerequisite is recorded as a
limitation, not filled with invented stakeholder decisions. Revisions retain the
original 80/20 split and select models only with training cross-validation. The
previously inspected test set cannot become an independent holdout again.

## Round 1 — review of the original notebook

| Phase and task coverage | Actual critique | Change made |
| --- | --- | --- |
| 1. Business: objectives, situation, mining goals, plan | Stakeholders and metrics are hypothetical; retrospective churn classification is conflated with prospective outreach. No business acceptance decision is stated. | Distinguished recorded-label classification from forecasting; specified eligibility/timestamp/window requirements, net incremental value and controlled-trial acceptance, and an offline-only decision. |
| 2. Understanding: collect, describe, explore, quality | Quality output counts whitespace charges as present; zero-tenure missingness and equal non-ID records lack an explicit account. | Saved conversion-aware quality counts; checked all 11 missing totals have zero tenure; reported 22 equal non-ID records without assuming duplicate persons. |
| 3. Preparation: select, clean, construct, integrate, format | The persisted pipeline does not contain numeric/age normalization. Inference silently depends on notebook preprocessing; median-only imputation erases missing-charge information. | Added shared `src/telco.py` normalization inside every candidate pipeline, raw-column schema checks, and a training-fitted numeric missing indicator. Documented no joins or synthetic records. |
| 4. Modeling: techniques, test design, build, assess | Balanced weights are assumed necessary without comparison. Fold SD can be overread; revised test results would reuse already-known evidence. | Added unweighted logistic regression under the same folds, saved fold scores, explained SD and AP selection, and labeled the old test partition a reused benchmark. |
| 5. Evaluation: results, process, next steps | Point estimates omit sampling uncertainty and workload. A technical pass can conceal false contacts and business uncertainty. | Added conditional stratified-bootstrap intervals, baseline accuracy, confusion/contact counts, subgroup denominators, and an explicit offline-only decision. |
| 6. Deployment: handoff, monitoring, final report, experience | Model reload alone is insufficient handoff; exact dependency versions are descriptive but inconvenient to replay. | Raw-input pipeline persistence, a project-specific pinned direct-dependency file, and an explicit release prohibition pending missing business/data prerequisites. |

Round 1 execution: all eight code cells completed. Training CV selected
unweighted logistic regression. On the reused benchmark: AP 0.634, ROC AUC 0.842,
precision 0.658, recall 0.561; 319 flagged, 109 false positives, 164 missed churners.
The inherited recall >= 0.60 check now fails. The original weighted model had
precision 0.503 and recall 0.783. This is an operating-point tradeoff, not a reason
to tune against the known test set or declare the revision universally better.

## Round 2 — critique of the executed first revision

| Phase | Actual follow-up critique | Justified revision |
| --- | --- | --- |
| 1. Business | AP selection and a recall gate answer different questions; the first revision fails recall, and an offline decision still lacks a measurable business bridge. | Keep the failed gate, separate ranking from binary acceptance, define an illustrative 20% review workload, and state the incremental-value equation with unknown costs/effectiveness. |
| 2. Understanding | Blank and duplicate counts alone do not examine label-ambiguous profiles, service consistency, or potential contemporaneous feature leakage. | Add training-only feature/missingness summary and profile-label ambiguity counts; check service semantics; document pre-outcome availability requirements for account, service, and demographic groups. |
| 3. Preparation | Schema/numeric validation still permits misspelled categories and inconsistent service states. Pipeline design alone does not prove fold isolation. | Reject invalid categories and contradictory phone/internet states; verify fitted imputation/scaling/category levels against each training fold; exercise malformed and missing-value inference cases. |
| 4. Modeling | A small AP difference is not established superiority; the model is selected but not explained, and a weighted model's scores are not directly calibrated risks. | Retain the modest CV winner without significance claims, report descriptive coefficient contrasts, and explicitly separate ranking choice, threshold tradeoff, and probability reliability. |
| 5. Evaluation | Conditional intervals are useful but omit workload-at-capacity, probability reliability, and subgroup false-positive burden. | Add a fixed illustrative top-20% ranking summary, Brier/reliability diagnostics, subgroup confusion denominators/FPR and recall intervals; retain reused-test and conditional-uncertainty warnings. |
| 6. Deployment | Raw-input persistence is improved, but round 1 lacks executable contract regression checks and artifact/source compatibility verification. | Add a reusable verification/replay command, source/model checksums, saved input contract, negative input tests, and a final decision artifact with concrete proposed maintenance actions. |

## Final evidence and assessment

Round 2 executed successfully, then a final replay reproduced all 13 compared
statistical JSON/CSV reports exactly. Eight code cells completed without cell
errors or stderr. Independent verification passed dataset/code/model checksums,
all 20 fitted-fold preprocessing checks, partition isolation, saved-model reload,
metric recomputation, column-order/normalized/raw/missing-value input scoring, and
10 invalid-input rejection cases. A separate verification command also passed.
Plots were inspected and README links checked. The pinned direct package versions
match the executing environment; a fresh-environment installation and complete
transitive dependency lock were not tested.

Final CV AP: unweighted logistic 0.6617, weighted logistic 0.6598. This small
difference does not establish superior population performance. Reused-benchmark
ROC AUC is 0.842 (conditional bootstrap interval 0.820–0.864), AP 0.634
(0.589–0.683), precision 0.658 and recall 0.561. The inherited recall gate remains
unmet. Top-20% review finds 187 of 374 churners among 282 customers; this describes
historical ranking, not preventable churn. Brier score is 0.138 versus prior
baseline 0.195; reliability diagnostics do not independently prove calibration.
Senior/non-senior false-positive rates are 0.234/0.088, requiring investigation.
Thirteen test records have the same predictor profiles as training records;
unique IDs do not exclude other dependence, and attribute equality alone does not
justify removing customers or claiming identity leakage.

| Phase | Final assessment |
| --- | --- |
| Business | Explicit purpose, assumptions, acceptance distinction and decision; stakeholder-approved costs, horizon and population remain unavailable. |
| Understanding | Source integrity, distributions, semantic consistency and ambiguity audited; representativeness and pre-outcome timestamps remain unknown. |
| Preparation | Reusable raw-input contract and fold-fitted transforms verified; imputation is an explicit baseline assumption. |
| Modeling | Simple reproducible comparisons and descriptive explanation; no significance or causal claim, and no threshold tuning against test labels. |
| Evaluation | Uncertainty, workload, reliability and subgroup burden reported; reused data and conditional intervals limit inference. |
| Deployment | Verified offline report/model handoff with replay and proposed maintenance; no actual service, monitoring history or demonstrated business benefit. |

Environment limitation: `pip check` still reports the pre-existing
`langgraph-sdk 0.4.8` requirement `websockets>=14,<17` against installed
`websockets 17.2`. No environment packages were changed; project execution passes.

Decision: retain the case study for offline education. Return to business/data
understanding for timestamped pre-outcome observations and agreed decision costs,
then use new validation data for threshold/calibration decisions and independent
temporal evidence plus a controlled intervention trial for prospective acceptance.
