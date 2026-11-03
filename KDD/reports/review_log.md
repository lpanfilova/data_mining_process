# Nine-step KDD review

Critical self-review, not independent expert endorsement. Framework:
[Fayyad, Piatetsky-Shapiro and Smyth (1996), detailed KDD process](https://www.ai.rug.nl/ki2/literature/kdd-fayyad-shapiro-smyth.pdf).
The framework supplies the step structure; judgments below are project-specific.
Keep the original equal-standardized-weight representation, k search, seed and
reference model. Diagnostic variants are not a search for better headline metrics.

## Round 1 — original implementation

| Step | Criticism | Revision |
| --- | --- | --- |
| 1. Domain/goals | Technical reproducibility could be mistaken for validated knowledge or stakeholder usefulness. | Separate reproducible description from score validity, natural types and unverified business acceptance. |
| 2. Target dataset | Two features are interpretable, but the rationale is not an optimal feature-selection result and field provenance is underspecified. | Add a field/role/unknown-metadata ledger; explicitly narrow the question to an income/score projection. |
| 3. Cleaning | Range/checksum checks cannot establish measurement validity; row accounting and unusual-value retention are not saved. | Add positive integer ID checks, duplicate/retention accounting, sorted-source observation and non-destructive IQR flags. |
| 4. Transformation | Equal standardized weights are asserted without checking the resulting dependence on that choice. | State the squared-distance weighting and compare fixed score contributions 0.5/1/2, preserving the primary model. |
| 5. Task matching | Unknown score inputs could make apparent income/score patterns partly circular. | Treat the task as description of supplied values, not discovery of spending behavior; require score-input audit before that interpretation. |
| 6. Exploration/model choice | The five-cluster argument does not quantify its runner-up or show that secondary criteria can disagree. | Report the silhouette gap, secondary-index preferences and whether the size guard actually changes eligibility. |
| 7. Mining | Numerical termination and a chosen k can be overread as a global or substantive optimum. | Explicitly limit convergence/selection claims and archive the unchanged reference rather than promote sensitivity variants. |
| 8. Interpretation/evaluation | Subsampling changes both records and optimizer seeds, and ARI includes records used to refit. | Fix the subsampling optimizer seed and add omitted-row ARI; retain the caveat that the reference saw all records. |
| 9. Knowledge use | A bounding-box range flag can be mistaken for support/novelty detection, and metadata prerequisites remain only narrative. | Clarify assignment diagnostics and save an explicit offline-only assessment with prerequisites for action. |

Round 1 execution passed eight code cells and existing verification. The primary
partition remained five clusters (silhouette 0.5547), with a 0.0148 silhouette gap
over k=6. Davies–Bouldin favors five; Calinski–Harabasz favors nine; the size guard
excludes none of k=2..10. Halving score's squared-distance weight selects seven,
not five; fixed-k=5 ARI remains 0.9696. Doubling it selects five (ARI 0.9712).
Fixed-seed subsampling yields all-row mean ARI 0.9892 and omitted-row ARI 0.9854.
There are four repeated two-feature profiles, no duplicate non-ID records, and
two income IQR flags. These results motivate the next critique rather than a
change to the primary model.

## Round 2 — executed first revision

| Step | Actual follow-up criticism | Revision |
| --- | --- | --- |
| 1. Domain/goals | A reproducible five-group partition is supportable, but weight-dependent k and unknown score semantics prevent a claim of validated knowledge. | Encode the distinction in the assessment/knowledge summary and limit the accepted deliverable to a conditional descriptive partition. |
| 2. Target dataset | Four repeated income/score profiles are not four duplicate persons; ID-based independence and profile multiplicity remain distinct issues. | Record profile multiplicity and retained-ID accounting; explain why deduplicating the projection would silently change observation weights. |
| 3. Cleaning | Two income IQR flags affect a variance-scaled geometry; retaining them is defensible but influence has not been examined. | Perform one fixed-k leave-IQR-flagged-records-out diagnostic, refitting its scaler; retain all 200 records in the primary fit. |
| 4. Transformation | Weight sensitivity changes the preferred k; interval-scale assumptions for the score remain untested even after rescaling. | Add one rank-score/equal-weight diagnostic, retaining ties and reporting it as a different geometry, not a validated replacement or cure for unknown score provenance. |
| 5. Task matching | Excluded age/gender are not external labels and could still be encoded inside an unknown score. | Make the projection/proxy limitation explicit in the final claim record; request audited measurement inputs before behavioral or fairness claims. |
| 6. Model/hypothesis selection | Fixed-k stability does not test stability of selecting five rather than another count. | Reselect k=2..10 on the first ten already-defined subsamples, preserving the 5% size guard, and save all candidates and choices. No reference-model reselection. |
| 7. Mining | Verifier recomputes only the selected candidate's indices; conclusions about neighboring k and sensitivity tables need traceable checks. | Independently refit all primary candidates, verify selection evidence and sensitivity tables, and retain the original model and parameters. |
| 8. Interpretation/evaluation | A global ARI can hide a less stable small cluster; omitted-row agreement still uses an all-data reference. | Add one-to-one matched per-cluster Jaccard summaries, retain all-/omitted-row ARIs, and state the shared-reference and non-independent-resampling limitations. |
| 9. Knowledge use | Coordinate bounds do not indicate joint support, and original unresolved-metadata text does not incorporate the new sensitivity evidence. | Include an inside-box corner and outside-box example, update the claim/unknowns record, and retain offline analyst-review use only. |

## Final evidence and assessment

Round 2 and the final replay passed all eight code cells without cell errors or
stderr. All 23 compared report/figure/example/assignment files reproduced
byte-for-byte. Verification independently refitted all primary and representation
sensitivity candidates, all 40 stability fits and the 90 subsample-k candidates;
it checked omitted-row ARIs, matched Jaccards, candidate sizes/indices/choices,
source/code/model hashes, original-unit centers, assignments, influence analysis,
new-profile inference, reordered columns and eight invalid-input rejections.
The primary model, assignments and centroids are byte-identical to the pre-review
staged artifacts. New diagnostics do not replace that model. Links passed.

Primary evidence: k=5, silhouette 0.5547, runner-up k=6 only 0.0148 lower. The size
guard excludes no original candidate; Davies–Bouldin favors five while
Calinski–Harabasz favors nine. Five was reselected in all ten subsample searches
within the original geometry, but reducing score weight to 0.5 selects seven.
The rank-score diagnostic retains five but changes the partition (ARI 0.9149).
Silhouettes across those geometries must not be used to choose a better representation.

Fixed optimizer-seed subsampling gives all-row mean ARI 0.9892 and omitted-row
mean ARI 0.9854. The minimum matched per-cluster Jaccard is 0.9545. Excluding the
two income IQR flags solely for a fixed-k diagnostic gives ARI 0.9847; all records
remain in the primary fit. These checks support limited perturbation repeatability,
not ground-truth accuracy, population stability or independence. Overlapping
subsamples share the same all-data reference and supply no confidence interval.

Final judgment: all nine procedural steps are demonstrated. **Five is a defensible
conditional silhouette-selected description, not a verified true or invariant
number of customer types.** Feature choice, weighting and the measurement model
matter. Four repeated two-feature profiles are legitimate retained observations
unless person-level duplication is established; a valid schema/checksum does not
establish an independent or representative sample.

Spending Score remains the substantive limitation: its formula, inputs, temporal
basis, comparability and interval distances are unknown. Rank transformation and
stable clusters cannot validate it or exclude circularity/proxies. Age/gender
summaries are descriptive and do not validate behavioral or fairness claims.
Use the outputs for qualified offline description and metadata/domain review;
obtain independent dated records and a relevant prospective outcome before
claiming actionable customer knowledge or intervention benefit.

Direct package versions are pinned, but fresh-environment installation and a full
transitive lock are not verified. `pip check` retains the pre-existing
`langgraph-sdk 0.4.8` requirement `websockets>=14,<17` against `websockets 17.2`.
No environment packages were changed; project execution and verification pass.
