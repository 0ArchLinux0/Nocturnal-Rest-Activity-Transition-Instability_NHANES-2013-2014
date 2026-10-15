# KAN-171 → KAN-172: P01 correction specification

## Scope boundary

KAN-171 is a read-only forensic audit. It did not modify P01 code, materialize PAXMIN records, calculate participant P01 values, rerun the 9.3 GB analysis, regenerate the cohort, fit models, or create figures. This document is the implementation contract for KAN-172; it does not authorize changing the cohort definition or manuscript claims.

KAN-172 should correct the time axis, valid-epoch predicate, overnight grouping, and transition calculation; preserve old and corrected values side by side; and invalidate any cache made under the prior algorithm. Keep the participant cohort and source-day window identical between old and corrected metrics for the comparison. Resolve the seven-full-day cohort window, survey design, and downstream inference in KAN-173 rather than conflating them with the P01 engine cutover.

## Canonical time axis

1. Use `SEQN` as the participant key and retain `PAXDAYM`, `PAXSSNMP`, `PAXTSM`, `PAXQFM`, `PAXMTSM`, and `PAXPREDM` as distinct source fields. Read participant `PAXFTIME` from `PAXHD_H` as the first-data-point clock-time anchor.
2. Interpret `PAXSSNMP` as the PAXMIN starting sample number at 80 Hz, not as an observed-row count. Derive a normalized local/device-clock timestamp from the header anchor and sample-number offset, e.g. `t_i = PAXFTIME + (PAXSSNMP_i - PAXSSNMP_first) / 80 seconds`, using the first PAXMIN sample that corresponds to the header anchor. Preserve the original sample number and derived timestamp in diagnostic output.
3. Before hard-coding a sample-number increment, verify the codebook's sample-number convention against its documented `9600 / 80 = 120 seconds` example and the note that 1/80 second of data is missing each minute. The missing sample is not a reason to infer time from `cumcount` or to bridge a real clock gap. The normalized timestamp is the continuity authority.
4. Preserve the source device/local-clock time zone semantics; do not invent a UTC offset. Verify each participant's first-row alignment to `PAXFTIME` and handle rollover/day fields from normalized timestamps, not a per-`PAXDAYM` row ordinal.
5. Define a minute record as valid for P01 only when `PAXQFM == 0`, `PAXMTSM` is finite and `>= 0` (exclude the official `-0.01` not-computable value), and the minute has complete duration (`PAXTSM == 60` seconds). Keep source timestamps for flagged or incomplete records for diagnostics, but do not use them as activity states.
   `PAXTSM` is rounded to integer seconds by the codebook procedure. Treat `PAXTSM == 60` as the available summary-level full-minute screen, combine it with aligned source timestamps and the expected minute grid, and do not claim subsecond completeness from this field alone.

## Complete blocks and exact adjacency

For each resolution `r ∈ {5, 10, 30, 60}` minutes:

- Align non-overlapping blocks to normalized clock-minute boundaries; block width is `r × 60` seconds. Derive the block ID from timestamp, not `cumcount` within a wear-day.
- A complete block contains exactly `r` unique, valid one-minute records at the expected one-minute timestamps, with complete `PAXTSM` for every minute. Sum the `PAXMTSM` values only after this predicate passes. No imputation and no partial-block sum.
- For any two block states considered for a transition, require `next_block_start - current_block_start == r × 60 seconds`, with both blocks complete. This is the explicit adjacency invariant. A missing/invalid/incomplete block, absent source row, or nonconsecutive timestamp breaks the sequence; never pair across it.
- Do not count a transition across the six-hour daytime interval. The 22:00–24:00 evening segment and 00:00–06:00 morning segment are separate contiguous sequences, joined only by their real midnight adjacency.

## `night_id` and transition definition

Use normalized local calendar date, not the wear-day row order:

- For a timestamp on date `d` in `[22:00, 24:00)`, assign `night_id = (SEQN, d)`.
- For a timestamp on date `d+1` in `[00:00, 06:00)`, assign `night_id = (SEQN, d)`.
- Keep evening and morning as separate segments within that `night_id`. A 23:50→00:00 pair at 10-minute resolution is eligible only if both complete blocks have exact 600-second start-time separation. A 05:50→22:00 pair is never eligible. A missing block or clock gap breaks the segment even if the remaining rows sort next to one another.
- For each valid segment, define state `0` when block activity is `<= participant threshold_r`, state `1` when it is `> threshold_r`. The eligible denominator is the number of state-0 blocks with an immediately following complete adjacent block in the same segment. The numerator is eligible 0→1 transitions. Compute participant P01 as the pooled numerator divided by pooled eligible denominator across valid nights; do not create an edge between nights or across the daylight gap. Emit per-night counts as diagnostics. Return missing (not zero) when the eligible denominator is zero or the prespecified minimum coverage is not met.

## Participant threshold and resolution sensitivity

- Preserve the existing intended threshold scope for comparability: for each participant and each resolution separately, `threshold_r` is the median activity sum over all valid, complete `r`-minute blocks in the same prespecified analysis-day window, all clock-times (not nocturnal-only). Do not include partial, QC-invalid, noncomputable, or missing blocks. Record the threshold and number of eligible blocks used. Because summed activity is resolution-dependent, calculate and retain a separate threshold for each of 5, 10, 30, and 60 minutes.
- Keep the current `len(minute_df) >= 200` guard only as a named legacy compatibility field if needed for the paired comparison; it is not a seven-day wear or nocturnal-coverage definition. Replace implicit row-count assumptions with an explicit, documented coverage rule. Do not silently switch from participant-wide to day-specific or nocturnal thresholds in this ticket.
- For the old-versus-corrected comparison, use identical participant IDs and analysis-day windows for both algorithms. The current cohort uses PAXDAYD 1–7, while current P01 reads PAXMIN days 1–9; record the exact window used for each side. The official seven-full-day window decision belongs to KAN-173 and must be applied consistently in a later final rerun.

## Required output contract

1. **Participant comparison:** `outputs/audit/p01_old_vs_corrected_participant.csv`, one row per `SEQN × resolution_min`, with at least:
   `SEQN, resolution_min, p01_old, p01_corrected, threshold_old, threshold_corrected, old_nocturnal_blocks, corrected_complete_blocks, corrected_valid_nights, corrected_valid_edges, corrected_rest_edges, corrected_0_to_1_edges, excluded_qc_minutes, excluded_uncomputed_minutes, incomplete_blocks, gap_breaks, legacy_200_minute_gate, status`.
   Preserve missing values and reason codes; do not coerce unavailable old metrics to zero.
2. **Resolution summary:** `outputs/audit/p01_old_vs_corrected_summary.csv`, one row per resolution, with `N_old`, `N_corrected`, `N_paired`, old/corrected mean and median, mean/median paired delta, mean absolute delta, correlation on paired finite values, and counts missing by reason. This is descriptive comparison, not an inferential model.
3. **Run manifest:** `outputs/audit/p01_correction_run_manifest.json`, recording code/version, source path and checksum, cohort artifact/checksum, participant/day window, time conversion rule, QC predicate, block completeness rule, night boundary, threshold scope, and execution timestamp.
4. **Production cache:** Rebuild `outputs/final_cohort_pam_metrics.csv` only with the corrected algorithm and include an algorithm-version marker or source/method fingerprint. The previous cache must not pass the corrected loader merely because its SEQN set matches.
5. Do not rerun OLS, logistic models, ROC curves, publication figures, or manuscript text in KAN-172. Downstream statistical interpretation and NHANES survey design belong to KAN-173 after the audited variables and estimand are fixed.

## Required synthetic regression cases

- Two complete adjacent 10-minute blocks across 23:50→00:00 on consecutive dates produce an eligible edge and the same `night_id`.
- 05:50→22:00 on the same date produces no edge.
- An absent minute, a positive `PAXQFM`, `PAXMTSM == -0.01`, incomplete `PAXTSM`, or incomplete block breaks the edge; no transition bridges across it.
- A QC flag of any positive value, including 3–14, excludes the minute.
- A block with fewer than `r` valid contiguous minutes is excluded at every resolution; 5/10/30/60-minute calculations each use their own threshold and width.
- A first/last partial wear-day does not become a complete day merely because its retained row count passes the legacy 200-minute guard.

## KAN-173 open decisions

- Confirm whether the final analytic wear window is the official seven full days (PAXDAYD 2–8) or the current 1–7 rule; apply the final choice to the cohort and all P01 sensitivity outputs.
- Confirm NHANES examined-weight treatment for PAX responders who pass valid-wear and PHQ-9 filters; use WTMEC2YR with SDMVSTRA/SDMVPSU as the documented candidate design, but do not assume no PAM nonresponse/valid-wear adjustment is needed without guidance.
- Finalize WASO/PAXHR overnight definitions, SLD010H naming, and SLQ060 interpretation before downstream models.
