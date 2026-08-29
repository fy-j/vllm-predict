# 06 — Prediction accuracy versus lookahead distance

**What to build:** Measure how well the cross-layer gate actually predicts the target layer's load, and how that degrades with prediction lookahead. This settles the lookahead default with evidence instead of an assumption, and supplies the accuracy input the cost-aware planner needs. It requires no replication machinery: the prediction path is read-only and the actual load is already recorded, so this is a comparison of two tensors that both already exist.

**Blocked by:** 02 — vLLM cross-layer prediction and global predicted-load snapshot.

**Status:** DONE — measured 2026-08-24, see `bench/RESULTS.md`. Both defaults confirmed; `prediction_skip_first_layers` could drop from 3 to 2.

- [x] Prediction accuracy is measured against the recorded actual load for the same layer and forward, for lookahead distances one, two, and three. One server per lookahead, 5500 to 5900 scorable layer pairs per run.
- [x] Accuracy is reported as hot-expert set overlap and per-expert count error, broken down by layer index. Recall at 2 is the headline because the planner replicates two experts by default: 0.828 / 0.781 / 0.726 for code at lookahead 1 / 2 / 3, and 0.793 / 0.740 / 0.696 for text. Count error, as total-variation distance between load shares, is 0.088 / 0.120 / 0.145. Gate-logit similarity is absent as specified.
- [x] Results confirm or revise the two defaults they exist to justify. **Lookahead 2 confirmed**: lookahead 1 is five points more accurate but hides only 15% of a transfer against 92% at 2, and 3 costs five more points for nothing. **Skip 3 confirmed, one layer conservative**: a dedicated skip-0 run shows target layers 2 and 3 unusable (0.028 and 0.262 against a 0.750 median, twenty to thirty tolerance-widths low) and 4 onward normal, which at lookahead 2 implicates source layers 0 and 1. Skip 2 is what the unambiguous evidence supports; 3 is the safer reading, because target 4 at 0.716 is itself marginal against a 0.75 to 0.83 median. Per-layer curves are computed per lookahead, never pooled - pooling inflates precisely the leading layers this turns on and hid them in the first analysis.
- [x] Accuracy is reported per content domain and per request shape, using the agreed shapes. Code at 1024 tokens predicts three to four points better than conversational text at 2048 at every distance, confirming the content dependence.
- [x] A recommendation: the accuracy at lookahead 2 **is** sufficient for a planner to act on, and the lookahead does not need reducing. Recall at 2 of 0.74 to 0.78 means the planner picks the right pair of experts about three times in four, and reducing the lookahead to gain five points would expose 85% of each transfer instead of 8%. This says nothing about whether acting is worthwhile — see ticket 00, which found the recoverable time on this node to be 0.08% to 0.15% of a decode step.


## Not established (2026-08-24)

- **Whether early-layer inaccuracy belongs to the source layer or the target
  layer.** Targets 2 and 3 are both the earliest targets and the ones fed by the
  earliest sources, and the runs cannot separate those. The experiment that would:
  skip 0 at lookahead 1, which reaches target 2 from source 1. Skipped because
  both attributions imply the same action.
- **Target layer 47, the last, scores 0.449** against a stable median near 0.76.
  Its source (45) is a legal source, so the trailing layers may deserve exclusion
  as well. One layer pooled over two runs — an observation, not a recommendation.
- Per-layer figures rest on a few hundred samples each and the curve is noisy
  (layer 5 at 0.526, layer 11 at 0.650, between neighbours near 0.75). Quote the
  bucketed and overall figures, not individual layers.
