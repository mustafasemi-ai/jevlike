# Research log: the calibration question on manufacturing sensor data

A chronological record of one working session (1–2 October 2026), kept in the
order things happened, including the wrong turns. Numbers are copied from the
runs; where a result rests on a single seed or a temporary script, it says so.

Worked on with an AI coding assistant (Claude Code): it wrote and ran the
scripts, I set the questions and decided what to try next.

## 0. Starting point

The main study in this repo measures calibration under distribution shift on
text, with a fine-tuned Qwen3-1.7B. Open question: is the finding specific to
text and LLMs, or does it show up where confidence gating is called *virtual
metrology* — predict quality from process sensors, auto-release what the model
is sure about, physically measure the rest?

Data: UCI SECOM. 1,567 wafers, 590 sensors, 104 failures, 19 July – 17 October
2008. Shift is not constructed: train on the first 60% of the period (940
wafers), test on the last 40% (627 wafers, 28 failures).

## 1. First model: the result does not replicate — it changes shape

`scripts/secom_study.py`. Small MLP, same loss as the main model
(CE + 0.5·Brier), 5-fold cross-fitting, temperature fitted on the early period.

| | early period | later period |
|---|---|---|
| accuracy | 0.911 | 0.947 |
| ECE | 0.042 [0.034, 0.066] | 0.060 [0.047, 0.080] |
| AUC | 0.664 [0.593, 0.733] | 0.512 [0.401, 0.623] |
| fail rate among released (P(pass) ≥ 0.95) | 3.7% | 5.0% |
| fail rate among held back | 12.2% | 3.9% |

Accuracy goes *up* and ECE barely moves, because the failure rate falls from
8.1% to 4.5%. Meanwhile ranking drops to chance and the gate stops selecting.
On text the failure was overconfidence, visible in ECE. Here it is a ranking
collapse that ECE does not see.

Check: 8 seeds, AUC 0.676 early vs 0.507 later on average. (Temporary check,
not kept as a script.)

## 2. Is it noise? Two controls

**Random-split control** (`--random-split`): same models, same sample sizes,
but "later" is no longer later in time. Over 8 seeds the held-out AUC is
0.57–0.70; with the chronological split it is 0.45–0.55. The ranges do not
overlap. The collapse is temporal.

**Simulator** (`scripts/synth_wafers.py`): wafers with a failure rule we wrote,
and one kind of shift switched on at a time. 20,000 wafers, 5 repeats.

| shift | accuracy | ECE | AUC | gate still selects? |
|---|---|---|---|---|
| none | 0.903 | 0.014 | 0.895 | yes |
| failure rate falls | 0.926 | 0.033 | 0.904 | yes |
| sensors drift | 0.740 | 0.135 | 0.887 | yes |
| failure mechanism changes | 0.818 | 0.083 | 0.494 | no |

Only a change of mechanism reproduces SECOM's symptom. This is a map from cause
to symptom, not evidence about the real line.

Added later: the combined scenario (mechanism change *and* falling failure
rate) gives accuracy 0.857, ECE 0.053, AUC 0.486. Closer to SECOM than
mechanism alone, but accuracy still falls from 0.903 where SECOM's rises, so
the simulator does not reproduce that part.

## 3. Is it the model? 28 models, 9 families

`scripts/secom_archs.py`, `secom_seq.py`, `secom_llm.py`. Same folds for all;
5 seeds unless noted. No hyperparameter search anywhere: one reasonable setting
per model. Sequence models were run at windows of 4 and 16 wafers; one of each
is listed, the other gave the same picture.

| model | family | AUC early | AUC later |
|---|---|---|---|
| TabPFN v2 | tabular foundation model | 0.740 | 0.524 |
| adaboost | boosted trees | 0.714 | 0.585 |
| lightgbm | boosted trees | 0.711 | 0.541 |
| gbdt | boosted trees | 0.708 | 0.571 |
| catboost | boosted trees | 0.700 | 0.469 |
| extra trees | bagged trees | 0.693 | 0.504 |
| xgboost | boosted trees | 0.686 | 0.555 |
| random forest | bagged trees | 0.683 | 0.496 |
| ensemble of 5 MLPs | neural | 0.683 | 0.517 |
| MLP | neural | 0.674 | 0.514 |
| single decision tree | tree | 0.662 | 0.476 |
| LDA | classic | 0.656 | 0.547 |
| kNN | classic | 0.653 | 0.479 |
| linear | neural | 0.646 | 0.551 |
| logistic regression (L2) | classic | 0.644 | 0.532 |
| deep MLP | neural | 0.641 | 0.509 |
| ResNet | neural | 0.622 | 0.488 |
| PLS-DA | classic | 0.619 | 0.534 |
| SVM (RBF) | classic | 0.618 | 0.517 |
| windowed MLP, 16 wafers | sequence | 0.615 | 0.483 |
| PCA + logistic | classic | 0.613 | 0.579 |
| GRU, 4 wafers | sequence | 0.611 | 0.530 |
| Qwen3-1.7B + LoRA, 30 sensors, 1 seed | language model | 0.578 | 0.538 |
| naive Bayes | classic | 0.577 | 0.480 |
| logistic regression (L1) | classic | 0.575 | 0.500 |
| FT-Transformer | neural | 0.561 | 0.492 |
| 1D-CNN, 16 wafers | sequence | 0.535 | 0.559 |
| isolation forest | anomaly detection | 0.485 | 0.518 |

- No model exceeds 0.59 in the later period. Seed-to-seed spread there is
  0.03–0.08, so the order within the top ten is not reliable.
- No model beats "always answer pass" on accuracy (0.919 early, 0.955 later).
- TabPFN is clearly best where it was trained and among the worst after the
  shift: the model that looks most trustworthy in-domain loses the most.
- Bigger models are worse. 750 training rows with ~50 failures is too little
  for a transformer, and far too little for a 1.7B language model reading
  sensors as text.
- Fine-tuning TabPFN changed nothing measurable: 0.763 / 0.584 against
  0.761 / 0.562 on the same seed.

**A false lead, corrected.** With one seed, boosted trees on the top 30 sensors
reached 0.618 in the later period, which looked like "fewer sensors are more
robust". Over 5 seeds it is 0.551. It was that seed.

## 4. Can the failure be announced? A drift alarm in series

`scripts/secom_drift.py`. A classifier two-sample test: trees try to tell a
window of 100 wafers from the training data. Needs no pass/fail labels.

| windows | drift AUC | alarm |
|---|---|---|
| random wafers from the training period (null) | 0.507 mean | threshold 0.572 |
| consecutive wafers inside the training period | 0.974 – 0.995 | 9 of 9 |
| consecutive wafers after it | 0.992 – 1.000 | 5 of 5 |

The detector is sound (the null is at chance) and useless: every week of this
line is distinguishable from every other, so the alarm never stops. 76 of 446
sensors shift their median by more than 0.5 sd between the periods, 27 make an
abrupt jump of more than 2 sd, and the jumps cluster on certain dates. Whether
that is sensor error or real process change cannot be told from this data.

This exposed a flaw in step 1. If consecutive weeks differ this much, the
"early period" score is optimistic: random folds let the model see each test
wafer's neighbours in time. Tested directly (`scripts/forward_checks.py`):

- Train on everything so far, predict the next 100 wafers: AUC 0.645
  [0.565, 0.729] for boosted trees and 0.616 [0.525, 0.703] for adaboost, not
  0.71.
- Retraining every 100 wafers does not beat a frozen model on the later
  period: boosted trees 0.569 against 0.604, adaboost 0.537 against 0.626, with
  wide overlapping intervals (28 failures).

So the natural response to an alarm — retrain on fresh data — does not help
either.

## 5. Prior work, checked late

Should have been step 0. Others have published the same core finding on SECOM:
random forest 0.739 under shuffled validation against 0.570 time-ordered,
adversarial validation at 0.999, recency weighting that does not rescue it
(github.com/sudharshan8683/SemiFab; two similar repositories report average
precision and a Brier score worse than a constant). Our numbers agree with
theirs closely, which is reassuring and means this part is not new.

What was not found elsewhere in a brief search: the 28-model comparison under
one protocol, the same-size random-split control, the retraining test, and the
confidence-gate analysis.

## 6. Things that broke

- **LLM fine-tune exhausted memory.** The script omitted the gradient
  checkpointing the main trainer uses. On Windows a full GPU spills into system
  memory instead of failing. Fixed, and a memory cap added so an overflow is an
  ordinary CUDA error.
- **TabPFN opened a browser login.** Its current release needs an account. The
  open v2 weights are used instead, with the browser prompt disabled.
- **TabPFN fine-tune exhausted memory the same way.** Every wafer-sensor cell
  is a token. Now 150 wafers per step and one estimator, with the cap.
- **Sequence windows misaligned.** Preprocessing dropped a sensor at some time
  steps only. A guard caught it before any number was produced.

## 7. Where this leaves the question

On SECOM, virtual metrology does not work with any model tried, and neither a
label-free alarm nor retraining repairs it. The link to the main study holds in
one respect only: a confidence gate validated under one condition stops working
under another without any error or warning. The form differs — overconfidence
on text, loss of ranking here.

SECOM is too small and too hard to say more: 28 failures in the later period.

## 8. A dataset where the model works before it breaks

UCI Gas Sensor Array Drift: 13,910 measurements, 16 chemical sensors (128
features), six gases, 36 months in ten batches, published for studying sensor
drift. `scripts/gas_study.py`. Same MLP and loss, trained on batches 1–3,
every later batch scored separately.

Five seeds, mean [min, max]:

| batch | n | accuracy | mean confidence | ECE | accuracy above the 0.99 gate |
|---|---|---|---|---|---|
| in-domain | 3275 | 0.990 | 0.985 | 0.008 | 0.998 [0.997, 0.998] |
| 4 | 161 | 0.814 | 0.965 | 0.151 | 0.893 [0.854, 0.914] |
| 5 | 197 | 0.987 | 0.989 | 0.010 | 0.994 [0.993, 0.994] |
| 6 | 2300 | 0.758 | 0.857 | 0.119 | 0.994 [0.992, 0.995] |
| 7 | 3613 | 0.662 | 0.810 | 0.155 | 0.999 [0.997, 1.000] |
| 8 | 294 | 0.595 | 0.718 | 0.157 | 0.846 [0.667, 0.955] |
| 9 | 470 | 0.633 | 0.773 | 0.156 | 1.000 [1.000, 1.000] |
| 10 | 3600 | 0.500 | 0.873 | 0.373 | 0.857 [0.798, 0.921] |

The overconfidence is the text result again, on sensors: calibrated where
trained (ECE 0.008), overconfident later (pooled ECE 0.21, range 0.18–0.24).
In the last batch the model is 87% confident and 50% right.

The gate is a different story, and a more interesting one. In batches 6, 7
and 9 overall accuracy is 0.63–0.76, yet the items above the 0.99 gate are
still at least 99.2% right: the model has lost a third of its accuracy and the
gate still keeps its promise, by releasing far fewer items. In batches 4, 8
and 10 the gate breaks (0.85–0.89). Miscalibration on average and failure of
the gate are not the same event.

- **Control holds.** With rows shuffled across batches every "later" batch has
  accuracy 0.97–0.99 and ECE 0.02–0.03.
- **Tightening the gate does not clearly widen the gap here.** Threshold minus
  realised accuracy over the later batches: +0.002 at 0.80, +0.040 at 0.90,
  +0.047 at 0.95, +0.041 at 0.99. Positive above 0.9, but flat rather than
  growing; the text result is not reproduced in this respect.
- **Not specific to the MLP.** Boosted trees are equally good in-domain (0.991)
  and worse later (0.405 in batch 10; one seed).

**Correction.** The first version of this section reported one seed (17). That
seed turned out to be the worst of six: its gate fell to 0.89–0.91 in batches
6 and 7 and to 0.68 in batch 10, its pooled ECE was 0.26, and its gap grew
with the threshold up to +0.166. None of the five other seeds shows a broken
gate in batches 6, 7 or 9. Accuracy and overconfidence were stable across
seeds from the start; everything about the gate was not, and should not have
been reported from one run.

- **The in-domain number is optimistic, as on SECOM.** It comes from random
  folds inside batches 1–3. Leaving a whole batch out gives 0.79–0.98 for the
  MLP instead of 0.992, and training on batches 1–2 to predict batch 3 gives
  0.789 (boosted trees: 0.927). Drift is already present inside the training
  period. "In-domain" here means what a random validation split would report
  — which is the number an operator would see before deploying.
  (`scripts/forward_checks.py`.)

## 9. How many measurements does it take to notice?

*This section applies a known method; see section 10. It was written before
the literature was checked.*

`scripts/gas_audit.py`. The gate exists to skip measurements, and the only
direct evidence that it has failed is a measurement. So: audit a fraction of
the released items anyway, and ask how small that fraction can be.

First, the signals that cost nothing:

Five seeds, mean [min, max]:

| batch | error among released (truth) | share released | drift AUC |
|---|---|---|---|
| in-domain | 0.002 [0.002, 0.003] | 0.841 | 0.514 |
| 4 | 0.107 [0.086, 0.146] | 0.743 | 1.000 |
| 5 | 0.006 [0.006, 0.007] | 0.811 | 1.000 |
| 6 | 0.006 [0.005, 0.008] | 0.401 | 1.000 |
| 7 | 0.001 [0.000, 0.003] | 0.324 | 1.000 |
| 8 | 0.154 [0.045, 0.333] | 0.071 | 1.000 |
| 9 | 0.000 [0.000, 0.000] | 0.238 | 1.000 |
| 10 | 0.143 [0.079, 0.202] | 0.313 | 1.000 |

Neither free signal identifies the bad batches. The drift detector is 1.000 on
every later batch, including the four where the gate is fine. The share
released drops in batches 6, 7 and 9, where the gate holds, as much as in
batch 10, where it does not.

Then the audit. Stream: in-domain items, then batches 4–10 in order. Each
released item is measured with probability f; a Bernoulli CUSUM (promised
error 1%, alarm-worthy 5%) watches the outcomes. 200 shuffled streams per seed.

| audited share | false alarm before the change | detected | wrong items released before the alarm: median [range of per-seed medians] | audits spent after the change |
|---|---|---|---|---|
| 0.2% | 0.0% | 0.9% | 216 [216, 238] | 11 |
| 0.5% | 0.0% | 6.8% | 178 [66, 228] | 19 |
| 1% | 0.0% | 20.5% | 160 [66, 205] | 36 |
| 2% | 0.0% | 46.9% | 128 [67, 177] | 66 |
| 5% | 0.0% | 87.6% | 82 [63, 90] | 148 |
| 10% | 0.0% | 97.2% | 51 [48, 53] | 274 |
| 20% | 0.0% | 100% | 30 [17, 34] | 487 |
| 100% | 0.3% | 99.7% | 4 [4, 4] | 33 |

Because the gate fails only in some batches, a low audit rate usually misses
it: at 1% the failure is caught in one stream out of five, and it takes 5–10%
to catch it reliably. The audits are mostly spent on batches where the gate is
fine, which is why the count rises with the rate (36 at 1%, 274 at 10%).

**Correction.** The first version of this table used seed 17 alone, where the
gate fails in nearly every later batch. There a 1% audit caught the failure
99% of the time with about 35 measurements. That was the easy case, not the
typical one.

Limits: rows inside a batch have no time order, so the stream is shuffled
within each batch; and the CUSUM settings were chosen once and not tuned.

## 10. Prior work on the audit question — checked after, again

The audit analysis was proposed as the new part. It is not new, in two fields.

**Machine learning.** Podkopaev and Ramdas, *Tracking the risk of a deployed
model and detecting harmful distribution shifts* (ICLR 2022), pose the same
problem: tell harmful shift from benign shift by monitoring the model's risk on
labels as they arrive, with sequential tests that control the false alarm rate.
Their confidence sequences come with guarantees; the CUSUM here has none, its
false alarm rate is only measured. Later work removes or reduces the labels
(Amoukou et al., NeurIPS 2024; prediction-powered risk monitoring; active
labelling under a budget). The observation in section 9 that a drift detector
cannot tell harmful from harmless drift is their starting point.

**Virtual metrology.** The question "how much real metrology can be skipped" is
a core topic of the field: the Reliance Index (Cheng et al., 2008) scores how
far each virtual measurement can be trusted, and Intelligent Sampling Decision
schemes choose which wafers get measured, raising the sampling rate when
virtual metrology accuracy drops.

So section 9 is a small, reproducible instance of an established idea on open
data, not a contribution. What it adds for this repo is the link: the same
confidence gate, examined on text, wafers and gas sensors, and a number for
what it costs to catch its failure.

**A caution about the gas data.** Dennler et al. (2022) showed that a related
dataset from the same group (the 2013 wind-tunnel recordings) has gases
recorded in temporal clusters, so sensor baseline alone identifies the gas and
published accuracies are inflated. That paper is not about the dataset used
here, and this version has no timestamps to repeat their test with. The
leave-one-batch-out check in section 8 is the closest available evidence, and
it does show the in-domain score depending on time neighbours.

Only abstracts and search summaries were read; the virtual metrology papers
were behind access walls.

## 11. Which run's gate will break? Fifty seeds

The correction in section 8 raised its own question: runs that are identical
in validation behaved very differently at the gate. `scripts/gas_seeds.py`
trains 50 runs (own folds, calibration split, initialisation, batch order).

**Spread.** The gate varies far more than anything else.

| quantity | mean | std | min | max |
|---|---|---|---|---|
| in-domain accuracy | 0.990 | 0.001 | 0.987 | 0.992 |
| in-domain error among released | 0.0025 | 0.0006 | 0.0015 | 0.0043 |
| later accuracy | 0.635 | 0.005 | 0.622 | 0.649 |
| later ECE | 0.217 | 0.021 | 0.162 | 0.267 |
| later error among released | 0.063 | 0.034 | 0.030 | 0.176 |

Later accuracy moves by under 1% of its value between runs; the error among
released items ranges over a factor of six. By batch, with "broken" meaning
more than 5% error among released items:

| batch | error among released: median [min, max] | runs with a broken gate |
|---|---|---|
| 4 | 0.096 [0.057, 0.168] | 50 of 50 |
| 5 | 0.006 [0.000, 0.014] | 0 of 50 |
| 6 | 0.007 [0.004, 0.096] | 3 of 50 |
| 7 | 0.001 [0.000, 0.107] | 3 of 50 |
| 8 | 0.075 [0.042, 0.337] | 23 of 24 (26 runs released fewer than 30 items) |
| 9 | 0.000 [0.000, 0.161] | 3 of 50 |
| 10 | 0.150 [0.079, 0.353] | 50 of 50 |

Two kinds of failure. Batches 4, 8 and 10 break the gate in every run: that is
the data. Batches 6, 7 and 9 break it in 3 runs of 50: that is the run. Seed
17, reported first, was one of those.

**Predictability.** Spearman correlation with the later error among released
items, across the 50 runs.

| signal | available | correlation | p |
|---|---|---|---|
| in-domain accuracy | before deployment | +0.39 | 0.005 |
| in-domain ECE | before deployment | -0.43 | 0.002 |
| in-domain mean confidence | before deployment | +0.62 | <0.001 |
| in-domain share released | before deployment | +0.55 | <0.001 |
| in-domain error among released | before deployment | +0.20 | 0.16 |
| temperature | before deployment | -0.39 | 0.005 |
| later mean confidence | after, no labels | +0.68 | <0.001 |
| later share released | after, no labels | +0.81 | <0.001 |
| later accuracy | after, needs labels | -0.13 | 0.37 |

- The thing one would check — the gate's own error in validation — says
  nothing (+0.20, not significant).
- What does carry signal is how confident a run is. Runs that are more
  confident in-domain, and release more, break more later. In-domain ECE has
  the *wrong* sign: the runs that look slightly worse calibrated are
  under-confident, and safer.
- After deployment the share released is the strongest signal (+0.81) and
  needs no labels: among equally validated runs, the one that keeps releasing
  the most is the one to distrust.
- Later accuracy is unrelated. Gate failure is not accuracy failure.

**Ensembles.**

| model | later error among released: median [min, max] | later share released |
|---|---|---|
| single run | 0.052 [0.030, 0.176] | 0.372 |
| average of 5 runs | 0.028 [0.025, 0.031] | 0.309 |
| average of all 50 | 0.024 | 0.292 |

An ensemble is less confident and so releases less; with every model made to
release the same 30% of later items the single runs are at 0.040 [0.029,
0.217] and the averages of 5 at 0.028 [0.025, 0.030]. Averaging five runs
removes the lottery: the worst case goes from 0.217 to 0.030. It does not fix
the gate: 0.024 with all 50 is still more than twice the promised 1%, because
batches 4, 8 and 10 defeat every run.

Limits: one dataset, one architecture, one threshold. The correlations are
across runs of the same pipeline and partly mechanical (a more confident run
releases more marginal items). That equally validated models differ out of
distribution is known as underspecification (D'Amour et al., 2020), and
run-to-run instability of confidence rankings has been noted for selective
classification; no study of this exact question was found, but only abstracts
were read.

## 12. Does section 11 hold for other thresholds and another model?

Same script, the MLP's cached runs at three gates, and 50 runs of boosted
trees. Boosted trees have no random initialisation; their runs differ only in
which rows they were trained and calibrated on.

Later error among released items, median [min, max], and correlations with it:

| | MLP 0.99 | MLP 0.95 | MLP 0.90 | trees 0.99 | trees 0.95 | trees 0.90 |
|---|---|---|---|---|---|---|
| in-domain error among released | 0.0025 | 0.0032 | 0.0037 | 0.0020 | 0.0039 | 0.0052 |
| later error among released | 0.052 [0.030, 0.176] | 0.105 [0.047, 0.208] | 0.150 [0.073, 0.232] | 0.264 [0.226, 0.300] | 0.319 [0.283, 0.376] | 0.363 [0.336, 0.408] |
| corr.: in-domain error among released | +0.20 | +0.04 | -0.03 | +0.20 | -0.12 | -0.05 |
| corr.: in-domain mean confidence | +0.62 | +0.68 | +0.69 | +0.26 | +0.33 | +0.26 |
| corr.: later share released | +0.81 | +0.86 | +0.86 | +0.35 | +0.69 | +0.65 |
| average of 5, later error among released | 0.028 | 0.063 | 0.107 | 0.250 | 0.257 | 0.279 |

What holds everywhere:

- **Validation says nothing about the gate later.** The gate's own in-domain
  error is uncorrelated with its later error in all six columns.
- **The share released after deployment carries signal**, without labels, in
  all six — strong for the MLP, weaker for trees at the 0.99 gate.

What does not carry over:

- **The lottery is the MLP's.** Its later gate error spans a factor of six
  across runs. For trees the spread is small (0.226–0.300): their gate is
  simply broken in every run, in nearly every batch (batch 6: 50 of 50 runs,
  median error 0.35).
- **Ensembling helps only where there is a lottery.** At the same share
  released, averaging five tree runs moves the error from 0.272 to 0.254.
- **In-domain confidence predicts the MLP's failures, barely the trees'.**

And one thing neither seed nor threshold shows: the two models are
indistinguishable in validation (accuracy 0.990 and 0.989, error among
released 0.0025 and 0.0020), and after drift the trees' gate is five times
worse (0.264 against 0.052). The choice of model matters far more than the
choice of run, and validation is blind to both.

Limits: still one dataset. "Broken" was counted at 5% for every gate, which
is the wrong bar for the 0.90 gate, so only the error values are compared
across thresholds.

## 13. Asking whether the drift is harmful, not whether there is drift

The drift alarm of sections 4 and 9 answers "has the data changed?", and the
answer is always yes. The question worth asking is whether the change is one
that breaks the gate. `scripts/gas_disagreement.py` tries one label-free
signal for it: deploy one run, keep four other runs of the same model as
companions, and measure how often they disagree with the deployed run on the
items it releases. Uses the 50 cached runs; no training.

MLP, gate at 0.99:

| batch | error among released (truth) | disagreement among released | share released | runs with a broken gate |
|---|---|---|---|---|
| in-domain | 0.002 | 0.001 | 0.851 | 0 of 50 |
| 4 | 0.101 | 0.015 | 0.759 | 50 of 50 |
| 5 | 0.006 | 0.001 | 0.838 | 0 of 50 |
| 6 | 0.012 | 0.001 | 0.435 | 3 of 50 |
| 7 | 0.009 | 0.002 | 0.354 | 3 of 50 |
| 8 | 0.097 | 0.023 | 0.160 | 23 of 24 |
| 9 | 0.011 | 0.006 | 0.270 | 3 of 50 |
| 10 | 0.169 | 0.009 | 0.340 | 50 of 50 |

Over all (run, later batch) pairs, telling a broken gate from an intact one:

| free signal | MLP 0.99 | MLP 0.95 | trees 0.99 |
|---|---|---|---|
| disagreement among released, AUC | 0.926 | 0.913 | 0.797 |
| share released, AUC | 0.519 | 0.595 | 0.494 |
| mean confidence, AUC | 0.435 | 0.529 | 0.474 |
| drift detector, AUC | 0.500 | 0.500 | 0.500 |

For the MLP, disagreement separates harmful from harmless drift where every
other free signal is at chance. As an alarm at "more than 0.3% of released
items disputed" it catches 81% of broken gates and flags 2.6% of intact ones.

Two limits, both visible in the tables.

- **It ranks, it does not measure.** Disagreement is 5–20 times smaller than
  the true error (batch 10: 0.9% disputed, 16.9% wrong). The runs mostly make
  the *same* mistakes, so it cannot be read as an error rate, and the alarm
  level has to be set relative to the in-domain value.
- **It fails when the companions are not independent.** Boosted trees have no
  random initialisation and their runs differ only in the rows they saw. In
  batch 6 their gate is wrong on 35% of released items and they disagree on
  0.2%: all fifty make the same errors. For trees the alarm at 0.3% catches
  77% of broken gates but flags 40% of intact ones.

Prior work: Jiang, Nagarajan, Baek and Kolter, *Assessing Generalization of
SGD via Disagreement* (ICLR 2022), show that the disagreement rate of two runs
of the same network estimates its test error in-distribution, and tie this to
ensembles being well calibrated. A follow-up note (Kirsch and Gal, 2022)
points out that an ensemble's calibration can deteriorate as disagreement
increases, so the estimate should not be trusted on new data -- which is the
first limit above, seen directly: under drift disagreement falls far short of
the error. Baek et al. (2022,
*Agreement-on-the-Line*) use agreement between models to predict accuracy
under shift. So the signal is known; what this section adds is its use on a
confidence gate under real sensor drift, and the failure case with trees.

The "broken" bar is 5% error among released items. One dataset.

## References

Each entry was checked against its publisher page, arXiv record or a search
result during this session; for most only the abstract was read.

**Datasets**
- McCann, M. and Johnston, A. (2008). *SECOM*. UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/179/secom
- Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L. and Huerta, R.
  (2012). Chemical gas sensor drift compensation using classifier ensembles.
  *Sensors and Actuators B: Chemical* 166-167, 320-329.
  doi:10.1016/j.snb.2012.01.074. Data: UCI *Gas Sensor Array Drift Dataset*.

**Calibration and shift**
- Ovadia, Y. et al. (2019). Can you trust your model's uncertainty? Evaluating
  predictive uncertainty under dataset shift. NeurIPS 2019.
- D'Amour, A. et al. (2020). Underspecification presents challenges for
  credibility in modern machine learning. arXiv:2011.03395 (JMLR 23, 2022).

**Monitoring a deployed model**
- Podkopaev, A. and Ramdas, A. (2022). Tracking the risk of a deployed model
  and detecting harmful distribution shifts. ICLR 2022. arXiv:2110.06177.
- Amoukou, S. I., Bewley, T., Mishra, S., Lecue, F., Magazzeni, D. and Veloso,
  M. (2024). Sequential harmful shift detection without labels. NeurIPS 2024.
  arXiv:2412.12910.
- Prediction-powered risk monitoring of deployed models for detecting harmful
  distribution shifts. arXiv:2602.02229.
- Incremental uncertainty-aware performance monitoring with active labeling
  intervention. arXiv:2505.07023.

**Disagreement between models**
- Jiang, Y., Nagarajan, V., Baek, C. and Kolter, J. Z. (2022). Assessing
  generalization of SGD via disagreement. ICLR 2022 (spotlight).
- Kirsch, A. and Gal, Y. (2022). A note on "Assessing generalization of SGD
  via disagreement". arXiv:2202.01851.
- Baek, C., Jiang, Y., Raghunathan, A. and Kolter, J. Z. (2022).
  Agreement-on-the-line: predicting the performance of neural networks under
  distribution shift. NeurIPS 2022.

**Virtual metrology**
- Cheng, Chen, Su and Zeng (2008). Evaluating
  reliance level of a virtual metrology system. *IEEE Transactions on
  Semiconductor Manufacturing* 21(1), 92-103.
- Cheng, F.-T., Chen, C.-F., Hsieh, Y.-S., Huang, H.-H. and Wu, C.-C. (2015).
  Intelligent sampling decision scheme based on the AVM system. *International
  Journal of Production Research* 53(7), 2073-2088.

**Gas sensor data caveat**
- Dennler, N., Rastogi, S., Fonollosa, J., van Schaik, A. and Schmuker, M.
  (2022). Drift in a popular metal oxide sensor dataset reveals limitations
  for gas classification benchmarks. *Sensors and Actuators B: Chemical*.
  arXiv:2108.08793. Concerns the 2013 wind-tunnel dataset, not the one used
  here.

**Earlier time-ordered analyses of SECOM (unreviewed repositories)**
- https://github.com/sudharshan8683/SemiFab
- https://github.com/bobora0802-bot/semiconductor-yield-defect-prediction
- https://github.com/williamtb710/semiconductor-yield-anomaly-triage

Not verified: the author lists of the two arXiv-only monitoring papers, which
are therefore cited by title, and the first names of the authors of the 2008
virtual metrology paper, cited by surname.
