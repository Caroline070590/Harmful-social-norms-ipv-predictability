# Reported past-year IPV classification in ENDES

The observed linked cohort contains 10267 women, including 954 reporting past-year physical or sexual IPV and 9313 reporting neither in that period. Only 169 endorse any justification. These are conditional cohort counts, not national prevalence estimates.

Five whole-sampling-cluster folds evaluate every included woman once. The demographic block includes age and native education, wealth, work, marital, residence and region categories. The added block contains the five individual justification responses. All violence items and summaries, symptom measures and identifiers are excluded from predictors and graph distances. Edges represent demographic similarity rather than observed social contact.

Primary Brier loss: logistic demographics 0.082845; logistic plus attitudes 0.082906. The added-attitude difference is 0.000061 [-0.000042, 0.000170]. The primary logistic comparison did not establish a clear improvement from adding justification responses.

GraphSAGE plus attitudes has Brier loss 0.082926, ROC-AUC 0.6145, average precision 0.1503 and sensitivity 0.0000 at the fixed probability threshold. Its Brier difference against logistic plus attitudes is 0.000019 [-0.000326, 0.000373]. No consistent graph-specific advantage was established across the primary logistic, self-message and random-neighbour comparisons.

## Later-cohort test

Training excludes the eight actual-2024 interviews from the earlier release and uses 10259 actual-2023 women. Testing includes 9847 actual-2024 women (851 positive), with 3250 women (304 positive) in sampling clusters absent from the entire earlier household source.

All 2024 respondents: logistic D/A Brier 0.077724/0.077661; GraphSAGE A 0.077656. Added-attitude difference -0.000063 [-0.000175, 0.000041].

New-PSU 2024 respondents: logistic D/A Brier 0.083264/0.083147; GraphSAGE A 0.083281. Added-attitude difference -0.000116 [-0.000358, 0.000107].

At the prespecified probability threshold of 0.5, every evaluated model identifies no positive women in each of the three domains. These results do not establish useful case detection at that operating point.

## Interpretation and limits

The target is classification of reported violence in the preceding year. Attitudes and violence were measured concurrently; this does not forecast future assaults or identify a causal effect of attitudes. The graph is constructed and provides no evidence of interpersonal transmission. The health-linked eligibility restriction selects a narrower cohort than the available violence module. Religion is unavailable and joint-module population weights remain unverified.

All model settings and the five-fold assignment were fixed before fitting this new study, after earlier outcomes had been examined. These remain retrospective exploratory comparisons of a bounded model family. Neural scores average three initialization probabilities; those runs are not independent participants. Pointwise intervals resample observed sampling clusters with predictions fixed and omit model-refitting, development and national survey-design uncertainty. Low positive prevalence makes accuracy alone inadequate; fixed-threshold operating rates and probabilistic scores are reported together. Average precision is a stepwise precision–recall summary. The pooled out-of-fold prevalence model varies across training folds, so its measured average precision need not equal the observed prevalence.

[Complete model scores](results/metrics.csv), [paired loss comparisons](results/paired_comparisons.csv), [calibration summaries](results/calibration.csv), and [figure](figures/endes_ipv_classification.pdf). Source-dependent fitting and stored-state verification are distinguished from the public aggregate replay.
