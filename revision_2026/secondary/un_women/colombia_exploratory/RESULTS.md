# Exploratory UN Women results: both author questions

These are executed analyses of a newly downloaded public source. They are not changes to the manuscript, a causal COVID/IPV effect, validated clinical PHQ/GAD scores, or an independent replication of the source report. The protocol was saved before the new fits.

## 1. Reports that violence or safety worsened

There are 16,154 released women across 13 countries. The table reports country-weighted proportions among valid answers. It measures reported perceptions/retrospective change, not observed incidence before and after COVID. The domestic-violence-vignette question differs from the broad local-violence question and must not be presented as the same indicator.

| Country | Local violence increased (%) | Valid n | Vignette worsened (%) | Valid n |
|---|---:|---:|---:|---:|
| Albania | 33.3 | 1037 | 82.5 | 1155 |
| Bangladesh | 62.6 | 1237 | 81.0 | 1239 |
| Cameroon | 19.2 | 966 | 44.9 | 1035 |
| Colombia | 24.4 | 1135 | 44.3 | 1183 |
| Cote d'Ivoire/Ivory Coast | 17.4 | 1040 | 65.1 | 1192 |
| Jordan | 11.4 | 1145 | 74.6 | 1191 |
| Kenya | 50.0 | 1169 | 93.5 | 1203 |
| Kyrgyz Republic | 10.7 | 1083 | 39.2 | 1005 |
| Morocco | 22.9 | 1098 | 69.3 | 1193 |
| Nigeria | 25.9 | 1415 | 74.5 | 1462 |
| Paraguay | 35.6 | 1018 | 58.1 | 1115 |
| Thailand | 6.1 | 1177 | 57.9 | 1188 |
| Ukraine | 14.7 | 1123 | 62.5 | 1142 |

In Colombia, **23.1%** of the **99** valid respondents to the conditional direct-IPV frequency-change question reported increased frequency (approximate 95% interval 15.8–32.4%). The raw yes count was 25. This is not the fraction of all Colombian women whose IPV increased. Refused/unknown responses were excluded and structural skips were not coded as no increase.

A timing flag remains: 20 direct-positive women reported before-only histories despite the direct question's last-12-month framing. Therefore these reports cannot establish incident post-COVID IPV or a causal change. No prospective violence time series has been constructed.

## 2. Direct reported violence and symptom items in Colombia

Of 1209 released Colombian women, 806 had a valid, non-inapplicable direct-IPV response. Excluding 24 incomplete symptom records and 3 additional incomplete covariate records left **779 women**, including **101 reporting direct IPV**. Full exclusions and coding are saved.

| Outcome | Adjusted difference, IPV versus never | Approximate 95% interval | Holm p, two primary tests |
|---|---:|---:|---:|
| depressive_item_sum | 0.516 | 0.128 to 0.904 | 0.0185 |
| worry_item | 0.025 | -0.228 to 0.277 | 0.8487 |

### Explicitly post-primary timing sensitivity

A separately saved amendment, written after inspecting the primary estimates and before the sensitivity fits, excludes the 20 complete women reporting before-only histories. Both models retain 759 women, including 81 reporting IPV. The depressive-item association remains positive and the worry-item interval still includes zero. This is a sensitivity in the same cohort, not a new primary discovery.

| Outcome | Adjusted difference | Approximate 95% interval | Unadjusted descriptive p |
|---|---:|---:|---:|
| depressive_item_sum | 0.679 | 0.253 to 1.106 | 0.0018 |
| worry_item | 0.128 | -0.160 to 0.415 | 0.3835 |

The depressive-symptom item sum is higher among women reporting direct partner violence after adjustment for age category, education, locality and marital category. The worry-item estimate is imprecise and does not provide clear evidence of a difference. It is not evidence that violence has no relationship with anxiety.

These Phase II questions use a since-COVID reference period. The two depressive items are summed 0–6 and the worry item ranges 0–3. This is not a standard two-week PHQ-2, a GAD-2, a clinical diagnosis or a change score. Phase I contains different last-month stress items and was not pooled into these outcomes. Report the actual item wording/reference periods in any manuscript use.

The direct Colombian exposure question concerns specific physical partner-violence acts; it is not a complete inventory of emotional, sexual and physical IPV. No religion/faith/attendance field was identified in the released column inventory, so this model does not claim religion adjustment.

## Uncertainty and interpretation

Supplied country weights are used. Original PSU/strata are not released, so uncertainty assumes independent respondent IDs and does not recover the complete original survey design or nonresponse uncertainty. Intervals are approximate and country intervals pointwise. The two primary model p values are Holm-adjusted. No global pooled estimate is calculated.

The cross-sectional association cannot establish whether violence preceded symptoms, whether COVID caused violence, or whether an individual treatment gap explains the association. Exposure and symptom recall periods differ, and income/food insecurity, pre-existing mental health and other unmeasured factors may matter. The direct results concern one selected Colombian analysis domain, not the full 13-country sample or all Colombian women.

These findings concern different measurements and units from the earlier GFS bereavement/country-context analyses. Keep earlier negative and inconclusive results. Do not claim the present individual association validates earlier country-level prediction or establishes CEBRA superiority. Additional algorithms were not used.

## Files and checks

`reported_worsening.csv`, `adjusted_symptom_associations.csv`, `all_model_coefficients.csv`, `unadjusted_symptom_means.csv`, `broad_mental_impact_descriptive.csv`, source/coding ledgers, `QC.json`, `R_sessionInfo.txt` and figures provide traceability. The independent Python check reproduces both adjusted coefficients and approximate covariance from weighted estimating equations, all descriptive weighted means, intervals and Holm correction. Respondent rows remain local.

Run `prepare_pilot.py`, then `analyze_pilot.R`, then `timing_sensitivity.R`, then `report_and_check.py` from the repository root using their project-relative paths. Source: [UN Women VAW rapid gender assessments](https://data.unwomen.org/publications/vaw-rga).
