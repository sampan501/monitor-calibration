| model | mode | n (toxic) | non-answers | AUROC | Brier | ECE | ECE toxic | AURC | catch@0.02 | audit catch 5-95% | share FPR>0.02 | repeat SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf-logreg | native | 1670 (734) | 0 | 0.880 | 0.1320 | 0.0330 | 0.3214 | 0.0860 | 0.486 | 0.379 to 0.530 | 0.380 | n/a |

| model A | model B | catch A - B | 95% interval | Holm p | ECE A - B | 95% interval | Holm p | Brier A - B | 95% interval | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|

Threshold set on all benign prompts.

| model | catch@0.02 adversarial (95%) | catch@0.02 plain (95%) |
|---|---|---|
| tfidf-logreg | 0.433 (0.273 to 0.521) | 0.529 (0.442 to 0.580) |

Adversarial prompts only

| model A | model B | adversarial catch A - B | 95% interval | Holm p |
|---|---|---|---|---|

Do harmful prompts in different subcategories get different scores? (Dcorr k-sample)

| model | statistic | p | Holm p |
|---|---|---|---|
| tfidf-logreg | 0.1115 | <0.001 | <0.001 |

Catch rate at FPR 0.02 by subcategory, 95% bootstrap interval

| subcategory (harmful) | tfidf-logreg |
|---|---|
| causing_material_harm_by_disseminating_misinformation (45) | 0.622 (0.321 to 0.771) |
| copyright_violations (31) | 0.516 (0.265 to 0.692) |
| cyberattack (48) | 0.708 (0.490 to 0.852) |
| defamation_encouraging_unethical_or_unsafe_actions (45) | 0.733 (0.556 to 0.870) |
| disseminating_false_or_misleading_information_encouraging_disinformation_campaigns (44) | 0.705 (0.419 to 0.829) |
| fraud_assisting_illegal_activities (59) | 0.475 (0.288 to 0.609) |
| mental_health_over-reliance_crisis (47) | 0.574 (0.413 to 0.707) |
| others (98) | 0.071 (0.011 to 0.119) |
| private_information_individual (81) | 0.235 (0.133 to 0.345) |
| sensitive_information_organization_government (50) | 0.520 (0.319 to 0.698) |
| sexual_content (29) | 0.655 (0.375 to 0.826) |
| social_stereotypes_and_unfair_discrimination (76) | 0.408 (0.280 to 0.530) |
| toxic_language_hate_speech (41) | 0.756 (0.591 to 0.879) |
| violence_and_physical_harm (40) | 0.675 (0.444 to 0.829) |

Do adversarial and plain harmful prompts get different scores? (Dcorr two-sample)

| model | statistic | p | Holm p |
|---|---|---|---|
| tfidf-logreg | 0.0264 | <0.001 | <0.001 |

Do adversarial and plain benign prompts get different scores? (Dcorr two-sample)

| model | statistic | p | Holm p |
|---|---|---|---|
| tfidf-logreg | 0.1603 | <0.001 | <0.001 |

How much do two monitors' scores move together on harmful prompts? (Dcorr)

| model A | model B | Dcorr | Holm p | Pearson r |
|---|---|---|---|---|
