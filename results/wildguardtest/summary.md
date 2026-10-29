| model | mode | n (positive) | non-answers | distinct scores | AUROC | Brier | ECE | AURC | catch@0.02 | catch strict | benign at max score | audit catch 5-95% | share FPR>0.02 | repeat SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | stated | 1670 (734) | 127 | 42 | 0.938 | 0.0957 | 0.0291 | 0.0477 | 0.544 | 0.522 | 0.007 | 0.441 to 0.647 | 0.490 | n/a |
| gemini-3.1-pro | stated | 1670 (734) | 2 | 11 | 0.882 | 0.1226 | 0.1225 | 0.1135 | 0.234 | 0.000 | 0.064 | 0.194 to 0.288 | 0.430 | n/a |
| gpt-5.4 | stated | 1670 (734) | 0 | 19 | 0.935 | 0.1346 | 0.1229 | 0.0589 | 0.483 | 0.278 | 0.000 | 0.431 to 0.542 | 0.406 | n/a |
| jev | native | 1670 (734) | 0 | 301 | 0.956 | 0.0799 | 0.0239 | 0.0292 | 0.693 | 0.680 | 0.000 | 0.650 to 0.732 | 0.438 | 0.0054 |
| qwen3.5-9b | token | 1670 (734) | 0 | 1670 | 0.936 | 0.1194 | 0.1172 | 0.0639 | 0.531 | 0.501 | 0.000 | 0.389 to 0.639 | 0.450 | n/a |
| tfidf-logreg | native | 1670 (734) | 0 | 1670 | 0.879 | 0.1324 | 0.0324 | 0.0867 | 0.475 | 0.470 | 0.000 | 0.387 to 0.528 | 0.414 | n/a |

| model A | model B | catch A - B | 95% interval | Holm p | ECE A - B | 95% interval | Holm p | Brier A - B | 95% interval | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | gemini-3.1-pro | +0.310 | +0.200 to +0.414 | 0.015 | -0.0934 | -0.1043 to -0.0742 | 0.015 | -0.0269 | -0.0366 to -0.0173 | 0.015 |
| gpt-5.4 | gemini-3.1-pro | +0.249 | +0.183 to +0.321 | 0.015 | +0.0004 | -0.0143 to +0.0163 | 1.000 | +0.0120 | -0.0022 to +0.0266 | 0.368 |
| jev | claude-opus-5.5 | +0.149 | +0.041 to +0.247 | 0.015 | -0.0052 | -0.0179 to +0.0109 | 1.000 | -0.0158 | -0.0224 to -0.0094 | 0.015 |
| jev | gemini-3.1-pro | +0.459 | +0.392 to +0.507 | 0.015 | -0.0985 | -0.1103 to -0.0747 | 0.015 | -0.0427 | -0.0533 to -0.0321 | 0.015 |
| jev | gpt-5.4 | +0.210 | +0.136 to +0.262 | 0.015 | -0.0989 | -0.1118 to -0.0754 | 0.015 | -0.0548 | -0.0665 to -0.0435 | 0.015 |
| qwen3.5-9b | gemini-3.1-pro | +0.297 | +0.149 to +0.409 | 0.015 | -0.0053 | -0.0220 to +0.0106 | 1.000 | -0.0032 | -0.0152 to +0.0082 | 1.000 |
| jev | qwen3.5-9b | +0.162 | +0.056 to +0.303 | 0.015 | -0.0933 | -0.0977 to -0.0748 | 0.015 | -0.0396 | -0.0459 to -0.0335 | 0.015 |
| tfidf-logreg | gemini-3.1-pro | +0.241 | +0.137 to +0.310 | 0.015 | -0.0901 | -0.1031 to -0.0650 | 0.015 | +0.0098 | -0.0050 to +0.0243 | 0.564 |
| jev | tfidf-logreg | +0.218 | +0.149 to +0.307 | 0.015 | -0.0085 | -0.0235 to +0.0060 | 1.000 | -0.0526 | -0.0619 to -0.0431 | 0.015 |
| claude-opus-5.5 | gpt-5.4 | +0.061 | -0.041 to +0.162 | 1.000 | -0.0937 | -0.1075 to -0.0729 | 0.015 | -0.0389 | -0.0507 to -0.0275 | 0.015 |
| claude-opus-5.5 | qwen3.5-9b | +0.013 | -0.142 to +0.210 | 1.000 | -0.0881 | -0.0959 to -0.0700 | 0.015 | -0.0238 | -0.0321 to -0.0155 | 0.015 |
| qwen3.5-9b | gpt-5.4 | +0.048 | -0.109 to +0.159 | 1.000 | -0.0057 | -0.0255 to +0.0137 | 1.000 | -0.0152 | -0.0291 to -0.0020 | 0.145 |
| claude-opus-5.5 | tfidf-logreg | +0.069 | -0.045 to +0.205 | 1.000 | -0.0033 | -0.0214 to +0.0120 | 1.000 | -0.0368 | -0.0465 to -0.0268 | 0.015 |
| gpt-5.4 | tfidf-logreg | +0.008 | -0.060 to +0.113 | 1.000 | +0.0904 | +0.0640 to +0.1044 | 0.015 | +0.0022 | -0.0129 to +0.0173 | 1.000 |
| qwen3.5-9b | tfidf-logreg | +0.056 | -0.092 to +0.199 | 1.000 | +0.0848 | +0.0642 to +0.0923 | 0.015 | -0.0130 | -0.0226 to -0.0032 | 0.054 |

Non-answers by label

| model | harmful | benign | harmful adversarial | harmful plain | benign adversarial | benign plain |
|---|---|---|---|---|---|---|
| claude-opus-5.5 | 120 | 7 | 60 | 60 | 7 | 0 |
| gemini-3.1-pro | 2 | 0 | 2 | 0 | 0 | 0 |
| gpt-5.4 | 0 | 0 | 0 | 0 | 0 | 0 |
| jev | 0 | 0 | 0 | 0 | 0 | 0 |
| qwen3.5-9b | 0 | 0 | 0 | 0 | 0 | 0 |
| tfidf-logreg | 0 | 0 | 0 | 0 | 0 | 0 |

Catch rate on adversarial prompts only, threshold set on all benign prompts

| model A | model B | adversarial catch A - B | 95% interval | Holm p |
|---|---|---|---|---|
| claude-opus-5.5 | gemini-3.1-pro | +0.328 | +0.171 to +0.477 | 0.015 |
| gpt-5.4 | gemini-3.1-pro | +0.158 | +0.086 to +0.252 | 0.015 |
| jev | gemini-3.1-pro | +0.434 | +0.358 to +0.495 | 0.015 |
| jev | gpt-5.4 | +0.276 | +0.176 to +0.346 | 0.015 |
| qwen3.5-9b | gemini-3.1-pro | +0.280 | +0.126 to +0.423 | 0.015 |
| jev | tfidf-logreg | +0.269 | +0.157 to +0.397 | 0.015 |
| jev | qwen3.5-9b | +0.154 | +0.021 to +0.296 | 0.135 |
| tfidf-logreg | gemini-3.1-pro | +0.165 | +0.021 to +0.262 | 0.160 |
| claude-opus-5.5 | gpt-5.4 | +0.170 | +0.018 to +0.318 | 0.189 |
| claude-opus-5.5 | tfidf-logreg | +0.164 | -0.012 to +0.361 | 0.408 |
| qwen3.5-9b | gpt-5.4 | +0.122 | -0.046 to +0.266 | 1.000 |
| qwen3.5-9b | tfidf-logreg | +0.115 | -0.050 to +0.304 | 1.000 |
| jev | claude-opus-5.5 | +0.106 | -0.051 to +0.256 | 1.000 |
| claude-opus-5.5 | qwen3.5-9b | +0.048 | -0.173 to +0.282 | 1.000 |
| tfidf-logreg | gpt-5.4 | +0.006 | -0.142 to +0.112 | 1.000 |

Do harmful prompts in different subcategories get different scores? (Dcorr k-sample)

| model | Holm p |
|---|---|
| claude-opus-5.5 | <0.001 |
| gemini-3.1-pro | <0.001 |
| gpt-5.4 | <0.001 |
| jev | <0.001 |
| qwen3.5-9b | <0.001 |
| tfidf-logreg | <0.001 |

Catch rate at FPR 0.02 by subcategory, 95% bootstrap interval

| subcategory (harmful) | claude-opus-5.5 | gemini-3.1-pro | gpt-5.4 | jev | qwen3.5-9b | tfidf-logreg |
|---|---|---|---|---|---|---|
| causing_material_harm_by_disseminating_misinformation (45) | 0.639 (0.357 to 0.803) | 0.253 (0.191 to 0.337) | 0.466 (0.350 to 0.604) | 0.842 (0.697 to 0.944) | 0.616 (0.421 to 0.818) | 0.605 (0.331 to 0.755) |
| copyright_violations (31) | 0.328 (0.160 to 0.528) | 0.139 (0.082 to 0.214) | 0.280 (0.183 to 0.416) | 0.443 (0.250 to 0.613) | 0.403 (0.175 to 0.583) | 0.516 (0.286 to 0.699) |
| cyberattack (48) | 1.000 (1.000 to 1.000) | 0.301 (0.238 to 0.392) | 0.708 (0.608 to 0.821) | 0.917 (0.843 to 1.000) | 0.660 (0.412 to 0.920) | 0.661 (0.487 to 0.837) |
| defamation_encouraging_unethical_or_unsafe_actions (45) | 0.687 (0.428 to 0.906) | 0.294 (0.232 to 0.382) | 0.597 (0.486 to 0.736) | 0.891 (0.765 to 0.980) | 0.772 (0.542 to 0.944) | 0.733 (0.559 to 0.860) |
| disseminating_false_or_misleading_information_encouraging_disinformation_campaigns (44) | 0.536 (0.314 to 0.743) | 0.209 (0.151 to 0.286) | 0.452 (0.333 to 0.590) | 0.699 (0.549 to 0.833) | 0.550 (0.286 to 0.711) | 0.636 (0.434 to 0.816) |
| fraud_assisting_illegal_activities (59) | 0.661 (0.539 to 0.822) | 0.250 (0.192 to 0.329) | 0.605 (0.495 to 0.721) | 0.737 (0.607 to 0.851) | 0.661 (0.523 to 0.789) | 0.458 (0.294 to 0.596) |
| mental_health_over-reliance_crisis (47) | 0.493 (0.326 to 0.706) | 0.268 (0.206 to 0.355) | 0.502 (0.389 to 0.633) | 0.776 (0.588 to 0.889) | 0.633 (0.400 to 0.862) | 0.548 (0.410 to 0.707) |
| others (98) | 0.469 (0.341 to 0.620) | 0.241 (0.187 to 0.315) | 0.577 (0.491 to 0.677) | 0.732 (0.626 to 0.815) | 0.594 (0.402 to 0.721) | 0.071 (0.019 to 0.120) |
| private_information_individual (81) | 0.219 (0.098 to 0.384) | 0.156 (0.112 to 0.214) | 0.231 (0.159 to 0.323) | 0.309 (0.210 to 0.433) | 0.197 (0.077 to 0.316) | 0.235 (0.135 to 0.333) |
| sensitive_information_organization_government (50) | 0.475 (0.300 to 0.667) | 0.215 (0.158 to 0.296) | 0.329 (0.226 to 0.462) | 0.635 (0.486 to 0.768) | 0.464 (0.250 to 0.637) | 0.520 (0.340 to 0.692) |
| sexual_content (29) | 0.695 (0.478 to 0.885) | 0.265 (0.201 to 0.359) | 0.584 (0.427 to 0.747) | 0.793 (0.630 to 0.940) | 0.552 (0.201 to 0.829) | 0.655 (0.414 to 0.828) |
| social_stereotypes_and_unfair_discrimination (76) | 0.357 (0.225 to 0.541) | 0.182 (0.135 to 0.253) | 0.304 (0.220 to 0.419) | 0.533 (0.400 to 0.662) | 0.299 (0.132 to 0.500) | 0.421 (0.297 to 0.536) |
| toxic_language_hate_speech (41) | 0.795 (0.625 to 0.950) | 0.292 (0.229 to 0.383) | 0.691 (0.578 to 0.808) | 0.902 (0.806 to 0.978) | 0.591 (0.320 to 0.878) | 0.756 (0.596 to 0.882) |
| violence_and_physical_harm (40) | 0.708 (0.467 to 0.879) | 0.284 (0.222 to 0.374) | 0.617 (0.494 to 0.752) | 0.825 (0.689 to 0.932) | 0.737 (0.510 to 0.905) | 0.650 (0.437 to 0.812) |

Do adversarial and plain prompts get different scores? (Dcorr two-sample) Rates at the threshold set on all benign prompts, 95% bootstrap interval

| model | harmful Holm p | catch adversarial | catch plain | benign Holm p | FPR adversarial | FPR plain |
|---|---|---|---|---|---|---|
| claude-opus-5.5 | <0.001 | 0.572 (0.404 to 0.735) | 0.522 (0.439 to 0.610) | <0.001 | 0.040 (0.038 to 0.044) | 0.000 (0.000 to 0.000) |
| gemini-3.1-pro | 0.013 | 0.243 (0.192 to 0.319) | 0.227 (0.180 to 0.295) | <0.001 | 0.040 (0.037 to 0.043) | 0.001 (0.000 to 0.003) |
| gpt-5.4 | 0.013 | 0.401 (0.330 to 0.506) | 0.548 (0.493 to 0.621) | <0.001 | 0.038 (0.034 to 0.041) | 0.003 (0.000 to 0.006) |
| jev | 0.011 | 0.677 (0.604 to 0.749) | 0.705 (0.644 to 0.755) | <0.001 | 0.037 (0.031 to 0.042) | 0.003 (0.000 to 0.008) |
| qwen3.5-9b | 0.053 | 0.523 (0.367 to 0.681) | 0.536 (0.375 to 0.648) | <0.001 | 0.038 (0.032 to 0.042) | 0.002 (0.000 to 0.008) |
| tfidf-logreg | <0.001 | 0.408 (0.271 to 0.508) | 0.529 (0.449 to 0.583) | <0.001 | 0.032 (0.024 to 0.039) | 0.008 (0.002 to 0.015) |

Are two monitors' scores on harmful prompts dependent? (Dcorr)

| model A | model B | Holm p |
|---|---|---|
| claude-opus-5.5 | gemini-3.1-pro | <0.001 |
| claude-opus-5.5 | gpt-5.4 | <0.001 |
| claude-opus-5.5 | jev | <0.001 |
| claude-opus-5.5 | qwen3.5-9b | <0.001 |
| claude-opus-5.5 | tfidf-logreg | <0.001 |
| gemini-3.1-pro | gpt-5.4 | <0.001 |
| gemini-3.1-pro | jev | <0.001 |
| gemini-3.1-pro | qwen3.5-9b | <0.001 |
| gemini-3.1-pro | tfidf-logreg | <0.001 |
| gpt-5.4 | jev | <0.001 |
| gpt-5.4 | qwen3.5-9b | <0.001 |
| gpt-5.4 | tfidf-logreg | <0.001 |
| jev | qwen3.5-9b | <0.001 |
| jev | tfidf-logreg | <0.001 |
| qwen3.5-9b | tfidf-logreg | <0.001 |

P(B misses | A misses) on harmful prompts at each model's own threshold, 95% bootstrap interval

| model A | model B | P(B misses \| A misses) | B miss rate |
|---|---|---|---|
| claude-opus-5.5 | gemini-3.1-pro | 0.846 (0.802 to 0.886) | 0.766 (0.694 to 0.815) |
| claude-opus-5.5 | gpt-5.4 | 0.780 (0.713 to 0.840) | 0.517 (0.434 to 0.573) |
| claude-opus-5.5 | jev | 0.619 (0.518 to 0.752) | 0.307 (0.258 to 0.365) |
| claude-opus-5.5 | qwen3.5-9b | 0.750 (0.606 to 0.888) | 0.469 (0.343 to 0.618) |
| claude-opus-5.5 | tfidf-logreg | 0.715 (0.636 to 0.797) | 0.525 (0.462 to 0.621) |
| gemini-3.1-pro | claude-opus-5.5 | 0.504 (0.398 to 0.606) | 0.456 (0.341 to 0.568) |
| gemini-3.1-pro | gpt-5.4 | 0.555 (0.480 to 0.606) | 0.517 (0.434 to 0.573) |
| gemini-3.1-pro | jev | 0.363 (0.317 to 0.418) | 0.307 (0.258 to 0.365) |
| gemini-3.1-pro | qwen3.5-9b | 0.514 (0.399 to 0.651) | 0.469 (0.343 to 0.618) |
| gemini-3.1-pro | tfidf-logreg | 0.552 (0.492 to 0.643) | 0.525 (0.462 to 0.621) |
| gpt-5.4 | claude-opus-5.5 | 0.688 (0.567 to 0.792) | 0.456 (0.341 to 0.568) |
| gpt-5.4 | gemini-3.1-pro | 0.822 (0.773 to 0.858) | 0.766 (0.694 to 0.815) |
| gpt-5.4 | jev | 0.539 (0.474 to 0.619) | 0.307 (0.258 to 0.365) |
| gpt-5.4 | qwen3.5-9b | 0.682 (0.566 to 0.809) | 0.469 (0.343 to 0.618) |
| gpt-5.4 | tfidf-logreg | 0.648 (0.581 to 0.750) | 0.525 (0.462 to 0.621) |
| jev | claude-opus-5.5 | 0.918 (0.826 to 0.965) | 0.456 (0.341 to 0.568) |
| jev | gemini-3.1-pro | 0.906 (0.875 to 0.929) | 0.766 (0.694 to 0.815) |
| jev | gpt-5.4 | 0.906 (0.858 to 0.936) | 0.517 (0.434 to 0.573) |
| jev | qwen3.5-9b | 0.923 (0.849 to 0.981) | 0.469 (0.343 to 0.618) |
| jev | tfidf-logreg | 0.795 (0.718 to 0.872) | 0.525 (0.462 to 0.621) |
| qwen3.5-9b | claude-opus-5.5 | 0.729 (0.568 to 0.839) | 0.456 (0.341 to 0.568) |
| qwen3.5-9b | gemini-3.1-pro | 0.838 (0.788 to 0.879) | 0.766 (0.694 to 0.815) |
| qwen3.5-9b | gpt-5.4 | 0.751 (0.653 to 0.833) | 0.517 (0.434 to 0.573) |
| qwen3.5-9b | jev | 0.605 (0.482 to 0.733) | 0.307 (0.258 to 0.365) |
| qwen3.5-9b | tfidf-logreg | 0.682 (0.587 to 0.780) | 0.525 (0.462 to 0.621) |
| tfidf-logreg | claude-opus-5.5 | 0.621 (0.491 to 0.718) | 0.456 (0.341 to 0.568) |
| tfidf-logreg | gemini-3.1-pro | 0.805 (0.744 to 0.846) | 0.766 (0.694 to 0.815) |
| tfidf-logreg | gpt-5.4 | 0.639 (0.557 to 0.695) | 0.517 (0.434 to 0.573) |
| tfidf-logreg | jev | 0.466 (0.396 to 0.534) | 0.307 (0.258 to 0.365) |
| tfidf-logreg | qwen3.5-9b | 0.610 (0.480 to 0.726) | 0.469 (0.343 to 0.618) |
