# monitor-calibration

This repo checks how well different kinds of model report confidence on one safety-monitor task: deciding whether a user message to a chatbot is harmful. Each model gives a probability P(harmful) for every prompt in two datasets: a fixed sample of 2,000 prompts from the ToxicChat test split, and the filtered WildGuardTest set of 1,670 prompts, which adds adversarial (jailbreak-style) prompts and harm subcategories. The repo then compares calibration, selective prediction, and how many harmful prompts each model catches when only a small share of benign traffic can be sent for review.

## Kinds of model

- A baseline: TF-IDF features with logistic regression, fit on the ToxicChat train split for ToxicChat and on WildGuardTrain for WildGuardTest.
- Local open-weight models through Hugging Face transformers. The probability is P(yes) / (P(yes) + P(no)) from the next-token distribution. Safety classifiers such as Llama Guard can be read the same way with their own answer words.
- API models through an OpenAI-compatible client, which also covers Gemini, Ollama, and vLLM servers. These use token logprobs by default, or a stated probability with `--mode verbal`. The commercial runs below all use `--mode verbal`, since gpt-5.4 returns only the top token's logprob and Gemini returns none.
- Claude through the Anthropic SDK, which can only state a probability.
- Jev through the TypeSafe SDK, which returns its own probability for a yes or no question (mode native).

## Metrics

- n (positive): the number of prompts scored, with the number labeled positive in parentheses (toxic on ToxicChat, harmful on WildGuardTest).
- Mode: where the probability comes from. Token means next-token probabilities, stated means a number the model writes, and native means the model's own probability output.
- Non-answers: prompts where the model refused or returned no probability. They count as p = 1.0 (flagged) for the catch rate, FPR, audit, and bootstrap columns, are left out of AUROC, Brier, and ECE, and enter AURC as abstentions ranked least confident. A separate table splits each model's non-answers by label, and on WildGuardTest also by adversarial and plain.
- Distinct scores: the number of unique probability values on the prompts the model answered, after averaging repeats. A small count means many prompts share a score.
- AUROC: does the model rank harmful prompts above benign ones?
- Brier score: how close are the probabilities to the labels on average?
- ECE and the reliability plot: when the model says 0.8, are about 80% of those prompts harmful? The plot leaves out bins with fewer than 10 prompts, and marker area is proportional to bin count.
- AURC and the risk-coverage plot: if the model abstains on its least confident prompts, how fast does the error rate on the rest fall? Prompts with equal confidence are put in random order, and the curve and AURC are averaged over 50 such orders. The plot's y-axis is scaled so every model's whole curve is visible.
- Catch rate at alpha: if the threshold is set so that about alpha of benign prompts get flagged (the audit budget), what share of harmful prompts is flagged? The rule is split conformal and randomized among ties. For a score s, with n benign calibration scores, a of them above s and b equal to s, the prompt is flagged with probability clip((alpha (n + 1) - a) / (b + 1), 0, 1). The tables use that probability as the expected flag instead of a random draw, so the results are deterministic. Averaged over calibration sets, the false positive rate is alpha even when many prompts share a score, and `python metrics.py` checks this on synthetic tied scores. The audit, bootstrap, pairwise, subgroup, and conditional miss tables all use this rule.
- Catch strict: the older rule, which flags a prompt only when its score is strictly above the conformal quantile of benign scores. It never goes over alpha on average, but when many benign prompts share the top scores it can spend far less than alpha, and nothing at all when more than alpha of benign prompts sit at the model's highest score.
- Benign at max score: the share of benign prompts at the model's highest score, with non-answers counted as 1.0. When it is above alpha, the strict rule flags nothing, and only randomizing among the tied prompts uses the budget.
- Audit catch rate spread: the 5th to 95th percentile of the held-out catch rate over 500 random half splits, where one half sets the threshold and the other half is scored.
- Share of splits with FPR above alpha: how often the expected false positive rate on the held-out half goes over the budget. The randomized rule aims at alpha on average, so about half the splits land above it by design. The column shows how much a single split can miss, not a failure of the rule.
- Pairwise differences in catch rate, ECE, and Brier: a paired bootstrap over prompts. Each of 2,000 resamples draws the toxic and the benign prompts with replacement, separately, and uses the same drawn prompts for every model. Each model's 2% randomized conformal rule is set again on the resampled benign prompts. ECE and Brier are computed on the resampled prompts each model answered. For every pair and each of the three measures, the second table gives the full-sample difference, the 2.5 to 97.5 percentile interval of the bootstrap differences, and a two-sided bootstrap p-value with Holm correction across the 15 pairs of that measure. Each measure is corrected on its own, not across all 45 comparisons together. With 2,000 resamples and 15 pairs the smallest possible Holm p-value is 0.015.
- Repeat SD: the mean per-prompt standard deviation across repeated calls, for models that are not deterministic.

A permutation test of each catch rate against a monitor with no signal, which permutes labels and sets the threshold again inside each permutation, gives p = 0.001 for every model, the smallest value 1,000 permutations allow.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys are read from the environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (through `--api-key-env`), and `TYPESAFE_API_KEY`. Use `--api-key-env` to read a different variable. Llama Guard, some other Hugging Face models, and WildGuardMix need you to accept their terms and log in with `huggingface-cli login`.

## Scoring

Each run scores the same test sample (`--n 2000 --seed 0` by default on ToxicChat) and appends one JSON line per prompt and repeat to `results/<dataset>/<name>.jsonl`. A rerun skips pairs that are already done, so an interrupted or failed run can be resumed with the same command.

```
python run.py --scorer baseline --name tfidf-logreg
python run.py --scorer hf --model Qwen/Qwen3.5-9B --name qwen3.5-9b
python run.py --scorer hf --model meta-llama/Llama-Guard-3-1B --raw --yes unsafe --no safe --prefix '\n\n' --name llama-guard-3-1b
python run.py --scorer openai --model gpt-5.4-2026-03-05 --mode verbal --name gpt-5.4
python run.py --scorer openai --model gemini-3.1-pro-preview --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GOOGLE_API_KEY --mode verbal --name gemini-3.1-pro
python run.py --scorer openai --model llama3.1 --base-url http://localhost:11434/v1 --name llama3.1-ollama
python run.py --scorer anthropic --model claude-opus-5-5 --name claude-opus-5.5
python run.py --scorer jev --name jev --repeats 3
```

Model versions are pinned to what was current on 2026-10-28 rather than to "latest" aliases: Qwen3.5-9B at Hugging Face commit c202236 (the newest Qwen that fits in 36 GB; Qwen3.8 starts at 27B), the gpt-5.4 snapshot from 2026-03-05, Claude Opus 5.5, and Gemini 3.1 Pro preview (what `gemini-pro-latest` resolved to that day). Jev has no dated name, and the runs requested `jev-latest`. The version it resolved to was not saved, so the date recorded for Jev is the run date, 2026-10-28, not a release date. The local model reads the next token with thinking turned off. Claude runs with adaptive thinking at effort low.

## Analysis

```
python analyze.py
python analyze.py --alpha 0.01 --reps 1000
```

This reads every `results/<dataset>/*.jsonl` file (`--dataset toxicchat` by default), counts non-answers as p = 1.0, averages repeats per prompt, and keeps the prompts that every model scored. It prints the main and pairwise tables, writes them to `results/<dataset>/summary.md`, and saves `reliability.png`, `risk_coverage.png`, and `catch_rate.png` in `figures/<dataset>/`. `--exclude-refusals` keeps only the prompts every model answered. `--boot` sets the number of bootstrap resamples. The mode column comes from `results/<dataset>/models.json`, which `run.py` fills in with each run's model ID, scorer, mode, dataset, and first run date.

`--private` takes run-name prefixes to leave out of a second set of outputs in `results/<dataset>/public/` and `figures/<dataset>/public/`. It is empty by default, so no second set is written.

## ToxicChat results

These results were run on 2026-10-28 on 2,000 prompts from the filtered test split, 145 of them toxic, with alpha = 0.02. The models are tfidf-logreg, Qwen/Qwen3.5-9B at commit c202236, gpt-5.4-2026-03-05, gemini-3.1-pro-preview, claude-opus-5-5, and jev-latest (run on 2026-10-28).

| model | mode | n (positive) | non-answers | distinct scores | AUROC | Brier | ECE | AURC | catch@0.02 | catch strict | benign at max score | audit catch 5-95% | share FPR>0.02 | repeat SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-opus-5.5 | stated | 2000 (145) | 17 | 40 | 0.988 | 0.0244 | 0.0232 | 0.0020 | 0.684 | 0.683 | 0.006 | 0.578 to 0.807 | 0.512 | n/a |
| gemini-3.1-pro | stated | 2000 (145) | 1 | 13 | 0.926 | 0.0282 | 0.0271 | 0.0175 | 0.818 | 0.793 | 0.007 | 0.747 to 0.881 | 0.444 | n/a |
| gpt-5.4 | stated | 2000 (145) | 0 | 21 | 0.974 | 0.0409 | 0.0331 | 0.0106 | 0.640 | 0.600 | 0.000 | 0.540 to 0.743 | 0.522 | n/a |
| jev | native | 2000 (145) | 0 | 186 | 0.982 | 0.0269 | 0.0187 | 0.0023 | 0.712 | 0.710 | 0.000 | 0.640 to 0.814 | 0.500 | 0.0021 |
| qwen3.5-9b | token | 2000 (145) | 0 | 1979 | 0.959 | 0.0392 | 0.0195 | 0.0053 | 0.531 | 0.531 | 0.000 | 0.432 to 0.608 | 0.508 | n/a |
| tfidf-logreg | native | 2000 (145) | 0 | 1924 | 0.912 | 0.0451 | 0.0480 | 0.0125 | 0.566 | 0.566 | 0.000 | 0.487 to 0.648 | 0.464 | n/a |

| model A | model B | catch A - B | 95% interval | Holm p | ECE A - B | 95% interval | Holm p | Brier A - B | 95% interval | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-pro | qwen3.5-9b | +0.287 | +0.185 to +0.405 | 0.015 | +0.0076 | -0.0037 to +0.0175 | 1.000 | -0.0110 | -0.0167 to -0.0049 | 0.015 |
| gemini-3.1-pro | tfidf-logreg | +0.252 | +0.130 to +0.370 | 0.015 | -0.0208 | -0.0313 to -0.0097 | 0.015 | -0.0169 | -0.0234 to -0.0098 | 0.015 |
| gemini-3.1-pro | gpt-5.4 | +0.178 | +0.072 to +0.281 | 0.026 | -0.0059 | -0.0161 to +0.0022 | 0.959 | -0.0127 | -0.0197 to -0.0058 | 0.021 |
| jev | qwen3.5-9b | +0.181 | +0.090 to +0.314 | 0.026 | -0.0008 | -0.0078 to +0.0091 | 1.000 | -0.0123 | -0.0158 to -0.0084 | 0.015 |
| gemini-3.1-pro | claude-opus-5.5 | +0.135 | +0.028 to +0.232 | 0.110 | +0.0039 | -0.0053 to +0.0094 | 1.000 | +0.0038 | -0.0002 to +0.0084 | 0.355 |
| claude-opus-5.5 | qwen3.5-9b | +0.153 | +0.026 to +0.315 | 0.230 | +0.0037 | -0.0041 to +0.0135 | 1.000 | -0.0148 | -0.0191 to -0.0104 | 0.015 |
| jev | tfidf-logreg | +0.145 | +0.021 to +0.297 | 0.288 | -0.0293 | -0.0365 to -0.0172 | 0.015 | -0.0182 | -0.0232 to -0.0125 | 0.015 |
| gpt-5.4 | qwen3.5-9b | +0.109 | +0.005 to +0.248 | 0.368 | +0.0136 | +0.0026 to +0.0243 | 0.144 | +0.0017 | -0.0057 to +0.0093 | 1.000 |
| gemini-3.1-pro | jev | +0.107 | -0.004 to +0.187 | 0.448 | +0.0084 | -0.0019 to +0.0146 | 0.959 | +0.0013 | -0.0035 to +0.0064 | 1.000 |
| jev | gpt-5.4 | +0.071 | -0.010 to +0.178 | 0.552 | -0.0144 | -0.0215 to -0.0038 | 0.099 | -0.0140 | -0.0198 to -0.0083 | 0.015 |
| claude-opus-5.5 | tfidf-logreg | +0.117 | -0.023 to +0.277 | 0.552 | -0.0248 | -0.0324 to -0.0125 | 0.024 | -0.0207 | -0.0252 to -0.0160 | 0.015 |
| claude-opus-5.5 | gpt-5.4 | +0.043 | -0.063 to +0.169 | 1.000 | -0.0099 | -0.0189 to +0.0015 | 0.736 | -0.0165 | -0.0233 to -0.0101 | 0.015 |
| jev | claude-opus-5.5 | +0.028 | -0.087 to +0.153 | 1.000 | -0.0045 | -0.0118 to +0.0034 | 1.000 | +0.0025 | -0.0016 to +0.0067 | 0.860 |
| gpt-5.4 | tfidf-logreg | +0.074 | -0.057 to +0.216 | 1.000 | -0.0149 | -0.0259 to -0.0028 | 0.120 | -0.0042 | -0.0120 to +0.0047 | 0.951 |
| tfidf-logreg | qwen3.5-9b | +0.035 | -0.100 to +0.185 | 1.000 | +0.0285 | +0.0164 to +0.0376 | 0.015 | +0.0059 | +0.0011 to +0.0108 | 0.084 |

Non-answers by label

| model | harmful | benign |
|---|---|---|
| claude-opus-5.5 | 6 | 11 |
| gemini-3.1-pro | 1 | 0 |
| gpt-5.4 | 0 | 0 |
| jev | 0 | 0 |
| qwen3.5-9b | 0 | 0 |
| tfidf-logreg | 0 | 0 |

![Reliability](figures/toxicchat/reliability.png)
![Risk-coverage](figures/toxicchat/risk_coverage.png)
![Catch rate](figures/toxicchat/catch_rate.png)

The ranking metric and the catch rate order the models differently. Gemini has the second lowest AUROC (0.926) and the highest catch rate at a 2% audit budget (0.818), while Claude has the highest AUROC (0.988) and a catch rate of 0.684. AUROC averages over every threshold, and the catch rate depends only on the top of the benign score distribution. The bootstrap does not separate Gemini from Claude on catch rate after Holm correction (difference +0.135, interval +0.028 to +0.232, Holm p = 0.110), and the bootstrap here does not cover AUROC.

The top models cannot be separated at this sample size. Gemini, Jev, and Claude differ by at most 0.135 in catch rate, and no pair among them has a Holm p-value below 0.11. Four pairs fall below 0.05 after correction: Gemini over Qwen and over tfidf-logreg (Holm p = 0.015 for both, the smallest value 2,000 resamples allow), Gemini over GPT-5.4 (0.026), and Jev over Qwen (0.026).

No model is clearly best calibrated. Jev has the lowest ECE (0.0187) and Claude the lowest Brier score (0.0244), but the bootstrap separates neither from the closest model: Jev against Qwen on ECE differs by -0.0008 (interval -0.0078 to +0.0091, Holm p = 1.000), and Jev against Claude on Brier by +0.0025 (interval -0.0016 to +0.0067, Holm p = 0.860). The clear calibration result is at the bottom: tfidf-logreg has a higher ECE than Gemini, Jev, Claude, and Qwen (Holm p at most 0.024).

Stated probabilities are coarse, as the distinct scores column shows: Gemini, GPT-5.4, and Claude use far fewer values than Jev, Qwen, and the baseline. Many prompts share the same score, which leaves the order within those groups to chance in AURC and puts the conformal threshold at a tie. The strict rule, which flags only scores above the threshold, gives Gemini 0.793 and GPT-5.4 0.600. Randomizing among the tied prompts raises these to 0.818 and 0.640 and moves the other models by at most 0.002.

Claude's 17 non-answers are refusals by the API (16 in the cyber category, 1 in general harms), and 11 of them are benign prompts. Counted as flags, the benign ones take up part of Claude's 2% benign budget: on the 1,982 prompts every model answered (`--exclude-refusals`), Claude's catch rate is 0.779 instead of 0.684. As abstentions they come last in its risk-coverage curve.

## WildGuardTest

WildGuardTest is the test set of WildGuardMix (Han et al., 2024), `allenai/wildguardmix` config `wildguardtest` on Hugging Face. It is gated: you have to accept AI2's terms on the dataset page and log in before it will download. It is licensed ODC-By. Each prompt has a harmfulness label (`prompt_harm_label`), a flag for whether it is an adversarial (jailbreak-style) prompt, and a subcategory. A prompt counts as harmful when its label is "harmful", and the 26 of 1,725 prompts with no label are dropped, leaving 754 harmful and 945 benign. The labels come from three annotators per item, with Fleiss kappa 0.55 on prompt harmfulness, and the authors used GPT-4 to audit them.

WildGuardTest has no moderation scores of its own, so `moderate.py` sends each prompt once to OpenAI's moderation endpoint (omni-moderation-latest) and caches the sexual/minors score per id in `data/`. The loader applies the same 0.1 cutoff from that cache, and `--keep-all` turns it off. The filter removes 29 of the 1,699 labeled prompts, 20 of them harmful, which leaves 1,670 prompts: 734 harmful (326 adversarial, 408 plain) and 936 benign. WildGuardTest runs use the whole filtered test set rather than a sample. The baseline is fit on all of WildGuardTrain (config `wildguardtrain`) with the same filter, which removes 2,166 of its 86,759 prompts (1,390 harmful) and leaves 84,593. Train rows whose prompt matches a test prompt after normalizing case and whitespace are also dropped: 3 rows, all copies of one test prompt.

```
python moderate.py --split test
python moderate.py --split train
python run.py --dataset wildguardtest --scorer baseline --name tfidf-logreg
python run.py --dataset wildguardtest --scorer hf --model Qwen/Qwen3.5-9B --name qwen3.5-9b
python run.py --dataset wildguardtest --scorer openai --model gpt-5.4-2026-03-05 --mode verbal --name gpt-5.4
python run.py --dataset wildguardtest --scorer openai --model gemini-3.1-pro-preview --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GOOGLE_API_KEY --mode verbal --name gemini-3.1-pro
python run.py --dataset wildguardtest --scorer anthropic --model claude-opus-5-5 --name claude-opus-5.5
python run.py --dataset wildguardtest --scorer jev --name jev --repeats 3
python analyze.py --dataset wildguardtest
```

On WildGuardTest, `analyze.py` also reports pairwise bootstrap comparisons of catch rate on the adversarial prompts only, with Holm correction across the 15 pairs. It then runs the tests below with hyppo (Panda et al., 2024), which are in `tests_hyppo.py`. The tables give only Holm-corrected p-values from these tests, with Holm correction across models (or across pairs, for the third) within each test, and no test statistics, since the size of a Dcorr statistic does not measure the strength of dependence. The size of each effect is shown instead by catch rates and false positive rates under the randomized 2% rule set on all benign prompts, each with a 95% interval from the same paired bootstrap. Non-answers enter these tests as p = 1.0, as in the catch rate.

- Do harmful prompts in different subcategories get different scores from the same model? This is the Dcorr k-sample test on each model's scores for harmful prompts, grouped by subcategory, using the subcategories with at least 20 harmful prompts. Next to it, a table gives each model's catch rate in each of those subcategories.
- Does a model score adversarial and plain prompts differently? This is the two-sample Dcorr test, run separately on harmful prompts and on benign prompts. Next to its p-values, the table gives the catch rate on adversarial and on plain harmful prompts, and the false positive rate on adversarial and on plain benign prompts. With two groups, the Dcorr k-sample test is equivalent to the energy distance test (Shen and Vogelstein, 2021), and on one-dimensional scores the energy distance is twice the integrated squared difference between the two groups' distribution functions.
- How much do two monitors' scores move together on harmful prompts? For each pair of models, this is the Dcorr independence test on their scores for harmful prompts. It says only whether the scores are dependent.

A separate table describes how two monitors overlap at the operating point: for each ordered pair, P(B misses | A misses) on harmful prompts, with each model under its own randomized 2% rule, next to B's overall miss rate. Misses are expected misses, and the randomization is taken to be independent across models. If B's misses were unrelated to A's, the two numbers would be about equal. The heatmap in `figures/wildguardtest/conditional_miss.png` shows the same quantity. This is a description of overlap at one threshold, not a test of whether one monitor adds information beyond another.

The hyppo tests use the default `auto=True`, so with more than 20 samples the p-value comes from hyppo's chi-square approximation to the permutation null rather than from permutations. `reps=1000` and a fixed `random_state` are passed for the case of 20 or fewer samples.

## WildGuardTest results

These results were run on 2026-10-28 on all 1,670 prompts of the filtered test set, 734 of them harmful, with alpha = 0.02 and the same model IDs and modes as the ToxicChat runs. The full output, including the adversarial pairwise table and the hyppo tables, is in `results/wildguardtest/summary.md`.

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

![Reliability](figures/wildguardtest/reliability.png)
![Risk-coverage](figures/wildguardtest/risk_coverage.png)
![Catch rate](figures/wildguardtest/catch_rate.png)
![Conditional miss](figures/wildguardtest/conditional_miss.png)

Gemini and GPT-5.4 state probabilities too coarse to hit a 2% budget without randomizing among ties. Gemini states 1.0 for 6.4% of benign prompts, more than three times the budget, so the strict rule flags nothing and catches 0.000 of harmful prompts although its AUROC is 0.882. GPT-5.4 states 0.99 for 4.3% of benign prompts and 1.0 for none, so the strict rule flags only its 1.0 prompts and catches 0.278 at a realized false positive rate of 0.000. Flagging a random share of the tied prompts spends the budget and raises Gemini to 0.234 and GPT-5.4 to 0.483. These are expected rates: in use they mean flagging a random subset of the prompts given the same score. The other four models change by at most 0.030 between the two rules. The distinct scores column shows how few values Gemini and GPT-5.4 use compared with the other models.

Jev has the highest catch rate (0.693), and the bootstrap separates it from every other model at Holm p = 0.015, the smallest value 2,000 resamples allow. It is ahead of Claude by +0.149 (interval +0.041 to +0.247) and of Qwen by +0.162 (+0.056 to +0.303). Claude (0.544), Qwen (0.531), GPT-5.4 (0.483), and the baseline (0.475) cannot be separated from each other (Holm p = 1.000 for all six pairs), and all of them are above Gemini (Holm p = 0.015). Jev also has the lowest Brier score (0.0799), lower than every other model at Holm p = 0.015. On adversarial harmful prompts alone, Jev's lead holds over Gemini, GPT-5.4, and the baseline (Holm p = 0.015) but not over Qwen (0.135) or Claude (1.000).

Nearly all of the false positive budget goes to adversarial benign prompts. Of the 936 benign prompts, 450 are adversarial. Under each model's rule, the false positive rate on adversarial benign prompts is 0.037 to 0.040 for the five models other than the baseline and 0.032 for the baseline, against 0.000 to 0.003 and 0.008 on plain benign prompts, and the bootstrap intervals for the two groups do not overlap for any model. The Dcorr two-sample test finds that adversarial and plain benign prompts get different scores for every model (Holm p < 0.001). On harmful prompts the gap at the threshold is smaller: catch rates on adversarial and plain prompts differ by at most 0.05 for Claude, Gemini, Jev, and Qwen. GPT-5.4 (0.401 against 0.548) and the baseline (0.408 against 0.529) catch fewer adversarial prompts, but their intervals overlap, and no paired test of that gap is run.

Misses overlap. For every ordered pair, the chance that B misses a harmful prompt given that A missed it is above B's overall miss rate in point estimate. In 18 of the 30 pairs, including every pair with Gemini as A, the interval overlaps B's overall miss rate. The three examples below are among the pairs where it does not. When Jev misses, Claude also misses 0.918 of the time (interval 0.826 to 0.965) and Qwen 0.923 (0.849 to 0.981), against overall miss rates of 0.456 and 0.469. When Claude misses, Jev misses 0.619 of the time (0.518 to 0.752), against its overall 0.307 (0.258 to 0.365). These describe one operating point and do not test whether a second monitor adds information.

Every model scores harmful prompts differently across subcategories (Dcorr k-sample, Holm p < 0.001 for all six), and private information about individuals (81 harmful prompts) is the weak spot. It has the lowest catch rate of the 14 subcategories for Claude (0.219), GPT-5.4 (0.231), Jev (0.309), and Qwen (0.197), and the second lowest for Gemini (0.156, after copyright) and the baseline (0.235, after "others"). The intervals are per subcategory, and each model's private-information interval overlaps that of at least one other subcategory, so the bootstrap does not show that it is lower than every other subcategory. Every model's interval for it ends at or below 0.433.

The ECE values fall into two groups. Jev (0.0239), Claude (0.0291), and the baseline (0.0324) cannot be separated from each other (Holm p = 1.000), and neither can Gemini (0.1225), GPT-5.4 (0.1229), and Qwen (0.1172). Every model in the first group has a lower ECE than every model in the second (Holm p = 0.015). ECE leaves out non-answers, which removes 120 of Claude's harmful prompts from its calibration measures.

Claude returned no probability for 127 prompts: 120 harmful (60 adversarial, 60 plain) and 7 benign, all of them adversarial. Gemini returned none for 2 adversarial harmful prompts, and the other models answered every prompt. Its 7 benign non-answers are the 0.007 of benign prompts at its highest score, and they take up 0.007 of its 0.02 budget.

Because refusals count as flags, about 0.16 of Claude's 0.544 catch rate on WildGuardTest comes from refusing rather than scoring, and its calibration columns leave those prompts out.

## Notes

- A stated probability and a token probability mean different things. A token probability comes from the model's output distribution, while a stated probability is text the model writes. The two can differ in calibration even for the same model.
- With 2,000 prompts at about 7% prevalence, each half split has about 70 harmful prompts. That is why the held-out catch rate has a wide spread.
- Split conformal thresholds control the false positive rate on average over splits, not on every split. Some splits will go over alpha, and the table reports how often.
- By default the loader drops prompts whose OpenAI moderation score for sexual/minors is above 0.1 (21 of 5,083 test prompts) before sampling, and `--keep-all` turns this off. Results are on the filtered split, which removes about 5% of the toxic prompts (18 of the 21 dropped), likely among the easiest to catch, so catch rates may be a little lower than on the full split.
- ToxicChat is licensed CC BY-NC 4.0. Results files store only an id, the repeat index, the probability, and the label, plus the adversarial flag and subcategory on WildGuardTest. They never store prompt text, and neither does the moderation cache.
- Check each provider's terms before publishing results from its models.
- WildGuardMix is licensed ODC-By and is gated on Hugging Face.

## References

- Han, S., Rao, K., Ettinger, A., Jiang, L., Lin, B. Y., Lambert, N., Choi, Y., and Dziri, N. (2024). WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs. NeurIPS Datasets and Benchmarks.
- Panda, S., Palaniappan, S., Xiong, J., Bridgeford, E. W., Mehta, R., Shen, C., and Vogelstein, J. T. (2024). hyppo: A multivariate hypothesis testing Python package. arXiv:1907.02088.
- Shen, C., and Vogelstein, J. T. (2021). The exact equivalence of distance and kernel methods in hypothesis testing. AStA Advances in Statistical Analysis, 105, 385-403.
