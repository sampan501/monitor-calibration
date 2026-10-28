# monitor-calibration

This repo checks how well different kinds of model report confidence on one safety-monitor task: deciding whether a user message to a chatbot is harmful. Each model gives a probability P(harmful) for every prompt in a fixed sample of the ToxicChat test split. The repo then compares calibration, selective prediction, and how many harmful prompts each model catches when only a small share of benign traffic can be sent for review.

## Kinds of model

- A baseline: TF-IDF features with logistic regression, fit on the ToxicChat train split.
- Local open-weight models through Hugging Face transformers. The probability is P(yes) / (P(yes) + P(no)) from the next-token distribution. Safety classifiers such as Llama Guard can be read the same way with their own answer words.
- API models through an OpenAI-compatible client, which also covers Ollama and vLLM servers. These use token logprobs by default, or a stated probability with `--mode verbal`.
- Models that state a probability only: Claude through the Anthropic SDK, and Jev through the TypeSafe SDK.

## Metrics

- AUROC: does the model rank harmful prompts above benign ones?
- Brier score: how close are the probabilities to the labels on average?
- ECE and the reliability plot: when the model says 0.8, are about 80% of those prompts harmful?
- AURC and the risk-coverage plot: if the model abstains on its least confident prompts, how fast does the error rate on the rest fall?
- Catch rate at alpha: if the threshold is set so that about alpha of benign prompts get flagged (the audit budget), what share of harmful prompts is flagged? The threshold is a split conformal quantile of benign scores.
- Audit catch rate spread: the 5th to 95th percentile of the held-out catch rate over many random half splits, where one half sets the threshold and the other half is scored.
- Share of splits with FPR above alpha: how often the realized false positive rate on the held-out half goes over the budget.
- Permutation p: is the catch rate better than a monitor with no signal? Labels are permuted and the threshold is set again inside each permutation.
- Repeat SD: the mean per-prompt standard deviation across repeated calls, for models that are not deterministic.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys are read from the environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `TYPESAFE_API_KEY`. Use `--api-key-env` to read a different variable. Llama Guard and some other Hugging Face models need you to accept their license and log in with `huggingface-cli login`.

## Scoring

Each run scores the same test sample (`--n 2000 --seed 0` by default) and appends one JSON line per prompt and repeat to `results/<name>.jsonl`. A rerun skips pairs that are already done, so an interrupted or failed run can be resumed with the same command.

```
python run.py --scorer baseline --name tfidf-logreg
python run.py --scorer hf --model Qwen/Qwen2.5-7B-Instruct --name qwen2.5-7b
python run.py --scorer hf --model meta-llama/Llama-Guard-3-1B --raw --yes unsafe --no safe --prefix '\n\n' --name llama-guard-3-1b
python run.py --scorer openai --model gpt-4.1-mini --name gpt-4.1-mini
python run.py --scorer openai --model llama3.1 --base-url http://localhost:11434/v1 --name llama3.1-ollama
python run.py --scorer anthropic --model <model-id> --name claude
python run.py --scorer jev --name jev --repeats 3
```

## Analysis

```
python analyze.py
python analyze.py --alpha 0.01 --reps 1000
```

This reads every `results/*.jsonl` file, averages repeats per prompt, and keeps only the prompts that every model scored. It prints a markdown table, writes it to `results/summary.md`, and saves `figures/reliability.png`, `figures/risk_coverage.png`, and `figures/catch_rate.png`.

## Notes

- A stated probability and a token probability mean different things. A token probability comes from the model's output distribution, while a stated probability is text the model writes. The two can differ in calibration even for the same model.
- With 2,000 prompts at about 7% prevalence, each half split has about 70 harmful prompts. That is why the held-out catch rate has a wide spread.
- Split conformal thresholds control the false positive rate on average over splits, not on every split. Some splits will go over alpha, and the table reports how often.
- By default the loader drops prompts whose OpenAI moderation score for sexual/minors is above 0.1 (21 of 5,083 test prompts) before sampling, and `--keep-all` turns this off.
- ToxicChat is licensed CC BY-NC 4.0. Results files store only an id, the repeat index, the probability, and the label. They never store prompt text.
- Check each provider's terms before publishing results from its models. Jev results are ignored by git until those terms are confirmed.
