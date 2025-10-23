# SQUISH benchmark

**SQUISH** (Summarization QUantifiable Information Saving Heuristic) is a **quantitative** benchmark for evaluating the ability of LLMs to compress information from documents into summaries.

## Overview
SQUISH adopts the perspective that an LLM can be thought of as a lossy decompression program, that generates some output given a prompt. Using this perspective, you can model a summary as a semantically encoded version of the original text. The goal of an optimal summary is to maximize the amount of information transferred to the original document, while being smaller than the original document.

This can be explicitly measured by using a reference base model LLM with the following high level steps:
1. Generate summaries for a document with some max summary size
2. Reject summaries over the maximum summary size
3. Measure the cross entropy in a target document `doc_loss`
4. Measure the cross entropy of the same target document prefixed with the summary `doc_loss_given_summary`
5. Find the information saved by including the summary as context `info_saved = doc_loss - doc_loss_given_summary`
6. Normalize by full document cross entropy

This method can be used on arbitrary text, obviating the need for hyper-curated datasets.

In addition to the benchmarking code, there is also a script included to RL a small model on this task using GRPO (-ish... it's all just policy gradient after all).

The most relevant previous work is in [Shannon Score (2021)](https://arxiv.org/pdf/2103.10918). The primary difference between SQUISH and Shannon Score is that SQUISH includes a requirement for summaries to be smaller than the original document, which allows for RL to improve summary quality without encouraging the trivial case of the document being its own summary.

## Results
You can run this benchmark at multiple different ratios of summary size, which is the primary parameter of interest in the SQUISH benchmark.

**Coming Soon!**
## How to reproduce results

The easiest way to use this repo is with `uv`. Install it using [UV's documentation](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Generate dataset
`uv run src/process_dataset.py`

### 2. Setup OpenRouter credentials
Add OpenRouter credentials in a `.env` file at the root of the repo (it's in `.gitignore` already). It needs to have the key `OPENROUTER_KEY`.

### 3. Run benchmarking script
`uv run src/benchmark_models.py`

### 4. Run RL script
`uv run src/rl_summarizer.py`
## FAQ
#### Why characters and not tokens?
3 different content size metrics were considered for this benchmark:
- Characters
- Tokens
- Information

Information and tokens are both valid ways of measuring the "size" of a summary, but is much harder to control for as a generation constraint across different models. For those looking to actually use this benchmark as an internal metric (and not compare to other model families), these alternative summary measures are still valuable. Tokens in particular has a great advantage of minimizing compute and KV-cache within a model family.
#### Why relative summary length?
Although it is desirable to define a fixed summary size in many applications (such as placement in UI elements like notifications), using a relative length for benchmarking minimizes bias in the value metric due to document length.

## Future work
#### Model ensemble
Currently only Qwen3-1.7B-base is being used as a reference model. However, it would be preferable to include a few different model families (and perhaps different model sizes), introduce variance to the biases inherent in pretraining data mixes.
#### Multiple languages
Currently the data corpus is only in English. It would be preferable to include other languages.
#### More document topics
Currently the data corpus only includes news-like data. It would be preferable to include document like: essays, transcripts, speeches, etc... Especially long form text.
#### Better handling of long contexts
Current models have demonstrated issues with in-context learning as context length increases substantially. As in-context learning is a primary measured factor of this benchmark, it would be preferable to update the reference model when such capabilities improve.
