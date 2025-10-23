from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, cast
from typing_extensions import Annotated

import httpx
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, load_from_disk
from pydantic import BaseModel, Field, BeforeValidator
from pydantic_settings import BaseSettings, CliApp
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from client import client_config
from data import format_context_for_training, get_summary_from_generation
from persistent_kv_store import PersistentKVStore
from policy import SummaryBudget, SummaryBudgetChars, SummaryBudgetRatio
from process_dataset import DATASET_PATH
from rl_summarizer import evaluate_summaries

DEFAULT_REFERENCE_MODEL = "Qwen/Qwen3-1.7B-Base"
REFERENCE_EVAL_MICRO_BATCH = 1

MIN_CHARS = 2**13
MAX_CHARS = 2**14
CONTEXT_EPS = 500
MODEL_EVAL_MIN_PASS_RATE = 0.8


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    model_id: str
    reasoning: Literal["minimal", "low", "medium", "high"] | None = None
    max_output_tokens: int = 2048
    unique_key: str = ""

    @property
    def spec_id(self) -> str:
        return f"{self.model_id}::{self.unique_key}" if self.unique_key else self.model_id

    @property
    def path_safe_spec_id(self) -> str:
        return self.spec_id.replace("/", "__").replace(":", "_")


MODEL_SPECS: list[ModelSpec] = [
    # OpenAI
    ModelSpec("GPT-5 (Reasoning)", model_id="openai/gpt-5", unique_key="reasoning", reasoning="medium"),
    ModelSpec("GPT-5 (Instruct)", model_id="openai/gpt-5", unique_key="instruct", reasoning="minimal"),
    ModelSpec("GPT-5-mini (Reasoning)", model_id="openai/gpt-5-mini", unique_key="reasoning", reasoning="medium"),
    ModelSpec("GPT-5-mini (Instruct)", model_id="openai/gpt-5-mini", unique_key="instruct", reasoning="minimal"),
    ModelSpec("GPT-5-nano (Reasoning)", model_id="openai/gpt-5-nano", unique_key="reasoning", reasoning="medium"),
    ModelSpec("GPT-5-nano (Instruct)", model_id="openai/gpt-5-nano", unique_key="instruct", reasoning="minimal"),
    ModelSpec("GPT-4.1 (Instruct)", model_id="openai/gpt-4.1"),
    ModelSpec("GPT-4.1 Mini (Instruct)", model_id="openai/gpt-4.1-mini"),
    ModelSpec("GPT-4.1 Nano (Instruct)", model_id="openai/gpt-4.1-nano"),
    ModelSpec("GPT-3.5 Turbo (Instruct)", model_id="openai/gpt-3.5-turbo"),
    # Anthropic
    ModelSpec("Claude 4.5 Sonnet (Reasoning)", model_id="anthropic/claude-sonnet-4.5", unique_key="reasoning", reasoning="medium"),
    ModelSpec("Claude 4.5 Sonnet (Instruct)", model_id="anthropic/claude-sonnet-4.5", unique_key="instruct"),
    ModelSpec("Claude 4.5 Haiku (Reasoning)", model_id="anthropic/claude-haiku-4.5", unique_key="reasoning", reasoning="medium"),
    ModelSpec("Claude 4.5 Haiku (Instruct)", model_id="anthropic/claude-haiku-4.5", unique_key="instruct"),
    ModelSpec("Claude 4 Sonnet (Reasoning)", model_id="anthropic/claude-sonnet-4", unique_key="reasoning", reasoning="medium"),
    ModelSpec("Claude 4 Sonnet (Instruct)", model_id="anthropic/claude-sonnet-4", unique_key="instruct"),
    ModelSpec("Claude 3.7 Sonnet (Reasoning)", model_id="anthropic/claude-3-7-sonnet:thinking"),
    ModelSpec("Claude 3.7 Sonnet (Instruct)", model_id="anthropic/claude-3-7-sonnet"),
    ModelSpec("Claude 3 Haiku (Instruct)", model_id="anthropic/claude-3-haiku"),
    # Gemini
    ModelSpec("Gemini 2.5 Pro (Reasoning)", "google/gemini-2.5-pro", reasoning="medium"),
    ModelSpec(
        "Gemini 2.5 Flash (Reasoning)",
        model_id="google/gemini-2.5-flash",
        reasoning="medium",
        unique_key="reasoning",
    ),
    ModelSpec(
        "Gemini 2.5 Flash (Instruct)",
        model_id="google/gemini-2.5-flash",
        unique_key="instruct",
    ),
    ModelSpec(
        "Gemini 2.5 Flash Lite (Reasoning)",
        model_id="google/gemini-2.5-flash-lite",
        reasoning="medium",
        unique_key="reasoning",
    ),
    ModelSpec(
        "Gemini 2.5 Flash Lite (Instruct)",
        model_id="google/gemini-2.5-flash-lite",
        unique_key="instruct",
    ),
    ModelSpec("Gemini 2.0 Flash (Instruct)", model_id="google/gemini-2.0-flash-001"),
    # Deepseek
    ModelSpec("DeepSeek R1 (Reasoning)", model_id="deepseek/deepseek-r1"),
    ModelSpec(
        "DeepSeek V3.1 (Reasoning)",
        model_id="deepseek/deepseek-chat-v3.1",
        reasoning="medium",
        unique_key="reasoning",
    ),
    ModelSpec(
        "DeepSeek V3.1 (Instruct)",
        model_id="deepseek/deepseek-chat-v3.1",
        reasoning="medium",
        unique_key="instruct",
    ),
    # Qwen
    ModelSpec("Qwen3 235B A22B (Reasoning)", model_id="qwen/qwen3-235b-a22b-thinking-2507"),
    ModelSpec("Qwen3 235B A22B (Instruct)", model_id="qwen/qwen3-235b-a22b-2507"),
    ModelSpec("Qwen3 30B A3B (Reasoning)", model_id="qwen/qwen3-30b-a3b-thinking-2507"),
    ModelSpec("Qwen3 30B A3B (Instruct)", model_id="qwen/qwen3-30b-a3b-instruct-2507"),
    # GLM
    ModelSpec("GLM-4.6 (Reasoning)", model_id="z-ai/glm-4.6", reasoning="medium"),
    ModelSpec("GLM-4 32B (Instruct)", model_id="z-ai/glm-4-32b"),
    # Kimi
    ModelSpec("Kimi K2 (Instruct)", model_id="moonshotai/kimi-k2-0905"),
    # LLaMA
    ModelSpec("Llama 3.3 70B Instruct", model_id="meta-llama/llama-3.3-70b-instruct"),
    ModelSpec("Llama 3.1 8B Instruct", model_id="meta-llama/llama-3.1-8b-instruct"),
    # Grok
    ModelSpec("Grok 4 (Reasoning)", model_id="x-ai/grok-4"),
    ModelSpec("Grok 4 Fast (Instruct)", model_id="x-ai/grok-4-fast"),
    ModelSpec("Grok 3 Mini (Instruct)", model_id="x-ai/grok-3-mini"),
    # Mistral
    ModelSpec("Mistral Magistral Medium 2506 (Reasoning)", model_id="mistralai/magistral-medium-2506"),
    ModelSpec("Mistral Medium 3.1 (Instruct)", model_id="mistralai/mistral-medium-3.1"),
    ModelSpec("Mistral Small 3.2 24B (Instruct)", model_id="mistralai/mistral-small-3.2-24b-instruct"),
]


def parse_summary_budget(summary_budget_input: str | SummaryBudget) -> SummaryBudget:
    if isinstance(summary_budget_input, SummaryBudget):
        return summary_budget_input
    type_, value = summary_budget_input.split(".")
    if type_ == "chars":
        return SummaryBudgetChars[value]
    elif type_ == "ratio":
        return SummaryBudgetRatio[value]
    raise ValueError(f"Invalid summary budget type: {type_}")


class BenchmarkConfig(BaseSettings, cli_parse_args=True, cli_kebab_case=True):
    limit: int = Field(description="Maximum number of documents to process", default=8, gt=0)
    summary_budget: Annotated[SummaryBudget, BeforeValidator(parse_summary_budget)] = Field(
        description="Summary budget", default=SummaryBudgetChars.LARGE
    )
    samples_per_doc: int = Field(description="Number of samples to generate per document", default=1, gt=0)
    cache_dir: Path = Field(description="Directory to cache model outputs", default=Path("cache"))
    force: bool = Field(description="Force re-generation of summaries", default=False)
    reference_model: str = Field(description="Reference model to use for comparison", default=DEFAULT_REFERENCE_MODEL)
    max_concurrent_requests: int = Field(description="Maximum number of concurrent requests", default=50, gt=0)
    max_retries: int = Field(description="Maximum number of retries for failed requests", default=3, gt=0)
    timeout: float = Field(description="Timeout for HTTP requests", default=20.0, gt=1)
    aggregate_only: bool = Field(description="Only aggregate existing summaries", default=False)


@dataclass(frozen=True)
class DocumentRecord:
    dataset_index: int
    id: str
    text: str


class SummaryRecord(BaseModel):
    model_spec: ModelSpec
    document_id: str
    sample_index: int = 0
    summary: str | None = None
    raw_response: str | None = None
    error: str | None = None


class BenchmarkRecord(BaseModel):
    document_index: int
    document_id: str
    sample_index: int = 0
    summary: str | None = None
    raw_response: str | None = None
    reward: float | None = None
    summary_length_chars: int = 0
    error: str | None = None


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_reference_model(model_name: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    device = detect_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "cuda":
        base_model.to(device)
    base_model.eval()
    return tokenizer, base_model


async def generate_summary(
    spec: ModelSpec,
    document: DocumentRecord,
    summary_budget: SummaryBudget,
    max_retries: int,
    sample_index: int,
) -> SummaryRecord:
    extra_args = {}
    if spec.reasoning is not None:
        extra_args["reasoning"] = {"effort": spec.reasoning}

    if isinstance(summary_budget, SummaryBudgetChars):
        budget_chars = summary_budget.value
    else:
        budget_chars = int(len(document.text) * summary_budget.value)
    prompt = format_context_for_training(document.text, budget_chars, is_base_model=False)
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.post(
                    url=f"{client_config.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {client_config.openrouter_key}"},
                    json={
                        "model": spec.model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 1.0,
                        "top_p": 1.0,
                        **extra_args,
                    },
                )
            response.raise_for_status()
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
            summary = get_summary_from_generation(raw_text, is_base_model=False)
            return SummaryRecord(
                model_spec=spec,
                document_id=document.id,
                sample_index=sample_index,
                summary=summary,
                raw_response=raw_text,
            )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, httpx.HTTPStatusError):
                print(f"Request error occurred during generation: {exc.response.text}")
            else:
                print(f"Error type: {type(exc).__name__}")
                print(f"Unexpected error occurred during generation: {str(exc)}")
            if attempt + 1 >= max_retries:
                return SummaryRecord(
                    model_spec=spec,
                    document_id=document.id,
                    sample_index=sample_index,
                    error=str(exc),
                )
            await asyncio.sleep(backoff)
            backoff *= 2
    return SummaryRecord(model_spec=spec, document_id=document.id, sample_index=sample_index, error="Exceeded retry limit")


def docs_iter(config: BenchmarkConfig) -> Iterator[DocumentRecord]:
    dataset = cast(Dataset, load_from_disk(DATASET_PATH)["validation"])
    idx = 0
    for elem in dataset:
        if idx == config.limit:
            break

        document_text: str = elem["text"]  # type: ignore[reportArgumentType]

        # TODO: consolidate document length filtering into the dataset preprocessing
        if isinstance(config.summary_budget, SummaryBudgetChars):
            summary_chars = config.summary_budget
        else:
            summary_chars = len(document_text) * config.summary_budget
        document_max_chars = MAX_CHARS - summary_chars - CONTEXT_EPS
        if len(document_text) > document_max_chars:
            continue
        document_id = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
        document = DocumentRecord(dataset_index=idx, id=document_id, text=document_text)
        yield document
        idx += 1


def aggregate_results(config: BenchmarkConfig) -> None:
    print("\nAggregate results for processed documents:")
    benchmark_store = PersistentKVStore(BenchmarkRecord, config.cache_dir / "benchmark")
    spec_id_to_rewards: dict[str, dict[str, list[float | None]]] = {
        spec.spec_id: defaultdict(list) for spec in MODEL_SPECS + [TRUNCATED_DOC_SPEC]
    }

    # Gather all rewards
    for document in docs_iter(config):
        for spec in MODEL_SPECS + [TRUNCATED_DOC_SPEC]:
            for sample_index in range(max(1, config.samples_per_doc)):
                record = benchmark_store[spec.path_safe_spec_id, config.summary_budget.name, document.id, str(sample_index)]

                if not record:
                    print(f"WARNING: missing cached document for ({spec.spec_id}, sample={sample_index}, {document.id})")
                    continue

                spec_id_to_rewards[spec.spec_id][document.id].append(record.reward)

    spec_id_to_reward_and_coverage: dict[str, tuple[float, float]] = dict()
    spec_id_to_display_name = {spec.spec_id: spec.display_name for spec in MODEL_SPECS + [TRUNCATED_DOC_SPEC]}

    # Apply statistics
    for spec in MODEL_SPECS + [TRUNCATED_DOC_SPEC]:
        no_valid_samples = 0
        spec_rewards: list[float] = []
        for doc_id, doc_rewards in spec_id_to_rewards.get(spec.spec_id, {}).items():
            # We want to exclude failed summaries
            filtered_rewards = []
            for reward in doc_rewards:
                if reward is not None and reward != 0:
                    filtered_rewards.append(reward)

            if len(filtered_rewards) == 0:
                no_valid_samples += 1
                continue

            doc_mean_reward = statistics.fmean(filtered_rewards)
            spec_rewards.append(doc_mean_reward)

        if len(spec_rewards) + no_valid_samples == 0:
            continue
        elif not spec_rewards:
            print(f"- {spec.display_name} ({spec.spec_id}): no successful summaries")
            continue

        if spec_rewards:
            mean_reward = statistics.fmean(spec_rewards)
            print(
                f"- {spec.display_name} ({spec.spec_id}): mean_reward={mean_reward:.4f} over {len(spec_rewards)}/{config.limit} documents"
            )
            spec_id_to_reward_and_coverage[spec.spec_id] = (mean_reward, len(spec_rewards))
        else:
            print(f"- {spec.display_name} ({spec.spec_id}): no successful summaries")

    spec_ids_at_high_coverage = set(
        spec_id
        for spec_id, (reward, coverage) in spec_id_to_reward_and_coverage.items()
        if coverage > config.limit * MODEL_EVAL_MIN_PASS_RATE
    )
    spec_ids_sorted_by_reward = sorted(
        spec_id_to_reward_and_coverage.keys(), key=lambda spec_id: spec_id_to_reward_and_coverage[spec_id][0], reverse=True
    )
    for spec_id in spec_ids_sorted_by_reward:
        if spec_id in spec_ids_at_high_coverage:
            print(
                f"- ({spec_id}): mean_reward={spec_id_to_reward_and_coverage[spec_id][0]:.4f} over {spec_id_to_reward_and_coverage[spec_id][1]}/{config.limit} documents"
            )

    # Plot results
    plot_spec_ids = [
        spec_id for spec_id in spec_ids_sorted_by_reward if spec_id != TRUNCATED_DOC_SPEC.spec_id and spec_id in spec_ids_at_high_coverage
    ]
    if not plot_spec_ids:
        print("\nNo non-baseline models met the pass-rate threshold for plotting.")
        return

    plot_labels = [spec_id_to_display_name[spec_id] for spec_id in plot_spec_ids]
    plot_rewards = [spec_id_to_reward_and_coverage[spec_id][0] for spec_id in plot_spec_ids]

    height = 0.3 * len(plot_spec_ids)
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(plot_labels, plot_rewards, color="#4c78a8")
    ax.margins(x=0.1, y=0.01)
    ax.invert_yaxis()  # best model appears at the top
    ax.set_xlabel("Mean reward")
    ax.set_title("Aggregate Model Performance (higher is better)")

    ax.bar_label(
        bars,
        labels=[f"{reward:.3f}" for reward in plot_rewards],
        padding=3,
        fontsize=8,
    )

    plt.tight_layout()
    plot_path = config.cache_dir / "aggregate_results.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


async def summarization_worker(
    config: BenchmarkConfig,
    store: PersistentKVStore[SummaryRecord],
    job_queue: asyncio.Queue[tuple[DocumentRecord, ModelSpec, int] | None,],
) -> None:
    while True:
        job = await job_queue.get()
        if job is None:
            job_queue.task_done()
            break
        document, spec, sample_index = job

        # We still want to re-attempt generating summaries for errors
        cache_key = (spec.path_safe_spec_id, config.summary_budget.name, document.id, str(sample_index))
        if (rec := store[cache_key]) and rec.error is None:
            job_queue.task_done()
            continue

        result = await generate_summary(spec, document, config.summary_budget, config.max_retries, sample_index)
        store[cache_key] = result

        job_queue.task_done()


TRUNCATED_DOC_SPEC = ModelSpec("Truncated Document Baseline", "baseline/truncated-document")


async def benchmark_models(config: BenchmarkConfig) -> None:
    summary_budget = config.summary_budget

    print(f"Preparing to evaluate up to {config.limit} documents.")
    print(f"Benchmarking {len(MODEL_SPECS)} model(s) with summary budget {summary_budget}.")

    tokenizer, reference_model = load_reference_model(config.reference_model)

    # Key structure: (spec_id, summary_budget, doc_id, sample_index)
    summary_store = PersistentKVStore(SummaryRecord, config.cache_dir / "summary")

    # Generate and cache summaries
    docs_queue: asyncio.Queue[tuple[DocumentRecord, ModelSpec, int] | None,] = asyncio.Queue(maxsize=config.max_concurrent_requests)
    summarization_workers = [
        asyncio.create_task(summarization_worker(config, summary_store, docs_queue)) for _ in range(config.max_concurrent_requests)
    ]
    for doc in docs_iter(config):
        for spec in MODEL_SPECS:
            for sample_index in range(config.samples_per_doc):
                await docs_queue.put((doc, spec, sample_index))

        # Include truncated doc baseline
        for sample_index in range(config.samples_per_doc):
            cache_key = (TRUNCATED_DOC_SPEC.path_safe_spec_id, summary_budget.name, doc.id, str(sample_index))
            if summary_store[cache_key] is not None:
                continue

            if isinstance(summary_budget, SummaryBudgetChars):
                budget_chars = summary_budget.value
            else:
                budget_chars = int(len(doc.text) * summary_budget.value)

            summary_store[cache_key] = SummaryRecord(
                model_spec=TRUNCATED_DOC_SPEC,
                document_id=doc.id,
                sample_index=sample_index,
                summary=doc.text[: int(budget_chars * 0.7)],
                raw_response=doc.text,
            )

    await docs_queue.join()
    for _ in range(config.max_concurrent_requests):
        await docs_queue.put(None)
    await asyncio.gather(*summarization_workers, return_exceptions=True)

    summary_store.flush()

    # Run benchmark and cache results
    processed = 0
    benchmark_store = PersistentKVStore(BenchmarkRecord, config.cache_dir / "benchmark")
    for doc in docs_iter(config):
        summary_records_for_doc: list[tuple[SummaryRecord, ModelSpec, int]] = []
        for spec in MODEL_SPECS + [TRUNCATED_DOC_SPEC]:
            for sample_index in range(config.samples_per_doc):
                cache_key = (spec.path_safe_spec_id, summary_budget.name, doc.id, str(sample_index))
                if not config.force and (benchmark := benchmark_store[cache_key]) and benchmark.error is None and benchmark.reward != 0:
                    continue
                summary_record = summary_store[cache_key]
                if not summary_record:
                    print("WARNING: Missing summary record for", (spec.spec_id, summary_budget, doc.id, sample_index))
                    continue
                summary_records_for_doc.append((summary_record, spec, sample_index))

        if not summary_records_for_doc:
            continue

        if isinstance(summary_budget, SummaryBudgetChars):
            budget_chars = summary_budget.value
        else:
            budget_chars = int(len(doc.text) * summary_budget.value)

        summaries_for_doc: list[str | None] = []
        for rec, _, _ in summary_records_for_doc:
            if rec.summary and 1.5 * budget_chars > len(rec.summary) > budget_chars:
                summaries_for_doc.append(rec.summary[: int(budget_chars * 0.7)])
            else:
                summaries_for_doc.append(rec.summary)

        rewards_tensor = evaluate_summaries(
            reference_model,
            tokenizer,
            doc.text,
            summaries_for_doc,
            summary_budget=budget_chars,
            micro_batch_size=REFERENCE_EVAL_MICRO_BATCH,
            is_training=False,
        )
        rewards = rewards_tensor.tolist()

        for idx, (summary_rec, spec, sample_index) in enumerate(summary_records_for_doc):
            if summary_rec.error is None:
                summary_len = len(summary_rec.summary) if summary_rec.summary else 0
                record = BenchmarkRecord(
                    document_index=doc.dataset_index,
                    document_id=doc.id,
                    sample_index=sample_index,
                    summary=summary_rec.summary,
                    raw_response=summary_rec.raw_response,
                    reward=rewards[idx],
                    summary_length_chars=summary_len,
                )
                print(f"  {summary_rec.model_spec.display_name}: reward={rewards[idx]:.4f} summary_length={summary_len}")
            else:
                record = BenchmarkRecord(
                    document_index=doc.dataset_index,
                    document_id=doc.id,
                    sample_index=sample_index,
                    raw_response=summary_rec.raw_response,
                    error=summary_rec.error,
                )
                print(f"  {summary_rec.model_spec.display_name}: failed -> {summary_rec.error}")
            benchmark_store[spec.path_safe_spec_id, summary_budget.name, doc.id, str(sample_index)] = record
        processed += len(summary_records_for_doc)

    benchmark_store.flush()

    if processed == 0:
        print("No documents processed.")
        return


if __name__ == "__main__":
    config = CliApp.run(BenchmarkConfig)
    if not config.aggregate_only:
        asyncio.run(benchmark_models(config))
    else:
        print("Skipping benchmarking, only generating aggregated statistics")
    aggregate_results(config)
