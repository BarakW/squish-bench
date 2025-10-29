import re

from evaluation import SummaryBudget


def format_context_for_training(document: str, budget: SummaryBudget, is_base_model: bool) -> str:
    budget_chars = int(len(document) * budget)
    maybe_prefill = "\n<summary>" if is_base_model else ""
    res = f"""Generate a summary of the provided document in strictly fewer than {budget_chars} characters. Target a length of around {int(budget_chars * 0.9)} characters. The document is within a <document> tag and the summary is within a <summary> tag.

<document>{document}</document>{maybe_prefill}"""
    return res


# There are 2 <summary> tags, the first one is for the instructions, the second one is for the actual summary
# TODO: get generated tokens from base model generations so there's only *one* regex
SUMMARY_EXTRACTOR_CHAT_MODEL = re.compile(r"<summary>(?P<summary>[\s\S]*?)</summary>")
SUMMARY_EXTRACTOR_BASE_MODEL = re.compile(r"<summary>[\s\S]*<summary>(?P<summary>[\s\S]*?)</summary>")


def get_summary_from_generation(text: str, is_base_model: bool) -> str | None:
    if is_base_model:
        match = SUMMARY_EXTRACTOR_BASE_MODEL.search(text)
    else:
        match = SUMMARY_EXTRACTOR_CHAT_MODEL.search(text)
    return match.group("summary") if match else None
