from __future__ import annotations

from torch import Tensor

from dataclasses import dataclass

from enum import Enum


class SummaryBudgetChars(int, Enum):
    """Summary budget is measured in number of characters"""

    EXTRA_SMALL = 64
    SMALL = 128
    MEDIUM = 256
    LARGE = 512
    EXTRA_LARGE = 1024


class SummaryBudgetRatio(float, Enum):
    """Summary budget is measured as a ratio of the document length"""

    EXTRA_SMALL = 0.01
    SMALL = 0.02
    MEDIUM = 0.05
    LARGE = 0.1
    EXTRA_LARGE = 0.2


SummaryBudget = SummaryBudgetChars | SummaryBudgetRatio


@dataclass
class Rollouts:
    tokens_BGT: Tensor  # Int32
    generation_mask_BGT: Tensor  # Int8
    old_policy_log_probs_BGT: Tensor  # Float
    summary_budgets_B: Tensor  # Int32
    # TODO: add advantages as a field for logical consolidation

    def get_group_slice(self, group_idx: int, group_size: int) -> Rollouts:
        return Rollouts(
            tokens_BGT=self.tokens_BGT[:, group_idx : group_idx + group_size, :],
            generation_mask_BGT=self.generation_mask_BGT[:, group_idx : group_idx + group_size, :],
            old_policy_log_probs_BGT=self.old_policy_log_probs_BGT[:, group_idx : group_idx + group_size, :],
            summary_budgets_B=self.summary_budgets_B,
        )
