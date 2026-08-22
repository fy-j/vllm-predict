# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed.parallel_state import get_eplb_group


class PredictiveLoadSnapshot:
    """Read-only predicted logical-expert load for one source MoE forward."""

    def __init__(
        self,
        target_gate: torch.nn.Module,
        target_router: object,
        num_experts: int,
    ):
        self.target_gate = target_gate
        self.target_router = target_router
        self.num_experts = num_experts
        self.snapshot: torch.Tensor | None = None
        self._work: object | None = None

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits, _ = self.target_gate(hidden_states)
        _, ids = self.target_router._select_experts(  # type: ignore[attr-defined]
            hidden_states, logits
        )
        return torch.bincount(ids.reshape(-1).to(torch.long), minlength=self.num_experts)

    def start(self, counts: torch.Tensor) -> None:
        group = get_eplb_group().device_group
        self.snapshot = torch.empty(
            (group.size(), self.num_experts), dtype=counts.dtype, device=counts.device
        )
        self._work = torch.distributed.all_gather_into_tensor(
            self.snapshot, counts.contiguous(), group=group, async_op=True
        )

    def finish(self) -> torch.Tensor | None:
        if self._work is not None:
            self._work.wait()  # type: ignore[attr-defined]
            self._work = None
        return self.snapshot
