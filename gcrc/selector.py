"""Local-MDL, compact proposals, counterfactual rollout, and GCR-C gate."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .config import GCRCConfig, PacketConfig
from .model import MaskedPrior, batch_reconstruct
from .packet import feasible_candidates, tx_breakdown


@torch.no_grad()
def current_local_scores(
    model: MaskedPrior,
    tokens: torch.Tensor,
    known: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token-wise NLL, entropy, and posterior probabilities."""
    inputs = tokens.clone()
    inputs[~known] = model.mask_id
    log_probabilities = F.log_softmax(model(inputs), dim=-1)
    probabilities = log_probabilities.exp()
    self_nll = -log_probabilities.gather(2, tokens.unsqueeze(-1)).squeeze(-1) / math.log(2)
    entropy = -(probabilities * log_probabilities / math.log(2)).sum(dim=-1)
    return self_nll, entropy, probabilities


@torch.no_grad()
def candidate_gains_and_visual(
    model: MaskedPrior,
    tokens: torch.Tensor,
    known: torch.Tensor,
    candidates: Sequence[int],
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residual codelength and patch-reconstruction gains."""
    if not candidates:
        empty = torch.empty(0, device=tokens.device)
        return empty, empty
    if tokens.shape[0] != 1 or known.shape[0] != 1:
        raise ValueError("candidate proposal construction currently expects batch size 1")
    target_tokens = tokens[0]
    known_row = known[0]
    base = target_tokens.clone()
    base[~known_row] = model.mask_id
    base_log_prob = F.log_softmax(model(base.unsqueeze(0)), dim=-1)[0]
    unknown = ~known_row
    base_nll = -(base_log_prob[unknown].gather(1, target_tokens[unknown, None]).squeeze(1) / math.log(2)).sum()
    codebook = codebook.to(tokens.device, dtype=torch.float32)
    base_mean_patch = base_log_prob.exp() @ codebook
    true_patch = codebook[target_tokens]
    base_visual = (base_mean_patch[unknown] - true_patch[unknown]).square().mean(dim=1)

    rows = []
    for candidate in candidates:
        row = base.clone()
        row[int(candidate)] = target_tokens[int(candidate)]
        rows.append(row)
    candidate_batch = torch.stack(rows)
    all_nll: list[torch.Tensor] = []
    all_visual: list[torch.Tensor] = []
    for start in range(0, len(candidate_batch), 128):
        chunk = candidate_batch[start : start + 128]
        log_prob = F.log_softmax(model(chunk), dim=-1)
        for row_index, candidate in enumerate(candidates[start : start + len(chunk)]):
            remaining = unknown.clone()
            remaining[int(candidate)] = False
            nll = -(log_prob[row_index, remaining].gather(
                1, target_tokens[remaining, None]
            ).squeeze(1) / math.log(2)).sum()
            mean_patch = log_prob[row_index].exp() @ codebook
            visual = base_visual.sum() - (mean_patch[remaining] - true_patch[remaining]).square().mean(dim=1).sum()
            all_nll.append(nll)
            all_visual.append(visual)
    return base_nll - torch.stack(all_nll), torch.stack(all_visual)


def coverage_candidates(
    selected: Sequence[int],
    feasible: Sequence[int],
    n_tokens: int,
    count: int = 1,
) -> list[int]:
    """Select spatially complementary positions on a square token grid."""
    if not feasible:
        return []
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError("coverage candidates require a square token grid")
    coordinates = {index: (index // side, index % side) for index in range(n_tokens)}
    remaining = list(feasible)
    anchors = [int(index) for index in selected]
    chosen: list[int] = []
    for _ in range(min(count, len(remaining))):
        if not anchors:
            center = ((side - 1) / 2.0, (side - 1) / 2.0)
            candidate = max(
                remaining,
                key=lambda index: (coordinates[index][0] - center[0]) ** 2
                + (coordinates[index][1] - center[1]) ** 2,
            )
        else:
            candidate = max(
                remaining,
                key=lambda index: min(
                    (coordinates[index][0] - coordinates[anchor][0]) ** 2
                    + (coordinates[index][1] - coordinates[anchor][1]) ** 2
                    for anchor in anchors
                ),
            )
        chosen.append(candidate)
        anchors.append(candidate)
        remaining.remove(candidate)
    return chosen


def proposal_set(
    model: MaskedPrior,
    tokens: torch.Tensor,
    known: torch.Tensor,
    selected: Sequence[int],
    feasible: Sequence[int],
    codebook: torch.Tensor,
    config: GCRCConfig | None = None,
) -> dict[str, object]:
    """Build the final compact ``Top-3 Local + Top-5 Coverage`` proposal."""
    config = config or GCRCConfig()
    if not feasible:
        return {"candidates": [], "sources": {"local": [], "coverage": []}}
    local_scores, _, _ = current_local_scores(model, tokens, known)
    local_ranked = sorted(feasible, key=lambda index: float(local_scores[0, index]), reverse=True)
    coverage = coverage_candidates(selected, feasible, int(tokens.shape[1]), config.top_coverage)
    candidates: list[int] = []
    sources: dict[str, list[int]] = {"local": [], "coverage": []}
    for index in local_ranked[: config.top_local]:
        if index not in candidates:
            candidates.append(index)
            sources["local"].append(index)
    for index in coverage:
        if index not in candidates:
            candidates.append(index)
            sources["coverage"].append(index)
    return {
        "candidates": candidates,
        "sources": sources,
        "local_scores": local_scores[0].detach(),
        "local_ranked": local_ranked,
        "coverage_ranked": coverage,
    }


def local_mdl_sequence(
    model: MaskedPrior,
    tokens: torch.Tensor,
    budget_bits: int,
    protocol: PacketConfig | None = None,
) -> list[int]:
    """Baseline Local-MDL sequence under the exact packet budget."""
    protocol = protocol or PacketConfig()
    token_batch = tokens[:1]
    n_tokens = int(token_batch.shape[1])
    code_bits = int(math.ceil(math.log2(model.vocab)))
    known = torch.zeros(1, n_tokens, dtype=torch.bool, device=token_batch.device)
    selected: list[int] = []
    while True:
        feasible = feasible_candidates(selected, n_tokens, budget_bits, code_bits, protocol)
        if not feasible:
            return selected
        scores, _, _ = current_local_scores(model, token_batch, known)
        choice = max(feasible, key=lambda index: float(scores[0, index]))
        known[0, choice] = True
        selected.append(int(choice))


def _local_action(
    model: MaskedPrior,
    tokens: torch.Tensor,
    known: torch.Tensor,
    selected: Sequence[int],
    budget_bits: int,
    protocol: PacketConfig,
) -> tuple[int | None, list[int], torch.Tensor]:
    n_tokens = int(tokens.shape[1])
    code_bits = int(math.ceil(math.log2(model.vocab)))
    feasible = feasible_candidates(selected, n_tokens, budget_bits, code_bits, protocol)
    if not feasible:
        return None, [], torch.empty(0, device=tokens.device)
    scores, _, _ = current_local_scores(model, tokens, known)
    return max(feasible, key=lambda index: float(scores[0, index])), feasible, scores[0]


@torch.no_grad()
def q_rollout(
    model: MaskedPrior,
    tokens: torch.Tensor,
    original: torch.Tensor,
    codebook: torch.Tensor,
    base_selected: Sequence[int],
    candidates: Sequence[int],
    budget_bits: int,
    horizon: int | None = None,
    protocol: PacketConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Evaluate candidates with one action plus Local-MDL continuation.

    ``horizon=None`` continues until the packet budget is exhausted and is
    the full-budget ``Q_B`` evaluator used by the paper.
    """
    protocol = protocol or PacketConfig()
    if not candidates:
        empty = torch.empty(0, device=tokens.device)
        return empty, empty, []
    token_batch = tokens[:1].expand(len(candidates), -1).clone()
    n_tokens = int(tokens.shape[1])
    code_bits = int(math.ceil(math.log2(model.vocab)))
    selections = [list(base_selected) + [int(index)] for index in candidates]
    known = torch.zeros(len(candidates), n_tokens, dtype=torch.bool, device=tokens.device)
    for row, selected in enumerate(selections):
        known[row, torch.tensor(selected, dtype=torch.long, device=tokens.device)] = True
    continuation_steps = 0
    while horizon is None or continuation_steps < max(0, int(horizon) - 1):
        scores, _, _ = current_local_scores(model, token_batch, known)
        choices: list[int | None] = []
        any_choice = False
        for row, selected in enumerate(selections):
            feasible = feasible_candidates(selected, n_tokens, budget_bits, code_bits, protocol)
            if not feasible:
                choices.append(None)
                continue
            choices.append(max(feasible, key=lambda index: float(scores[row, index])))
            any_choice = True
        if not any_choice:
            break
        for row, choice in enumerate(choices):
            if choice is not None:
                selections[row].append(int(choice))
                known[row, int(choice)] = True
        continuation_steps += 1
    reconstructions = batch_reconstruct(model, token_batch, selections, codebook)
    target = original[:1].expand_as(reconstructions)
    mse = (reconstructions - target).square().mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    return psnr.detach(), mse.detach(), selections


def select_gcrc(
    model: MaskedPrior,
    codebook: torch.Tensor,
    image: torch.Tensor,
    tokens: torch.Tensor,
    budget_bits: int,
    nominal_bpp: float | None = None,
    config: GCRCConfig | None = None,
    protocol: PacketConfig | None = None,
    return_trace: bool = False,
) -> tuple[list[int], dict[str, object]]:
    """Run the deterministic GCR-C correction layer on one image.

    The Local-MDL action remains the fallback. At an eligible state, a compact
    proposal is evaluated with the same budget and Local continuation. At most
    ``max_interventions`` accepted replacements are made.
    """
    config = config or GCRCConfig()
    protocol = protocol or PacketConfig()
    token_batch = tokens[:1]
    original = image[:1]
    n_tokens = int(token_batch.shape[1])
    known = torch.zeros(1, n_tokens, dtype=torch.bool, device=token_batch.device)
    selected: list[int] = []
    interventions = 0
    counterfactual_calls = 0
    trace: list[dict[str, object]] = []
    while True:
        local, feasible, _ = _local_action(model, token_batch, known, selected, budget_bits, protocol)
        if local is None:
            break
        rate_allowed = nominal_bpp is None or any(abs(float(nominal_bpp) - rate) < 1e-9 for rate in config.trigger_rates)
        early_allowed = len(selected) / max(1, n_tokens) <= config.max_fraction
        gate_evaluated = rate_allowed and early_allowed and interventions < config.max_interventions
        choice = local
        best_advantage = 0.0
        proposal = [local]
        if gate_evaluated:
            proposal_info = proposal_set(model, token_batch, known, selected, feasible, codebook, config)
            proposal = list(proposal_info["candidates"])
            horizon = config.horizon
            q_values, _, _ = q_rollout(
                model, token_batch, original, codebook, selected, proposal,
                budget_bits, horizon, protocol,
            )
            counterfactual_calls += len(proposal)
            local_quality = float(q_values[proposal.index(local)])
            best_index = int(torch.argmax(q_values))
            best_advantage = float(q_values[best_index]) - local_quality
            if best_advantage > config.delta_db:
                choice = int(proposal[best_index])
                interventions += 1
        if return_trace:
            trace.append({
                "state_id": len(selected),
                "set_fraction": len(selected) / max(1, n_tokens),
                "local_action": int(local),
                "chosen": int(choice),
                "advantage_db": float(best_advantage),
                "gate_evaluated": int(gate_evaluated),
                "intervention": int(choice != local),
                "candidate_count": len(proposal),
            })
        known[0, choice] = True
        selected.append(int(choice))
    return selected, {
        "interventions": interventions,
        "counterfactual_calls": counterfactual_calls,
        "trace": trace,
    }


def selection_summary(selected: Sequence[int], n_tokens: int, code_bits: int, protocol: PacketConfig | None = None) -> dict[str, object]:
    """Return a serializable packet summary for a caller-supplied selection."""
    bits = tx_breakdown(selected, n_tokens, code_bits, protocol)
    return {"selected": [int(index) for index in selected], "tokens": len(selected), "packet": bits}
