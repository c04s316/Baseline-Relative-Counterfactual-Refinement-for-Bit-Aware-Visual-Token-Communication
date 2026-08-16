"""Source-only reference implementation of GCR-C."""

from .config import GCRCConfig, PacketConfig
from .model import MaskedPrior, batch_reconstruct, load_masked_prior, reconstruct, train_masked_prior
from .metrics import psnr, ssim
from .packet import decode_positions, encode_positions, feasible_candidates, position_roundtrip_tests, tx_breakdown
from .representation import fit_codebook, patchify, quantize, unpatchify
from .selector import (
    coverage_candidates,
    current_local_scores,
    proposal_set,
    q_rollout,
    selection_summary,
    select_gcrc,
)

__version__ = "0.1.0"

__all__ = [
    "GCRCConfig",
    "PacketConfig",
    "MaskedPrior",
    "batch_reconstruct",
    "load_masked_prior",
    "reconstruct",
    "train_masked_prior",
    "psnr",
    "ssim",
    "decode_positions",
    "encode_positions",
    "feasible_candidates",
    "position_roundtrip_tests",
    "tx_breakdown",
    "fit_codebook",
    "patchify",
    "quantize",
    "unpatchify",
    "coverage_candidates",
    "current_local_scores",
    "proposal_set",
    "q_rollout",
    "selection_summary",
    "select_gcrc",
]
