"""Configuration objects for the packet protocol and GCR-C policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PacketConfig:
    """Bit-accurate core packet configuration used by the paper method."""

    header_bits: int = 32
    crc_bits: int = 16
    fec_ratio: float = 1.25
    position_mode: str = "adaptive_min"


@dataclass(frozen=True)
class GCRCConfig:
    """Deterministic selector settings.

    ``horizon=None`` is the full-budget evaluator used in the paper. Set an
    integer to study a shorter Local-MDL continuation.
    """

    max_fraction: float = 0.30
    trigger_rates: tuple[float, ...] = (0.20, 0.32)
    max_interventions: int = 1
    delta_db: float = 0.0
    horizon: int | None = None
    top_local: int = 3
    top_coverage: int = 5
