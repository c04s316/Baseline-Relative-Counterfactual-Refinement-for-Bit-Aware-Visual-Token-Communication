"""Dependency-only smoke test; it creates no files or experiment artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcrc import GCRCConfig, MaskedPrior, PacketConfig, encode_positions, decode_positions, position_roundtrip_tests, select_gcrc, tx_breakdown
from gcrc.representation import unpatchify


def main() -> None:
    torch.manual_seed(7)
    protocol = PacketConfig(fec_ratio=1.0)
    assert decode_positions(encode_positions([0, 3, 7], 16), 16) == [0, 3, 7]
    position_roundtrip_tests(n_tokens=16, trials=20, seed=7)

    model = MaskedPrior(vocab=8, n_tokens=16, dim=16, layers=1, heads=2).eval()
    codebook = torch.rand(8, 3 * 2 * 2)
    tokens = torch.randint(0, 8, (1, 16))
    image = unpatchify(codebook[tokens])
    selected, stats = select_gcrc(
        model=model,
        codebook=codebook,
        image=image,
        tokens=tokens,
        budget_bits=200,
        nominal_bpp=0.20,
        config=GCRCConfig(delta_db=0.0, horizon=1, top_coverage=2),
        protocol=protocol,
        return_trace=True,
    )
    assert len(selected) == len(set(selected))
    assert int(tx_breakdown(selected, 16, 3, protocol)["total"]) <= 200
    assert "interventions" in stats
    print(f"smoke test passed: selected={len(selected)} tokens, interventions={stats['interventions']}")


if __name__ == "__main__":
    main()
