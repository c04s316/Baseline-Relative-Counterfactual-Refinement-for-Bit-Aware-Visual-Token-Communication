"""Run GCR-C on caller-supplied tensors and write an optional JSON trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcrc import GCRCConfig, PacketConfig, load_masked_prior, select_gcrc, selection_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="caller-supplied [B,3,H,W] float tensor .pt")
    parser.add_argument("--tokens", type=Path, required=True, help="caller-supplied [B,N] token-index tensor .pt")
    parser.add_argument("--codebook", type=Path, required=True, help="caller-supplied [V,D] codebook tensor .pt")
    parser.add_argument("--checkpoint", type=Path, required=True, help="caller-supplied MaskedPrior state .pt")
    parser.add_argument("--budget-bits", type=int, required=True)
    parser.add_argument("--nominal-bpp", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=Path, default=None, help="optional output JSON; ignored unless supplied")
    parser.add_argument("--delta-db", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=None, help="use full-budget rollout when omitted")
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must contain a torch.Tensor")
    return value


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    images = load_tensor(args.images).float()
    tokens = load_tensor(args.tokens).long()
    codebook = load_tensor(args.codebook).float()
    if images.ndim != 4 or tokens.ndim != 2 or len(images) != len(tokens):
        raise ValueError("images and tokens must be batched tensors with matching first dimensions")
    model = load_masked_prior(args.checkpoint, device=device)
    protocol = PacketConfig()
    config = GCRCConfig(delta_db=args.delta_db, horizon=args.horizon)
    rows = []
    for index in range(len(images)):
        selected, stats = select_gcrc(
            model=model,
            codebook=codebook.to(device),
            image=images[index : index + 1].to(device),
            tokens=tokens[index : index + 1].to(device),
            budget_bits=args.budget_bits,
            nominal_bpp=args.nominal_bpp,
            config=config,
            protocol=protocol,
            return_trace=True,
        )
        row = selection_summary(selected, int(tokens.shape[1]), int(torch.ceil(torch.log2(torch.tensor(model.vocab))).item()), protocol)
        row.update({"image_index": index, "stats": stats, "device": str(device)})
        rows.append(row)
    payload = {"config": vars(args) | {"device": str(device)}, "rows": rows}
    payload["config"]["images"] = str(args.images)
    payload["config"]["tokens"] = str(args.tokens)
    payload["config"]["codebook"] = str(args.codebook)
    payload["config"]["checkpoint"] = str(args.checkpoint)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
