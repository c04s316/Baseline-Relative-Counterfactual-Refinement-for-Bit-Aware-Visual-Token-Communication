# GCR-C

Source-only reference implementation of **Gated Counterfactual Refinement for Communication (GCR-C)** for bit-aware visual-token communication.

This directory contains the method code only. It intentionally contains no datasets, images, generated results, result tables, logs, cached files, pretrained weights, or training checkpoints. The `.gitignore` file also blocks these artifacts from being committed accidentally.

## Repository layout

```text
code/
├── gcrc/
│   ├── config.py          # packet and selector configuration
│   ├── model.py           # masked-token prior and reconstruction interface
│   ├── metrics.py         # PSNR and tensor-only SSIM diagnostic
│   ├── packet.py          # bitmap/gap-list syntax and bit accounting
│   ├── representation.py  # patchify, codebook fitting, and quantization
│   └── selector.py        # Local-MDL, proposal, Q_B rollout, and GCR-C gate
├── scripts/
│   ├── run_selector.py    # run on user-supplied tensors/checkpoint
│   └── smoke_test.py      # dependency-only method smoke test
├── tests/                 # reserved for lightweight unit tests
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

Use Python 3.10--3.12. Install a PyTorch build that matches the target CPU or CUDA runtime, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the source-only smoke test from this directory:

```bash
python scripts/smoke_test.py
```

The smoke test creates tensors in memory and writes no files.

## Method interface

GCR-C keeps Local-MDL as the default action. At an eligible state it constructs the final compact proposal `Top-3 Local ∪ Top-5 Coverage`. Each candidate is taken once and the remaining budget is completed with the same Local-MDL continuation. The candidate replaces the baseline when its full-budget reconstruction advantage is positive (`delta_db=0`); the default policy accepts at most one intervention per image.

The defaults match the final paper configuration: full-budget `Q_B` continuation, `rho_max=0.30` eligibility, intervention rates `0.20` and `0.32` bpp, and at most one accepted intervention. The earlier P4 set-gain union is not the released default method.

The packet protocol charges one 32-bit header, one position description (adaptive bitmap or gap list), the token payload, a 16-bit CRC, and the configured FEC expansion. Position encoding and feasibility checks are implemented in `gcrc.packet` so selection and rate accounting use the same protocol.

## Running on caller-supplied artifacts

The repository does not ship data or a model. Prepare your own tensors and checkpoint:

- `images.pt`: float tensor with shape `[B, 3, H, W]` and values in `[0, 1]`;
- `tokens.pt`: integer tensor with shape `[B, N]`;
- `codebook.pt`: float tensor with shape `[V, C * patch_size**2]`;
- `masked_prior.pt`: a `gcrc.model.MaskedPrior` state dictionary.

Then run:

```bash
python scripts/run_selector.py \
  --images /path/to/images.pt \
  --tokens /path/to/tokens.pt \
  --codebook /path/to/codebook.pt \
  --checkpoint /path/to/masked_prior.pt \
  --budget-bits 204 \
  --nominal-bpp 0.20 \
  --output outputs/gcrc_selections.json
```

The output path is user-selected and is ignored by Git. Omit `--output` to print the JSON trace instead. Omit `--horizon` to use the full-budget `Q_B` continuation; pass an integer to inspect a shorter rollout.

For a library call:

```python
import torch

from gcrc import GCRCConfig, PacketConfig, load_masked_prior, select_gcrc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_masked_prior("/path/to/masked_prior.pt", device=device)
selected, stats = select_gcrc(
    model=model,
    codebook=torch.load("/path/to/codebook.pt", map_location=device),
    image=torch.load("/path/to/images.pt", map_location=device)[:1],
    tokens=torch.load("/path/to/tokens.pt", map_location=device)[:1],
    budget_bits=204,
    nominal_bpp=0.20,
    config=GCRCConfig(),
    protocol=PacketConfig(),
    return_trace=True,
)
```

## Reproducibility and scope

Record the PyTorch version, CUDA version, device, packet configuration, selector configuration, and the command line used for each run. The source release deliberately does not assert that a checkpoint, dataset split, generated table, or figure is included. Dataset preparation, prior/codebook training, channel simulation, and statistical analysis should be kept in a separate experiment workspace or added later as independently reviewed scripts.

The implementation is deterministic conditional on the supplied model, codebook, tensors, and configuration. The full-budget evaluator is computation-intensive; batching and learned value approximations are possible extensions but are not silently substituted for the paper method.
