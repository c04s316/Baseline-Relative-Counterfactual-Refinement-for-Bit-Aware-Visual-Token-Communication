"""Bit-accurate packet accounting and position syntax."""

from __future__ import annotations

import math
import random
from typing import Sequence

from .config import PacketConfig


def fixed_bits(n_values: int) -> int:
    return max(1, int(math.ceil(math.log2(max(2, n_values)))))


def gamma_bits(value: int) -> int:
    if value < 1:
        raise ValueError("Elias-gamma values must be positive")
    return 2 * int(math.floor(math.log2(value))) + 1


def gamma_encode(value: int) -> str:
    code = format(int(value), "b")
    return "0" * (len(code) - 1) + code


def gamma_decode(bits: str, cursor: int) -> tuple[int, int]:
    zeros = 0
    while cursor + zeros < len(bits) and bits[cursor + zeros] == "0":
        zeros += 1
    if cursor + 2 * zeros + 1 > len(bits):
        raise ValueError("truncated Elias-gamma code")
    start = cursor + zeros
    value = int(bits[start : start + zeros + 1], 2)
    return value, start + zeros + 1


def _normalise(selected: Sequence[int], n_tokens: int) -> list[int]:
    values = sorted(set(int(index) for index in selected))
    if any(index < 0 or index >= n_tokens for index in values):
        raise ValueError("position outside token grid")
    return values


def gap_position_bits(selected: Sequence[int], n_tokens: int) -> int:
    """Length excluding the one-bit bitmap/gap mode flag."""
    values = _normalise(selected, n_tokens)
    total = fixed_bits(n_tokens + 1)
    if values:
        total += fixed_bits(n_tokens)
        total += sum(gamma_bits(right - left) for left, right in zip(values, values[1:]))
    return total


def position_encoding_mode(
    selected: Sequence[int], n_tokens: int, mode: str = "adaptive_min"
) -> str:
    if mode not in {"bitmap", "gap", "adaptive_min"}:
        raise ValueError(f"unknown position mode: {mode}")
    if mode == "adaptive_min":
        bitmap_length = 1 + n_tokens
        gap_length = 1 + gap_position_bits(selected, n_tokens)
        return "bitmap" if bitmap_length <= gap_length else "gap"
    return mode


def encode_positions(selected: Sequence[int], n_tokens: int, mode: str = "adaptive_min") -> str:
    values = _normalise(selected, n_tokens)
    actual = position_encoding_mode(values, n_tokens, mode)
    if actual == "bitmap":
        present = set(values)
        return "0" + "".join("1" if index in present else "0" for index in range(n_tokens))
    count_bits = fixed_bits(n_tokens + 1)
    output = "1" + format(len(values), f"0{count_bits}b")
    if values:
        position_bits = fixed_bits(n_tokens)
        output += format(values[0], f"0{position_bits}b")
        output += "".join(gamma_encode(right - left) for left, right in zip(values, values[1:]))
    return output


def decode_positions(bitstream: str, n_tokens: int) -> list[int]:
    if not bitstream:
        raise ValueError("empty position bitstream")
    if bitstream[0] == "0":
        if len(bitstream) != 1 + n_tokens:
            raise ValueError("invalid bitmap length")
        return [index for index, flag in enumerate(bitstream[1:]) if flag == "1"]
    if bitstream[0] != "1":
        raise ValueError("unknown position mode flag")
    cursor = 1
    count_bits = fixed_bits(n_tokens + 1)
    if cursor + count_bits > len(bitstream):
        raise ValueError("truncated cardinality field")
    count = int(bitstream[cursor : cursor + count_bits], 2)
    cursor += count_bits
    if count > n_tokens:
        raise ValueError("invalid cardinality")
    if count == 0:
        if cursor != len(bitstream):
            raise ValueError("trailing bits for empty position set")
        return []
    position_bits = fixed_bits(n_tokens)
    if cursor + position_bits > len(bitstream):
        raise ValueError("truncated first position")
    first = int(bitstream[cursor : cursor + position_bits], 2)
    cursor += position_bits
    if first >= n_tokens:
        raise ValueError("invalid first position")
    values = [first]
    for _ in range(count - 1):
        gap, cursor = gamma_decode(bitstream, cursor)
        next_value = values[-1] + gap
        if next_value >= n_tokens:
            raise ValueError("decoded position outside token grid")
        values.append(next_value)
    if cursor != len(bitstream):
        raise ValueError("trailing bits in gap-list stream")
    return values


def position_roundtrip_tests(n_tokens: int = 64, trials: int = 1000, seed: int = 0) -> dict[str, object]:
    rng = random.Random(seed)
    cases: list[list[int]] = [[], [0], [n_tokens - 1], list(range(n_tokens)), list(range(0, n_tokens, 2))]
    for _ in range(trials):
        size = rng.randrange(n_tokens + 1)
        cases.append(sorted(rng.sample(range(n_tokens), size)))
    lengths: dict[str, list[int]] = {key: [] for key in ("bitmap", "gap", "adaptive_min")}
    for values in cases:
        for mode in lengths:
            encoded = encode_positions(values, n_tokens, mode)
            if decode_positions(encoded, n_tokens) != values:
                raise AssertionError(f"position roundtrip failed: {values}")
            lengths[mode].append(len(encoded))
    return {
        "trials": trials,
        "cases": len(cases),
        "max_length": {key: max(values) for key, values in lengths.items()},
        "mean_length": {key: sum(values) / len(values) for key, values in lengths.items()},
    }


def tx_breakdown(
    selected: Sequence[int],
    n_tokens: int,
    code_bits: int,
    protocol: PacketConfig | None = None,
) -> dict[str, float | str | int]:
    """Charge the position description exactly once in the core packet."""
    protocol = protocol or PacketConfig()
    values = _normalise(selected, n_tokens)
    actual = position_encoding_mode(values, n_tokens, protocol.position_mode)
    gap_bits = float(gap_position_bits(values, n_tokens))
    if actual == "bitmap":
        position_bits = float(1 + n_tokens)
        mask_bits, index_bits = float(n_tokens), 0.0
    else:
        position_bits = float(1 + gap_bits)
        mask_bits, index_bits = 0.0, gap_bits
    packet_bits = float(protocol.header_bits)
    payload_bits = float(len(values) * code_bits)
    crc_bits = float(protocol.crc_bits)
    raw_bits = packet_bits + position_bits + payload_bits + crc_bits
    return {
        "packet": packet_bits,
        "mask": mask_bits,
        "index": index_bits,
        "position": position_bits,
        "gap_index": gap_bits,
        "position_mode": actual,
        "requested_position_mode": protocol.position_mode,
        "position_encoding": actual,
        "payload": payload_bits,
        "crc": crc_bits,
        "fec": raw_bits * (protocol.fec_ratio - 1.0),
        "raw": raw_bits,
        "total": int(math.ceil(raw_bits * protocol.fec_ratio)),
    }


def feasible_candidates(
    selected: Sequence[int],
    n_tokens: int,
    budget_bits: int,
    code_bits: int,
    protocol: PacketConfig | None = None,
) -> list[int]:
    """Return every not-yet-selected position that fits the packet budget."""
    protocol = protocol or PacketConfig()
    current = set(int(index) for index in selected)
    return [
        index
        for index in range(n_tokens)
        if index not in current
        and int(tx_breakdown([*selected, index], n_tokens, code_bits, protocol)["total"]) <= budget_bits
    ]
