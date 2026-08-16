"""Unit tests for the source-only packet protocol."""

from __future__ import annotations

import unittest

from gcrc.packet import decode_positions, encode_positions, position_roundtrip_tests, tx_breakdown


class PacketProtocolTest(unittest.TestCase):
    def test_roundtrip_for_both_position_syntaxes(self) -> None:
        for mode in ("bitmap", "gap", "adaptive_min"):
            encoded = encode_positions([0, 3, 7, 15], 16, mode)
            self.assertEqual(decode_positions(encoded, 16), [0, 3, 7, 15])

    def test_packet_charges_one_position_description(self) -> None:
        bits = tx_breakdown([0, 3, 7], 16, 3)
        self.assertGreater(bits["position"], 0)
        self.assertEqual(bits["raw"], bits["packet"] + bits["position"] + bits["payload"] + bits["crc"])

    def test_random_roundtrips(self) -> None:
        summary = position_roundtrip_tests(n_tokens=16, trials=20, seed=11)
        self.assertEqual(summary["trials"], 20)


if __name__ == "__main__":
    unittest.main()
