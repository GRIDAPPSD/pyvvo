import unittest
from unittest.mock import patch

from pyvvo import ga
from pyvvo.equipment import capacitor, regulator
from tests.test_sparql import CAPACITORS, REGULATORS

import pandas as pd
import numpy as np


class MapChromosomeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Get capacitors and regulators."""
        reg_df = pd.read_csv(REGULATORS)
        cap_df = pd.read_csv(CAPACITORS)

        cls.regs = regulator.initialize_regulators(reg_df)
        cls.caps = capacitor.initialize_capacitors(cap_df)

        cls.map, cls.len = ga.map_chromosome(cls.regs, cls.caps)

    def test_length(self):
        """4 three phase regs, 9 single phase caps."""
        self.assertEqual(4 * 3 * 6 + 9, self.len)

    def test_map_length(self):
        self.assertEqual(4 * 3 + 9, len(self.map))

    def test_map_idx(self):
        """Ensure we have purely non-overlapping indices that cover
        the expected chromosome length.
        """
        # Initialize list of zeros.
        chrom = np.array([0] * self.len)

        # By the end, we had better have a list of ones.
        expected = np.array([1] * self.len)

        # Loop over each item in the map.
        for m in self.map.values():
            chrom[m['idx'][0]:m['idx'][1]] += 1

        # Ensure our arrays are equal.
        np.testing.assert_array_equal(chrom, expected)


class IntBinLengthTestCase(unittest.TestCase):
    """Test int_bin_length"""
    def test_0(self):
        self.assertEqual(1, ga.int_bin_length(0))

    def test_1(self):
        self.assertEqual(1, ga.int_bin_length(1))

    def test_2(self):
        self.assertEqual(2, ga.int_bin_length(2))

    def test_3(self):
        self.assertEqual(2, ga.int_bin_length(3))

    def test_32(self):
        self.assertEqual(6, ga.int_bin_length(32))

    def test_63(self):
        self.assertEqual(6, ga.int_bin_length(63))

    def test_255(self):
        self.assertEqual(8, ga.int_bin_length(255))


@patch('pyvvo.equipment.regulator.RegulatorSinglePhase', autospec=True)
class RegBinLengthTestCase(unittest.TestCase):
    """Test reg_bin_length."""
    def test_16_16(self, reg_mock):
        reg_mock.raise_taps = 16
        reg_mock.lower_taps = 16
        self.assertEqual(6, ga.reg_bin_length(reg_mock))

    def test_16_0(self, reg_mock):
        reg_mock.raise_taps = 16
        reg_mock.lower_taps = 0
        self.assertEqual(5, ga.reg_bin_length(reg_mock))

    def test_32_0(self, reg_mock):
        reg_mock.raise_taps = 32
        reg_mock.lower_taps = 0
        self.assertEqual(6, ga.reg_bin_length(reg_mock))


if __name__ == '__main__':
    unittest.main()
