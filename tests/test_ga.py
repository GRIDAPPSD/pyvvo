import unittest
from unittest.mock import patch
import os
from datetime import datetime

from pyvvo import ga
from pyvvo import equipment
from tests.test_sparql import CAPACITORS, REGULATORS
from pyvvo.glm import GLMManager
from pyvvo.utils import run_gld

import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IEEE_8500 = os.path.join(THIS_DIR, 'ieee_8500.glm')


class MapChromosomeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Get capacitors and regulators."""
        reg_df = pd.read_csv(REGULATORS)
        cap_df = pd.read_csv(CAPACITORS)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

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


@patch('pyvvo.equipment.RegulatorSinglePhase', autospec=True)
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


class PrepGLMMGRTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_8500)
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 5)
        ga.prep_glm_mgr(cls.glm_mgr, cls.starttime, cls.stoptime)

        # Write file.
        cls.out_file = 'tmp.glm'
        cls.glm_mgr.write_model(cls.out_file)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.out_file)
        try:
            os.remove('gridlabd.xml')
        except FileNotFoundError:
            pass

    def test_clock_present(self):
        """This is a so-so way to ensure add_run_components ran.

        Really, we should be using mock to ensure the method is called.
        """
        c = self.glm_mgr.get_items_by_type(item_type='clock')
        self.assertIn('clock', c)

    def test_regs_manual(self):
        regs = self.glm_mgr.get_objects_by_type(
            object_type='regulator_configuration')
        for r in regs:
            self.assertEqual('MANUAL', r['Control'])

    def test_caps_manual(self):
        caps = self.glm_mgr.get_objects_by_type(object_type='capacitor')
        for c in caps:
            self.assertEqual('MANUAL', c['control'])

    def test_triplex_load_has_group(self):
        tls = self.glm_mgr.get_objects_by_type(object_type='triplex_load')
        for tl in tls:
            self.assertEqual(ga.TRIPLEX_GROUP, tl['groupid'])

    def test_mysql_present(self):
        m = self.glm_mgr.get_items_by_type(item_type='module')
        self.assertIn('mysql', m)

    def test_mysql_group_recorder_present(self):
        r = self.glm_mgr.get_objects_by_type(
            object_type='mysql.recorder')
        self.assertIsNotNone(r)

    def test_model_runs(self):
        result = run_gld(self.out_file)
        self.assertEqual(0, result.returncode)


if __name__ == '__main__':
    unittest.main()
