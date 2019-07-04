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
        cls.out_file = 'tmp.glm'

        # Run the prep function.
        ga.prep_glm_mgr(cls.glm_mgr, cls.starttime, cls.stoptime)

        # Write file.
        cls.glm_mgr.write_model(cls.out_file)

    @classmethod
    def tearDownClass(cls):
        # TODO: DROP DATABASE TABLES
        # Why is PyCharm telling me this attribute isn't defined? It
        # clearly is!
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
        r = self.glm_mgr.find_object(obj_type='mysql.recorder',
                                     obj_name='triplex_load_recorder')
        self.assertIsNotNone(r)

    def test_swing_meter_present(self):
        sm = self.glm_mgr.find_object(obj_type='meter',
                                      obj_name='"sourcebus_meter"')
        self.assertIsNotNone(sm)

    def test_swing_recorder_present(self):
        sr = self.glm_mgr.find_object(obj_type='mysql.recorder',
                                      obj_name='substation_recorder')

    def test_model_runs(self):
        result = run_gld(self.out_file)
        self.assertEqual(0, result.returncode)


class IndividualTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ind = ga.Individual(uid=0, chrom_len=50)

    def test_bad_uid(self):
        with self.assertRaisesRegex(TypeError, 'uid should be an integer.'):
            ga.Individual(uid='1', chrom_len=10)

    def test_bad_uid_negative(self):
        with self.assertRaisesRegex(ValueError, 'uid must be greater than 0.'):
            ga.Individual(uid=-2, chrom_len=7)

    def test_bad_chrom_len(self):
        with self.assertRaisesRegex(TypeError, 'chrom_len should be an int'):
            ga.Individual(uid=3, chrom_len='4')

    def test_bad_chrom_len_negative(self):
        with self.assertRaisesRegex(ValueError, 'chrom_len must be greater'):
            ga.Individual(uid=999, chrom_len=-5)

    def test_chromosome_length(self):
        """Ensure the chromosome matches the given length."""
        self.assertEqual(self.ind.chrom_len, self.ind.chromosome.shape[0])
        self.assertEqual(1, len(self.ind.chromosome.shape))

    def test_chromosome_values(self):
        """Ensure all values are in bounds."""
        self.assertTrue((self.ind.chromosome <= 1).all())
        self.assertTrue((self.ind.chromosome >= 0).all())
        x = self.ind.chromosome == 1
        y = self.ind.chromosome == 0
        self.assertTrue((x | y).all())

    def test_fitness_none(self):
        self.assertIsNone(self.ind.fitness)

    def test_chrom_override_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'chromosome must be a np.'):
            ga.Individual(uid=0, chrom_len=10, chrom_override=[1] * 10)

    def test_chrom_override_bad_dtype(self):
        with self.assertRaisesRegex(ValueError, 'chromosome must have dtype'):
            ga.Individual(uid=0, chrom_len=10,
                          chrom_override=np.array([1] * 10, dtype=np.float))

    def test_override_chromosome_bad_length(self):
        with self.assertRaisesRegex(ValueError, 'chromosome shape must match'):
            ga.Individual(uid=0, chrom_len=10,
                          chrom_override=np.array([1] * 9, dtype=np.bool))

    def test_crossover_1(self):
        """Very simple crossover test."""

        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=10)
        ind2 = ga.Individual(uid=1, chrom_len=10)

        # Override their chromosomes.
        ind1._chromosome = np.ones_like(ind1.chromosome)
        ind2._chromosome = np.zeros_like(ind2.chromosome)

        # Patch numpy's random randint.
        with patch('numpy.random.randint',
                   return_value=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                         dtype=np.bool)):
            child1, child2 = ind1.crossover_uniform(ind2, 2, 3)

        # Check uid's.
        self.assertEqual(child1.uid, 2)
        self.assertEqual(child2.uid, 3)

        # Check chromosomes
        np.testing.assert_array_equal(
            child1.chromosome,
            np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.bool))

        np.testing.assert_array_equal(
            child2.chromosome,
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.bool))

        # Check fitnesses.
        self.assertIsNone(child1.fitness)
        self.assertIsNone(child2.fitness)

    def test_crossover_2(self):
        """Slightly less simple crossover test."""

        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=6)
        ind2 = ga.Individual(uid=1, chrom_len=6)

        # Override their chromosomes.
        ind1._chromosome = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
        ind2._chromosome = np.array([1, 0, 0, 0, 1, 0], dtype=bool)

        # Patch numpy's random randint.
        with patch('numpy.random.randint',
                   return_value=np.array([0, 0, 1, 1, 1, 1],
                                         dtype=np.bool)):
            child1, child2 = ind1.crossover_uniform(ind2, 2, 3)

        # Check uid's.
        self.assertEqual(child1.uid, 2)
        self.assertEqual(child2.uid, 3)

        # Check chromosomes
        np.testing.assert_array_equal(
            child1.chromosome,
            np.array([1, 0, 0, 1, 0, 0], dtype=np.bool))

        np.testing.assert_array_equal(
            child2.chromosome,
            np.array([1, 1, 0, 0, 1, 0], dtype=np.bool))

        # Check fitnesses.
        self.assertIsNone(child1.fitness)
        self.assertIsNone(child2.fitness)

    def test_mutate_1(self):
        """Simple mutation test."""
        ind1 = ga.Individual(uid=0, chrom_len=5,
                             chrom_override=np.array([1, 1, 1, 1, 1],
                                                     dtype=np.bool))

        with patch('numpy.random.random_sample',
                   return_value=np.array([0.1, 0.5, 0.2, 0.7, 0.4])):
            ind1.mutate(mut_prob=0.2)

        np.testing.assert_array_equal(ind1.chromosome,
                                      np.array([0, 1, 0, 1, 1], dtype=np.bool))

    def test_mutate_2(self):
        """Slightly less simple mutation test."""
        ind1 = ga.Individual(uid=0, chrom_len=8,
                             chrom_override=np.array([0, 0, 1, 0, 1, 1, 0, 1],
                                                     dtype=np.bool))

        with patch('numpy.random.random_sample',
                   return_value=np.array(
                       [0.1, 0.5, 0.2, 0.7, 0.4, 0.21, 0.01, 0.9])):
            ind1.mutate(mut_prob=0.2)

        np.testing.assert_array_equal(ind1.chromosome,
                                      np.array([1, 0, 0, 0, 1, 1, 1, 1],
                                               dtype=np.bool))

    def test_mutate_bad_type(self):
        with self.assertRaises(TypeError):
            self.ind.mutate(mut_prob='0.2')

    def test_mutate_too_high(self):
        with self.assertRaisesRegex(ValueError, 'mut_prob must be on the'):
            self.ind.mutate(mut_prob=1.00001)

    def test_mutate_too_low(self):
        with self.assertRaisesRegex(ValueError, 'mut_prob must be on the'):
            self.ind.mutate(mut_prob=-0.00001)




if __name__ == '__main__':
    unittest.main()
