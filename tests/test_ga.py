import unittest
from unittest.mock import patch, create_autospec, NonCallableMagicMock
import os
from datetime import datetime
from copy import deepcopy
import multiprocessing as mp
import threading
from time import sleep
import itertools

import tests.data_files as _df
from tests.models import IEEE_9500, IEEE_13
from pyvvo import ga
from pyvvo import equipment
from pyvvo.glm import GLMManager
from pyvvo.utils import run_gld, time_limit
from pyvvo import db

import numpy as np
import pandas as pd
import MySQLdb

np.random.seed(42)


class MapChromosomeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Get capacitors and regulators."""
        cls.reg_df = _df.read_pickle(_df.REGULATORS_9500)
        cls.cap_df = _df.read_pickle(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(cls.reg_df)
        cls.caps = equipment.initialize_capacitors(cls.cap_df)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

    def test_length(self):
        """6 three phase regs, 9 single phase caps."""
        self.assertEqual(6 * 3 * 6 + 9, self.len)

    def test_map_length(self):
        """6 * 3 individual phase regs, 9 caps."""
        self.assertEqual(6 * 3 + 9, len(self.map))

    def test_map_idx(self):
        """Ensure we have purely non-overlapping indices that cover
        the expected chromosome length.
        """
        # Initialize list of zeros.
        chrom = np.array([0] * self.len)

        # By the end, we had better have a list of ones.
        expected = np.array([1] * self.len)

        # Loop over each item in the map.
        for m1 in self.map.values():
            for m2 in m1.values():
                chrom[m2['idx'][0]:m2['idx'][1]] += 1

        # Ensure our arrays are equal.
        np.testing.assert_array_equal(chrom, expected)

    def test_map_num_eq(self):
        """6x3 regs, 9 caps"""
        self.assertEqual(6 * 3 + 9, self.num_eq)

    def test_map_phasing(self):
        """Ensure each element in the map has valid phases under it."""
        for eq_name, phase_dict in self.map.items():
            for phase in phase_dict.keys():
                self.assertIn(phase, equipment.EquipmentSinglePhase.PHASES)

    def test_map_eq_obj(self):
        """Ensure each 'eq_obj' element is EquipmentSinglePhase."""
        for eq_name, phase_dict in self.map.items():
            for sd in phase_dict.values():
                self.assertIsInstance(sd['eq_obj'],
                                      equipment.EquipmentSinglePhase)

    def test_map_range(self):
        for eq_name, phase_dict in self.map.items():
            for sd in phase_dict.values():
                if isinstance(sd['eq_obj'], equipment.CapacitorSinglePhase):
                    self.assertEqual((0, 1), sd['range'])
                elif isinstance(sd['eq_obj'], equipment.RegulatorSinglePhase):
                    self.assertEqual(
                        (0, sd['eq_obj'].raise_taps + sd['eq_obj'].lower_taps),
                        sd['range'])
                else:
                    raise TypeError('Bad equipment type!')

    def test_map_bad_reg_type(self):
        with self.assertRaisesRegex(TypeError, 'regulators must be a dict'):
            ga.map_chromosome(regulators=self.reg_df,
                              capacitors=self.caps)

    def test_map_bad_cap_type(self):
        with self.assertRaisesRegex(TypeError, 'capacitors must be a dict'):
            ga.map_chromosome(regulators=self.regs, capacitors=self.cap_df)

    def test_map_current_state(self):
        for eq_name, phase_dict in self.map.items():
            for sd in phase_dict.values():
                if isinstance(sd['eq_obj'], equipment.CapacitorSinglePhase):
                    self.assertEqual(sd['eq_obj'].state, sd['current_state'])
                elif isinstance(sd['eq_obj'], equipment.RegulatorSinglePhase):
                    self.assertEqual(
                        sd['eq_obj'].tap_pos + sd['eq_obj'].lower_taps,
                        sd['current_state'])
                else:
                    raise TypeError('Bad equipment type!')


class BinaryArrayToScalarTestCase(unittest.TestCase):
    """Test _binary_array_to_scalar"""

    def test_zero(self):
        a = np.array([0])
        self.assertEqual(0, ga._binary_array_to_scalar(a))

    def test_one(self):
        a = np.array([1])
        self.assertEqual(1, ga._binary_array_to_scalar(a))

    def test_two(self):
        a = np.array([1, 0])
        self.assertEqual(2, ga._binary_array_to_scalar(a))

    def test_three(self):
        a = np.array([1, 1])
        self.assertEqual(3, ga._binary_array_to_scalar(a))

    def test_four(self):
        a = np.array([1, 0, 0])
        self.assertEqual(4, ga._binary_array_to_scalar(a))

    def test_five(self):
        a = np.array([1, 0, 1])
        self.assertEqual(5, ga._binary_array_to_scalar(a))

    def test_six(self):
        a = np.array([1, 1, 0])
        self.assertEqual(6, ga._binary_array_to_scalar(a))

    def test_seven(self):
        a = np.array([1, 1, 1])
        self.assertEqual(7, ga._binary_array_to_scalar(a))

    def test_eight(self):
        a = np.array([1, 0, 0, 0])
        self.assertEqual(8, ga._binary_array_to_scalar(a))

    def test_big_number(self):
        n = 239034
        b = bin(n)
        # bin is prefixed with '0b', so start at 2.
        a = np.array([int(i) for i in b[2:]])
        self.assertEqual(n, ga._binary_array_to_scalar(a))


class CIMToGLMNameTestCase(unittest.TestCase):
    """Test _cim_to_glm_name"""

    def test_one(self):
        self.assertEqual('"my_name"',
                         ga._cim_to_glm_name(prefix='my',
                                             cim_name='name'))

    def test_two(self):
        self.assertEqual('"pfah_stuff"',
                         ga._cim_to_glm_name(prefix='pfah',
                                             cim_name='"stuff"'))

    def test_three(self):
        self.assertEqual('"bad_people"',
                         ga._cim_to_glm_name(prefix='"bad"',
                                             cim_name='"people"'))


class IntBinLengthTestCase(unittest.TestCase):
    """Test _int_bin_length"""

    def test_0(self):
        self.assertEqual(1, ga._int_bin_length(0))

    def test_1(self):
        self.assertEqual(1, ga._int_bin_length(1))

    def test_2(self):
        self.assertEqual(2, ga._int_bin_length(2))

    def test_3(self):
        self.assertEqual(2, ga._int_bin_length(3))

    def test_32(self):
        self.assertEqual(6, ga._int_bin_length(32))

    def test_63(self):
        self.assertEqual(6, ga._int_bin_length(63))

    def test_255(self):
        self.assertEqual(8, ga._int_bin_length(255))


class IntToBinaryListTestCase(unittest.TestCase):
    """Test _int_to_binary_list."""

    def test_0_1(self):
        self.assertListEqual([0],
                             ga._int_to_binary_list(0, 1))

    def test_0_0(self):
        self.assertListEqual([0],
                             ga._int_to_binary_list(0, 0))

    def test_0_2(self):
        self.assertListEqual([0, 0],
                             ga._int_to_binary_list(0, 2))

    def test_16_32(self):
        self.assertListEqual([0, 1, 0, 0, 0, 0],
                             ga._int_to_binary_list(16, 32))


@patch('pyvvo.equipment.RegulatorSinglePhase', autospec=True)
class RegBinLengthTestCase(unittest.TestCase):
    """Test _reg_bin_length."""

    def test_16_16(self, reg_mock):
        reg_mock.raise_taps = 16
        reg_mock.lower_taps = 16
        self.assertEqual(6, ga._reg_bin_length(reg_mock))

    def test_16_0(self, reg_mock):
        reg_mock.raise_taps = 16
        reg_mock.lower_taps = 0
        self.assertEqual(5, ga._reg_bin_length(reg_mock))

    def test_32_0(self, reg_mock):
        reg_mock.raise_taps = 32
        reg_mock.lower_taps = 0
        self.assertEqual(6, ga._reg_bin_length(reg_mock))


class PrepGLMMGRTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_13)
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 5)
        cls.out_file = 'tmp.glm'

        # Run the prep function.
        ga.prep_glm_mgr(cls.glm_mgr, cls.starttime, cls.stoptime)

        # Write file.
        cls.glm_mgr.write_model(cls.out_file)

    @classmethod
    def tearDownClass(cls):
        # Get a connection to the database.
        db_conn = db.connect_loop()

        # Truncate the tables.
        db.truncate_table(db_conn=db_conn, table=ga.TRIPLEX_TABLE)
        db.truncate_table(db_conn=db_conn, table=ga.SUBSTATION_TABLE)

        try:
            # Why is PyCharm telling me this attribute isn't defined? It
            # clearly is!
            # noinspection PyUnresolvedReferences
            os.remove(cls.out_file)
        except FileNotFoundError:
            pass

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

    def test_mysql_recorder_present(self):
        r = self.glm_mgr.find_object(obj_type='mysql.recorder',
                                     obj_name=ga.TRIPLEX_RECORDER)
        self.assertIsNotNone(r)

    def test_swing_meter_present(self):
        sm = self.glm_mgr.find_object(obj_type='meter',
                                      obj_name='"sourcebus_meter"')
        self.assertIsNotNone(sm)

    def test_swing_recorder_present(self):
        sr = self.glm_mgr.find_object(obj_type='mysql.recorder',
                                      obj_name=ga.SUBSTATION_RECORDER)
        self.assertIsNotNone(sr)

    def test_model_runs(self):
        result = run_gld(self.out_file)
        self.assertEqual(0, result.returncode)


class PrepGLMMGR9500TestCase(unittest.TestCase):
    """Test running the 9500 node model. We may need to skip this in
    general, as it's going to be really slow.
    """

    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_9500)
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 5)
        cls.out_file = 'tmp.glm'

        # Run the prep function.
        ga.prep_glm_mgr(cls.glm_mgr, cls.starttime, cls.stoptime)

        # # Testing a hack.
        # cls.glm_mgr.modify_item({"object": "switch",
        #                          "name": '"swt_tsw803273_sw"',
        #                          "status": "CLOSED"})

        # Write file.
        cls.glm_mgr.write_model(cls.out_file)

    @classmethod
    def tearDownClass(cls):
        # Get a connection to the database.
        db_conn = db.connect_loop()

        # Truncate the tables.
        db.truncate_table(db_conn=db_conn, table=ga.TRIPLEX_TABLE)
        db.truncate_table(db_conn=db_conn, table=ga.SUBSTATION_TABLE)

        try:
            # Why is PyCharm telling me this attribute isn't defined? It
            # clearly is!
            # noinspection PyUnresolvedReferences
            os.remove(cls.out_file)
        except FileNotFoundError:
            pass

        try:
            os.remove('gridlabd.xml')
        except FileNotFoundError:
            pass

    def test_model_runs(self):
        result = run_gld(self.out_file)
        self.assertEqual(0, result.returncode)


class IndividualTestCase(unittest.TestCase):
    """Test everything that doesn't involve a glm.GLMManager. Those
    tests are more involved and will be done elsewhere.
    """

    @classmethod
    def setUpClass(cls):
        # Get capacitor and regulator information.
        reg_df = _df.read_pickle(_df.REGULATORS_9500)
        cap_df = _df.read_pickle(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

        # It seems we don't have a way of getting capacitor state from
        # the CIM (which is where those DataFrames originate from). So,
        # let's randomly command each capacitor.
        for c in cls.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                c.state = np.random.randint(low=0, high=2, size=None,
                                            dtype=int)
            elif isinstance(c, dict):
                for cc in c.values():
                    cc.state = np.random.randint(low=0, high=2, size=None,
                                                 dtype=int)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)
        cls.ind = ga.Individual(uid=0, chrom_len=cls.len, chrom_map=cls.map,
                                num_eq=cls.num_eq)

    def test_bad_uid(self):
        with self.assertRaisesRegex(TypeError, 'uid should be an integer.'):
            ga.Individual(uid='1', chrom_len=10, chrom_map=self.map,
                          num_eq=self.num_eq)

    def test_bad_uid_negative(self):
        with self.assertRaisesRegex(ValueError, 'uid must be greater than 0.'):
            ga.Individual(uid=-2, chrom_len=7, chrom_map=self.map,
                          num_eq=self.num_eq)

    def test_bad_chrom_len(self):
        with self.assertRaisesRegex(TypeError, 'chrom_len should be an int'):
            ga.Individual(uid=3, chrom_len='4', chrom_map=self.map,
                          num_eq=self.num_eq)

    def test_bad_chrom_len_negative(self):
        with self.assertRaisesRegex(ValueError, 'chrom_len must be greater'):
            ga.Individual(uid=999, chrom_len=-6, chrom_map=self.map,
                          num_eq=self.num_eq)

    def test_bad_num_eq_type(self):
        with self.assertRaisesRegex(TypeError, 'num_eq should be an integer'):
            ga.Individual(uid=3, chrom_len=self.len, chrom_map=self.map,
                          num_eq={'hi': 'friend'})

    def test_bad_num_eq_negative(self):
        with self.assertRaisesRegex(ValueError, 'There must be at least one'):
            ga.Individual(uid=999, chrom_len=self.len, chrom_map=self.map,
                          num_eq=-33)

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
            ga.Individual(uid=0, chrom_len=10, chrom_override=[1] * 10,
                          chrom_map=self.map, num_eq=self.num_eq)

    def test_chrom_override_bad_dtype(self):
        with self.assertRaisesRegex(ValueError, 'chromosome must have dtype'):
            ga.Individual(uid=0, chrom_len=10,
                          chrom_override=np.array([1] * 10, dtype=np.float),
                          chrom_map=self.map, num_eq=self.num_eq)

    def test_override_chromosome_bad_length(self):
        with self.assertRaisesRegex(ValueError, 'chromosome shape must match'):
            ga.Individual(uid=0, chrom_len=10,
                          chrom_override=np.array([1] * 9, dtype=np.bool),
                          chrom_map=self.map, num_eq=self.num_eq)

    def test_check_and_fix_chromosome(self):
        # Grab a copy of the individual's chromosome.
        c = self.ind.chromosome.copy()

        # Splice in values for VREG2 and VREG3.
        idx2 = self.ind._chrom_map['vreg2_a']['A']['idx']
        idx3 = self.ind._chrom_map['vreg3_c']['C']['idx']
        # The call below looks wrong, but it's fine, I promise :)
        # the 'm' argument is just used for the width formatting.
        c[idx2[0]:idx2[1]] = ga._int_to_binary_list(50, m=32)
        c[idx3[0]:idx3[1]] = ga._int_to_binary_list(25, m=32)

        # Patch the 'VREG3' range and run the check. No need to patch
        # the 'VREG2' range since it's above the maximum.
        with patch.dict(self.ind._chrom_map['vreg3_c']['C'],
                        {'range': (31, 32)}):
            c_new = self.ind._check_and_fix_chromosome(c)

        # Violations above should be cut down to the top of the range.
        self.assertEqual(ga._binary_array_to_scalar(c_new[idx2[0]:idx2[1]]),
                         32)
        # Violations below should be brought up to the bottom of the range.
        self.assertEqual(ga._binary_array_to_scalar(c_new[idx3[0]:idx3[1]]),
                         31)
        pass

    def test_override_calls_check_and_fix_chromosome(self):
        with patch('pyvvo.ga.Individual._check_and_fix_chromosome',
                   autospec=True) as p:
            ga.Individual(uid=0, chrom_len=self.len,
                          chrom_override=np.ones(self.len, dtype=np.bool),
                          chrom_map=self.map, num_eq=self.num_eq)

        p.assert_called_once()

    def test_chromosome_values_in_range(self):
        """Loop over the map, and ensure translating chromosome values
        to integers results in a valid value.

        This effectively tests an Individual's _initialize_chromosome
        method.
        """
        for phase_dict in self.map.values():
            for eq_dict in phase_dict.values():
                idx = eq_dict['idx']
                # Grab the binary sliver.
                b = self.ind.chromosome[idx[0]:idx[1]]
                # Get it as a number.
                n = ga._binary_array_to_scalar(b)
                # Ensure it's in range.
                with self.subTest(msg=eq_dict['eq_obj'].name):
                    self.assertGreaterEqual(eq_dict['range'][1], n)
                    self.assertLessEqual(eq_dict['range'][0], n)

    def test_crossover_uniform(self):
        """Very simple crossover test. This test is a little
        over-loaded, and actually covers both crossover_uniform and
        _crossover.
        """

        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)
        ind2 = ga.Individual(uid=1, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)

        # Create an array where the first half is ones/True, and the
        # second half is zeros/False
        first_half = int(np.ceil(self.len / 2))
        second_half = int(np.floor(self.len / 2))

        patched_array = np.array([1] * first_half + [0] * second_half,
                                 dtype=np.bool)

        # Patch numpy's random randint.
        with patch('numpy.random.randint', return_value=patched_array):
            child1, child2 = ind1.crossover_uniform(ind2, 2, 3)

        # Check uid's.
        self.assertEqual(child1.uid, 2)
        self.assertEqual(child2.uid, 3)

        # Check fitnesses.
        self.assertIsNone(child1.fitness)
        self.assertIsNone(child2.fitness)

        # Check that the children are properly taking values from each
        # parent. NOTE: This is fragile because
        # _check_and_fix_chromosome will be called for each child.
        # HOWEVER, for the 123 node system, cutting the chromosome in
        # half will never result in a value out of bounds.
        np.testing.assert_array_equal(child1.chromosome[0:first_half],
                                      ind1.chromosome[0:first_half])
        np.testing.assert_array_equal(child1.chromosome[first_half:],
                                      ind2.chromosome[first_half:])

        np.testing.assert_array_equal(child2.chromosome[0:first_half],
                                      ind2.chromosome[0:first_half])
        np.testing.assert_array_equal(child2.chromosome[first_half:],
                                      ind1.chromosome[first_half:])

    def test_crossover_uniform_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'other must be an Individual'):
            self.ind.crossover_uniform(other='spam', uid1=10, uid2=400)

    def test_crossover_uniform_runs(self):
        """No patching, just run it and ensure we get individuals back."""
        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)
        ind2 = ga.Individual(uid=1, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)

        ind3, ind4 = ind1.crossover_uniform(ind2, 2, 3)

        self.assertIsInstance(ind3, ga.Individual)
        self.assertIsInstance(ind4, ga.Individual)

    def test_crossover_by_gene_runs(self):
        """No patching, just run it and ensure we get individuals back."""
        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)
        ind2 = ga.Individual(uid=1, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)

        ind3, ind4 = ind1.crossover_by_gene(ind2, 2, 3)

        self.assertIsInstance(ind3, ga.Individual)
        self.assertIsInstance(ind4, ga.Individual)

    def test_crossover_by_gene_expected_behavior(self):
        """Patch our random draw, ensure everything is as expected."""
        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)
        ind2 = ga.Individual(uid=1, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)

        # Simplest case: draw only True values.
        patched_array = np.array([True] * ind1.num_eq, dtype=np.bool)

        with patch('numpy.random.randint', return_value=patched_array):
            with patch.object(ind1, '_crossover', wraps=ind1._crossover) as p:
                child1, child2 = \
                    ind1.crossover_by_gene(other=ind2, uid1=2, uid2=3)

        p.assert_called_once()

        np.testing.assert_array_equal(ind1.chromosome, child1.chromosome)
        np.testing.assert_array_equal(ind2.chromosome, child2.chromosome)

    def test_crossover_by_gene_expected_behavior_2(self):
        """Slightly more sophisticated test to ensure we're getting the
        behavior we want.
        """
        # Initialize two individuals.
        ind1 = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)
        ind2 = ga.Individual(uid=1, chrom_len=self.len, chrom_map=self.map,
                             num_eq=self.num_eq)

        # Patch the array in such a way that the first parent gives the
        # first and last pieces of equipment, and the second parent
        # gives the rest (to the first child).
        patched_array = np.array([False] * ind1.num_eq, dtype=np.bool)
        patched_array[0] = True
        patched_array[-1] = True

        with patch('numpy.random.randint', return_value=patched_array):
            with patch.object(ind1, '_crossover', wraps=ind1._crossover) as p:
                child1, child2 = \
                    ind1.crossover_by_gene(other=ind2, uid1=2, uid2=3)

        p.assert_called_once()

        # Get the indices of the first and last pieces of equipment.
        # Recent versions of Python consistently iterate over
        # dictionaries.
        all_values = list(ind2.chrom_map.values())

        # Extract indices for equipment 0 and equipment end (e).
        v0 = all_values[0]
        v0_p = list(v0.values())
        v0_idx = v0_p[0]['idx']

        ve = all_values[-1]
        ve_p = list(ve.values())
        ve_idx = ve_p[-1]['idx']

        # Ensure the chromosomes match up.

        # Start by testing the beginning and end of child1, ensuring
        # those stretches are the same as ind1.
        np.testing.assert_array_equal(child1.chromosome[v0_idx[0]:v0_idx[1]],
                                      ind1.chromosome[v0_idx[0]:v0_idx[1]])

        np.testing.assert_array_equal(child1.chromosome[ve_idx[0]:ve_idx[1]],
                                      ind1.chromosome[ve_idx[0]:ve_idx[1]])

        # Ensure the rest of child1 matches ind2.
        np.testing.assert_array_equal(
            child1.chromosome[(v0_idx[1] + 1):(ve_idx[0] - 1)],
            ind2.chromosome[(v0_idx[1] + 1):(ve_idx[0] - 1)]
        )

        # Test the beginning and end of child2, ensuring a match with
        # ind2
        np.testing.assert_array_equal(child2.chromosome[v0_idx[0]:v0_idx[1]],
                                      ind2.chromosome[v0_idx[0]:v0_idx[1]])

        np.testing.assert_array_equal(child2.chromosome[ve_idx[0]:ve_idx[1]],
                                      ind2.chromosome[ve_idx[0]:ve_idx[1]])

        # Ensure the rest of child2 matches ind1.
        np.testing.assert_array_equal(
            child2.chromosome[(v0_idx[1] + 1):(ve_idx[0] - 1)],
            ind1.chromosome[(v0_idx[1] + 1):(ve_idx[0] - 1)]
        )

    def test_mutate(self):
        """Simple mutation test."""
        ind1 = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                             chrom_override=np.zeros(self.len, dtype=np.bool),
                             num_eq=self.num_eq)

        patched_array = np.ones(self.len)
        patched_array[7] = 0
        patched_array[10] = 0
        with patch('numpy.random.random_sample', return_value=patched_array):
            with patch.object(ind1, '_check_and_fix_chromosome',
                              wraps=ind1._check_and_fix_chromosome) as p:
                ind1.mutate(mut_prob=0.01)

        # Ensure _check_and_fix chromosome is called.
        p.assert_called_once()

        # Our new chromosome should have ones in positions 7 and 10.
        expected = np.zeros(self.len, dtype=np.bool)
        expected[7] = True
        expected[10] = True

        np.testing.assert_array_equal(ind1.chromosome, expected)

    def test_mutate_bad_type(self):
        with self.assertRaises(TypeError):
            self.ind.mutate(mut_prob='0.2')

    def test_mutate_too_high(self):
        with self.assertRaisesRegex(ValueError, 'mut_prob must be on the'):
            self.ind.mutate(mut_prob=1.00001)

    def test_mutate_too_low(self):
        with self.assertRaisesRegex(ValueError, 'mut_prob must be on the'):
            self.ind.mutate(mut_prob=-0.00001)

    def test_special_init_bad_value(self):
        with self.assertRaisesRegex(ValueError, 'special_init must be one of'):
            ga.Individual(uid=0, chrom_len=self.len, chrom_override=None,
                          chrom_map=self.map, num_eq=self.num_eq,
                          special_init='MAX')

    def test_special_init_and_chrom_override_warns(self):
        with self.assertLogs(level='WARN') as log:
            ga.Individual(uid=0, chrom_len=self.len,
                          chrom_override=np.array([False] * self.len,
                                                  dtype=np.bool),
                          chrom_map=self.map, num_eq=self.num_eq,
                          special_init='max')

        self.assertIn('The given value of special_init, max, is being ignored',
                      log.output[0])

    def test_special_init_none(self):
        """Ensure np.random.randint is called once for each piece of
        equipment."""

        with patch('numpy.random.randint', wraps=np.random.randint) as p:
            ga.Individual(uid=0, chrom_len=self.len,
                          chrom_override=None,
                          chrom_map=self.map, num_eq=self.num_eq,
                          special_init=None)

        # 6 three phase regs, 9 single phase caps.
        self.assertEqual(6 * 3 + 9, p.call_count)

    def test_special_init_max(self):
        """Ensure each piece of equipment is at its max."""
        ind = ga.Individual(uid=0, chrom_len=self.len,
                            chrom_override=None,
                            chrom_map=self.map, num_eq=self.num_eq,
                            special_init='max')

        c = ind.chromosome

        for phase_dict in self.map.values():
            for eq_dict in phase_dict.values():
                i = eq_dict['idx']
                c_i = c[i[0]:i[1]]
                if isinstance(eq_dict['eq_obj'],
                              equipment.RegulatorSinglePhase):
                    self.assertEqual(ga._binary_array_to_scalar(c_i),
                                     abs(eq_dict['eq_obj'].high_step
                                         - eq_dict['eq_obj'].low_step))
                elif isinstance(eq_dict['eq_obj'],
                                equipment.CapacitorSinglePhase):
                    self.assertEqual(c_i[0], 1)
                else:
                    raise ValueError('Unexpected equipment type.')

    def test_special_init_min(self):
        """Ensure each piece of equipment is at its min."""
        ind = ga.Individual(uid=0, chrom_len=self.len,
                            chrom_override=None,
                            chrom_map=self.map, num_eq=self.num_eq,
                            special_init='min')

        c = ind.chromosome

        for phase_dict in self.map.values():
            for eq_dict in phase_dict.values():
                i = eq_dict['idx']
                c_i = c[i[0]:i[1]]
                # Regs and caps will ALWAYS have a minimum of zero.
                self.assertEqual(ga._binary_array_to_scalar(c_i), 0)

    def test_special_init_current_state(self):
        """Ensure each piece of equipment respects the current state."""
        ind = ga.Individual(uid=0, chrom_len=self.len,
                            chrom_override=None,
                            chrom_map=self.map, num_eq=self.num_eq,
                            special_init='current_state')

        c = ind.chromosome

        for phase_dict in self.map.values():
            for eq_dict in phase_dict.values():
                i = eq_dict['idx']
                c_i = c[i[0]:i[1]]

                if isinstance(eq_dict['eq_obj'],
                              equipment.CapacitorSinglePhase):
                    self.assertEqual(ga._binary_array_to_scalar(c_i),
                                     eq_dict['eq_obj'].state)
                elif isinstance(eq_dict['eq_obj'],
                                equipment.RegulatorSinglePhase):
                    # Note that we must shift the interval. Also, it's
                    # important that in this test we use the GridLAB-D
                    # regulator attributes, as that's what the
                    # map_chromosome method uses.
                    self.assertEqual(ga._binary_array_to_scalar(c_i),
                                     (eq_dict['eq_obj'].tap_pos
                                      + eq_dict['eq_obj'].lower_taps))
                else:
                    raise ValueError('Unexpected type!')


class IndividualUpdateModelComputeCostsTestCase(unittest.TestCase):
    """Test the _update_model_compute_costs method of an individual.

    Additionally, we'll test _update_reg and _update_cap.
    """

    @classmethod
    def setUpClass(cls):
        cls.reg_df = _df.read_pickle(_df.REGULATORS_9500)
        cls.caps_df = _df.read_pickle(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(cls.reg_df)
        cls.caps = equipment.initialize_capacitors(cls.caps_df)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

        cls.glm_mgr = GLMManager(IEEE_9500)

        # Force all the capacitors to be open.
        for c in cls.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                c.state = 0
            elif isinstance(c, dict):
                for cc in c.values():
                    cc.state = 0

        # Force all regulators to be at their minimum.
        for r in cls.regs.values():
            if isinstance(r, equipment.RegulatorSinglePhase):
                r.tap_pos = -r.lower_taps
            elif isinstance(r, dict):
                for rr in r.values():
                    rr.tap_pos = -rr.lower_taps

        # Force the individual to have all capacitors closed and all
        # regulators at their maximum.
        # noinspection PyUnusedLocal
        def patch_randint(arg1, arg2, arg3):
            return arg2 - 1

        with patch('numpy.random.randint', new=patch_randint):
            cls.ind = ga.Individual(uid=0, chrom_len=cls.len,
                                    chrom_map=cls.map, num_eq=cls.num_eq)

    def setUp(self):
        self.fresh_mgr = deepcopy(self.glm_mgr)

    def test_update_reg(self):
        """Test _update_reg"""
        penalty = 0
        phase_dicts = [self.map['feeder_reg1a'], self.map['feeder_reg1b'],
                       self.map['feeder_reg1c']]
        with patch.dict(ga.CONFIG, {'costs': {'regulator_tap': 10}}):
            with patch.object(self.fresh_mgr, 'update_reg_taps',
                              wraps=self.fresh_mgr.update_reg_taps) as p:
                for phase_dict in phase_dicts:
                    penalty += self.ind._update_reg(phase_dict=phase_dict,
                                                    glm_mgr=self.fresh_mgr)

        # 3 phases, each phase moved 32 positions with a cost of 10 per
        # position.
        self.assertEqual(3 * 32 * 10, penalty)

        # Ensure the taps are being updated properly in the model.
        # The actual 'update_reg_taps' method is tested elsewhere, so
        # no need to actually look up the objects and confirm.
        self.assertEqual(3, p.call_count)
        self.assertEqual(('"reg_feeder_reg1a"', {'A': 16}),
                         p.call_args_list[0][0])
        self.assertEqual(('"reg_feeder_reg1b"', {'B': 16}),
                         p.call_args_list[1][0])
        self.assertEqual(('"reg_feeder_reg1c"', {'C': 16}),
                         p.call_args_list[2][0])

    def test_update_cap(self):
        """Test _update_cap"""
        phase_dict = self.map['capbank0a']
        with patch.dict(ga.CONFIG, {'costs': {'capacitor_switch': 10}}):
            with patch.object(self.fresh_mgr, 'update_cap_switches',
                              wraps=self.fresh_mgr.update_cap_switches) as p:
                penalty = self.ind._update_cap(phase_dict=phase_dict,
                                               glm_mgr=self.fresh_mgr)

        # One phase, one switch --> 10 penalty.
        self.assertEqual(10, penalty)

        # Ensure capacitor switch position is being updated correctly.
        p.assert_called_once()
        p.assert_called_with(ga._cim_to_glm_name(prefix=ga.CAP_PREFIX,
                                                 cim_name='capbank0a'),
                             {'A': 'CLOSED'})

    @patch.dict(ga.CONFIG, {'costs': {'regulator_tap': 10,
                                      'capacitor_switch': 10}})
    def test_update_model_compute_costs(self):
        with patch.object(self.ind, '_update_reg',
                          wraps=self.ind._update_reg) as pr:
            with patch.object(self.ind, '_update_cap',
                              wraps=self.ind._update_cap) as pc:
                reg_penalty, cap_penalty = \
                    self.ind._update_model_compute_costs(
                        glm_mgr=self.fresh_mgr)

        # 9 single phase caps switching.
        self.assertEqual(9 * 10, cap_penalty)

        # 3 phases per regulator, 6 regulators, each moving 32 taps,
        # with a cost of 10 per tap
        self.assertEqual(3 * 6 * 32 * 10, reg_penalty)

        # Ensure our helper methods are called the appropriate number of
        # times. We have 9 capacitor phases, and the regulators are
        # now modeled as single phase (rather than 3 phase). 6 * 3 = 18.
        self.assertEqual(9, pc.call_count)
        self.assertEqual(18, pr.call_count)


class IndividualUpdateCapBadStateTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reg_df = _df.read_pickle(_df.REGULATORS_9500)
        cls.caps_df = _df.read_pickle(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(cls.reg_df)
        cls.caps = equipment.initialize_capacitors(cls.caps_df)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

        cls.glm_mgr = GLMManager(IEEE_9500)

        cls.ind = ga.Individual(uid=0, chrom_len=cls.len,
                                chrom_map=cls.map, num_eq=cls.num_eq)

    def setUp(self):
        self.fresh_mgr = deepcopy(self.glm_mgr)

    def test_update_cap_bad_state(self):
        """If we screw up and get capacitors straight from the CIM, they
        don't have state information. Ensure this blows up.
        """
        with self.assertRaisesRegex(ValueError, 'Equipment .* has invali'):
            self.ind._update_cap(self.map['capbank0a'],
                                 glm_mgr=self.fresh_mgr)


# Dictionary used by the tests below.
PARTIAL_DICT = {'voltage_high': 1, 'voltage_low': 2, 'power_factor_lead': 3,
                'power_factor_lag': 4, 'energy': 5}


class IndividualEvaluateTestCase(unittest.TestCase):
    """Test the evaluate method of an Individual.
    """

    @classmethod
    def setUpClass(cls):
        # Get capacitor and regulator information.
        reg_df = _df.read_pickle(_df.REGULATORS_13)
        cap_df = _df.read_pickle(_df.CAPACITORS_13)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

    def setUp(self):
        # Initialize an individual.
        self.ind = ga.Individual(uid=0, chrom_len=self.len, chrom_map=self.map,
                                 num_eq=self.num_eq)

        # Patch inputs to the Individual's evaluate method.
        self.mock_glm = unittest.mock.create_autospec(GLMManager,
                                                      spec_set=True)
        self.mock_db = unittest.mock.create_autospec(MySQLdb.connection,
                                                     set_spec=True)

    @patch('pyvvo.ga._Evaluator.evaluate', autospec=True,
           return_value=PARTIAL_DICT)
    @patch('pyvvo.ga._Evaluator.__init__', autospec=True, return_value=None)
    def test_evaluate(self, eval_init_patch, eval_evaluate_patch):
        """Patch everything, ensure the correct methods are called.

        The individual functions have been well tested, so no need to
        actually call them.

        Patching __init__ feels fragile, but it's the only way I could
        get this thing working.

        https://stackoverflow.com/q/57044593/11052174
        """
        with patch.object(self.ind, '_update_model_compute_costs',
                          autospec=True, return_value=(6, 7)) as p_update:
            self.ind.evaluate(glm_mgr=self.mock_glm, db_conn=self.mock_db)

        # Assertion time.
        # Ensure _update_model_compute_costs is called and called
        # correctly.
        p_update.assert_called_once()
        p_update.assert_called_with(glm_mgr=self.mock_glm)

        # Ensure our _Evaluator is constructor appropriately.
        eval_init_patch.assert_called_once()
        self.assertDictEqual(eval_init_patch.call_args[1],
                             {'uid': self.ind.uid, 'glm_mgr': self.mock_glm,
                              'db_conn': self.mock_db})

        # Ensure _Evaluator._evaluate is called.
        eval_evaluate_patch.assert_called_once()

        # Total fitness is the sum of the values in the penalties dict.
        self.assertEqual(28, self.ind.fitness)

        # Ensure our penalties dict comes back as expected.
        expected = {**PARTIAL_DICT, 'regulator_tap': 6, 'capacitor_switch': 7}
        self.assertDictEqual(expected, self.ind.penalties)

    @patch('pyvvo.ga._Evaluator.evaluate', autospec=True,
           side_effect=UserWarning('Dummy exception for testing.'))
    @patch('pyvvo.ga._Evaluator.__init__', autospec=True, return_value=None)
    def test_error(self, eval_init_patch, eval_evaluate_patch):
        """Ensure the behavior is correct when an exception is raised
        during evaluation.
        """
        with patch.object(self.ind, '_update_model_compute_costs',
                          autospec=True, return_value=(6, 7)):
            with self.assertRaisesRegex(UserWarning, 'Dummy exception for t'):
                self.ind.evaluate(glm_mgr=self.mock_glm, db_conn=self.mock_db)

        # A failed evaluation should result in an infinite fitness, and
        # penalties should be None.
        self.assertEqual(self.ind.fitness, np.inf)
        self.assertIsNone(self.ind.penalties)


class PatchSubprocessResult:
    def __init__(self):
        self.returncode = 0


class EvaluatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_13)
        # 20 second model runtime.
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 0, 20)

        # Prep the GLMManager, as required to run an individual's
        # "evaluate" method.
        ga.prep_glm_mgr(cls.glm_mgr, starttime=cls.starttime,
                        stoptime=cls.stoptime)

    def setUp(self):
        """Get a database connection and a copy of the GLMManager."""
        # Mock the db connection for speed. Though this doesn't actually
        # seem to help with speed...
        self.db_conn = \
            unittest.mock.create_autospec(MySQLdb.connection)
        self.glm_fresh = deepcopy(self.glm_mgr)
        # Patch the truncate table call for speed.
        with patch('pyvvo.db.truncate_table', autospec=True):
            self.evaluator = ga._Evaluator(uid=23, glm_mgr=self.glm_fresh,
                                           db_conn=self.db_conn)

    def test_init_bad_glm_mgr(self):
        with self.assertRaisesRegex(TypeError, 'glm_mgr must be a glm\.GLM'):
            ga._Evaluator(uid=7, glm_mgr={'hello': 'there'},
                          db_conn=self.db_conn)

    def test_init_bad_db_conn(self):
        with self.assertRaisesRegex(TypeError, 'db_conn must be a MySQLdb'):
            ga._Evaluator(uid=7, glm_mgr=self.glm_fresh,
                          db_conn=7)

    def test_init_uid(self):
        # hard-coding for the win!
        self.assertEqual(self.evaluator.uid, 23)

    def test_init_starttime(self):
        self.assertEqual(self.starttime.strftime(self.glm_fresh.DATE_FORMAT),
                         self.evaluator.starttime)

    def test_init_triplex_table(self):
        self.assertEqual(self.evaluator.triplex_table,
                         '{}_{}'.format(ga.TRIPLEX_TABLE, self.evaluator.uid))

    def test_init_substation_table(self):
        self.assertEqual(self.evaluator.substation_table,
                         '{}_{}'.format(ga.SUBSTATION_TABLE,
                                        self.evaluator.uid))

    def test_init_truncates_tables(self):
        with patch('pyvvo.db.truncate_table', autospec=True) as p:
            self.evaluator = ga._Evaluator(uid=23, glm_mgr=self.glm_fresh,
                                           db_conn=self.db_conn)

        self.assertEqual(2, p.call_count)
        self.assertEqual(p.call_args_list,
                         [({'db_conn': self.db_conn,
                            'table': self.evaluator.triplex_table},),
                          ({'db_conn': self.db_conn,
                            'table': self.evaluator.substation_table},)])

    def test_init_modifies_glm_tables(self):
        tt = self.glm_fresh.find_object(obj_type='mysql.recorder',
                                        obj_name=ga.TRIPLEX_RECORDER)
        st = self.glm_fresh.find_object(obj_type='mysql.recorder',
                                        obj_name=ga.SUBSTATION_RECORDER)

        self.assertEqual(tt['table'], self.evaluator.triplex_table)
        self.assertEqual(st['table'], self.evaluator.substation_table)

    def test_voltage_penalty_none(self):
        with patch('pyvvo.db.execute_and_fetch_all', return_value=((None,),)):
            penalty = self.evaluator._voltage_penalty(query='who cares')

        self.assertEqual(penalty, 0)

    def test_voltage_penalty_float(self):
        p = 34856.2
        with patch('pyvvo.db.execute_and_fetch_all', return_value=((p,),)):
            penalty = self.evaluator._voltage_penalty(query='who cares')

        self.assertEqual(penalty, p)

    def test_voltage_penalty_bad_return_1(self):
        with patch('pyvvo.db.execute_and_fetch_all',
                   return_value=((7, 12),)):
            with self.assertRaisesRegex(ValueError, 'Voltage penalty queries'):
                self.evaluator._voltage_penalty(query='who cares')

    def test_voltage_penalty_bad_return_2(self):
        with patch('pyvvo.db.execute_and_fetch_all',
                   return_value=((7,), (8,))):
            with self.assertRaisesRegex(ValueError, 'Voltage penalty queries'):
                self.evaluator._voltage_penalty(query='who cares')

    def test_low_voltage_penalty(self):
        with patch.object(self.evaluator, '_voltage_penalty', autospec=True,
                          return_value=10) as p:
            with patch.dict(ga.CONFIG['costs'], {'voltage_violation_low': 2}):
                penalty = self.evaluator._low_voltage_penalty()

        p.assert_called_once()
        self.assertEqual(penalty, 10)

        # Do a fragile hard-coded assertion.
        self.assertEqual(p.call_args[1]['query'],
                         ("SELECT SUM((240 - measured_voltage_12_mag) * 2) "
                          "as penalty FROM triplex_23 WHERE "
                          "(measured_voltage_12_mag < 228.0 AND "
                          "t > '2013-04-01 12:00:00')"))

    def test_high_voltage_penalty(self):
        with patch.object(self.evaluator, '_voltage_penalty',
                          autospec=True,
                          return_value=42) as p:
            with patch.dict(ga.CONFIG['costs'],
                            {'voltage_violation_high': 3}):
                penalty = self.evaluator._high_voltage_penalty()

        p.assert_called_once()
        self.assertEqual(penalty, 42)

        # Do a fragile hard-coded assertion.
        self.assertEqual(p.call_args[1]['query'],
                         (
                             "SELECT SUM((measured_voltage_12_mag - 240) * 3) "
                             "as penalty FROM triplex_23 WHERE "
                             "(measured_voltage_12_mag > 252.0 AND "
                             "t > '2013-04-01 12:00:00')"))

    def test_get_substation_data_query(self):
        r = pd.DataFrame([[1, 2, 3, 9], [4, 5, 6, 10]],
                         columns=list(ga.SUBSTATION_COLUMNS))
        with patch('pandas.read_sql_query', autospec=True,
                   return_value=r) as p:
            data = self.evaluator._get_substation_data()

        p.assert_called_once()
        self.assertEqual(
            p.call_args[1]['sql'],
            "SELECT * FROM substation_23 WHERE t > '2013-04-01 12:00:00';")
        pd.testing.assert_frame_equal(r, data)

    def test_get_substation_data_empty_return(self):
        r = pd.DataFrame()
        with patch('pandas.read_sql_query', autospec=True,
                   return_value=r):
            with self.assertRaisesRegex(ValueError, 'No substation data was'):
                self.evaluator._get_substation_data()

    def test_get_substation_data_bad_columns(self):
        r = pd.DataFrame([[1, 3, 4], [4, 59, 1]])
        with patch('pandas.read_sql_query', autospec=True,
                   return_value=r):
            with self.assertRaisesRegex(ValueError,
                                        'Unexpected substation data columns.'):
                self.evaluator._get_substation_data()

    def test_pf_lag_penalty(self):
        pf = np.array([0.95, -0.97, 0.93, 0.99, 0.98])
        with patch.dict(ga.CONFIG, {'limits': {'power_factor_lag': 0.98},
                                    'costs': {'power_factor_lag': 3}}):
            penalty = self.evaluator._pf_lag_penalty(pf)

        self.assertAlmostEqual(penalty, 3 * 3 + 5 * 3, places=10)

    def test_pf_lead_penalty(self):
        pf = np.array([0.95, -0.97, 0.93, 0.99, 0.98])
        with patch.dict(ga.CONFIG, {'limits': {'power_factor_lead': 0.98},
                                    'costs': {'power_factor_lead': 16}}):
            penalty = self.evaluator._pf_lead_penalty(pf)

        self.assertAlmostEqual(penalty, 16, places=10)

    def test_power_factor_penalty(self):
        """Ensure helper methods are called correctly. No need to check
        accuracy, as all the helper functions themselves have been
        tested.
        """
        data = pd.DataFrame([[1, 1], [2, -2]],
                            columns=[ga.SUBSTATION_REAL_POWER,
                                     ga.SUBSTATION_REACTIVE_POWER])
        a = np.array([0.99, -0.99])
        with patch('pyvvo.utils.power_factor', autospec=True,
                   return_value=a) as pf:
            with patch.object(self.evaluator, '_pf_lead_penalty',
                              autospec=True, return_value=3) as p_lead:
                with patch.object(self.evaluator, '_pf_lag_penalty',
                                  autospec=True, return_value=7) as p_lag:
                    lead, lag = self.evaluator._power_factor_penalty(data)

        pf.assert_called_once()
        np.testing.assert_array_equal(np.array([1 + 1j, 2 + 1j * -2]),
                                      pf.call_args[0][0])
        p_lead.assert_called_once()
        np.testing.assert_array_equal(a, p_lead.call_args[0][0])
        p_lag.assert_called_once()
        np.testing.assert_array_equal(a, p_lag.call_args[0][0])
        self.assertEqual(3, lead)
        self.assertEqual(7, lag)

    def test_energy_penalty(self):
        data = pd.DataFrame([1, 2, 3, 4, 5, 6, 7000],
                            columns=[ga.SUBSTATION_ENERGY])
        with patch.dict(ga.CONFIG['costs'], {'energy': 3}):
            penalty = self.evaluator._energy_penalty(data)

        self.assertEqual(penalty, 21)

    @patch('pyvvo.ga._Evaluator._energy_penalty', autospec=True,
           return_value=5)
    @patch('pyvvo.ga._Evaluator._power_factor_penalty', autospec=True,
           return_value=(3, 4))
    @patch('pyvvo.ga._Evaluator._get_substation_data', autospec=True,
           return_value=pd.DataFrame([[1, 2, 3], ],
                                     columns=[ga.SUBSTATION_REAL_POWER,
                                              ga.SUBSTATION_REACTIVE_POWER,
                                              ga.SUBSTATION_ENERGY]))
    @patch('pyvvo.ga._Evaluator._low_voltage_penalty', autospec=True,
           return_value=2)
    @patch('pyvvo.ga._Evaluator._high_voltage_penalty', autospec=True,
           return_value=1)
    @patch('pyvvo.utils.run_gld', autospec=True,
           return_value=PatchSubprocessResult())
    @patch('pyvvo.glm.GLMManager.write_model', autospec=True,
           return_value=None)
    def test_evaluate(self, write_model_patch, run_gld_patch, hv_patch,
                      lv_patch, sub_data_patch, pf_patch, e_patch):
        """Patch everything and just make sure the appropriate helper
        method results get assigned to the correct penalties.
        """
        penalties = self.evaluator.evaluate()

        write_model_patch.assert_called_once()
        write_model_patch.assert_called_with(self.glm_fresh, 'model_23.glm')

        run_gld_patch.assert_called_once()
        run_gld_patch.assert_called_with('model_23.glm')

        hv_patch.assert_called_once()

        lv_patch.assert_called_once()

        sub_data_patch.assert_called_once()

        df = pd.DataFrame([[1, 2, 3], ],
                          columns=[ga.SUBSTATION_REAL_POWER,
                                   ga.SUBSTATION_REACTIVE_POWER,
                                   ga.SUBSTATION_ENERGY])

        pf_patch.assert_called_once()
        pd.testing.assert_frame_equal(df, pf_patch.call_args[1]['data'])

        e_patch.assert_called_once()
        pd.testing.assert_frame_equal(df, e_patch.call_args[1]['data'])

        self.assertDictEqual({'voltage_high': 1, 'voltage_low': 2,
                              'power_factor_lead': 3, 'power_factor_lag': 4,
                              'energy': 5},
                             penalties)


class MockIndividual:
    """Mock objects don't like being pickled.

    https://github.com/testing-cabal/mock/issues/139
    https://bugs.python.org/issue14577
    https://code.google.com/archive/p/mock/issues/139

    This is unfortunate, because this hard-coded mock is going to make
    testing more fragile.
    """

    def __init__(self, *args, **kwargs):
        self.fitness = None
        self.uid = 1
        self.penalties = None
        # self.chromosome = np.random.randint(0, 2, 100, dtype=np.bool)

    def evaluate(self, *args, **kwargs):
        self.fitness = 1
        self.penalties = {'p1': 10, 'p2': 30, 'p3': 0.1}


class MockIndividual2(MockIndividual):
    """Same as MockIndividual, except evaluate throws and exception."""
    def evaluate(self, *args, **kwargs):
        self.fitness = np.inf
        self.penalties = None
        raise RuntimeError('Dummy error for testing.')


class SleepyMockIndividual(MockIndividual):
    """Same as MockIndividual, except evaluate sleeps."""

    sleep_time = 0.02

    def evaluate(self, *args, **kwargs):
        sleep(self.sleep_time)
        self.fitness = 1
        self.penalties = {'p1': 10, 'p2': 30, 'p3': 0.1}


class EvaluateWorkerBadInputTestCase(unittest.TestCase):

    def test_bad_input_queue(self):
        with self.assertRaisesRegex(TypeError, 'input_queue must be '):
            ga._evaluate_worker(input_queue=['hi'], logging_queue=[],
                                output_queue=[], glm_mgr=None)


class EvaluateWorkerTestCase(unittest.TestCase):
    """Test _evaluate_worker function."""

    def setUp(self) -> None:
        # Create queues.
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()
        self.logging_queue = mp.Queue()
        self.glm_mgr = create_autospec(GLMManager)

        self.p = mp.Process(target=ga._evaluate_worker,
                            kwargs={'input_queue': self.input_queue,
                                    'output_queue': self.output_queue,
                                    'logging_queue': self.logging_queue,
                                    'glm_mgr': self.glm_mgr})

        self.p.start()

    def tearDown(self) -> None:
        self.input_queue.put(None)
        # Sleep to ensure we don't encounter a race condition.
        sleep(0.01)
        self.assertFalse(self.p.is_alive())

    def test_expected_behavior(self):
        """Mock up an individual and glm_mgr. IMPORTANT NOTE: Since
        Mock objects don't seem to be pickleable, we need to use the
        pretty meh class, MockIndividual.

        TODO: This test could cause tests to hang indefinitely if
            something gets broken. The .join() call should probably
            be wrapped with some sort of timeout.
        """
        ind_in = MockIndividual()

        self.input_queue.put(ind_in)

        self.input_queue.join()

        log_out = self.logging_queue.get()

        ind_out = self.output_queue.get_nowait()

        self.assertEqual(ind_out.fitness, 1)
        self.assertIn('time', log_out)
        log_out.pop('time')
        self.assertDictEqual({'uid': 1, 'fitness': 1,
                              'penalties': {'p1': 10, 'p2': 30, 'p3': 0.1}},
                             log_out)

    def test_exception_handling(self):
        """When an exception is thrown during evaluation, the worker
        should catch it and put the exception into the queue.
        """
        # Create a mock individual which will raise an exception upon
        # evaluation.
        ind_in = MockIndividual2()

        self.input_queue.put(ind_in)

        self.input_queue.join()

        # Extract the logging output and the individual.
        log_out = self.logging_queue.get()
        ind_out = self.output_queue.get_nowait()

        # Errors should result in infinite fitness.
        self.assertEqual(ind_out.fitness, np.inf)
        # There should be an error and uid field.
        self.assertIn('error', log_out)
        self.assertIn('uid', log_out)

        with self.assertRaisesRegex(RuntimeError,
                                    'Dummy error for testing.'):
            raise log_out['error']

        # Despite the error, the process should still be alive.
        self.assertTrue(self.p.is_alive())

    def test_none_terminates_processes(self):
        # Process should start out alive.
        self.assertTrue(self.p.is_alive())
        # Putting None in should terminate the process.
        self.input_queue.put_nowait(None)
        # The task should be marked as done, so join should work. Use a
        # time limit so our test doesn't hang on failure.
        with time_limit(1):
            self.input_queue.join()

        # Give it a moment to die, then ensure it's dead.
        sleep(0.01)
        self.assertFalse(self.p.is_alive())


class LoggingThreadTestCase(unittest.TestCase):
    """Test _logging_thread function."""

    def setUp(self) -> None:
        self.q = mp.Queue()

        self.t = threading.Thread(target=ga._logging_thread,
                                  kwargs={'logging_queue': self.q})

        self.t.start()

    def tearDown(self) -> None:
        self.assertTrue(self.t.is_alive())

        # Kill the thread.
        self.q.put(None)

        # Sleep to ensure we don't encounter a race condition.
        sleep(0.01)

        # Ensure the thread is dead.
        self.assertFalse(self.t.is_alive())

    def test_expected_normal_behavior(self):
        """NOTE: Couldn't get assertLogs to work here, so this test is
        close to useless. I guess it makes sure things don't error out,
        which is something.
        """
        self.assertTrue(self.t.is_alive())

        # Ensure we get a debug message.
        with self.assertLogs(level='DEBUG', logger=ga.LOG):
            self.q.put({'uid': 7, 'fitness': 16, 'penalties': {'p': 16},
                        'time': 5})
            # Sleep to ensure we get logs out.
            sleep(0.01)

    def test_exception_behavior(self):
        """If given an 'error' key, the behavior is different."""
        self.assertTrue(self.t.is_alive())

        # Generate an exception.
        my_e = None
        try:
            raise UserWarning('Dummy exception')
        except UserWarning as e:
            my_e = e

        with self.assertLogs(level='ERROR', logger=ga.LOG):
            # Put exception in queue.
            self.q.put({'uid': 42, 'error': my_e})
            # Sleep to allow logging to occur.
            sleep(0.01)


class TournamentTestCase(unittest.TestCase):
    """Test _tournament"""
    @classmethod
    def setUpClass(cls):
        # Initialize a population of mock individuals.
        cls.population = [create_autospec(ga.Individual) for _ in range(5)]

        # Set their fitnesses.
        cls.population[0].fitness = 5
        cls.population[1].fitness = 3
        cls.population[2].fitness = 7
        cls.population[3].fitness = 2
        cls.population[4].fitness = 17

    def test_correct_return(self):
        best = ga._tournament(population=self.population, tournament_size=5,
                              n=2)

        self.assertEqual(2, len(best))
        self.assertEqual(2, self.population[best[0]].fitness)
        self.assertEqual(3, self.population[best[1]].fitness)

    def test_sort(self):
        with patch('numpy.random.choice', return_value=[4, 1]) as p:
            best = ga._tournament(population=self.population,
                                  tournament_size=2, n=2)

        p.assert_called_once()

        self.assertEqual(2, len(best))
        self.assertEqual(3, self.population[best[0]].fitness)
        self.assertEqual(17, self.population[best[1]].fitness)

    def test_choice(self):
        with patch('numpy.random.choice', wraps=np.random.choice) as p:
            best = ga._tournament(population=self.population,
                                  tournament_size=4, n=3)

        p.assert_called_once()

        np.testing.assert_array_equal(np.array([0, 1, 2, 3, 4]),
                                      p.call_args[1]['a'])
        self.assertEqual(4, p.call_args[1]['size'])
        self.assertFalse(p.call_args[1]['replace'])

        self.assertEqual(3, len(best))


class DumpQueueTestCase(unittest.TestCase):
    """Test _dump_queue"""

    @classmethod
    def setUpClass(cls):
        cls.i_orig = [1, 2, 3]

    def setUp(self):
        self.i = [*self.i_orig]
        self.q = mp.Queue()

        for n in range(4, 7):
            self.q.put(n)

    def test_correct(self):
        # Need to sleep due to the small delay for the background thread
        # which stuff things into the queue.
        sleep(0.01)
        self.assertFalse(self.q.empty())

        i2 = ga._dump_queue(q=self.q, i=self.i)

        self.assertIs(i2, self.i)
        self.assertListEqual([1, 2, 3, 4, 5, 6], self.i)

        sleep(0.01)

        self.assertTrue(self.q.empty())


class DrainQueueTestCase(unittest.TestCase):
    """Test _drain_queue."""

    def test_drain_makes_empty_joinable(self):

        q = mp.JoinableQueue()

        for n in range(4, 7):
            q.put(n)

        # Need to sleep so that empty won't return False while the
        # background thread dumps stuff into the queue.
        sleep(0.01)
        self.assertFalse(q.empty())

        ga._drain_queue(q)

        sleep(0.01)

        self.assertTrue(q.empty())

        # Ensure all tasks were marked as done.
        with time_limit(1):
            q.join()

    def test_drain_makes_empty_not_joinable(self):
        q = mp.Queue()

        for n in range(4, 7):
            q.put(n)

        # Need to sleep so that empty won't return False while the
        # background thread dumps stuff into the queue.
        sleep(0.01)
        self.assertFalse(q.empty())

        ga._drain_queue(q)

        sleep(0.01)

        self.assertTrue(q.empty())


class PopulationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_9500)
        # 20 second model runtime.
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 0, 20)

        # Get regulators and capacitors.
        reg_df = pd.read_csv(_df.REGULATORS_9500)
        cap_df = pd.read_csv(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

        # It seems we don't have a way of getting capacitor state from
        # the CIM (which is where those DataFrames originate from). So,
        # let's randomly command each capacitor.
        for c in cls.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                c.state = np.random.randint(low=0, high=2, size=None,
                                            dtype=int)
            elif isinstance(c, dict):
                for cc in c.values():
                    cc.state = np.random.randint(low=0, high=2, size=None,
                                                 dtype=int)

        cls.ga_config = {
            "probabilities": {
                "mutate_individual": 0.2,
                "mutate_bit": 0.05,
                "crossover": 0.7
            },
            "intervals": {
                "sample": 5,
                "minimum_timestep": 1
            },
            "population_size": 14,
            "generations": 3,
            "top_fraction": 0.1,
            "total_fraction": 0.5,
            "tournament_fraction": 0.2}

        cls.pop_obj = cls.helper_create_pop_obj()

    # noinspection PyUnresolvedReferences
    @classmethod
    def helper_create_pop_obj(cls, ga_dict=None):
        """Helper to create a population object if we're concerned about
        altering state.
        """
        if ga_dict is None:
            ga_dict = cls.ga_config

        with patch.dict(ga.CONFIG['ga'], ga_dict):
            pop_obj = ga.Population(regulators=cls.regs,
                                    capacitors=cls.caps,
                                    glm_mgr=deepcopy(cls.glm_mgr),
                                    starttime=cls.starttime,
                                    stoptime=cls.stoptime)

        return pop_obj

    def test_prob_mutate_individual(self):
        self.assertEqual(0.2, self.pop_obj.prob_mutate_individual)

    def test_prob_mutate_bit(self):
        self.assertEqual(0.05, self.pop_obj.prob_mutate_bit)

    def test_prob_crossover(self):
        self.assertEqual(0.7, self.pop_obj.prob_crossover)

    def test_population_size(self):
        self.assertEqual(14, self.pop_obj.population_size)

    def test_generations(self):
        self.assertEqual(3, self.pop_obj.generations)

    def test_top_fraction(self):
        self.assertEqual(0.1, self.pop_obj.top_fraction)

    def test_total_fraction(self):
        self.assertEqual(0.5, self.pop_obj.total_fraction)

    def test_tournament_fraction(self):
        self.assertEqual(0.2, self.pop_obj.tournament_fraction)

    def test_top_keep(self):
        self.assertEqual(2, self.pop_obj.top_keep)

    def test_total_keep(self):
        self.assertEqual(7, self.pop_obj.total_keep)

    def test_tournament_size(self):
        self.assertEqual(3, self.pop_obj.tournament_size)

    def test_regulators(self):
        self.assertIs(self.regs, self.pop_obj.regulators)

    def test_capacitors(self):
        self.assertIs(self.caps, self.pop_obj.capacitors)

    def test_glm_mgr(self):
        self.assertIsInstance(self.pop_obj.glm_mgr, GLMManager)

    def test_starttime(self):
        self.assertIs(self.starttime, self.pop_obj.starttime)

    def test_stoptime(self):
        self.assertIs(self.stoptime, self.pop_obj.stoptime)

    def test_map_chromosome_called(self):
        fresh_mgr = deepcopy(self.glm_mgr)
        with patch('pyvvo.ga.map_chromosome', wraps=ga.map_chromosome) as p:
            pop_obj = ga.Population(regulators=self.regs,
                                    capacitors=self.caps,
                                    glm_mgr=fresh_mgr,
                                    starttime=self.starttime,
                                    stoptime=self.stoptime)

        p.assert_called_once()
        p.assert_called_with(regulators=self.regs, capacitors=self.caps)

    def test_map_chromosome_outputs(self):
        self.assertIsInstance(self.pop_obj.chrom_map, dict)
        self.assertIsInstance(self.pop_obj.chrom_len, int)
        self.assertIsInstance(self.pop_obj.num_eq, int)

    def test_prep_glm_mgr_called(self):
        fresh_mgr = deepcopy(self.glm_mgr)
        with patch('pyvvo.ga.prep_glm_mgr', wraps=ga.prep_glm_mgr) as p:
            pop_obj = ga.Population(regulators=self.regs,
                                    capacitors=self.caps,
                                    glm_mgr=fresh_mgr,
                                    starttime=self.starttime,
                                    stoptime=self.stoptime)

        p.assert_called_once()
        p.assert_called_with(glm_mgr=pop_obj.glm_mgr,
                             starttime=pop_obj.starttime,
                             stoptime=pop_obj.stoptime)

    def test_uid_counter(self):
        self.assertTrue(hasattr(self.pop_obj.uid_counter, '__next__'))

    def test_input_queue(self):
        """Haven't found a good way to type check, so check attributes.
        """
        self.assertTrue(hasattr(self.pop_obj.input_queue, 'put'))
        self.assertTrue(hasattr(self.pop_obj.input_queue, 'put_nowait'))
        self.assertTrue(hasattr(self.pop_obj.input_queue, 'get'))
        self.assertTrue(hasattr(self.pop_obj.input_queue, 'get_nowait'))
        self.assertTrue(hasattr(self.pop_obj.input_queue, 'join'))
        self.assertTrue(hasattr(self.pop_obj.input_queue, 'task_done'))

    def test_output_queue(self):
        """Haven't found a good way to type check, so check attributes.
        """
        self.assertTrue(hasattr(self.pop_obj.output_queue, 'put'))
        self.assertTrue(hasattr(self.pop_obj.output_queue, 'put_nowait'))
        self.assertTrue(hasattr(self.pop_obj.output_queue, 'get'))
        self.assertTrue(hasattr(self.pop_obj.output_queue, 'get_nowait'))

    def test_logging_queue(self):
        """Haven't found a good way to type check, so check attributes.
        """
        self.assertTrue(hasattr(self.pop_obj.logging_queue, 'put'))
        self.assertTrue(hasattr(self.pop_obj.logging_queue, 'put_nowait'))
        self.assertTrue(hasattr(self.pop_obj.logging_queue, 'get'))
        self.assertTrue(hasattr(self.pop_obj.logging_queue, 'get_nowait'))

    def test_logging_thread(self):
        self.assertIsInstance(self.pop_obj.logging_thread, threading.Thread)
        self.assertTrue(self.pop_obj.logging_thread.is_alive())
        self.assertIs(self.pop_obj.logging_queue,
                      self.pop_obj.logging_thread._kwargs['logging_queue'])

    def test_processes(self):
        self.assertIsInstance(self.pop_obj.processes, list)

        for p in self.pop_obj.processes:
            self.assertIsInstance(p, mp.Process)

        self.assertTrue(self.pop_obj.all_processes_alive)

    def test_ind_init(self):
        self.assertDictEqual(self.pop_obj.ind_init,
                             {'chrom_len': self.pop_obj.chrom_len,
                              'num_eq': self.pop_obj.num_eq,
                              'chrom_map': self.pop_obj.chrom_map})

    def test_all_chromosomes(self):
        self.assertIsInstance(self.pop_obj.all_chromosomes, list)

    def test_population(self):
        self.assertIsInstance(self.pop_obj.population, list)

    def test_chrom_already_existed(self):
        """Test _chrom_already_existed"""
        # Create a list to patch our population object's all_chromosomes
        # attribute.
        all_chrom = [np.array([1, 2, 3, 4]),
                     np.array([1., 2., 5., 6.]),
                     np.array([23, 23, 89, 42, 78])]

        with patch.object(self.pop_obj, '_all_chromosomes', all_chrom):
            self.assertFalse(
                self.pop_obj._chrom_already_existed(np.array([1, 2, 3, 5])))

            self.assertFalse(
                self.pop_obj._chrom_already_existed(np.array([1.01, 2, 3, 4])))

            self.assertTrue(
                self.pop_obj._chrom_already_existed(
                    np.array([1, 2, 5, 6])))

    def test_init_individual(self):
        """Large, relatively comprehensive test of _init_individual"""
        # Get fresh population object so we don't contaminate state.
        pop_obj = self.helper_create_pop_obj()

        # Entering this test, there should be no chromosomes in
        # all_chromosomes.
        self.assertEqual(0, len(pop_obj.all_chromosomes))

        # Initialize one.
        ind = pop_obj._init_individual(special_init='max')
        self.assertIsInstance(ind, ga.Individual)
        for key, value in pop_obj.ind_init.items():
            self.assertEqual(value, getattr(ind, key))

        # The length of all_chromosomes should now be one.
        self.assertEqual(1, len(pop_obj.all_chromosomes))

        # The chromosome should be the same as that of our ind.
        np.testing.assert_array_equal(ind.chromosome,
                                      pop_obj.all_chromosomes[0])

        # However, it should be a copy.
        self.assertIsNot(ind.chromosome, pop_obj.all_chromosomes[0])

        # Initialize another, ensuring we get an exception.
        with self.assertRaises(ga.ChromosomeAlreadyExistedError):
            pop_obj._init_individual(special_init='max')

        # The length of all_chromosomes should still be one.
        self.assertEqual(1, len(pop_obj.all_chromosomes))

        # Now, we should be able to alternatively initialize an
        # individual with a chrom_override even if it's chromosome is
        # already in the master list.

        new_ind = pop_obj._init_individual(
            chrom_override=ind.chromosome.copy())

        # The length of all_chromosomes should still be one.
        self.assertEqual(1, len(pop_obj.all_chromosomes))

        # This individual should have an identical chromosome to our
        # original.
        np.testing.assert_array_equal(ind.chromosome, new_ind.chromosome)

        # Finally, we should be able to create a new individual with a
        # chromosome override and have it tracked.
        c = ind.chromosome.copy()
        # Flip the first bit.
        c[0] = 1 - c[0]
        pop_obj._init_individual(chrom_override=c)

        self.assertEqual(2, len(pop_obj.all_chromosomes))

        np.testing.assert_array_equal(c, pop_obj.all_chromosomes[1])

    def test_init_individual_loop_limit(self):
        """Ensure we don't loop forever in _init_individual."""
        # Get fresh population object so we don't contaminate state.
        pop_obj = self.helper_create_pop_obj()

        # Mock up an individual.
        mock_ind = create_autospec(ga.Individual)
        mock_ind.chromosome = np.array([0, 1, 0, 1], dtype=np.bool)

        # Ensure everytime we ask for an individual we get the same
        # one.
        with patch('pyvvo.ga.Individual', return_value=mock_ind) as p:
            # First creation should be successful.
            i1 = pop_obj._init_individual()

            # Second should fail.
            with self.assertRaisesRegex(ga.ChromosomeAlreadyExistedError,
                                        'After 100 attempts, we failed to'):
                pop_obj._init_individual()

        # Including our first initialization, we should have attempted
        # to create 101 individuals.
        self.assertEqual(101, p.call_count)

        # Really just a safety net to ensure our patch worked.
        self.assertIs(mock_ind, i1)

        # There should be a single chromosome in the tracker.
        self.assertEqual(1, len(pop_obj.all_chromosomes))

        # The first element should be equal to our mock_ind's
        # chromosome.
        np.testing.assert_array_equal(mock_ind.chromosome,
                                      pop_obj.all_chromosomes[0])

        # However, it should be a copy.
        self.assertIsNot(mock_ind.chromosome, pop_obj.all_chromosomes[0])

    def test_initialize_population_error_1(self):
        with patch.object(self.pop_obj, '_population', [1, 2, 3]):
            with self.assertRaisesRegex(ValueError, 'initialize_population'):
                self.pop_obj.initialize_population()

    def test_initialize_population_error_2(self):
        # Get the initialization configuration so we can override
        # part of it.
        # noinspection PyDictCreation
        config = {**self.ga_config}
        config['population_size'] = 2

        # Get population object.
        pop_obj = self.helper_create_pop_obj(ga_dict=config)

        with self.assertRaisesRegex(ValueError, 'seeds the population with 3'):
            pop_obj.initialize_population()

    def test_initialize_population(self):

        # Get a fresh Population object so we don't mess up state for
        # other tests.
        pop_obj = self.helper_create_pop_obj()

        with patch.object(pop_obj, '_init_individual',
                          wraps=pop_obj._init_individual) as p:
            pop_obj.initialize_population()

        self.assertEqual(pop_obj.population_size, p.call_count)
        self.assertEqual(pop_obj.population_size, len(pop_obj.population))

        for i in pop_obj.population:
            self.assertIsInstance(i, ga.Individual)

        # Check the seeding.
        self.assertDictEqual(p.call_args_list[0][1],
                             {'special_init': 'max'})
        self.assertDictEqual(p.call_args_list[1][1],
                             {'special_init': 'min'})
        self.assertDictEqual(p.call_args_list[2][1],
                             {'special_init': 'current_state'})

    def test_evaluate_population_simple(self):
        """Given a population with no evaluated individuals, ensure
        successful evaluation.
        """
        # Get a new population object to avoid state contamination.
        pop_obj = self.helper_create_pop_obj()

        # Initialize the population with mocked individuals.
        for _ in range(pop_obj.population_size):
            pop_obj.population.append(MockIndividual())

        with self.assertLogs(logger=pop_obj.log, level='INFO'):
            pop_obj.evaluate_population()

        # evaluate_population overrides the part of the population which
        # has not yet been evaluated, so make sure we get the same
        # number of individuals back.
        self.assertEqual(pop_obj.population_size, len(pop_obj.population))

        # Ensure all the individuals now have a fitness which is not
        # None.
        for i in pop_obj.population:
            self.assertIsNotNone(i.fitness)

    def test_evaluate_population_partial_evaluation(self):
        """Given a population with some evaluated individuals and some
        not evaluated individuals, ensure the function behaves
        correctly.
        """
        # Get a new population object to avoid state contamination.
        pop_obj = self.helper_create_pop_obj()

        # Override the population size.
        pop_obj._population_size = 5

        # Hand make 5 mock individuals.
        ind0 = MockIndividual()
        ind1 = MockIndividual()
        ind2 = MockIndividual()
        ind3 = MockIndividual()
        ind4 = MockIndividual()

        # Set fitness values for ind1 and ind3, leaving the rest None.
        ind1.fitness = 7
        ind3.fitness = 2

        # Put all these individuals in the population.
        pop_obj._population = [ind0, ind1, ind2, ind3, ind4]

        # Run the evaluation.
        pop_obj.evaluate_population()

        # We should have 5 members still.
        self.assertEqual(5, len(pop_obj.population))

        # ind1 and ind3 should not have been evaluated. Thus they
        # should be in positions 0 and 1.
        self.assertIs(ind1, pop_obj.population[0])
        self.assertIs(ind3, pop_obj.population[1])

        # Due to the pickling that occurs in multiprocessing, the rest
        # of the individuals should not be present.
        for i in [ind0, ind2, ind4]:
            self.assertNotIn(i, pop_obj.population)

        # All the individuals should have a fitness which is not None.
        for i in pop_obj.population:
            self.assertIsNotNone(i.fitness)

    def test_evaluate_population_error(self):
        with self.assertRaisesRegex(ValueError, 'evaluate_population should '):
            self.pop_obj.evaluate_population()

    def test_natural_selection(self):
        # Get a new population object to avoid state contamination.
        n = 10
        # noinspection PyDictCreation
        d = {**self.ga_config}
        d['population_size'] = n

        # Get a Population object.
        pop_obj = self.helper_create_pop_obj(ga_dict=d)

        # Add mocked individuals to the population.
        for _ in range(n):
            pop_obj._population.append(MockIndividual())

        # Manually assign fitnesses.
        pop_obj._population[0].fitness = 7
        pop_obj._population[1].fitness = 3  # 1
        pop_obj._population[2].fitness = 5  # 3
        pop_obj._population[3].fitness = 10
        pop_obj._population[4].fitness = 1  # 0
        pop_obj._population[5].fitness = 8
        pop_obj._population[6].fitness = 4  # 2
        pop_obj._population[7].fitness = 35
        pop_obj._population[8].fitness = 15
        pop_obj._population[9].fitness = 9

        # Run the selection.
        with patch('pyvvo.ga._tournament', wraps=ga._tournament) as p:
            pop_obj.natural_selection()

        # Ensure our population has the number of individuals we expect.
        self.assertEqual(pop_obj.total_keep, len(pop_obj.population))

        # Ensure the individual in the first position has the expected
        # fitness.
        self.assertEqual(1, pop_obj.population[0].fitness)

        # In this case, we should be keeping one individual via
        # elitism.
        self.assertEqual(1, pop_obj.top_keep)

        # Ensure _tournament was called four times.
        self.assertEqual(4, p.call_count)

        # Ensure the call args to _tournament are as expected.
        for idx in range(4):
            # In this case, all 4 calls should have a tournament size of 2.
            self.assertEqual(2, p.call_args_list[idx][1]['tournament_size'])
            # We're always just asking for a single individual.
            self.assertEqual(1, p.call_args_list[idx][1]['n'])
            # The population should be shrinking as we go.
            # Unfortunately this test doesn't work, as the mock points
            # to the object which was passed in, and doesn't make a
            # copy. So, it just sees the current population. However,
            # I manually verified the population is shrinking as we go.
            # self.assertEqual(9 - idx,
            #                  len(p.call_args_list[idx][1]['population']))
            #

    def test_mutate(self):
        """Test _mutate."""
        # Get a fresh population object.
        pop_obj = self.helper_create_pop_obj()

        # Create a chromosome.
        c = np.array([1, 2, 3, 4])

        # Create mock individual.
        mock_ind = create_autospec(ga.Individual)
        mock_ind.chromosome = c
        mock_ind.uid = 0

        # Call mutation.
        with patch.object(pop_obj, '_chrom_already_existed',
                          wraps=pop_obj._chrom_already_existed) as p:
            pop_obj._mutate(mock_ind)

        p.assert_called_once()

        # Now add the chromosome to the record of chromosomes.
        pop_obj._all_chromosomes = [c.copy()]

        # Since the mock mutate method won't actually do anything, we'll
        # get an error here.
        with self.assertRaisesRegex(ga.ChromosomeAlreadyExistedError,
                                    'attempted mutations, the individual'):
            pop_obj._mutate(mock_ind)

        # Ensure mutate was called 101 times. Once on the first call to
        # _mutate, and then 100 times on the second.
        c = 0
        for call in mock_ind.mock_calls:
            if call[0] == 'mutate':
                c += 1

        self.assertEqual(101, c)

    def test_crossover_and_mutate(self):
        """Test crossover_and_mutate. The pieces of this function have
        been factored to use helper functions, so get ready for plenty
        of mocking.
        """
        # Get the initialization configuration so we can override
        # part of it.
        # noinspection PyDictCreation
        config = {**self.ga_config}
        # Use a population size of 10.
        config['population_size'] = 10
        # Patch the probaility of crossover.
        # noinspection PyUnresolvedReferences
        config['probabilities']['crossover'] = 0.2

        # Get a population object.
        pop_obj = self.helper_create_pop_obj(ga_dict=config)

        # Give us a half-full population. Just use numbers for
        # simplicity.
        pop_obj._population = [1, 2, 3, 4, 5]

        # We'll be patching np.random.rand with a simple counter.
        counter = itertools.count(start=0.1, step=0.1)

        # Begin the patching bonanza!
        with patch('numpy.random.rand', wraps=counter.__next__) as p_rand:
            with patch.object(pop_obj, '_get_two_parents',
                              return_value=[8, 9]) as p_parents:
                with patch.object(pop_obj, '_crossover_and_mutate',
                                  return_value=[1, 2]) as p_cm:
                    with patch.object(pop_obj, '_asexual_reproduction',
                                      return_value=[3, 4]) as p_ar:
                        pop_obj.crossover_and_mutate()

        # Assertion time.
        # For starters, the random function should be called three times
        # since we need to replace 5 individuals and each loop creates
        # two.
        self.assertEqual(3, p_rand.call_count)
        # Same goes for _get_two_parents.
        self.assertEqual(3, p_parents.call_count)

        # Since we made the crossover probability 0.2 and we patched the
        # random call with a counter with a step of 0.1, we can expect
        # _crossover_and_mutate to be called once, and
        # _asexual_reproduction to be called twice.
        self.assertEqual(1, p_cm.call_count)
        self.assertEqual(2, p_ar.call_count)

        # Finally, ensure the truncation worked correctly. Three loops
        # would result in 6 individuals, but we only want to keep 5 so
        # that the population is the correct size.
        self.assertEqual(10, len(pop_obj.population))

    def test_sort_population_simple(self):
        """Test sort_population."""
        # Get a population object to which we can add individuals to
        # without messing up state for other tests.
        pop_obj = self.helper_create_pop_obj()

        # Create a few mock individuals.
        ind1 = MockIndividual()
        ind2 = MockIndividual()
        ind3 = MockIndividual()
        ind4 = MockIndividual()

        # Set their fitnesses.
        ind1.fitness = 10
        ind2.fitness = 7
        ind3.fitness = np.inf
        ind4.fitness = 2

        # Put them in the population.
        pop_obj._population = [ind1, ind2, ind3, ind4]

        # Sort.
        pop_obj.sort_population()

        self.assertIs(pop_obj.population[0], ind4)
        self.assertIs(pop_obj.population[1], ind2)
        self.assertIs(pop_obj.population[2], ind1)
        self.assertIs(pop_obj.population[3], ind3)

    def test_sort_population_with_Nones(self):
        pop_obj = self.helper_create_pop_obj()

        # Create a few mock individuals.
        ind1 = MockIndividual()
        ind2 = MockIndividual()

        ind1.fitness = 1
        ind2.fitness = None

        pop_obj._population = [ind1, ind2]

        with self.assertRaisesRegex(TypeError, 'While attempting to sort the'):
            pop_obj.sort_population()

    def test_get_two_parents(self):
        """Test _get_two_parents."""
        # Get population object
        pop_obj = self.helper_create_pop_obj()

        # Put some mock individuals in the population.
        for k in range(pop_obj.population_size):
            pop_obj._population.append(MockIndividual())

        # Loop over the population and assign random fitness values.
        for i in pop_obj.population:
            i.fitness = np.random.rand()

        # Patch the call to tournament so we can ensure it's being
        # called correctly.
        with patch('pyvvo.ga._tournament', wraps=ga._tournament) as p:
            parent1, parent2 = pop_obj._get_two_parents()

        p.assert_called_once()
        p.assert_called_with(population=pop_obj.population,
                             tournament_size=pop_obj.tournament_size,
                             n=2)

        self.assertIsInstance(parent1, MockIndividual)
        self.assertIn(parent1, pop_obj.population)

        self.assertIsInstance(parent2, MockIndividual)
        self.assertIn(parent2, pop_obj.population)

    def test__crossover_and_mutate(self):
        """Massive test of _crossover_and_mutate."""
        # Get the initialization configuration so we can override
        # part of it.
        # noinspection PyDictCreation
        config = {**self.ga_config}
        # Use a population size of 3 so we don't get an error in
        # initialize_population.
        config['population_size'] = 3
        # noinspection PyUnresolvedReferences
        config['probabilities']['mutate_individual'] = 0.3

        # Create a population object.
        pop_obj = self.helper_create_pop_obj(ga_dict=config)

        # Initialize the population with real individuals.
        pop_obj.initialize_population()

        # Grab parents.
        p1 = pop_obj.population[0]
        p2 = pop_obj.population[1]

        # Patch the random call, and call _crossover_and_mutate
        with patch('numpy.random.rand', return_value=np.array([0.2, 0.4])) \
                as p_rand:
            # Patch calls to _mutate so we can count how many times it's
            # called.
            with patch.object(pop_obj, '_mutate', wraps=pop_obj._mutate) \
                    as p_mutate:
                children = pop_obj._crossover_and_mutate(p1, p2)

        # We only do one random draw here.
        p_rand.assert_called_once()

        # _mutate should only have been called once. Note that this is
        # a little on the fragile side, as there is a chance it gets
        # called twice if the crossover resulted in a non-unique
        # individual. If this starts failing, consider changing the
        # random seed.
        p_mutate.assert_called_once()

        # Ensure we're getting individuals back and that they are
        # different than their parents.
        for c in children:
            self.assertIsInstance(c, ga.Individual)
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     p1.chromosome,
                                     c.chromosome)
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     p2.chromosome,
                                     c.chromosome)

        # Incest should result in forced mutations, as the children
        # would otherwise be identical to the parents. Genetic algorithm
        # jokes/irony, nice.
        with patch('numpy.random.rand', return_value=np.array([0.4, 0.4])) \
                as p_rand:
            with patch.object(pop_obj, '_mutate', wraps=pop_obj._mutate) \
                    as p_mutate:
                pop_obj._crossover_and_mutate(p1, p1)

        p_rand.assert_called_once()
        self.assertEqual(2, p_mutate.call_count)

    def test_asexual_reproduction(self):
        """Test _asexual_reproduction. This function is quite simple,
        so really we just need to ensure the right functions are called
        in the right order.
        """
        p1 = create_autospec(ga.Individual)
        p2 = create_autospec(ga.Individual)

        with patch.object(self.pop_obj, '_init_individual') as p_init:
            with patch.object(self.pop_obj, '_mutate') as p_mutate:
                children = self.pop_obj._asexual_reproduction(p1, p2)

        # Start with simple call counts.
        self.assertEqual(2, p_init.call_count)
        self.assertEqual(2, p_mutate.call_count)

        # Ensure the first call to _init_individual used p1, and the
        # second used p2. Start with the chrom_override.
        self.assertIs(p1.chromosome.copy(),
                      p_init.call_args_list[0][1]['chrom_override'])
        self.assertIs(p2.chromosome.copy(),
                      p_init.call_args_list[1][1]['chrom_override'])
        # On to special_init.
        self.assertIsNone(p_init.call_args_list[0][1]['special_init'])
        self.assertIsNone(p_init.call_args_list[1][1]['special_init'])

        # Two parents in, two children out.
        self.assertEqual(2, len(children))

        # _mutate should first be called with child1, then child2.
        self.assertIs(children[0], p_mutate.call_args_list[0][0][0])
        self.assertIs(children[1], p_mutate.call_args_list[1][0][0])

    def test_lock(self):
        # These will error out if the lock doesn't have the
        # attributes.
        ac = self.pop_obj._lock.acquire
        re = self.pop_obj._lock.release

        try:
            # Acquire the lock.
            ac()

            # Try to get the lock.
            self.assertFalse(ac(blocking=False))
        finally:
            # Release it.
            re()

    # Couldn't get this method to work in a reasonable time and just
    # need to move on.
    # def test_dump_queue_into_population_locks(self):
    #     sleep_1 = 0.5
    #     sleep_2 = 0.2
    #
    #     # Create function to sleep
    #     def sleep_a_bit():
    #         sleep(sleep_1)
    #
    #     try:
    #         # Patch the _dump_queue method so it sleeps.
    #         with patch('pyvvo.ga._dump_queue', new_callable=sleep_a_bit):
    #             with patch.object(self.pop_obj, '_population'):
    #                 # Start a thread to run the method.
    #                 t = threading.Thread(
    #                     target=self.pop_obj._dump_queue_into_population)
    #
    #                 t.start()
    #
    #                 # Sleep less than sleep_a_bit sleeps.
    #                 sleep(sleep_2)
    #
    #                 # Try to acquire the lock.
    #                 self.assertFalse(self.pop_obj._lock.acquire(
    #                     blocking=False))
    #
    #                 # Sleep a bit.
    #                 sleep(sleep_1 - sleep_2 + 0.005)
    #
    #                 # Now we should be able to acquire the lock.
    #                 self.assertTrue(self.pop_obj._lock.acquire(blocking=False))
    #
    #     finally:
    #         # Release the lock.
    #         self.pop_obj._lock.release()

    def test_graceful_shutdown(self):
        """Ensure the graceful shutdown method empties out the queues
        and stops the processes.
        """
        # Initialize fresh population object.
        pop = self.helper_create_pop_obj()

        # Sleep for a bit since multiprocessing stuff takes finite
        # time.
        sleep(0.01)

        # Ensure all our processes are running and good to go.
        self.assertTrue(pop.all_processes_alive)

        # Load up the input queue with mock individuals that sleep when
        # evaluate is called. Note we put 2 times as many individuals
        # in the queue as there are processes.
        for _ in range(2 * len(pop.processes)):
            pop.input_queue.put_nowait(SleepyMockIndividual())

        # Shut things down. Sleep to allow the processes to extract the
        # individuals.
        sleep(0.01)
        with self.assertLogs(logger=pop.log, level='WARNING'):
            pop.graceful_shutdown()

        # The input queue should be empty.
        self.assertTrue(pop.input_queue.empty())

        # After waiting for all tasks in the input queue to be marked
        # as complete, the output queue should have the same size as
        # the number of processes.
        with time_limit(1):
            pop.input_queue.join()

        self.assertEqual(len(pop.processes), pop.output_queue.qsize())

        # At this point, all the processes should be dead. Sleep to
        # ensure processes have time to die.
        sleep(0.02)
        self.assertTrue(pop.all_processes_dead)

    def test_forceful_shutdown(self):
        self.assertRaises(NotImplementedError, self.pop_obj.forceful_shutdown)


class UpdateEquipmentWithIndividualTestCase(unittest.TestCase):
    """Test _update_equipment_with_individual"""

    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_9500)
        # 20 second model runtime.
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 1, 0)

        # Get regulators and capacitors.
        # TODO: update to 9500 node
        reg_df = pd.read_csv(_df.REGULATORS_9500)
        cap_df = pd.read_csv(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

        # It seems we don't have a way of getting capacitor state from
        # the CIM (which is where those DataFrames originate from). So,
        # let's randomly command each capacitor.
        for c in cls.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                c.state = np.random.randint(low=0, high=2, size=None,
                                            dtype=int)
            elif isinstance(c, dict):
                for cc in c.values():
                    cc.state = np.random.randint(low=0, high=2, size=None,
                                                 dtype=int)

        # Map the chromosome.
        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

    def test_update_max(self):
        # Create an individual with all equipment "maxed out"
        ind = ga.Individual(uid=0, chrom_len=self.len, num_eq=self.num_eq,
                            chrom_map=self.map, special_init='max')

        # Run the update function
        ga._update_equipment_with_individual(ind=ind,
                                             regs=self.regs,
                                             caps=self.caps)
        # Ensure all regulators are at their maximum position in CIM
        # terms.
        for r in self.regs.values():
            if isinstance(r, equipment.RegulatorSinglePhase):
                # GA only works with controllable assets.
                if not r.controllable:
                    continue
                self.assertEqual(r.high_step, r.state)
            elif isinstance(r, dict):
                for sr in r.values():
                    if not sr.controllable:
                        continue
                    self.assertEqual(sr.high_step, sr.state)
            else:
                raise TypeError('Unexpected type!')

        # Ensure all capacitors are closed
        for c in self.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                if not c.controllable:
                    continue
                self.assertEqual(1, c.state)
            elif isinstance(c, dict):
                for sc in c.values():
                    if not sc.controllable:
                        continue
                    self.assertEqual(1, sc.state)

    def test_update_min(self):
        """I was bad/lazy and copy + pasted this from test_update_max
        and updated it...
        """
        # Create an individual with all equipment at its minimum.
        ind = ga.Individual(uid=0, chrom_len=self.len, num_eq=self.num_eq,
                            chrom_map=self.map, special_init='min')

        # Run the update function
        ga._update_equipment_with_individual(ind=ind,
                                             regs=self.regs,
                                             caps=self.caps)
        # Ensure all regulators are at their maximum position in CIM
        # terms.
        for r in self.regs.values():
            if isinstance(r, equipment.RegulatorSinglePhase):
                # GA only works with controllable assets.
                if not r.controllable:
                    continue
                self.assertEqual(r.low_step, r.state)
            elif isinstance(r, dict):
                for sr in r.values():
                    if not sr.controllable:
                        continue
                    self.assertEqual(sr.low_step, sr.state)
            else:
                raise TypeError('Unexpected type!')

        # Ensure all capacitors are closed
        for c in self.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                if not c.controllable:
                    continue
                self.assertEqual(0, c.state)
            elif isinstance(c, dict):
                for sc in c.values():
                    if not sc.controllable:
                        continue
                    self.assertEqual(0, sc.state)


class MainTestCase(unittest.TestCase):
    """Test the 'main' method in ga.py."""

    @classmethod
    def setUpClass(cls):
        cls.glm_mgr = GLMManager(IEEE_9500)
        # 20 second model runtime.
        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 1, 0)

        # Get regulators and capacitors.
        reg_df = pd.read_csv(_df.REGULATORS_9500)
        cap_df = pd.read_csv(_df.CAPACITORS_9500)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

        # It seems we don't have a way of getting capacitor state from
        # the CIM (which is where those DataFrames originate from). So,
        # let's randomly command each capacitor.
        for c in cls.caps.values():
            if isinstance(c, equipment.CapacitorSinglePhase):
                c.state = np.random.randint(low=0, high=2, size=None,
                                            dtype=int)
            elif isinstance(c, dict):
                for cc in c.values():
                    cc.state = np.random.randint(low=0, high=2, size=None,
                                                 dtype=int)

    def test_methods_called_correctly(self):
        """Do plenty of patching, and just ensure everything is called
        correctly.
        """

        # Create a mock individual and assign it a fitness value.
        mock_ind = MockIndividual()
        mock_ind.fitness = 10

        # Create a mock to be returned by Population()
        mock_pop = NonCallableMagicMock()
        mock_pop.population = [mock_ind]

        # Number of generations:
        g = 3

        with patch('pyvvo.ga.Population', return_value=mock_pop,
                   autospec=True) as p_pop:
            with patch('pyvvo.ga._update_equipment_with_individual',
                       autospec=True) as p_update:
                with patch.dict(ga.CONFIG['ga'], {'generations': g}):
                    regs, caps = \
                        ga.main(regulators=self.regs, capacitors=self.caps,
                                glm_mgr=self.glm_mgr, starttime=self.starttime,
                                stoptime=self.stoptime)

        p_pop.assert_called_once()
        p_pop.assert_called_with(regulators=self.regs, capacitors=self.caps,
                                 glm_mgr=self.glm_mgr,
                                 starttime=self.starttime,
                                 stoptime=self.stoptime)

        # We only initialize the population once.
        mock_pop.initialize_population.assert_called_once()

        # All the GA methods should be called once per generation,
        # except for evaluate which gets called once extra.
        self.assertEqual(g+1, mock_pop.evaluate_population.call_count)
        self.assertEqual(g, mock_pop.natural_selection.call_count)
        self.assertEqual(g, mock_pop.crossover_and_mutate.call_count)

        # If we weren't mocking the Population object, sort_population
        # would be called more since it's called within
        # natural_selection. However, for the purposes of this test we
        # need to ensure it's just called once.
        mock_pop.sort_population.assert_called_once()

        # Ensure we correctly call _update_equipment_with_individual.
        p_update.assert_called_once()
        p_update.assert_called_with(ind=mock_ind, regs=self.regs,
                                    caps=self.caps)

        # Finally, ensure our returns are indeed the given regulators
        # and capacitors.
        # IMPORTANT NOTE: IN THE NON-PATCHED IMPLEMENTATION, THIS WILL
        # NOT BE TRUE, AS THE OBJECTS WILL GET PICKLED FOR
        # MULTIPROCESSING.
        self.assertIs(regs, self.regs)
        self.assertIs(caps, self.caps)


if __name__ == '__main__':
    unittest.main()
