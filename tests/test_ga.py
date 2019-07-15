import unittest
from unittest.mock import patch
import os
from datetime import datetime
from copy import deepcopy

import tests.data_files as _df
from pyvvo import ga
from pyvvo import equipment
from pyvvo.glm import GLMManager
from pyvvo.utils import run_gld
from pyvvo import db

import numpy as np
import pandas as pd
import MySQLdb

np.random.seed(42)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, 'models')
IEEE_8500 = os.path.join(MODEL_DIR, 'ieee_8500.glm')
IEEE_13 = os.path.join(MODEL_DIR, 'ieee_13.glm')


class MapChromosomeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Get capacitors and regulators."""
        cls.reg_df = _df.read_pickle(_df.REGULATORS_8500)
        cls.cap_df = _df.read_pickle(_df.CAPACITORS_8500)

        cls.regs = equipment.initialize_regulators(cls.reg_df)
        cls.caps = equipment.initialize_capacitors(cls.cap_df)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

    def test_length(self):
        """4 three phase regs, 9 single phase caps."""
        self.assertEqual(4 * 3 * 6 + 9, self.len)

    def test_map_length(self):
        """4 regs, 9 caps."""
        self.assertEqual(4 + 9, len(self.map))

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
        """4x3 regs, 9 caps"""
        self.assertEqual(4 * 3 + 9, self.num_eq)

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


class IndividualTestCase(unittest.TestCase):
    """Test everything that doesn't involve a glm.GLMManager. Those
    tests are more involved and will be done elsewhere.
    """

    @classmethod
    def setUpClass(cls):
        # Get capacitor and regulator information.
        reg_df = _df.read_pickle(_df.REGULATORS_123)
        cap_df = _df.read_pickle(_df.CAPACITORS_123)

        cls.regs = equipment.initialize_regulators(reg_df)
        cls.caps = equipment.initialize_capacitors(cap_df)

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

        # Splice in values for reg2 and reg3.
        idx2 = self.ind._chrom_map['reg2']['A']['idx']
        idx3 = self.ind._chrom_map['reg3']['C']['idx']
        # The call below looks wrong, but it's fine, I promise :)
        # the 'm' argument is just used for the width formatting.
        c[idx2[0]:idx2[1]] = ga._int_to_binary_list(50, m=32)
        c[idx3[0]:idx3[1]] = ga._int_to_binary_list(25, m=32)

        # Patch the 'reg2' and 'reg3' ranges and run the check.
        with patch.dict(self.ind._chrom_map['reg3']['C'], {'range': (31, 32)}):
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


class IndividualUpdateModelComputeCostsTestCase(unittest.TestCase):
    """Test the _update_model_compute_costs method of an individual.

    Additionally, we'll test _update_reg and _update_cap.
    """

    @classmethod
    def setUpClass(cls):
        cls.reg_df = _df.read_pickle(_df.REGULATORS_8500)
        cls.caps_df = _df.read_pickle(_df.CAPACITORS_8500)

        cls.regs = equipment.initialize_regulators(cls.reg_df)
        cls.caps = equipment.initialize_capacitors(cls.caps_df)

        cls.map, cls.len, cls.num_eq = ga.map_chromosome(cls.regs, cls.caps)

        cls.glm_mgr = GLMManager(IEEE_8500)

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
        phase_dict = self.map['FEEDER_REG']
        with patch.dict(ga.CONFIG, {'costs': {'regulator_tap': 10}}):
            with patch.object(self.fresh_mgr, 'update_reg_taps',
                              wraps=self.fresh_mgr.update_reg_taps) as p:
                penalty = self.ind._update_reg(phase_dict=phase_dict,
                                               glm_mgr=self.fresh_mgr)

        # 3 phases, each phase moved 32 positions with a cost of 10 per
        # position.
        self.assertEqual(3 * 32 * 10, penalty)

        # Ensure the taps are being updated properly in the model.
        # The actual 'update_reg_taps' method is tested elsewhere, so
        # no need to actually look up the objects and confirm.
        p.assert_called_once()
        p.assert_called_with(ga._cim_to_glm_name(prefix=ga.REG_PREFIX,
                                                 cim_name='FEEDER_REG'),
                             {'A': 16, 'B': 16, 'C': 16})

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

        # 3 phases per regulator, 4 regulators, each moving 32 taps,
        # with a cost of 10 per tap
        self.assertEqual(3 * 4 * 32 * 10, reg_penalty)

        # Ensure our helper methods are called the appropriate number of
        # times.
        self.assertEqual(9, pc.call_count)
        self.assertEqual(4, pr.call_count)


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
        cls.ind = ga.Individual(uid=0, chrom_len=cls.len, chrom_map=cls.map,
                                num_eq=cls.num_eq)

    @patch('pyvvo.ga._Evaluator.evaluate', autospec=True,
           return_value={'voltage_high': 1, 'voltage_low': 2,
                         'power_factor_lead': 3,
                         'power_factor_lag': 4,
                         'energy': 5})
    @patch('pyvvo.ga._Evaluator.__init__', autospec=True, return_value=None)
    def test_evaluate(self, eval_init_patch, eval_evaluate_patch):
        """Patch everything, ensure the correct methods are called.

        The individual functions have been well tested, so no need to
        actually call them.

        Patching __init__ feels fragile, but it's the only way I could
        get this thing working.

        https://stackoverflow.com/q/57044593/11052174
        """
        # Patch inputs to the Individual's evaluate method.
        mock_glm = unittest.mock.create_autospec(GLMManager, spec_set=True)
        mock_db = unittest.mock.create_autospec(MySQLdb.connection,
                                                set_spec=True)

        # partial_dict is going to be the patched return from
        # _Evaluator.evaluate. This is hard-coded to match the
        # return_value in the patch above.
        partial_dict = {'voltage_high': 1, 'voltage_low': 2,
                        'power_factor_lead': 3, 'power_factor_lag': 4,
                        'energy': 5}

        with patch.object(self.ind, '_update_model_compute_costs',
                          autospec=True, return_value=(6, 7)) as p_update:
            results = self.ind.evaluate(glm_mgr=mock_glm, db_conn=mock_db)

        # Assertion time.
        # Ensure _update_model_compute_costs is called and called
        # correctly.
        p_update.assert_called_once()
        p_update.assert_called_with(glm_mgr=mock_glm)

        # Ensure our _Evaluator is constructor appropriately.
        eval_init_patch.assert_called_once()
        self.assertDictEqual(eval_init_patch.call_args[1],
                             {'uid': self.ind.uid, 'glm_mgr': mock_glm,
                              'db_conn': mock_db})

        # Ensure _Evaluator._evaluate is called.
        eval_evaluate_patch.assert_called_once()

        # Total fitness is the sum of the values in the penalties dict.
        self.assertEqual(28, self.ind.fitness)

        # Ensure our penalties dict comes back as expected.
        expected = {**partial_dict, 'regulator_tap': 6, 'capacitor_switch': 7}
        self.assertDictEqual(expected, results)


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
        r = pd.DataFrame([[1, 2, 3], [4, 5, 6]],
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


if __name__ == '__main__':
    unittest.main()
