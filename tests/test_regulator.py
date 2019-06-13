import unittest
from pyvvo.equipment import regulator
from pyvvo.equipment import equipment
from copy import copy, deepcopy
import os
import pandas as pd
import json
from random import randint

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REGULATORS = os.path.join(THIS_DIR, 'query_regulators.csv')
REG_MEAS = os.path.join(THIS_DIR, 'query_reg_meas.csv')
REG_MEAS_MSG = os.path.join(THIS_DIR, 'reg_meas_message.json')


class InitializeControllableRegulatorsTestCase(unittest.TestCase):
    """Test initialize_controllable_regulators"""

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(REGULATORS)
        cls.regs = regulator.initialize_controllable_regulators(cls.df)

    def test_twelve_tap_changers(self):
        """There should be 12 single phase regulators (4 x 3 phase)"""
        self.assertEqual(len(self.regs), 12)

    def test_all_regs(self):
        """Every item should be a RegulatorSinglePhase"""
        for key, reg in self.regs.items():
            with self.subTest('reg = {}'.format(reg)):
                self.assertIsInstance(reg, regulator.RegulatorSinglePhase)

    def test_ltc_filter(self):
        """If a regulator's ltc_flag is false, it shouldn't be included.
        """
        # Get a copy of the DataFrame.
        df = self.df.copy(deep=True)

        # Set the first three ltc_flags to False.
        # NOTE: This is hard-coding based on the DataFrame have regs in
        # order.
        df.loc[0:2, 'ltc_flag'] = False

        # Create regulators. NOTE: This should log.
        with self.assertLogs(regulator.LOG, 'INFO'):
            regs = regulator.initialize_controllable_regulators(df)

        # There should be nine now instead of twelve.
        self.assertEqual(len(regs), 9)


# noinspection PyProtectedMember
class TapCIMToGLDTestCase(unittest.TestCase):

    def test_tap_cim_to_gld_1(self):
        actual = regulator._tap_cim_to_gld(step=16, neutral_step=16)
        self.assertEqual(0, actual)

    def test_tap_cim_to_gld_2(self):
        actual = regulator._tap_cim_to_gld(step=1, neutral_step=8)
        self.assertEqual(-7, actual)

    def test_tap_cim_to_gld_3(self):
        actual = regulator._tap_cim_to_gld(step=24, neutral_step=16)
        self.assertEqual(8, actual)

    def test_tap_cim_to_gld_4(self):
        actual = regulator._tap_cim_to_gld(step=0, neutral_step=16)
        self.assertEqual(-16, actual)

    def test_tap_cim_to_gld_5(self):
        actual = regulator._tap_cim_to_gld(step=-2, neutral_step=-1)
        self.assertEqual(-1, actual)


# noinspection PyProtectedMember
class TapGLDToCIMTestCase(unittest.TestCase):

    def test_tap_gld_to_cim_1(self):
        actual = regulator._tap_gld_to_cim(tap_pos=2, neutral_step=8)
        self.assertEqual(10, actual)

    def test_tap_gld_to_cim_2(self):
        actual = regulator._tap_gld_to_cim(tap_pos=-10, neutral_step=16)

        self.assertEqual(6, actual)

    def test_tap_gld_to_cim_3(self):
        actual = regulator._tap_gld_to_cim(tap_pos=0, neutral_step=10)
        self.assertEqual(10, actual)

    def test_tap_gld_to_cim_4(self):
        actual = regulator._tap_gld_to_cim(tap_pos=5, neutral_step=-5)
        self.assertEqual(0, actual)


class RegulatorSinglePhaseInitializationTestCase(unittest.TestCase):
    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.inputs = \
            {'control_mode': 'voltage',
             'enabled': True, 'high_step': 32, 'low_step': 0,
             'mrid': '_3E73AD1D-08AF-A34B-33D2-1FCE3533380A',
             'name': 'FEEDER_REG', 'neutral_step': 16, 'phase': 'A',
             'tap_changer_mrid': '_330E7EDE-2C70-8F72-B183-AA4BA3C5E221',
             'step': 18, 'step_voltage_increment': 0.625}

        cls.reg = regulator.RegulatorSinglePhase(**cls.inputs)

    def test_attributes(self):
        """The inputs should match the attributes."""
        for key, value in self.inputs.items():
            with self.subTest('attribute: {}'.format(key)):
                self.assertEqual(getattr(self.reg, key), value)

    def test_raise_taps(self):
        self.assertEqual(self.reg.raise_taps, 16)

    def test_lower_taps(self):
        self.assertEqual(self.reg.lower_taps, 16)

    def test_tap_pos(self):
        self.assertEqual(self.reg.tap_pos, 2)

    def test_tap_pos_old(self):
        self.assertIsNone(self.reg.tap_pos_old)

    def test_step_old(self):
        self.assertIsNone(self.reg.step_old)

    def test_update_step(self):
        reg = regulator.RegulatorSinglePhase(**self.inputs)

        self.assertIsNone(reg.step_old)
        self.assertIsNone(reg.tap_pos_old)
        self.assertEqual(reg.step, 18)
        self.assertEqual(reg.tap_pos, 2)

        reg.step = 15
        self.assertEqual(reg.step, 15)
        self.assertEqual(reg.tap_pos, -1)
        self.assertEqual(reg.step_old, 18)
        self.assertEqual(reg.tap_pos_old, 2)

    def test_update_tap_pos(self):
        reg = regulator.RegulatorSinglePhase(**self.inputs)

        self.assertIsNone(reg.step_old)
        self.assertIsNone(reg.tap_pos_old)
        self.assertEqual(reg.step, 18)
        self.assertEqual(reg.tap_pos, 2)

        reg.tap_pos = -15
        self.assertEqual(reg.step, 1)
        self.assertEqual(reg.tap_pos, -15)
        self.assertEqual(reg.step_old, 18)
        self.assertEqual(reg.tap_pos_old, 2)

    def test_update_step_bad_type(self):
        reg = regulator.RegulatorSinglePhase(**self.inputs)
        with self.assertRaisesRegex(TypeError, 'step must be an integer'):
            reg.step = 1.0

    def test_update_tap_pos_bad_type(self):
        reg = regulator.RegulatorSinglePhase(**self.inputs)
        with self.assertRaisesRegex(TypeError, 'tap_pos must be an integer'):
            reg.tap_pos = -1.0

    def test_update_step_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'step must be between'):
            self.reg.step = 100

    def test_update_tap_pos_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'tap_pos must be between'):
            self.reg.tap_pos = 17


class RegulatorSinglePhaseBadInputsTestCase(unittest.TestCase):

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.inputs = \
            {'control_mode': 'voltage',
             'enabled': True, 'high_step': 32, 'low_step': 0,
             'mrid': '_3E73AD1D-08AF-A34B-33D2-1FCE3533380A',
             'name': 'FEEDER_REG', 'neutral_step': 16, 'phase': 'A',
             'tap_changer_mrid': '_330E7EDE-2C70-8F72-B183-AA4BA3C5E221',
             'step': 1.0125, 'step_voltage_increment': 0.625}

    def test_bad_mrid_type(self):
        i = deepcopy(self.inputs)
        i['mrid'] = 10
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_name_type(self):
        i = deepcopy(self.inputs)
        i['name'] = {'name': 'reg'}
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_phase_type(self):
        i = deepcopy(self.inputs)
        i['name'] = ['name', 'yo']
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_phase_value(self):
        i = deepcopy(self.inputs)
        i['phase'] = 'N'
        self.assertRaises(ValueError, regulator.RegulatorSinglePhase, **i)

    def test_bad_tap_changer_mrid_type(self):
        i = deepcopy(self.inputs)
        i['tap_changer_mrid'] = 111
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_step_voltage_increment_type(self):
        i = deepcopy(self.inputs)
        i['step_voltage_increment'] = 1
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_control_mode_type(self):
        i = deepcopy(self.inputs)
        i['control_mode'] = (0, 0, 1)
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_control_mode_value(self):
        i = deepcopy(self.inputs)
        i['control_mode'] = 'my mode'
        self.assertRaises(ValueError, regulator.RegulatorSinglePhase, **i)

    def test_bad_enabled_type(self):
        i = deepcopy(self.inputs)
        i['enabled'] = 'true'
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_high_step_type(self):
        i = deepcopy(self.inputs)
        i['high_step'] = 10.1
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_low_step_type(self):
        i = deepcopy(self.inputs)
        i['low_step'] = 10+1j
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_neutral_step_type(self):
        i = deepcopy(self.inputs)
        i['neutral_step'] = '16'
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_bad_step_values_1(self):
        i = deepcopy(self.inputs)
        i['low_step'] = 17
        i['neutral_step'] = 16
        i['high_step'] = 20
        with self.assertRaisesRegex(ValueError, 'The following is not True'):
            regulator.RegulatorSinglePhase(**i)

    def test_bad_step_values_2(self):
        i = deepcopy(self.inputs)
        i['low_step'] = 0
        i['neutral_step'] = 21
        i['high_step'] = 20
        with self.assertRaisesRegex(ValueError, 'The following is not True'):
            regulator.RegulatorSinglePhase(**i)

    def test_bad_step_values_3(self):
        i = deepcopy(self.inputs)
        i['low_step'] = 0
        i['neutral_step'] = 0
        i['high_step'] = -1
        with self.assertRaisesRegex(ValueError, 'The following is not True'):
            regulator.RegulatorSinglePhase(**i)

    def test_bad_step_type(self):
        i = deepcopy(self.inputs)
        i['step'] = 2.0
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)

    def test_step_out_of_range_1(self):
        i = deepcopy(self.inputs)
        i['step'] = 33
        with self.assertRaisesRegex(ValueError, 'between low_step '):
            regulator.RegulatorSinglePhase(**i)

    def test_step_out_of_range_2(self):
        i = deepcopy(self.inputs)
        i['step'] = -1
        with self.assertRaisesRegex(ValueError, 'between low_step '):
            regulator.RegulatorSinglePhase(**i)


class RegulatorManagerTestCase(unittest.TestCase):
    """Test RegulatorManager"""

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.reg_meas = pd.read_csv(REG_MEAS)
        with open(REG_MEAS_MSG, 'r') as f:
            cls.reg_meas_msg = json.load(f)

    # noinspection PyPep8Naming
    def setUp(self):
        # Gotta be careful with these mutable types... Get fresh
        # instances each time. It won't be that slow, I promise.
        self.reg_dict = \
            regulator.initialize_controllable_regulators(
                pd.read_csv(REGULATORS))
        self.reg_mgr = \
            regulator.RegulatorManager(reg_dict=self.reg_dict,
                                       reg_meas=self.reg_meas)

    def test_reg_dict_attribute(self):
        self.assertIs(self.reg_dict, self.reg_mgr.reg_dict)

    def test_inconsistent_inputs(self):
        """Ensure we get an exception if inputs are not consistent.
        """
        meas = self.reg_meas.copy(deep=True)
        # Create a duplicate entry.
        meas.iloc[0] = meas.iloc[1]

        s = 'inputs are inconsistent'
        with self.assertRaisesRegex(ValueError, s):
            _ = regulator.RegulatorManager(reg_dict=self.reg_dict,
                                           reg_meas=meas)

    def test_all_measurements_mapped(self):
        """Ensure all measurements are in the map."""
        for meas_mrid in self.reg_meas['pos_meas_mrid'].values:
            with self.subTest():
                self.assertIn(meas_mrid, self.reg_mgr.meas_reg_map.keys())

    def test_no_meas_for_reg(self):
        """Remove measurements for given regulator, ensure we get
        proper exception.
        """
        meas_view = self.reg_meas[
            ~(self.reg_meas['tap_changer_mrid']
              == self.reg_meas['tap_changer_mrid'][0])
        ]

        with self.assertRaisesRegex(ValueError, 'do not match'):
            _ = regulator.RegulatorManager(reg_dict=self.reg_dict,
                                           reg_meas=meas_view)

    def test_bad_reg_dict_type(self):
        with self.assertRaisesRegex(TypeError,
                                    'reg_dict must be a dictionary'):
            _ = regulator.RegulatorManager(reg_dict=10,
                                           reg_meas=self.reg_meas)

    def test_bad_reg_meas_type(self):
        with self.assertRaisesRegex(TypeError,
                                    'reg_measurements must be a Pandas'):
            _ = regulator.RegulatorManager(reg_dict=self.reg_dict,
                                           reg_meas=pd.Series())

    def test_no_meas_for_single_phase_reg(self):
        meas_view = self.reg_meas.drop(0, axis=0)
        with self.assertRaisesRegex(ValueError, 'do not match'):
            _ = regulator.RegulatorManager(reg_dict=self.reg_dict,
                                           reg_meas=meas_view)

    def test_two_meas_for_single_phase_reg(self):
        reg_meas = self.reg_meas.append(self.reg_meas.iloc[0])

        with self.assertRaisesRegex(ValueError, 'do not match'):
            _ = regulator.RegulatorManager(reg_dict=self.reg_dict,
                                           reg_meas=reg_meas)

    def test_update_regs_simple(self):
        """Just ensure it runs without error."""

        # At the time of writing, the debug line is the last one in the
        # function, so ensuring it gets hit is adequate.
        with self.assertLogs(level='DEBUG'):
            self.reg_mgr.update_regs(self.reg_meas_msg)

    def test_update_regs_changes_taps(self):
        """Ensure our taps changed appropriately. We'll hard-code
        this for simplicity.
        """
        self.reg_mgr.update_regs(self.reg_meas_msg)

        # Loop over the message.
        for msg_dict in self.reg_meas_msg:
            # Grab the MRID.
            meas_mrid = msg_dict['measurement_mrid']
            meas_value = msg_dict['value']

            # Look up the measurement mrid.
            row = self.reg_meas[self.reg_meas['pos_meas_mrid'] == meas_mrid]

            # Grab regulator mrid and phase.
            reg_mrid = row['tap_changer_mrid'].values[0]

            # Ensure this regulator got updated.
            with self.subTest(meas_mrid=meas_mrid):
                # noinspection PyUnboundLocalVariable
                self.assertEqual(self.reg_dict[reg_mrid].tap_pos, meas_value)

    def test_update_regs_bad_mrid(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrid': '1234', 'value': 12})

        with self.assertLogs(level='WARNING'):
            self.reg_mgr.update_regs(reg_meas_msg)

    def test_update_regs_bad_entry_1(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrId': '1234', 'value': 12})

        with self.assertRaisesRegex(KeyError, 'measurement_mrid'):
            self.reg_mgr.update_regs(reg_meas_msg)

    def test_update_regs_bad_entry_2(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrid': '1234', 'valu3': 12})

        with self.assertRaisesRegex(KeyError, 'value'):
            self.reg_mgr.update_regs(reg_meas_msg)

    def test_update_regs_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'msg must be a list'):
            self.reg_mgr.update_regs(msg='hello there')

    def test_update_regs_bad_type_2(self):
        with self.assertRaisesRegex(TypeError, 'must be a dict'):
            self.reg_mgr.update_regs(msg=['hello there'])

    def test_build_regulator_commands(self):
        """One stop big function which probably should be spun off into
        its own test case.

        NOTE: This test is fragile as it currently relies on how the
        looping is performed in build_regulator_commands.
        """
        reg_dict_forward = deepcopy(self.reg_dict)
        # For some reason the new reg_dict won't pickle?
        # reg_dict_forward = \
        #     regulator.initialize_controllable_regulators(
        #         pd.read_csv(REGULATORS))

        # Randomly update steps.
        forward_vals = []
        for reg_single in reg_dict_forward.values():
            new_step = randint(reg_single.low_step, reg_single.high_step)
            reg_single.step = new_step
            forward_vals.append(new_step)

        # Grab reverse values.
        reverse_vals = []
        for reg_single in self.reg_dict.values():
            reverse_vals.append(reg_single.step)

        # Just use the same dictionary to make a "do nothing" command.
        out = self.reg_mgr.build_regulator_commands(
            reg_dict_forward=reg_dict_forward)

        # Ensure we're getting the fields we need.
        self.assertIn('object_ids', out)
        self.assertIn('attributes', out)
        self.assertIn('forward_values', out)
        self.assertIn('reverse_values', out)

        # Ensure our forward values match. WARNING: this is quite
        # fragile as it depends on looping order.
        self.assertListEqual(forward_vals, out['forward_values'])
        # Ensure reverse values match (also fragile).
        self.assertListEqual(reverse_vals, out['reverse_values'])

        # Ensure the lengths are equal to all our single phases.
        # I'm just going to hard-code the fact that the 8500 node model
        # has 4 3-phase regs.
        for v in out.values():
            self.assertIsInstance(v, list)
            self.assertEqual(len(v), 12)

    def test_build_regulator_commands_mismatch(self):
        """Send mismatched reg dicts in."""
        reg_dict_forward = deepcopy(self.reg_dict)
        reg_dict_forward['blah'] = \
            reg_dict_forward.pop(list(reg_dict_forward.keys())[0])

        with self.assertRaisesRegex(ValueError, 'not matching up with'):
            self.reg_mgr.build_regulator_commands(reg_dict_forward)


if __name__ == '__main__':
    unittest.main()
