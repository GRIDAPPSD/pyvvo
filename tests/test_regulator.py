import unittest
from pyvvo.equipment import regulator
from copy import deepcopy
import os
import pandas as pd

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


if __name__ == '__main__':
    unittest.main()
