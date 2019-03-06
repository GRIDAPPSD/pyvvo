import unittest
from unittest.mock import patch
import pyvvo.equipment.regulator as regulator
from copy import deepcopy
from inspect import signature
import os
import pandas as pd

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REGULATORS = os.path.join(THIS_DIR, 'query_regulators.csv')


class InitializeControllableRegulatorsTestCase(unittest.TestCase):
    """Test initialize_controllable_regulators"""
    def setUp(self):
        self.df = pd.read_csv(REGULATORS)
        self.regs = regulator.initialize_controllable_regulators(self.df)

    def test_four_regs(self):
        """There should be 4 three phase regulators"""
        self.assertEqual(len(self.regs), 4)

    def test_all_regs(self):
        """Every item should be a RegulatorThreePhase"""
        for reg in self.regs:
            with self.subTest('reg = {}'.format(reg)):
                self.assertIsInstance(reg, regulator.RegulatorThreePhase)

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

        # There should be three now instead of four.
        self.assertEqual(len(regs), 3)


class TapCIMToGLDTestCase(unittest.TestCase):
    """NOTE: The way the platform currently handles 'step' is likely
    wrong. Test this after there's a way forward identified.
    """

    def test_tap_cim_to_gld_1(self):
        actual = regulator._tap_cim_to_gld(step=1.0125,
                                           step_voltage_increment=0.625)
        self.assertEqual(2, actual)

    def test_tap_cim_to_gld_2(self):
        actual = regulator._tap_cim_to_gld(step=1.00625,
                                           step_voltage_increment=0.625)

        self.assertEqual(1, actual)

    def test_tap_cim_to_gld_3(self):
        actual = regulator._tap_cim_to_gld(step=0.99375,
                                           step_voltage_increment=0.625)
        self.assertEqual(-1, actual)

    def test_tap_cim_to_gld_4(self):
        actual = regulator._tap_cim_to_gld(step=0.9,
                                           step_voltage_increment=0.625)
        self.assertEqual(-16, actual)

    def test_tap_cim_to_gld_5(self):
        actual = regulator._tap_cim_to_gld(step=1.1,
                                           step_voltage_increment=0.625)
        self.assertEqual(16, actual)


class TapGLDToCIMTestCase(unittest.TestCase):
    """NOTE: The way the platform currently handles 'step' is likely
    wrong. Test this after there's a way forward identified.
    """

    def test_tap_gld_to_cim_1(self):
        actual = regulator._tap_gld_to_cim(tap_pos=2,
                                           step_voltage_increment=0.625)
        self.assertEqual(1.0125, actual)

    def test_tap_gld_to_cim_2(self):
        actual = regulator._tap_gld_to_cim(tap_pos=1,
                                           step_voltage_increment=0.625)

        self.assertEqual(1.00625, actual)

    def test_tap_gld_to_cim_3(self):
        actual = regulator._tap_gld_to_cim(tap_pos=-1,
                                           step_voltage_increment=0.625)
        self.assertEqual(0.99375, actual)

    def test_tap_gld_to_cim_4(self):
        actual = regulator._tap_gld_to_cim(tap_pos=-16,
                                           step_voltage_increment=0.625)
        self.assertEqual(0.9, actual)

    def test_tap_gld_to_cim_5(self):
        actual = regulator._tap_gld_to_cim(tap_pos=16,
                                           step_voltage_increment=0.625)
        self.assertEqual(1.1, actual)


class MockRegulatorSinglePhase:
    """Simple class for mocking RegulatorSinglePhase for the purpose of
    testing RegulatorThreePhase
    """
    def __init__(self, name, mrid, phase):
        self.name = name
        self.mrid = mrid
        self.phase = phase


class RegulatorThreePhaseInitializationTestCase(unittest.TestCase):

    def test_bad_input_type(self):
        self.assertRaises(TypeError, regulator.RegulatorThreePhase,
                          'hello')

    def test_bad_input_list_length(self):
        self.assertRaises(ValueError, regulator.RegulatorThreePhase,
                          [1, 2])

    def test_bad_input_list_type(self):
        self.assertRaises(TypeError, regulator.RegulatorThreePhase,
                          (1, 2, 3))

    # noinspection PyArgumentList
    @patch(target='pyvvo.equipment.regulator.RegulatorSinglePhase',
           new=MockRegulatorSinglePhase)
    def test_successful_init(self):
        """Pass three mocked single phase regs."""
        reg1 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='a')
        reg2 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='b')
        reg3 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='C')

        reg_3_phs = regulator.RegulatorThreePhase((reg1, reg2, reg3))

        self.assertEqual(reg_3_phs.name, 'reg')
        self.assertEqual(reg_3_phs.mrid, '123')
        self.assertIs(reg_3_phs.a, reg1)
        self.assertIs(reg_3_phs.b, reg2)
        self.assertIs(reg_3_phs.c, reg3)

    # noinspection PyArgumentList
    @patch(target='pyvvo.equipment.regulator.RegulatorSinglePhase',
           new=MockRegulatorSinglePhase)
    def test_mismatched_names(self):
        """All single phase regs should have the same name."""
        reg1 = regulator.RegulatorSinglePhase(name='reg2', mrid='123',
                                              phase='a')
        reg2 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='b')
        reg3 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='C')

        self.assertRaises(ValueError, regulator.RegulatorThreePhase,
                          (reg1, reg2, reg3))

    # noinspection PyArgumentList
    @patch(target='pyvvo.equipment.regulator.RegulatorSinglePhase',
           new=MockRegulatorSinglePhase)
    def test_mismatched_mrids(self):
        """All single phase regs should have the same name."""
        reg1 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='a')
        reg2 = regulator.RegulatorSinglePhase(name='reg', mrid='123',
                                              phase='b')
        reg3 = regulator.RegulatorSinglePhase(name='reg', mrid='1234',
                                              phase='C')

        self.assertRaises(ValueError, regulator.RegulatorThreePhase,
                          (reg1, reg2, reg3))


class RegulatorSinglePhaseInitializationTestCase(unittest.TestCase):
    def setUp(self):
        self.inputs = \
            {'control_mode': 'voltage',
             'enabled': True, 'high_step': 32, 'low_step': 0,
             'mrid': '_3E73AD1D-08AF-A34B-33D2-1FCE3533380A',
             'name': 'FEEDER_REG', 'neutral_step': 16, 'phase': 'A',
             'tap_changer_mrid': '_330E7EDE-2C70-8F72-B183-AA4BA3C5E221',
             'step': 1.0125, 'step_voltage_increment': 0.625}

        self.reg = regulator.RegulatorSinglePhase(**self.inputs)

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


class RegulatorSinglePhaseBadInputsTestCase(unittest.TestCase):

    def setUp(self):
        self.inputs = \
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
        self.assertRaises(ValueError, regulator.RegulatorSinglePhase, **i)

    def test_bad_step_values_2(self):
        i = deepcopy(self.inputs)
        i['low_step'] = 0
        i['neutral_step'] = 21
        i['high_step'] = 20
        self.assertRaises(ValueError, regulator.RegulatorSinglePhase, **i)

    def test_bad_step_values_3(self):
        i = deepcopy(self.inputs)
        i['low_step'] = 0
        i['neutral_step'] = 0
        i['high_step'] = -1
        self.assertRaises(ValueError, regulator.RegulatorSinglePhase, **i)

    def test_bad_step_type(self):
        # TODO: Update when platform is updated.
        i = deepcopy(self.inputs)
        i['step'] = 2
        self.assertRaises(TypeError, regulator.RegulatorSinglePhase, **i)


if __name__ == '__main__':
    unittest.main()
