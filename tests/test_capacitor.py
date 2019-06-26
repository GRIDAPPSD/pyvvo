import unittest
from pyvvo.equipment import capacitor, equipment
import os
import pandas as pd
import numpy as np

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CAPACITORS = os.path.join(THIS_DIR, 'query_capacitors.csv')


class CapacitorSinglePhaseTestCase(unittest.TestCase):
    """Basic property tests for CapacitorSinglePhase."""

    @classmethod
    def setUpClass(cls):
        """Create CapacitorSinglePhase object."""
        cls.cap = \
            capacitor.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state=1, mode='ACTIVEpower',
                                           controllable=True)

    def test_equipment(self):
        self.assertIsInstance(self.cap, equipment.EquipmentSinglePhase)

    def test_name(self):
        self.assertEqual(self.cap.name, 'cap1')

    def test_mrid(self):
        self.assertEqual('1', self.cap.mrid)

    def test_mode(self):
        """Mode is case insensitive, and is cast to lower case."""
        self.assertEqual('activepower', self.cap.mode)

    def test_phase(self):
        self.assertEqual('C', self.cap.phase)

    def test_controllable(self):
        self.assertTrue(self.cap.controllable)

    def test_state(self):
        self.assertEqual(1, self.cap.state)

    def test_state_none(self):
        """None is a valid state to initialize a capacitor."""
        cap = \
            capacitor.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state=None, mode='voltage',
                                           controllable=True)
        self.assertIsNone(cap.state)

    def test_repr(self):
        self.assertIn(self.cap.name, str(self.cap))
        self.assertIn(self.cap.phase, str(self.cap))
        self.assertIn('CapacitorSinglePhase', str(self.cap))

    def test_state_update(self):
        cap = \
            capacitor.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state=None, mode='voltage',
                                           controllable=True)

        self.assertIsNone(cap.state)

        cap.state = 0

        self.assertEqual(cap.state, 0)
        self.assertEqual(cap.state_old, None)

        cap.state = 1

        self.assertEqual(cap.state, 1)
        self.assertEqual(cap.state_old, 0)


class CapacitorSinglePhaseBadInputsTestCase(unittest.TestCase):
    """Test bad inputs to CapacitorSinglePhase"""

    def test_name_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name=[1, 2, 3], mrid='1', phase='A', state='OPEN',
                          mode='admittance', controllable=True)

    def test_mrid_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid={'a': 1}, phase='A', state='OPEN',
                          mode='admittance', controllable=True)

    def test_phase_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase=7, state='OPEN',
                          mode='admittance', controllable=True)

    def test_phase_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='N', state='OPEN',
                          mode='admittance', controllable=True)

    def test_state_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='c', state=1.0,
                          mode='admittance', controllable=True)

    def test_state_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state=3,
                          mode='admittance', controllable=True)

    def test_mode_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state=0,
                          mode=0, controllable=True)

    def test_mode_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state=0,
                          mode='vvo', controllable=True)

    def test_controllable_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'controllable must be a bool'):
            capacitor.CapacitorSinglePhase(
                name='1', mrid='1', phase='b', state=0,
                mode='temperature', controllable='True')

    def test_mode_controllable_mismatch_1(self):
        with self.assertRaisesRegex(ValueError, 'seem to conflict'):
            capacitor.CapacitorSinglePhase(
                name='1', mrid='1', phase='b', state=None,
                mode=None, controllable=True)

    def test_mode_controllable_mismatch_2(self):
        with self.assertRaisesRegex(ValueError, 'seem to conflict'):
            capacitor.CapacitorSinglePhase(
                name='1', mrid='1', phase='b', state=None,
                mode='voltage', controllable=False)


class InitializeCapacitors(unittest.TestCase):
    """Test initialize_capacitors"""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CAPACITORS)
        cls.caps = capacitor.initialize_capacitors(cls.df)

    def test_length(self):
        """The return should have 10 items, as there are 3 controllable
        3 phase caps, and 1 non-controllable 3 phase cap.
        """
        self.assertEqual(len(self.caps), 10)

    def test_is_capacitor_or_dict(self):
        """Ensure each result is CapacitorSinglePhase or dict of them."""
        cap_count = 0
        dict_count = 0
        dict_cap_count = 0
        for _, cap in self.caps.items():
            self.assertIsInstance(cap, (capacitor.CapacitorSinglePhase, dict))
            # If we have a dict, ensure all values are
            # CapacitorSinglePhase.
            if isinstance(cap, dict):
                dict_count += 1
                for c in cap.values():
                    dict_cap_count += 1
                    self.assertIsInstance(c, capacitor.CapacitorSinglePhase)
            else:
                cap_count += 1

        self.assertEqual(cap_count, 9)
        self.assertEqual(dict_count, 1)
        self.assertEqual(dict_cap_count, 3)


if __name__ == '__main__':
    unittest.main()
