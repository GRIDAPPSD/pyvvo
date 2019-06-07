import unittest
from pyvvo.equipment import capacitor
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
                                           state='OPEN', mode='ACTIVEpower')

    def test_name(self):
        self.assertEqual(self.cap.name, 'cap1')

    def test_mrid(self):
        self.assertEqual('1', self.cap.mrid)

    def test_mode(self):
        """Mode is case insensitive, and is cast to lower case."""
        self.assertEqual('activepower', self.cap.mode)

    def test_phase(self):
        """Lower case phase should be cast to upper case."""
        self.assertEqual('C', self.cap.phase)

    def test_state(self):
        """State should be cast to upper case."""
        self.assertEqual('OPEN', self.cap.state)

    def test_state_none(self):
        """None is a valid state to initialize a capacitor."""
        cap = \
            capacitor.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state=None, mode='voltage')
        self.assertIsNone(cap.state)

    def test_repr(self):
        self.assertEqual(str(self.cap), self.cap.name)

    def test_state_update(self):
        cap = \
            capacitor.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state=None, mode='voltage')

        self.assertIsNone(cap.state)

        cap.state = 'OPEN'

        self.assertEqual(cap.state, 'OPEN')
        self.assertEqual(cap.state_old, None)

        cap.state = 'CLOSED'

        self.assertEqual(cap.state, 'CLOSED')
        self.assertEqual(cap.state_old, 'OPEN')

class CapacitorSinglePhaseBadInputsTestCase(unittest.TestCase):
    """Test bad inputs to CapacitorSinglePhase"""

    def test_name_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name=[1, 2, 3], mrid='1', phase='A', state='OPEN',
                          mode='admittance')

    def test_mrid_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid={'a': 1}, phase='A', state='OPEN',
                          mode='admittance')

    def test_phase_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase=7, state='OPEN',
                          mode='admittance')

    def test_phase_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='N', state='OPEN',
                          mode='admittance')

    def test_state_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='c', state=True,
                          mode='admittance')

    def test_state_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state='stuck',
                          mode='admittance')

    def test_mode_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state='stuck',
                          mode=0)

    def test_mode_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state='stuck',
                          mode='vvo')


class InitializeControllableCapacitors(unittest.TestCase):
    """Test initialize_controllable_capacitors"""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CAPACITORS)
        cls.caps = capacitor.initialize_controllable_capacitors(cls.df)

    def test_length(self):
        """There should be 10-1=9 capacitors, because one capacitor is
        not controllable"""
        self.assertEqual(len(self.caps), 9)

    def test_is_capacitor(self):
        """Ensure each result is indeed a SinglePhaseCapacitor."""

        for _, cap in self.caps.items():
            self.assertIsInstance(cap, capacitor.CapacitorSinglePhase)

    def test_multi_phase_capacitor(self):
        """Not supporting multi-phase controllable capacitors until
        necessary to."""
        df = self.df.copy(deep=True)
        df.loc[3, 'phase'] = np.nan

        self.assertRaises(NotImplementedError,
                          capacitor.initialize_controllable_capacitors,
                          df=df)

    def test_value(self):
        self.assertEqual(
            self.caps['capbank1c'].mrid,
            self.df[self.df['name'] == 'capbank1c']['mrid'].iloc[0])

    def test_no_controllable_caps(self):
        """If the ctrlenabled field isn't present, no caps are
        controllable."""
        df = self.df.drop(labels='ctrlenabled', axis=1)
        self.assertDictEqual({},
                             capacitor.initialize_controllable_capacitors(df))


class CapacitorSinglePhaseUpdateControlMode(unittest.TestCase):
    """Test the 'update_control' method."""
    def test_stub(self):
        self.assertTrue(False, 'This method is not complete yet.')


if __name__ == '__main__':
    unittest.main()
