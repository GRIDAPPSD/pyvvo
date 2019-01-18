import unittest
from pyvvo.equipment import capacitor


class CapacitorSinglePhaseTestCase(unittest.TestCase):
    """Basic property tests for CapacitorSinglePhase."""

    def setUp(self):
        """Create CapacitorSinglePhase object."""
        self.cap = \
            capacitor.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state='OpEn', name_prefix='cap_')

    def test_name_prefix(self):
        self.assertEqual('cap_', self.cap.name_prefix)

    def test_name(self):
        """Ensure the prefix was added to the name."""
        self.assertEqual(self.cap.name, 'cap_cap1')

    def test_mrid(self):
        self.assertEqual('1', self.cap.mrid)

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
                                           state=None, name_prefix='cap_')
        self.assertIsNone(cap.state)


class CapacitorSinglePhaseBadInputsTestCase(unittest.TestCase):
    """Test bad inputs to CapacitorSinglePhase"""

    def test_name_prefix_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='cap', mrid='1', phase='A', state='OPEN',
                          name_prefix=1)

    def test_name_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name=[1, 2, 3], mrid='1', phase='A', state='OPEN',
                          name_prefix='blah')

    def test_mrid_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid={'a': 1}, phase='A', state='OPEN',
                          name_prefix='blah')

    def test_phase_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase=7, state='OPEN',
                          name_prefix='blah')

    def test_phase_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='N', state='OPEN',
                          name_prefix='blah')

    def test_state_bad_type(self):
        self.assertRaises(TypeError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='c', state=True,
                          name_prefix='blah')

    def test_state_bad_value(self):
        self.assertRaises(ValueError, capacitor.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state='stuck',
                          name_prefix='blah')


if __name__ == '__main__':
    unittest.main()
