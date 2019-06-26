import unittest
import os
from pyvvo.equipment import switch, equipment
import pandas as pd

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SWITCHES = os.path.join(THIS_DIR, 'query_switches.csv')


class SwitchSinglePhaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.switch = switch.SwitchSinglePhase(name='my_switch', mrid='xyz',
                                              phase='C', controllable=False)

    def test_equipment(self):
        self.assertIsInstance(self.switch, equipment.EquipmentSinglePhase)

    def test_name(self):
        self.assertEqual('my_switch', self.switch.name)

    def test_mrid(self):
        self.assertEqual('xyz', self.switch.mrid)

    def test_phase(self):
        self.assertEqual('C', self.switch.phase)

    def test_controllable(self):
        self.assertEqual(False, self.switch.controllable)

    def test_state(self):
        self.assertIsNone(self.switch.state)

        self.switch.state = 0

        self.assertEqual(0, self.switch.state)
        self.assertEqual(None, self.switch.state_old)

        self.switch.state = 1
        self.assertEqual(1, self.switch.state)
        self.assertEqual(0, self.switch.state_old)


class InitializeSwitchesTestCase(unittest.TestCase):
    """Test initialize_switches"""
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(SWITCHES)
        cls.switches = switch.initialize_switches(cls.df)

    def test_length(self):
        """Hard-code number of expected switches."""
        self.assertEqual(len(self.switches), 38)

    def test_switch_or_dict_of_switches(self):
        """Ensure all objects are the correct type."""
        for item in self.switches.values():
            try:
                self.assertIsInstance(item, switch.SwitchSinglePhase)
            except AssertionError:
                self.assertIsInstance(item, dict)

                for key, value in item.items():
                    self.assertIn(key, switch.SwitchSinglePhase.PHASES)
                    self.assertIsInstance(value, switch.SwitchSinglePhase)

    def test_controllable(self):
        """For now, all switches are hard-coded to be not controllable.
        """
        for item in self.switches.values():
            try:
                self.assertFalse(item.controllable)
            except AttributeError:
                for key, value in item.items():
                    self.assertFalse(value.controllable)


if __name__ == '__main__':
    unittest.main()
