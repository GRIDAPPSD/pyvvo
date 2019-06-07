import unittest
from unittest.mock import patch
from copy import copy, deepcopy
from pyvvo.equipment import equipment


# Ensure abstract methods are set.
# https://stackoverflow.com/a/28738073/11052174
@patch.multiple(equipment.EquipmentSinglePhase, __abstractmethods__=set())
class EquipmentMultiPhaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up some default inputs to RegulatorSinglePhase.
        cls.inputs = \
            {'mrid': '_3E73AD1D-08AF-A34B-33D2-1FCE3533380A',
             'name': 'FEEDER_REG', 'phase': 'a'}

    def test_bad_input_type(self):
        self.assertRaises(TypeError, equipment.EquipmentMultiPhase,
                          'hello')

    def test_bad_input_list_length_1(self):
        with self.assertRaisesRegex(ValueError,
                                    r'1 <= len\(equipment_list\) <= 3'):
            equipment.EquipmentMultiPhase([])

    def test_bad_input_list_length_2(self):
        with self.assertRaisesRegex(ValueError,
                                    r'1 <= len\(equipment_list\) <= 3'):
            equipment.EquipmentMultiPhase([1, 2, 3, 4])

    def test_bad_input_list_type(self):
        self.assertRaises(TypeError, equipment.EquipmentMultiPhase,
                          (1, 2, 3))

    def test_successful_init_1(self):
        """Pass three single phase regs."""
        input1 = self.inputs
        input2 = copy(self.inputs)
        input3 = copy(self.inputs)
        input2['phase'] = 'b'
        input3['phase'] = 'C'

        reg1 = equipment.EquipmentSinglePhase(**input1)
        reg2 = equipment.EquipmentSinglePhase(**input2)
        reg3 = equipment.EquipmentSinglePhase(**input3)

        reg_multi_phase = equipment.EquipmentMultiPhase((reg1, reg2, reg3))

        self.assertEqual(reg_multi_phase.name, self.inputs['name'])
        self.assertEqual(reg_multi_phase.mrid, self.inputs['mrid'])
        self.assertIs(reg_multi_phase.a, reg1)
        self.assertIs(reg_multi_phase.b, reg2)
        self.assertIs(reg_multi_phase.c, reg3)

    def test_successful_init_2(self):
        """Pass two single phase regs."""
        input1 = self.inputs
        input3 = copy(self.inputs)
        input3['phase'] = 'C'

        reg1 = equipment.EquipmentSinglePhase(**input1)
        reg3 = equipment.EquipmentSinglePhase(**input3)

        reg_multi_phase = equipment.EquipmentMultiPhase((reg1, reg3))

        # noinspection SpellCheckingInspection
        self.assertEqual(reg_multi_phase.name, self.inputs['name'])
        self.assertEqual(reg_multi_phase.mrid, self.inputs['mrid'])
        self.assertIs(reg_multi_phase.a, reg1)
        self.assertIs(reg_multi_phase.c, reg3)

    def test_successful_init_3(self):
        """Pass a single mocked single phase regs."""
        reg1 = equipment.EquipmentSinglePhase(**self.inputs)

        reg_multi_phase = equipment.EquipmentMultiPhase((reg1, ))

        # noinspection SpellCheckingInspection
        self.assertEqual(reg_multi_phase.name, self.inputs['name'])
        self.assertEqual(reg_multi_phase.mrid, self.inputs['mrid'])
        self.assertIs(reg_multi_phase.a, reg1)

    def test_mismatched_names(self):
        """All single phase regs should have the same name."""
        input1 = self.inputs
        input2 = copy(self.inputs)
        input3 = copy(self.inputs)
        input2['phase'] = 'b'
        input2['name'] = 'just kidding'
        input3['phase'] = 'C'
        reg1 = equipment.EquipmentSinglePhase(**input1)
        reg2 = equipment.EquipmentSinglePhase(**input2)
        reg3 = equipment.EquipmentSinglePhase(**input3)

        with self.assertRaisesRegex(ValueError, 'matching "name" attributes'):
            equipment.EquipmentMultiPhase((reg1, reg2, reg3))

    def test_mismatched_mrids(self):
        """All single phase regs should have the same name."""
        input1 = self.inputs
        input2 = copy(self.inputs)
        input3 = copy(self.inputs)
        input2['phase'] = 'b'
        input2['mrid'] = 'whoops'
        input3['phase'] = 'C'
        reg1 = equipment.EquipmentSinglePhase(**input1)
        reg2 = equipment.EquipmentSinglePhase(**input2)
        reg3 = equipment.EquipmentSinglePhase(**input3)

        with self.assertRaisesRegex(ValueError, 'matching "mrid" attributes'):
            equipment.EquipmentMultiPhase((reg1, reg2, reg3))

    def test_multiple_same_phases(self):
        """Passing multiple RegulatorSinglePhase objects on the same phase
        is not allowed.
        """
        input1 = self.inputs
        input2 = copy(self.inputs)
        input3 = copy(self.inputs)
        input3['phase'] = 'C'
        reg1 = equipment.EquipmentSinglePhase(**input1)
        reg2 = equipment.EquipmentSinglePhase(**input2)
        reg3 = equipment.EquipmentSinglePhase(**input3)

        with self.assertRaisesRegex(ValueError,
                                    'Multiple equipments for phase'):
            equipment.EquipmentMultiPhase((reg1, reg2, reg3))

    
if __name__ == '__main__':
    unittest.main()