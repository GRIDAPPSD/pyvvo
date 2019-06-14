import unittest
from random import randint
from copy import deepcopy
import os
import pandas as pd
import json

from pyvvo.equipment import regulator, equipment
from pyvvo.sparql import REG_MEAS_MEAS_MRID_COL, REG_MEAS_REG_MRID_COL,\
    CAP_MEAS_MEAS_MRID_COL, CAP_MEAS_CAP_MRID_COL

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REGULATORS = os.path.join(THIS_DIR, 'query_regulators.csv')
REG_MEAS = os.path.join(THIS_DIR, 'query_reg_meas.csv')
REG_MEAS_MSG = os.path.join(THIS_DIR, 'reg_meas_message.json')


class EquipmentManagerRegulatorTestCase(unittest.TestCase):
    """Test EquipmentManager with regulator data."""

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
            equipment.EquipmentManager(
                eq_dict=self.reg_dict, eq_meas=self.reg_meas,
                meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                eq_mrid_col=REG_MEAS_REG_MRID_COL
            )

    def test_reg_dict_attribute(self):
        self.assertIs(self.reg_dict, self.reg_mgr.eq_dict)

    def test_inconsistent_inputs(self):
        """Ensure we get an exception if inputs are not consistent.
        """
        meas = self.reg_meas.copy(deep=True)
        # Create a duplicate entry.
        meas.iloc[0] = meas.iloc[1]

        s = 'inputs are inconsistent'
        with self.assertRaisesRegex(ValueError, s):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.reg_dict, eq_meas=meas,
                    meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=REG_MEAS_REG_MRID_COL
                )

    def test_all_measurements_mapped(self):
        """Ensure all measurements are in the map."""
        for meas_mrid in self.reg_meas['pos_meas_mrid'].values:
            with self.subTest():
                self.assertIn(meas_mrid, self.reg_mgr.meas_eq_map.keys())

    def test_no_meas_for_reg(self):
        """Remove measurements for given regulator, ensure we get
        proper exception.
        """
        meas_view = self.reg_meas[
            ~(self.reg_meas['tap_changer_mrid']
              == self.reg_meas['tap_changer_mrid'][0])
        ]

        with self.assertRaisesRegex(ValueError, 'do not match'):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.reg_dict, eq_meas=meas_view,
                    meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=REG_MEAS_REG_MRID_COL
                )

    def test_bad_reg_dict_type(self):
        with self.assertRaisesRegex(TypeError,
                                    'eq_dict must be a dictionary'):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=10, eq_meas=self.reg_meas,
                    meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=REG_MEAS_REG_MRID_COL
                )

    def test_bad_reg_meas_type(self):
        with self.assertRaisesRegex(TypeError,
                                    'eq_meas must be a Pandas'):
            _ = equipment.EquipmentManager(
                eq_dict=self.reg_dict, eq_meas=pd.Series(),
                meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                eq_mrid_col=REG_MEAS_REG_MRID_COL
            )

    def test_no_meas_for_single_phase_reg(self):
        meas_view = self.reg_meas.drop(0, axis=0)
        with self.assertRaisesRegex(ValueError, 'do not match'):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.reg_dict, eq_meas=meas_view,
                    meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=REG_MEAS_REG_MRID_COL
                )

    def test_two_meas_for_single_phase_reg(self):
        reg_meas = self.reg_meas.append(self.reg_meas.iloc[0])

        with self.assertRaisesRegex(ValueError, 'do not match'):
            _ = equipment.EquipmentManager(
                eq_dict=self.reg_dict, eq_meas=reg_meas,
                meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                eq_mrid_col=REG_MEAS_REG_MRID_COL
            )

    def test_update_state_simple(self):
        """Just ensure it runs without error."""

        # At the time of writing, the debug line is the last one in the
        # function, so ensuring it gets hit is adequate.
        with self.assertLogs(level='DEBUG'):
            self.reg_mgr.update_state(self.reg_meas_msg)

    def test_update_state_changes_taps(self):
        """Ensure our taps changed appropriately. We'll hard-code
        this for simplicity.
        """
        self.reg_mgr.update_state(self.reg_meas_msg)

        # Loop over the message.
        for msg_dict in self.reg_meas_msg:
            # Grab the MRID.
            meas_mrid = msg_dict['measurement_mrid']
            meas_value = msg_dict['value']

            # Look up the measurement mrid.
            row = self.reg_meas[
                self.reg_meas[REG_MEAS_MEAS_MRID_COL] == meas_mrid]

            self.assertGreater(row.shape[0], 0)

            # Grab regulator mrid and phase.
            reg_mrid = row[REG_MEAS_REG_MRID_COL].values[0]

            # Ensure this regulator got updated.
            with self.subTest(meas_mrid=meas_mrid):
                # noinspection PyUnboundLocalVariable
                self.assertEqual(self.reg_dict[reg_mrid].tap_pos, meas_value)

    def test_update_state_bad_mrid(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrid': '1234', 'value': 12})

        with self.assertLogs(level='WARNING'):
            self.reg_mgr.update_state(reg_meas_msg)

    def test_update_state_bad_entry_1(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrId': '1234', 'value': 12})

        with self.assertRaisesRegex(KeyError, 'measurement_mrid'):
            self.reg_mgr.update_state(reg_meas_msg)

    def test_update_state_bad_entry_2(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrid': '1234', 'valu3': 12})

        with self.assertRaisesRegex(KeyError, 'value'):
            self.reg_mgr.update_state(reg_meas_msg)

    def test_update_state_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'msg must be a list'):
            self.reg_mgr.update_state(msg='hello there')

    def test_update_state_bad_type_2(self):
        with self.assertRaisesRegex(TypeError, 'must be a dict'):
            self.reg_mgr.update_state(msg=['hello there'])

    def test_build_equipment_commands(self):
        """One stop big function which probably should be spun off into
        its own test case.

        NOTE: This test is fragile as it currently relies on how the
        looping is performed in build_equipment_commands.
        """
        reg_dict_forward = deepcopy(self.reg_dict)
        # For some reason the new eq_dict won't pickle?
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
        out = self.reg_mgr.build_equipment_commands(
            eq_dict_forward=reg_dict_forward)

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

    def test_build_equipment_commands_mismatch(self):
        """Send mismatched reg dicts in."""
        reg_dict_forward = deepcopy(self.reg_dict)
        reg_dict_forward['blah'] = \
            reg_dict_forward.pop(list(reg_dict_forward.keys())[0])

        with self.assertRaisesRegex(ValueError, 'not matching up with'):
            self.reg_mgr.build_equipment_commands(reg_dict_forward)


# # Ensure abstract methods are set.
# # https://stackoverflow.com/a/28738073/11052174
# @patch.multiple(equipment.EquipmentSinglePhase, __abstractmethods__=set())
# class EquipmentMultiPhaseTestCase(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         # Set up some default inputs to RegulatorSinglePhase.
#         cls.inputs = \
#             {'mrid': '_3E73AD1D-08AF-A34B-33D2-1FCE3533380A',
#              'name': 'FEEDER_REG', 'phase': 'a'}
#
#     def test_bad_input_type(self):
#         self.assertRaises(TypeError, equipment.EquipmentMultiPhase,
#                           'hello')
#
#     def test_bad_input_list_length_1(self):
#         with self.assertRaisesRegex(ValueError,
#                                     r'1 <= len\(equipment_list\) <= 3'):
#             equipment.EquipmentMultiPhase([])
#
#     def test_bad_input_list_length_2(self):
#         with self.assertRaisesRegex(ValueError,
#                                     r'1 <= len\(equipment_list\) <= 3'):
#             equipment.EquipmentMultiPhase([1, 2, 3, 4])
#
#     def test_bad_input_list_type(self):
#         self.assertRaises(TypeError, equipment.EquipmentMultiPhase,
#                           (1, 2, 3))
#
#     def test_successful_init_1(self):
#         """Pass three single phase regs."""
#         input1 = self.inputs
#         input2 = copy(self.inputs)
#         input3 = copy(self.inputs)
#         input2['phase'] = 'b'
#         input3['phase'] = 'C'
#
#         reg1 = equipment.EquipmentSinglePhase(**input1)
#         reg2 = equipment.EquipmentSinglePhase(**input2)
#         reg3 = equipment.EquipmentSinglePhase(**input3)
#
#         reg_multi_phase = equipment.EquipmentMultiPhase((reg1, reg2, reg3))
#
#         self.assertEqual(reg_multi_phase.name, self.inputs['name'])
#         self.assertEqual(reg_multi_phase.mrid, self.inputs['mrid'])
#         self.assertIs(reg_multi_phase.a, reg1)
#         self.assertIs(reg_multi_phase.b, reg2)
#         self.assertIs(reg_multi_phase.c, reg3)
#
#     def test_successful_init_2(self):
#         """Pass two single phase regs."""
#         input1 = self.inputs
#         input3 = copy(self.inputs)
#         input3['phase'] = 'C'
#
#         reg1 = equipment.EquipmentSinglePhase(**input1)
#         reg3 = equipment.EquipmentSinglePhase(**input3)
#
#         reg_multi_phase = equipment.EquipmentMultiPhase((reg1, reg3))
#
#         # noinspection SpellCheckingInspection
#         self.assertEqual(reg_multi_phase.name, self.inputs['name'])
#         self.assertEqual(reg_multi_phase.mrid, self.inputs['mrid'])
#         self.assertIs(reg_multi_phase.a, reg1)
#         self.assertIs(reg_multi_phase.c, reg3)
#
#     def test_successful_init_3(self):
#         """Pass a single mocked single phase regs."""
#         reg1 = equipment.EquipmentSinglePhase(**self.inputs)
#
#         reg_multi_phase = equipment.EquipmentMultiPhase((reg1, ))
#
#         # noinspection SpellCheckingInspection
#         self.assertEqual(reg_multi_phase.name, self.inputs['name'])
#         self.assertEqual(reg_multi_phase.mrid, self.inputs['mrid'])
#         self.assertIs(reg_multi_phase.a, reg1)
#
#     def test_mismatched_names(self):
#         """All single phase regs should have the same name."""
#         input1 = self.inputs
#         input2 = copy(self.inputs)
#         input3 = copy(self.inputs)
#         input2['phase'] = 'b'
#         input2['name'] = 'just kidding'
#         input3['phase'] = 'C'
#         reg1 = equipment.EquipmentSinglePhase(**input1)
#         reg2 = equipment.EquipmentSinglePhase(**input2)
#         reg3 = equipment.EquipmentSinglePhase(**input3)
#
#         with self.assertRaisesRegex(ValueError, 'matching "name" attributes'):
#             equipment.EquipmentMultiPhase((reg1, reg2, reg3))
#
#     def test_mismatched_mrids(self):
#         """All single phase regs should have the same name."""
#         input1 = self.inputs
#         input2 = copy(self.inputs)
#         input3 = copy(self.inputs)
#         input2['phase'] = 'b'
#         input2['mrid'] = 'whoops'
#         input3['phase'] = 'C'
#         reg1 = equipment.EquipmentSinglePhase(**input1)
#         reg2 = equipment.EquipmentSinglePhase(**input2)
#         reg3 = equipment.EquipmentSinglePhase(**input3)
#
#         with self.assertRaisesRegex(ValueError, 'matching "mrid" attributes'):
#             equipment.EquipmentMultiPhase((reg1, reg2, reg3))
#
#     def test_multiple_same_phases(self):
#         """Passing multiple RegulatorSinglePhase objects on the same phase
#         is not allowed.
#         """
#         input1 = self.inputs
#         input2 = copy(self.inputs)
#         input3 = copy(self.inputs)
#         input3['phase'] = 'C'
#         reg1 = equipment.EquipmentSinglePhase(**input1)
#         reg2 = equipment.EquipmentSinglePhase(**input2)
#         reg3 = equipment.EquipmentSinglePhase(**input3)
#
#         with self.assertRaisesRegex(ValueError,
#                                     'Multiple equipments for phase'):
#             equipment.EquipmentMultiPhase((reg1, reg2, reg3))
#
#
if __name__ == '__main__':
    unittest.main()
