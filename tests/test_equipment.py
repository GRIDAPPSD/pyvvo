import unittest
from unittest.mock import patch
from random import randint, choice
from copy import deepcopy
import numpy as np
import pandas as pd
import simplejson as json
from datetime import datetime

from pyvvo import equipment, utils

from pyvvo.sparql import REG_MEAS_MEAS_MRID_COL, REG_MEAS_REG_MRID_COL,\
    CAP_MEAS_MEAS_MRID_COL, CAP_MEAS_CAP_MRID_COL, SWITCH_MEAS_MEAS_MRID_COL,\
    SWITCH_MEAS_SWITCH_MRID_COL
import tests.data_files as _df


class EquipmentManagerRegulatorTestCase(unittest.TestCase):
    """Test EquipmentManager with regulator data."""

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.reg_meas = _df.read_pickle(_df.REG_MEAS_9500)
        with open(_df.REG_MEAS_MSG_9500, 'r') as f:
            cls.reg_meas_msg = json.load(f)

        # Do some bad, fragile stuff: loop over all the measurements
        # and increment the value by one. This way, we ensure the
        # regulator actually changes state (I'm pretty sure the original
        # message has the regulators in their original state).
        for d in cls.reg_meas_msg:
            d['value'] += 1

        # Just create a bogus datetime.
        cls.sim_dt = datetime(2019, 9, 2, 17, 8)

    # noinspection PyPep8Naming
    def setUp(self):
        # Gotta be careful with these mutable types... Get fresh
        # instances each time. It won't be that slow, I promise.
        self.reg_dict = \
            equipment.initialize_regulators(
                _df.read_pickle(_df.REGULATORS_9500))
        self.reg_mgr = \
            equipment.EquipmentManager(
                eq_dict=self.reg_dict, eq_meas=self.reg_meas,
                meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                eq_mrid_col=REG_MEAS_REG_MRID_COL
            )

    @staticmethod
    def random_update(reg_dict_in):
        """Helper to randomly update tap steps."""
        # Randomly update steps.
        forward_vals = []
        for reg_single in reg_dict_in.values():

            new_step = reg_single.state

            while new_step == reg_single.state:
                new_step = randint(reg_single.low_step, reg_single.high_step)

            reg_single.state = new_step
            forward_vals.append(new_step)

        return forward_vals

    def test_reg_dict_attribute(self):
        self.assertIs(self.reg_dict, self.reg_mgr.eq_dict)

    def test_missing_meas(self):
        """Ensure we get an exception if missing an input."""
        meas = self.reg_meas.copy(deep=True)
        meas = meas.drop(index=meas.index[-1])

        s = 'The eq_meas input is missing equipment'
        with self.assertRaisesRegex(ValueError, s):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.reg_dict, eq_meas=meas,
                    meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=REG_MEAS_REG_MRID_COL
                )

    def test_duplicate_meas(self):
        """Ensure we get an exception if inputs are not consistent.
        """
        meas = self.reg_meas.copy(deep=True)
        # Create a duplicate entry.
        meas = meas.append(meas.iloc[0])

        s = 'Received 2 measurements for equipment with mrid'
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

        with self.assertRaisesRegex(ValueError, 'The eq_meas input is miss'):
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
        with self.assertRaisesRegex(ValueError, 'The eq_meas input is miss'):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.reg_dict, eq_meas=meas_view,
                    meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=REG_MEAS_REG_MRID_COL
                )

    def test_two_meas_for_single_phase_reg(self):
        reg_meas = self.reg_meas.append(self.reg_meas.iloc[0])

        with self.assertRaisesRegex(ValueError, 'Received 2 measurements for'):
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
            self.reg_mgr.update_state(self.reg_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_changes_taps(self):
        """Ensure our taps changed appropriately. We'll hard-code
        this for simplicity.
        """
        self.reg_mgr.update_state(self.reg_meas_msg, sim_dt=self.sim_dt)

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
                self.assertEqual(self.reg_dict[reg_mrid].state, meas_value)

    def test_update_state_bad_mrid(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrid': '1234', 'value': 12})

        with self.assertLogs(level='WARNING'):
            self.reg_mgr.update_state(reg_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_bad_entry_1(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrId': '1234', 'value': 12})

        with self.assertRaisesRegex(KeyError, 'measurement_mrid'):
            self.reg_mgr.update_state(reg_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_bad_entry_2(self):
        reg_meas_msg = deepcopy(self.reg_meas_msg)
        reg_meas_msg.append({'measurement_mrid': '1234', 'valu3': 12})

        with self.assertRaisesRegex(KeyError, 'value'):
            self.reg_mgr.update_state(reg_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'msg must be a list'):
            self.reg_mgr.update_state(msg='hello there', sim_dt=self.sim_dt)

    def test_update_state_bad_type_2(self):
        with self.assertRaisesRegex(TypeError, 'string indices must be'):
            self.reg_mgr.update_state(msg=['hello there'], sim_dt=self.sim_dt)

    def test_build_equipment_commands_no_op(self):
        """Ensure that if equipment has the same state, no update
        command comes out.
        """
        # Just use the same dictionary to make a "do nothing" command.
        out = self.reg_mgr.build_equipment_commands(
            eq_dict_forward=self.reg_dict)

        for v in out.values():
            self.assertEqual(0, len(v))

    def test_build_equipment_commands(self):
        """One stop big function which probably should be spun off into
        its own test case.

        NOTE: This test is fragile as it currently relies on how the
        looping is performed in build_equipment_commands.
        """
        reg_dict_forward = deepcopy(self.reg_dict)
        # For some reason the new eq_dict won't pickle?
        # reg_dict_forward = \
        #     equipment.initialize_regulators(
        #         _df.read_pickle(_df.REGULATORS_9500))

        # Randomly update steps.
        forward_vals = self.random_update(reg_dict_forward)

        # Grab reverse values.
        reverse_vals = []
        for reg_single in self.reg_dict.values():
            reverse_vals.append(reg_single.state)

        # Command the regulators.
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
        # I'm just going to hard-code the fact that the 9500 node model
        # has 6 3-phase regs.
        for v in out.values():
            self.assertIsInstance(v, list)
            self.assertEqual(len(v), 18)

    def test_build_equipment_commands_mismatch(self):
        """Send mismatched reg dicts in."""
        reg_dict_forward = deepcopy(self.reg_dict)
        reg_dict_forward['blah'] = \
            reg_dict_forward.pop(list(reg_dict_forward.keys())[0])

        with self.assertRaisesRegex(ValueError, 'not matching up with'):
            self.reg_mgr.build_equipment_commands(reg_dict_forward)

    def test_build_equipment_commands_json_serializable(self):
        """If any Numpy data types leak in, we've got a problem in that
        we can't serialize the data into json. Inject numpy types and
        attempt to serialize the result.
        """
        # Make a copy of the message
        msg = deepcopy(self.reg_meas_msg)

        # Change all the data types.
        for d in msg:
            d['value'] = np.int64(d['value'])

        def assert_64(eq):
            assert isinstance(eq.state, np.int64)

        def assert_32(eq):
            assert isinstance(eq.state, np.int32)

        # Update the regulators.
        self.reg_mgr.update_state(msg=msg, sim_dt='high noon')

        # Ensure the equipment actually has int64 states.
        equipment.loop_helper(self.reg_dict, assert_64)

        # Similarly, update all the equipment in the dictionary.

        # Helper for casting to int32.
        def to_int32(eq):
            eq.state = np.int32(eq.state)

        # Get a copy of the regulators and randomly change their states.
        reg_dict_copy = deepcopy(self.reg_dict)
        self.random_update(reg_dict_copy)

        # Now cast the now randomized states to int32.
        equipment.loop_helper(eq_dict=reg_dict_copy, func=to_int32)
        equipment.loop_helper(reg_dict_copy, assert_32)

        # Build the equipment commands. This returns a dictionary.
        cmd = self.reg_mgr.build_equipment_commands(reg_dict_copy)

        # Ensure we actually get commands out, otherwise we're giving
        # ourselves false confidence in this test.
        for v in cmd.values():
            self.assertGreater(len(v), 0)

        # Attempt to serialize cmd.
        j_str = json.dumps(cmd)

        # Put in an assert just for good measure.
        self.assertIsInstance(j_str, str)

    def test_lookup_locked(self):
        """Ensure lookup_eq_by_mrid_and_phase uses the lock."""
        with patch.object(self.reg_mgr, '_lock') as p_lock:
            # Unfortunately we cannot patch the method here, because
            # that also patches the wrapping. So, call the method.
            with self.assertRaises(KeyError):
                self.reg_mgr.lookup_eq_by_mrid_and_phase('abc')

        # Ensure that acquire and release were called.
        self.assertEqual('acquire', p_lock.method_calls[0][0])
        self.assertEqual('release', p_lock.method_calls[1][0])

    def test_update_state_locked(self):
        """Ensure update_state uses the lock."""
        with patch.object(self.reg_mgr, '_lock') as p_lock:
            # Unfortunately we cannot patch the method here, because
            # that also patches the wrapping. So, call the method.
            with self.assertRaisesRegex(TypeError, 'msg must be a list'):
                self.reg_mgr.update_state('abc', 'def')

        # Ensure that acquire and release were called.
        self.assertEqual('acquire', p_lock.method_calls[0][0])
        self.assertEqual('release', p_lock.method_calls[1][0])

    def test_build_equipment_commands_locked(self):
        """Ensure build_equipment_commands uses the lock."""
        with patch.object(self.reg_mgr, '_lock') as p_lock:
            # Unfortunately we cannot patch the method here, because
            # that also patches the wrapping. So, call the method.
            with self.assertRaises(AttributeError):
                self.reg_mgr.build_equipment_commands('abc')

        # Ensure that acquire and release were called.
        self.assertEqual('acquire', p_lock.method_calls[0][0])
        self.assertEqual('release', p_lock.method_calls[1][0])


class EquipmentManagerCapacitorTestCase(unittest.TestCase):
    """Test EquipmentManager with capacitor data."""

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.cap_meas = _df.read_pickle(_df.CAP_MEAS_9500)
        with open(_df.CAP_MEAS_MSG_9500, 'r') as f:
            cls.cap_meas_msg = json.load(f)

        # Just create a bogus datetime.
        cls.sim_dt = datetime(2019, 9, 2, 17, 8)

    # noinspection PyPep8Naming
    def setUp(self):
        # Gotta be careful with these mutable types... Get fresh
        # instances each time. It won't be that slow, I promise.
        self.cap_dict = \
            equipment.initialize_capacitors(
                _df.read_pickle(_df.CAPACITORS_9500))
        self.cap_mgr = \
            equipment.EquipmentManager(
                eq_dict=self.cap_dict, eq_meas=self.cap_meas,
                meas_mrid_col=CAP_MEAS_MEAS_MRID_COL,
                eq_mrid_col=CAP_MEAS_CAP_MRID_COL
            )

    def test_cap_dict_attribute(self):
        self.assertIs(self.cap_dict, self.cap_mgr.eq_dict)

    def test_inconsistent_inputs(self):
        """Ensure we get an exception if inputs are not consistent.
        """
        meas = self.cap_meas.copy(deep=True)
        # Create a duplicate entry.
        meas = meas.append(meas.iloc[0])

        s = 'Received 2 measurements for equipment with mrid'
        with self.assertRaisesRegex(ValueError, s):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.cap_dict, eq_meas=meas,
                    meas_mrid_col=CAP_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=CAP_MEAS_CAP_MRID_COL
                )

    def test_all_measurements_mapped(self):
        """Ensure all measurements are in the map."""
        for meas_mrid in self.cap_meas[CAP_MEAS_MEAS_MRID_COL].values:
            with self.subTest():
                self.assertIn(meas_mrid, self.cap_mgr.meas_eq_map.keys())

    def test_no_meas_for_cap(self):
        """Remove measurements for given capacitor, ensure we get
        proper exception.
        """
        meas_view = self.cap_meas[
            ~(self.cap_meas[CAP_MEAS_CAP_MRID_COL]
              == self.cap_meas[CAP_MEAS_CAP_MRID_COL].iloc[-1])
        ]

        with self.assertRaisesRegex(ValueError, 'The eq_meas input is miss'):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=self.cap_dict, eq_meas=meas_view,
                    meas_mrid_col=CAP_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=CAP_MEAS_CAP_MRID_COL
                )

    def test_bad_cap_dict_type(self):
        with self.assertRaisesRegex(TypeError,
                                    'eq_dict must be a dictionary'):
            _ = \
                equipment.EquipmentManager(
                    eq_dict=10, eq_meas=self.cap_meas,
                    meas_mrid_col=CAP_MEAS_MEAS_MRID_COL,
                    eq_mrid_col=CAP_MEAS_CAP_MRID_COL
                )

    def test_bad_cap_meas_type(self):
        with self.assertRaisesRegex(TypeError,
                                    'eq_meas must be a Pandas'):
            _ = equipment.EquipmentManager(
                eq_dict=self.cap_dict, eq_meas=pd.Series(),
                meas_mrid_col=CAP_MEAS_MEAS_MRID_COL,
                eq_mrid_col=CAP_MEAS_CAP_MRID_COL
            )

    def test_update_state_simple(self):
        """Just ensure it runs without error."""

        # At the time of writing, the debug line is the last one in the
        # function, so ensuring it gets hit is adequate.
        with self.assertLogs(level='DEBUG'):
            self.cap_mgr.update_state(self.cap_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_changes_state(self):
        """Ensure our states changed appropriately. We'll hard-code
        this for simplicity.
        """
        self.cap_mgr.update_state(self.cap_meas_msg, sim_dt=self.sim_dt)

        # Loop over the message.
        for msg_dict in self.cap_meas_msg:
            # Grab the MRID.
            meas_mrid = msg_dict['measurement_mrid']
            meas_value = msg_dict['value']

            # Look up the measurement mrid.
            row = self.cap_meas[
                self.cap_meas[CAP_MEAS_MEAS_MRID_COL] == meas_mrid]

            self.assertGreater(row.shape[0], 0)

            # Grab capacitor mrid and phase.
            cap_mrid = row[CAP_MEAS_CAP_MRID_COL].values[0]
            cap_phase = row['phase'].values[0]

            # Ensure this capacitor got updated.
            with self.subTest(meas_mrid=meas_mrid):
                # Lookup the object.
                eq = self.cap_mgr.lookup_eq_by_mrid_and_phase(mrid=cap_mrid,
                                                              phase=cap_phase)
                # noinspection PyUnboundLocalVariable
                self.assertEqual(eq.state, meas_value)

    def test_update_state_bad_mrid(self):
        cap_meas_msg = deepcopy(self.cap_meas_msg)
        cap_meas_msg.append({'measurement_mrid': '1234', 'value': 12})

        with self.assertLogs(level='WARNING'):
            self.cap_mgr.update_state(cap_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_bad_entry_1(self):
        cap_meas_msg = deepcopy(self.cap_meas_msg)
        cap_meas_msg.append({'measurement_mrId': '1234', 'value': 12})

        with self.assertRaisesRegex(KeyError, 'measurement_mrid'):
            self.cap_mgr.update_state(cap_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_bad_entry_2(self):
        cap_meas_msg = deepcopy(self.cap_meas_msg)
        cap_meas_msg.append({'measurement_mrid': '1234', 'valu3': 12})

        with self.assertRaisesRegex(KeyError, 'value'):
            self.cap_mgr.update_state(cap_meas_msg, sim_dt=self.sim_dt)

    def test_update_state_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'msg must be a list'):
            self.cap_mgr.update_state(msg='hello there', sim_dt=self.sim_dt)

    def test_update_state_bad_type_2(self):
        with self.assertRaisesRegex(TypeError, 'string indices must'):
            self.cap_mgr.update_state(msg=['hello there'], sim_dt=self.sim_dt)

    def test_build_equipment_commands(self):
        """One stop big function which probably should be spun off into
        its own test case.

        NOTE: This test is fragile as it currently relies on how the
        looping is performed in build_equipment_commands.
        """
        cap_dict_forward = deepcopy(self.cap_dict)
        # For some reason the new eq_dict won't pickle?
        # cap_dict_forward = \
        #     equipment.initialize_regulators(
        #         _df.read_pickle(REGULATORS))

        def update_state(cap, forward):
            """Nested helper function."""
            new_state = choice(equipment.CapacitorSinglePhase.STATES)
            cap.state = new_state
            forward.append(new_state)

        # Randomly update steps.
        forward_vals = []
        for cap_or_dict in cap_dict_forward.values():
            if isinstance(cap_or_dict, equipment.EquipmentSinglePhase):
                update_state(cap_or_dict, forward_vals)
            elif isinstance(cap_or_dict, dict):
                for phase, cap_obj in cap_or_dict.items():
                    if cap_obj.controllable:
                        update_state(cap_obj, forward_vals)
            else:
                raise ValueError('What has gone wrong?')

        # Grab reverse values.
        reverse_vals = []
        for cap_or_dict in self.cap_dict.values():
            if isinstance(cap_or_dict, equipment.EquipmentSinglePhase):
                reverse_vals.append(cap_or_dict.state)
            elif isinstance(cap_or_dict, dict):
                for cap_obj in cap_or_dict.values():
                    if cap_obj.controllable:
                        reverse_vals.append(cap_obj.state)

        # Just use the same dictionary to make a "do nothing" command.
        out = self.cap_mgr.build_equipment_commands(
            eq_dict_forward=cap_dict_forward)

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

        # Ensure the lengths are equal to all our controllable
        # capacitors. Hard-code the fact there are 9.
        for v in out.values():
            self.assertIsInstance(v, list)
            self.assertEqual(len(v), 9)

    def test_build_equipment_commands_mismatch(self):
        """Send mismatched cap dicts in."""
        cap = deepcopy(self.cap_dict)
        cap['blah'] = \
            cap.pop(list(cap.keys())[0])

        with self.assertRaisesRegex(ValueError, 'not matching up with'):
            self.cap_mgr.build_equipment_commands(cap)


class EquipmentManagerSwitchTestCase(unittest.TestCase):
    """Test EquipmentManager with switch data. Since the "Regulator"
    and "Capacitor" versions of this test go pretty in-depth, we'll
    keep this one light and simple.
    """

    @classmethod
    def setUpClass(cls):
        cls.switch_meas = _df.read_pickle(_df.SWITCH_MEAS_9500)
        with open(_df.SWITCH_MEAS_MSG_9500, 'r') as f:
            cls.switch_meas_msg = json.load(f)

        # Just create a bogus datetime.
        cls.sim_dt = datetime(2019, 9, 2, 17, 8)

    # noinspection PyPep8Naming
    def setUp(self):
        # Gotta be careful with these mutable types... Get fresh
        # instances each time. It won't be that slow, I promise.
        self.switch_dict = \
            equipment.initialize_switches(
                _df.read_pickle(_df.SWITCHES_9500))
        self.switch_mgr = \
            equipment.EquipmentManager(
                eq_dict=self.switch_dict, eq_meas=self.switch_meas,
                meas_mrid_col=SWITCH_MEAS_MEAS_MRID_COL,
                eq_mrid_col=SWITCH_MEAS_SWITCH_MRID_COL
            )

    def state_none(self, switch):
        """Helper to ensure a switch state is None."""
        self.assertIsNone(switch.state)

    def state_valid(self, switch):
        """Helper to ensure a switch state is valid."""
        self.assertIn(switch.state, equipment.SwitchSinglePhase.STATES)

    def test_update(self):
        """Send in an update message and ensure state changed."""
        # Start by ensuring all switches start with a status of None.
        equipment.loop_helper(eq_dict=self.switch_mgr.eq_dict,
                              func=self.state_none)

        # Now that we've ensure all switches start with None status,
        # update them all.
        self.switch_mgr.update_state(self.switch_meas_msg, sim_dt=self.sim_dt)

        # Loop again and ensure the states are now not None and are
        # valid.
        equipment.loop_helper(eq_dict=self.switch_mgr.eq_dict,
                              func=self.state_valid)


class InverterEquipmentManagerTestCase(unittest.TestCase):
    """Test the InverterEquipmentManager."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.inv_meas = _df.read_pickle(_df.INVERTER_MEAS_9500)
        with open(_df.INVERTER_MEAS_MSG_9500, 'r') as f:
            cls.inv_meas_msg = json.load(f)

        # Override all the measurement magintudes and angles so we can
        # ensure they result in a change when the message is sent in
        # as an update command.
        n = len(cls.inv_meas_msg)
        # Magnitude values between 1 and 1000.
        cls.mag_values = (1000 - 1) * np.random.random(n) + 1
        # Angle values between -90 and 90.
        cls.angle_values = (90 + 90) * np.random.random(n) - 90

        for mag, ang, m in zip(cls.mag_values, cls.angle_values,
                               cls.inv_meas_msg):
            m['magnitude'] = mag.item()
            m['angle'] = ang.item()

        # Just create a bogus datetime.
        cls.sim_dt = datetime(2019, 9, 2, 17, 8)

    def setUp(self) -> None:
        self.inv_dict = equipment.initialize_inverters(
            _df.read_pickle(_df.INVERTERS_9500))

        self.mgr = \
            equipment.InverterEquipmentManager(eq_dict=self.inv_dict,
                                               eq_meas=self.inv_meas,
                                               meas_mrid_col='meas_mrid',
                                               eq_mrid_col='inverter_mrid')

    def test_init(self):
        """Ensure initialization occurs without errors, and that we
        get the properties expected.
        """

        self.assertIsInstance(self.mgr, equipment.EquipmentManager)
        self.assertIs(self.inv_meas, self.mgr.eq_meas)
        self.assertIs(self.inv_dict, self.mgr.eq_dict)

        # This is a bit of a kludge, but oh well. I want to ensure that
        # the initializer is not overridden, so we'll compare the doc
        # strings. Not great, but maybe better than nothing.
        self.assertEqual(self.mgr.__init__.__doc__,
                         equipment.EquipmentManager.__init__.__doc__)

    def test_build_equipment_commands(self):
        """This hasn't been implemented yet, but will need tested when
        it is.
        """
        with self.assertRaisesRegex(NotImplementedError, 'build_equipment_co'):
            self.mgr.build_equipment_commands('bleh')

    def test_update_state(self):
        """Ensure update_state works."""
        # Start by ensuring all inverters do not have a previous state.
        current_state = []
        for inv in self.mgr.meas_eq_map.values():
            current_state.append(inv.state)
            self.assertIsNone(inv.state_old)

        self.mgr.update_state(msg=self.inv_meas_msg, sim_dt=self.sim_dt)

        # Ensure that the current states match the former current states
        # after update. Python now ensures that looping over the same
        # dictionary twice will loop in the same order, so we'll count
        # on that fact here.
        for s, inv in zip(current_state, self.mgr.meas_eq_map.values()):
            self.assertEqual(s, inv.state_old)
            
        # Make sure our fancy looping didn't screw something up.
        for m in self.inv_meas_msg:
            rect = utils.get_complex(r=m['magnitude'], phi=m['angle'],
                                     degrees=True)
            s = (rect.real, rect.imag)

            eq = self.mgr.meas_eq_map[m['measurement_mrid']]

            self.assertEqual(s, eq.state)


class InitializeRegulatorsTestCase(unittest.TestCase):
    """Test initialize_regulators"""

    # noinspection PyPep8Naming
    @classmethod
    def setUpClass(cls):
        cls.df = _df.read_pickle(_df.REGULATORS_9500)
        cls.regs = equipment.initialize_regulators(cls.df)

    def test_eighteen_tap_changers(self):
        """There should be 18 single phase regulators (6 x 3 phase)"""
        self.assertEqual(len(self.regs), 18)

    def test_all_regs(self):
        """Every item should be a RegulatorSinglePhase"""
        for key, reg in self.regs.items():
            with self.subTest('reg = {}'.format(reg)):
                self.assertIsInstance(reg, equipment.RegulatorSinglePhase)


# noinspection PyProtectedMember
class TapCIMToGLDTestCase(unittest.TestCase):

    def test_tap_cim_to_gld_1(self):
        actual = equipment._tap_cim_to_gld(step=16, neutral_step=16)
        self.assertEqual(0, actual)

    def test_tap_cim_to_gld_2(self):
        actual = equipment._tap_cim_to_gld(step=1, neutral_step=8)
        self.assertEqual(-7, actual)

    def test_tap_cim_to_gld_3(self):
        actual = equipment._tap_cim_to_gld(step=24, neutral_step=16)
        self.assertEqual(8, actual)

    def test_tap_cim_to_gld_4(self):
        actual = equipment._tap_cim_to_gld(step=0, neutral_step=16)
        self.assertEqual(-16, actual)

    def test_tap_cim_to_gld_5(self):
        actual = equipment._tap_cim_to_gld(step=-2, neutral_step=-1)
        self.assertEqual(-1, actual)


# noinspection PyProtectedMember
class TapGLDToCIMTestCase(unittest.TestCase):

    def test_tap_gld_to_cim_1(self):
        actual = equipment._tap_gld_to_cim(tap_pos=2, neutral_step=8)
        self.assertEqual(10, actual)

    def test_tap_gld_to_cim_2(self):
        actual = equipment._tap_gld_to_cim(tap_pos=-10, neutral_step=16)

        self.assertEqual(6, actual)

    def test_tap_gld_to_cim_3(self):
        actual = equipment._tap_gld_to_cim(tap_pos=0, neutral_step=10)
        self.assertEqual(10, actual)

    def test_tap_gld_to_cim_4(self):
        actual = equipment._tap_gld_to_cim(tap_pos=5, neutral_step=-5)
        self.assertEqual(0, actual)

    def test_tap_gld_to_cim_5(self):
        actual = equipment._tap_gld_to_cim(tap_pos=2, neutral_step=0)
        self.assertEqual(2, actual)


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
             'step': 18, 'step_voltage_increment': 0.625,
             'controllable': True}

        cls.reg = equipment.RegulatorSinglePhase(**cls.inputs)

    def test_equipment(self):
        self.assertIsInstance(self.reg, equipment.EquipmentSinglePhase)

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
        reg = equipment.RegulatorSinglePhase(**self.inputs)

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
        reg = equipment.RegulatorSinglePhase(**self.inputs)

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
        reg = equipment.RegulatorSinglePhase(**self.inputs)
        with self.assertRaisesRegex(TypeError, 'step must be an integer'):
            reg.step = 1.0

    def test_update_tap_pos_bad_type(self):
        reg = equipment.RegulatorSinglePhase(**self.inputs)
        with self.assertRaisesRegex(TypeError, 'tap_pos must be an integer'):
            reg.tap_pos = -1.0

    def test_update_step_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'step must be between'):
            self.reg.step = 100

    def test_update_tap_pos_out_of_range(self):
        with self.assertRaisesRegex(ValueError, 'tap_pos must be between'):
            self.reg.tap_pos = 17

    def test_controllable(self):
        self.assertEqual(self.inputs['controllable'], self.reg.controllable)


class RegulatorSinglePhaseBadInputsTestCase(unittest.TestCase):

    def setUp(self):
        """We want fresh inputs each time as we'll be modifying fields.
        """
        self.i = \
            {'control_mode': 'voltage',
             'enabled': True, 'high_step': 32, 'low_step': 0,
             'mrid': '_3E73AD1D-08AF-A34B-33D2-1FCE3533380A',
             'name': 'FEEDER_REG', 'neutral_step': 16, 'phase': 'A',
             'tap_changer_mrid': '_330E7EDE-2C70-8F72-B183-AA4BA3C5E221',
             'step': 1.0125, 'step_voltage_increment': 0.625,
             'controllable': True}

    def test_bad_mrid_type(self):
        self.i['mrid'] = 10
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_name_type(self):
        self.i['name'] = {'name': 'reg'}
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_phase_type(self):
        self.i['name'] = ['name', 'yo']
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_phase_value(self):
        self.i['phase'] = 'N'
        self.assertRaises(ValueError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_tap_changer_mrid_type(self):
        self.i['tap_changer_mrid'] = 111
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_step_voltage_increment_type(self):
        self.i['step_voltage_increment'] = 1
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_control_mode_type(self):
        self.i['control_mode'] = (0, 0, 1)
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_control_mode_value(self):
        self.i['control_mode'] = 'my mode'
        self.assertRaises(ValueError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_enabled_type(self):
        self.i['enabled'] = 'true'
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_high_step_type(self):
        self.i['high_step'] = 10.1
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_low_step_type(self):
        self.i['low_step'] = 10 + 1j
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_neutral_step_type(self):
        self.i['neutral_step'] = '16'
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_bad_step_values_1(self):
        self.i['low_step'] = 17
        self.i['neutral_step'] = 16
        self.i['high_step'] = 20
        with self.assertRaisesRegex(ValueError, 'The following is not True'):
            equipment.RegulatorSinglePhase(**self.i)

    def test_bad_step_values_2(self):
        self.i['low_step'] = 0
        self.i['neutral_step'] = 21
        self.i['high_step'] = 20
        with self.assertRaisesRegex(ValueError, 'The following is not True'):
            equipment.RegulatorSinglePhase(**self.i)

    def test_bad_step_values_3(self):
        self.i['low_step'] = 0
        self.i['neutral_step'] = 0
        self.i['high_step'] = -1
        with self.assertRaisesRegex(ValueError, 'The following is not True'):
            equipment.RegulatorSinglePhase(**self.i)

    def test_bad_step_type(self):
        self.i['step'] = 2.0
        self.assertRaises(TypeError, equipment.RegulatorSinglePhase, **self.i)

    def test_step_out_of_range_1(self):
        self.i['step'] = 33
        with self.assertRaisesRegex(ValueError, 'between low_step '):
            equipment.RegulatorSinglePhase(**self.i)

    def test_step_out_of_range_2(self):
        self.i['step'] = -1
        with self.assertRaisesRegex(ValueError, 'between low_step '):
            equipment.RegulatorSinglePhase(**self.i)

    def test_controllable_bad_type(self):
        self.i['controllable'] = 0
        with self.assertRaisesRegex(TypeError, 'controllable'):
            equipment.RegulatorSinglePhase(**self.i)


class CapacitorSinglePhaseTestCase(unittest.TestCase):
    """Basic property tests for CapacitorSinglePhase."""

    @classmethod
    def setUpClass(cls):
        """Create CapacitorSinglePhase object."""
        cls.cap = \
            equipment.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
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
            equipment.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
                                           state=None, mode='voltage',
                                           controllable=True)
        self.assertIsNone(cap.state)

    def test_repr(self):
        self.assertIn(self.cap.name, str(self.cap))
        self.assertIn(self.cap.phase, str(self.cap))
        self.assertIn('CapacitorSinglePhase', str(self.cap))

    def test_state_update(self):
        cap = \
            equipment.CapacitorSinglePhase(name='cap1', mrid='1', phase='c',
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
        self.assertRaises(TypeError, equipment.CapacitorSinglePhase,
                          name=[1, 2, 3], mrid='1', phase='A', state='OPEN',
                          mode='admittance', controllable=True)

    def test_mrid_bad_type(self):
        self.assertRaises(TypeError, equipment.CapacitorSinglePhase,
                          name='1', mrid={'a': 1}, phase='A', state='OPEN',
                          mode='admittance', controllable=True)

    def test_phase_bad_type(self):
        self.assertRaises(TypeError, equipment.CapacitorSinglePhase,
                          name='1', mrid='1', phase=7, state='OPEN',
                          mode='admittance', controllable=True)

    def test_phase_bad_value(self):
        self.assertRaises(ValueError, equipment.CapacitorSinglePhase,
                          name='1', mrid='1', phase='N', state='OPEN',
                          mode='admittance', controllable=True)

    def test_state_bad_type(self):
        self.assertRaises(TypeError, equipment.CapacitorSinglePhase,
                          name='1', mrid='1', phase='c', state=1.0,
                          mode='admittance', controllable=True)

    def test_state_bad_value(self):
        self.assertRaises(ValueError, equipment.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state=3,
                          mode='admittance', controllable=True)

    def test_mode_bad_type(self):
        self.assertRaises(TypeError, equipment.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state=0,
                          mode=0, controllable=True)

    def test_mode_bad_value(self):
        self.assertRaises(ValueError, equipment.CapacitorSinglePhase,
                          name='1', mrid='1', phase='b', state=0,
                          mode='vvo', controllable=True)

    def test_controllable_bad_type(self):
        with self.assertRaisesRegex(TypeError, 'controllable must be a bool'):
            equipment.CapacitorSinglePhase(
                name='1', mrid='1', phase='b', state=0,
                mode='temperature', controllable='True')

    def test_mode_controllable_mismatch_1(self):
        with self.assertRaisesRegex(ValueError, 'seem to conflict'):
            equipment.CapacitorSinglePhase(
                name='1', mrid='1', phase='b', state=None,
                mode=None, controllable=True)

    def test_mode_controllable_mismatch_2(self):
        with self.assertRaisesRegex(ValueError, 'seem to conflict'):
            equipment.CapacitorSinglePhase(
                name='1', mrid='1', phase='b', state=None,
                mode='voltage', controllable=False)


class InitializeCapacitorsTestCase(unittest.TestCase):
    """Test initialize_capacitors for the 9500 node model."""

    @classmethod
    def setUpClass(cls):
        cls.df = _df.read_pickle(_df.CAPACITORS_9500)
        cls.caps = equipment.initialize_capacitors(cls.df)

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
            self.assertIsInstance(cap, (equipment.CapacitorSinglePhase, dict))
            # If we have a dict, ensure all values are
            # CapacitorSinglePhase.
            if isinstance(cap, dict):
                dict_count += 1
                for c in cap.values():
                    dict_cap_count += 1
                    self.assertIsInstance(c, equipment.CapacitorSinglePhase)
            else:
                cap_count += 1

        self.assertEqual(cap_count, 9)
        self.assertEqual(dict_count, 1)
        self.assertEqual(dict_cap_count, 3)

    def test_controllable_capacitors(self):
        """We should be able to control 9 of the 10 capacitor objects.
        """
        control_count = 0
        for _, cap in self.caps.items():
            if isinstance(cap, dict):
                for c in cap.values():
                    if c.controllable:
                        control_count += 1
            elif isinstance(cap, equipment.CapacitorSinglePhase):
                if cap.controllable:
                    control_count += 1
            else:
                raise TypeError('Unexpected type!')

        self.assertEqual(9, control_count)


class InitializeCapacitors13TestCase(unittest.TestCase):
    """Test initialize_capacitors, but use the 13 bus data.
    There shouldn't be any controllable capacitors.
    """
    @classmethod
    def setUpClass(cls):
        cls.df = _df.read_pickle(_df.CAPACITORS_13)
        cls.caps = equipment.initialize_capacitors(cls.df)

    def test_length(self):
        """The return should have 2 items - 1 three phase cap and 1
        single phase."""
        self.assertEqual(len(self.caps), 2)

    def test_no_controllable_caps(self):
        """None of these capacitors are controllable."""
        for _, cap in self.caps.items():
            self.assertIsInstance(cap, (equipment.CapacitorSinglePhase, dict))

            if isinstance(cap, dict):
                for c in cap.values():
                    self.assertFalse(c.controllable)
            else:
                self.assertFalse(cap.controllable)


class SwitchSinglePhaseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.switch = equipment.SwitchSinglePhase(name='my_switch', mrid='xyz',
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
        cls.df = _df.read_pickle(_df.SWITCHES_9500)
        cls.switches = equipment.initialize_switches(cls.df)

    def test_length(self):
        """Hard-code number of expected switches."""
        self.assertEqual(len(self.switches), 107)

    def test_switch_or_dict_of_switches(self):
        """Ensure all objects are the correct type."""
        for item in self.switches.values():
            try:
                self.assertIsInstance(item, equipment.SwitchSinglePhase)
            except AssertionError:
                self.assertIsInstance(item, dict)

                for key, value in item.items():
                    self.assertIn(key, equipment.SwitchSinglePhase.PHASES)
                    self.assertIsInstance(value, equipment.SwitchSinglePhase)

    def test_controllable(self):
        """For now, all switches are hard-coded to be not controllable.
        """
        for item in self.switches.values():
            try:
                self.assertFalse(item.controllable)
            except AttributeError:
                for key, value in item.items():
                    self.assertFalse(value.controllable)


class InverterSinglePhaseTestCase(unittest.TestCase):
    """Test InverterSinglePhase class"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.p = 12.5
        cls.q = 18
        cls.inverter = equipment.InverterSinglePhase(mrid='abcd',
                                                     name='my_inv',
                                                     phase='s2',
                                                     controllable=True,
                                                     p=cls.p, q=cls.q)

    def test_state_expected(self):
        self.assertEqual(self.inverter.state, (self.p, self.q))

    def test_p_expected(self):
        self.assertEqual(self.inverter.p, self.p)

    def test_q_expected(self):
        self.assertEqual(self.inverter.q, self.q)

    def test_valid_state_update(self):
        inv_copy = deepcopy(self.inverter)
        inv_copy.state = (9, 3.1)
        self.assertEqual(inv_copy.state, (9, 3.1))

    def test_init_invalid_state(self):
        with self.assertRaisesRegex(ValueError, 'The contents of the state'):
            # noinspection PyUnusedLocal
            inv = equipment.InverterSinglePhase(mrid='blah', name='bleh',
                                                phase='C', controllable=False,
                                                p=3.2, q='a')

    def test_set_invalid_state_bad_length(self):
        inv_copy = deepcopy(self.inverter)
        with self.assertRaisesRegex(ValueError, 'state must be a two element'):
            inv_copy.state = (3, 6, 9)

    def test_set_invalid_state_non_tuple(self):
        inv_copy = deepcopy(self.inverter)
        with self.assertRaisesRegex(TypeError, 'state must be a two element'):
            inv_copy.state = [13.5, -22.7]


class InitializeInvertersTestCase(unittest.TestCase):
    """Test initialize_inverters."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = _df.read_pickle(_df.INVERTERS_9500)
        cls.inverters = equipment.initialize_inverters(cls.df)

    def test_length(self):
        """Length should match original DataFrame."""
        self.assertEqual(len(self.inverters), self.df.shape[0])

    def test_dict_or_inverters(self):
        """Ensure all instances are dictionaries or Inverters."""
        equipment.loop_helper(self.inverters, self.assertIsInstance,
                              equipment.InverterSinglePhase)

    def test_all_controllable(self):
        """At this point in time, all inverters are controllable."""
        def assert_controllable(inverter):
            self.assertTrue(inverter.controllable)

        equipment.loop_helper(self.inverters, assert_controllable)

    def test_three_phase_count(self):
        """Ensure we get the expected number of three phase inverters.
        """
        # We should get a three phase inverter if the phases is NaN.
        expected_3 = self.df['phases'].isna().values.sum()
        
        # For the 9500 node model, there should be more than 0 three
        # phase inverters.
        self.assertGreater(expected_3, 0)

        # Loop and count.
        actual_3 = 0
        for i in self.inverters.values():
            if isinstance(i, dict):
                actual_3 += 1
            elif isinstance(i, equipment.InverterSinglePhase):
                pass
            else:
                raise ValueError('huh?')

        self.assertEqual(actual_3, expected_3)


if __name__ == '__main__':
    unittest.main()
