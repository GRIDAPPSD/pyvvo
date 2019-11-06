import unittest
from unittest.mock import patch, Mock
from random import randint, choice
from copy import deepcopy
import numpy as np
import pandas as pd
import simplejson as json
from datetime import datetime
import threading
import queue
import time

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

        cls.reg_df = _df.read_pickle(_df.REGULATORS_9500)

    # noinspection PyPep8Naming
    def setUp(self):
        # Gotta be careful with these mutable types... Get fresh
        # instances each time. It won't be that slow, I promise.
        self.reg_dict = \
            equipment.initialize_regulators(self.reg_df)
        self.reg_mgr = \
            equipment.EquipmentManager(
                eq_dict=self.reg_dict, eq_meas=self.reg_meas,
                meas_mrid_col=REG_MEAS_MEAS_MRID_COL,
                eq_mrid_col=REG_MEAS_REG_MRID_COL
            )

    @staticmethod
    def random_update(reg_in, list_in):
        """Helper to randomly update tap steps. Use this with the
        loop_helper.
        """
        new_step = reg_in.state

        while new_step == reg_in.state:
            new_step = randint(reg_in.low_step, reg_in.high_step)

        reg_in.state = new_step
        list_in.append(new_step)

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
            # noinspection PyTypeChecker
            self.reg_mgr.update_state(msg='hello there', sim_dt=self.sim_dt)

    def test_update_state_bad_type_2(self):
        with self.assertRaisesRegex(TypeError, 'string indices must be'):
            # noinspection PyTypeChecker
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

        # Initialize list to hold regulator positions.
        forward_vals = []

        # Randomly update all regulators.
        equipment.loop_helper(eq_dict=reg_dict_forward,
                              func=self.random_update,
                              list_in=forward_vals)

        # Get reverse values.
        reverse_vals = []

        def get_state(reg_in):
            reverse_vals.append(reg_in.state)

        # Get reverse values.
        equipment.loop_helper(eq_dict=self.reg_dict, func=get_state)

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

        # Ensure the given MRID's all correspond to tap changers.
        tap_mrids = self.reg_df['tap_changer_mrid']
        self.assertTrue(tap_mrids.isin(out['object_ids']).values.all())

        # Ensure the lengths are equal to all our single phases.
        # I'm just going to hard-code the fact that the 9500 node model
        # has 6 3-phase regs.
        for v in out.values():
            self.assertIsInstance(v, list)
            self.assertEqual(len(v), 18)

        # All our regulators should have had their expected_state
        # updated.
        expected_state = []

        def get_expected_state(reg_in):
            expected_state.append(reg_in.expected_state)

        equipment.loop_helper(eq_dict=self.reg_dict, func=get_expected_state)

        self.assertListEqual(forward_vals, expected_state)

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
        # noinspection PyTypeChecker
        self.reg_mgr.update_state(msg=msg, sim_dt='high noon')

        # Ensure the equipment actually has int64 states.
        equipment.loop_helper(self.reg_dict, assert_64)

        # Similarly, update all the equipment in the dictionary.

        # Helper for casting to int32.
        def to_int32(eq):
            eq.state = np.int32(eq.state)

        # Get a copy of the regulators and randomly change their states.
        reg_dict_copy = deepcopy(self.reg_dict)

        equipment.loop_helper(eq_dict=reg_dict_copy, func=self.random_update,
                              list_in=[])

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
                # noinspection PyTypeChecker
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

    def test_expected_not_equal_to_actual(self):
        """Test helper function _expected_not_equal_to_actual."""
        eq = Mock(spec=equipment.RegulatorSinglePhase)
        eq.expected_state = 7
        eq.state = None

        # Ensure with a state of None we get a False return.
        self.assertFalse(equipment._expected_not_equal_to_actual(eq))

        # Flop 'em, and should still get None.
        eq.expected_state = None
        eq.state = 5

        self.assertFalse(equipment._expected_not_equal_to_actual(eq))

        # With different settings, we should get True.
        eq.expected_state = 6

        self.assertTrue(equipment._expected_not_equal_to_actual(eq))

        # With the same settings, we should get False.
        eq.state = 6
        self.assertFalse(equipment._expected_not_equal_to_actual(eq))

    def test_wait_and_get_delta(self):
        """Test _wait_and_get_delta."""
        # Create times which are 60 seconds apart.
        old_t = datetime(2019, 11, 4, 9, 0)
        self.reg_mgr.last_time = datetime(2019, 11, 4, 9, 1)

        # We should get a timeout error if the event isn't toggled.
        with self.assertRaisesRegex(TimeoutError, 'The update_state method '):
            self.reg_mgr._wait_and_get_delta(old_t=old_t, timeout=0.01)

        # Now, spin up a thread to get the time delta.
        def get_delta(mgr, dt, q):
            delta = mgr._wait_and_get_delta(old_t=dt, timeout=1)
            q.put(delta)

        mq = queue.Queue()

        t = threading.Thread(target=get_delta,
                             args=(self.reg_mgr, old_t, mq))
        t.start()

        self.reg_mgr._toggle_update_state_event()

        delta_out = mq.get(timeout=1)

        # Hard code 60 second difference.
        self.assertEqual(delta_out, 60)

    def test_verify_command(self):
        """Test the verify_command method."""
        # We should get a ValueError if last_time is not set (which at
        # this point, it shouldn't be).
        with self.assertRaisesRegex(ValueError, 'verify_command has been cal'):
            self.reg_mgr.verify_command(wait_duration=0.1, timeout=0.1)

        # Grab the first piece of equipment and put into a dictionary.
        mrid = list(self.reg_mgr.eq_dict.keys())[0]
        eq_or_dict = self.reg_mgr.eq_dict[mrid]
        if isinstance(eq_or_dict, dict):
            phase = list(self.reg_mgr.eq_dict[mrid].keys())[0]
            eq = self.reg_mgr.eq_dict[mrid][phase]
            single_eq_dict = {mrid: {phase: eq}}
        else:
            eq = eq_or_dict
            single_eq_dict = {mrid: eq_or_dict}

        # Set the last_time, and get a time 60 seconds later.
        dt = datetime(2019, 11, 4, 9, 15)
        dt2 = datetime(2019, 11, 4, 9, 16)
        self.reg_mgr.last_time = dt

        # We should get a timeout error if the update_state_event never
        # gets toggled.
        with self.assertRaisesRegex(TimeoutError, 'The update_state method'):
            self.reg_mgr.verify_command(wait_duration=0.01, timeout=0.01)

        # If no equipment has a mismatch between their expected_state
        # and their state (excluding Nones), we should get a None
        # return.
        mq = queue.Queue()

        def put_result_in_queue(mgr, q, wait_duration, timeout):
            result = mgr.verify_command(wait_duration=wait_duration,
                                        timeout=timeout)
            q.put(result)

        t = threading.Thread(target=put_result_in_queue,
                             args=(self.reg_mgr, mq, 60, 1))
        t.start()

        # Wait a tiny bit for the thread to kick off properly.
        time.sleep(0.05)

        # Update the last_time and toggle the event to simulate a
        # message coming in.
        self.reg_mgr.last_time = dt2
        self.reg_mgr._toggle_update_state_event()

        # Grab element out of the queue.
        output = mq.get(timeout=1)

        # No equipment has a mismatch between expected_state and state,
        # so this result should be None.
        self.assertIsNone(output)

        # Reset the time.
        self.reg_mgr.last_time = dt

        # Tweak the expected_state for our equipment to ensure it's
        # different from the state.
        eq.expected_state = eq.state + 1

        # Get a time which is thirty seconds after the first. Hard
        # coding for the win.
        dt3 = datetime(2019, 11, 4, 9, 15, 30)

        # Fire up another thread to run verify_command again.
        mq = queue.Queue()
        t = threading.Thread(target=put_result_in_queue,
                             args=(self.reg_mgr, mq, 60, 1))
        t.start()
        time.sleep(0.05)

        # Update time and toggle event. However, note this time is
        # less than the wait_duration.
        self.reg_mgr.last_time = dt3
        self.reg_mgr._toggle_update_state_event()

        # We shouldn't have gotten a return value yet.
        with self.assertRaises(queue.Empty):
            mq.get(timeout=0.1)

        # Update time and toggle event, but this time we should get a
        # return.
        time.sleep(0.05)
        self.reg_mgr.last_time = dt2
        self.reg_mgr._toggle_update_state_event()

        # Extract output.
        actual_dict = mq.get(timeout=1)

        # Output should match our single_eq_dict.
        self.assertDictEqual(single_eq_dict, actual_dict)

        # The equipment should be inoperable.
        def is_inoperable(eq_in):
            self.assertFalse(eq_in.operable)

        equipment.loop_helper(eq_dict=actual_dict, func=is_inoperable)


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
            # noinspection PyTypeChecker
            self.cap_mgr.update_state(msg='hello there', sim_dt=self.sim_dt)

    def test_update_state_bad_type_2(self):
        with self.assertRaisesRegex(TypeError, 'string indices must'):
            # noinspection PyTypeChecker
            self.cap_mgr.update_state(msg=['hello there'], sim_dt=self.sim_dt)

    def test_build_equipment_commands(self):
        """One stop big function which probably should be spun off into
        its own test case.

        NOTE: This test is fragile as it currently relies on how the
        looping is performed in build_equipment_commands.
        """
        cap_dict_forward = deepcopy(self.cap_mgr.eq_dict)

        forward_vals = []

        def update_state(cap):
            """Nested helper function."""
            if cap.controllable:
                new_state = choice(equipment.CapacitorSinglePhase.STATES)
                cap.state = new_state
                forward_vals.append(new_state)

        # Randomly update steps.
        equipment.loop_helper(eq_dict=cap_dict_forward, func=update_state)

        # Grab reverse values.
        reverse_vals = []

        def get_state(cap):
            if cap.controllable:
                reverse_vals.append(cap.state)

        equipment.loop_helper(eq_dict=self.cap_mgr.eq_dict, func=get_state)

        # Build equipment commands..
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

        # Ensure our expected_state matches the state of our forward
        # items.

        expected_state = []

        def get_expected_state(eq):
            if eq.controllable:
                expected_state.append(eq.expected_state)

        equipment.loop_helper(eq_dict=self.cap_mgr.eq_dict,
                              func=get_expected_state)

        self.assertListEqual(expected_state, forward_vals)

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
        """Send in an update message and ensure that state changed and
        callbacks are called.
        """
        # Add a callback.
        m = Mock()
        self.switch_mgr.add_callback(m)

        # Start by ensuring all switches start with a status of None.
        equipment.loop_helper(eq_dict=self.switch_mgr.eq_dict,
                              func=self.state_none)

        # The update_state_event should not be set.
        self.assertFalse(self.switch_mgr.update_state_event.is_set())

        # Before receiving an update message, last_time should be None.
        self.assertIsNone(self.switch_mgr.last_time)

        # Start up a thread which will flip a variable when the
        # update_state_event is set.
        event_queue = queue.Queue()

        def toggle_state_event_set(q):
            result = self.switch_mgr.update_state_event.wait(timeout=0.5)
            if result:
                q.put(True)
            else:
                q.put(False)

        t = threading.Thread(target=toggle_state_event_set,
                             args=(event_queue,))
        t.start()

        # Now that we've ensure all switches start with None status,
        # update them all.
        self.switch_mgr.update_state(self.switch_meas_msg, sim_dt=self.sim_dt)

        # Ensure the last_time attribute has been updated.
        self.assertEqual(self.switch_mgr.last_time, self.sim_dt)

        # Loop again and ensure the states are now not None and are
        # valid.
        equipment.loop_helper(eq_dict=self.switch_mgr.eq_dict,
                              func=self.state_valid)

        # The callback should have been called.
        m.assert_called_once()
        m.assert_called_with(self.sim_dt)

        # Ensure our state_event_set got toggled.
        self.assertTrue(event_queue.get(timeout=0.5))

    def test_add_and_call_callbacks(self):
        """Test add_callback and _call_callbacks."""
        # Ensure callbacks start empty.
        self.assertEqual(len(self.switch_mgr._callbacks), 0)

        # Add a callback.
        m = Mock()
        self.switch_mgr.add_callback(m)

        # Callbacks should have a length of 1.
        self.assertEqual(len(self.switch_mgr._callbacks), 1)

        # Call the callbacks.
        self.switch_mgr._call_callbacks('bananas')

        # Our mock should have been called once.
        m.assert_called_once()
        m.assert_called_with('bananas')

        # Add another callback.
        m2 = Mock()
        self.switch_mgr.add_callback(m2)
        self.assertEqual(len(self.switch_mgr._callbacks), 2)
        self.switch_mgr._call_callbacks('oranges')

        self.assertEqual(m.call_count, 2)
        m2.assert_called_once()
        m2.assert_called_with('oranges')

    def test_callback_not_called(self):
        """Ensure that for no state changes, callbacks are not called.
        """
        # Update the states.
        self.switch_mgr.update_state(self.switch_meas_msg, sim_dt=self.sim_dt)

        # Add a callback.
        m = Mock()
        self.switch_mgr.add_callback(m)

        # Update the states again with the same message. So, no states
        # should change.
        self.switch_mgr.update_state(self.switch_meas_msg, sim_dt=self.sim_dt)

        # The callback should not have been called.
        self.assertEqual(0, m.call_count)

    def test_callback_dies(self):
        """Ensure our callbacks are held as weak references that die
        when the method reference is deleted.
        """
        m = Mock()
        self.switch_mgr.add_callback(m)
        self.assertEqual(len(self.switch_mgr._callbacks), 1)

        # Delete the object and force garbage collection.
        del m
        import gc
        gc.collect()

        self.assertEqual(len(self.switch_mgr._callbacks), 0)


class EquipmentManagerBuildEquipmentCommandsInvertTestCase(unittest.TestCase):
    """Ensure build_equipment_commands acts appropriately depending on
    the equipment's INVERT_STATES_FOR_COMMANDS attribute.
    """
    def helper(self, invert):
        """Create equipment manager and equipment dictionaries."""
        # Create dictionary with a single piece of equipment.
        eq_dict = {
            'mrid1': equipment.SwitchSinglePhase(
                name='switch1', mrid='mrid1', phase='A', controllable=True,
                state=1)
        }

        # Create DataFrame with measurement information.
        eq_meas = pd.DataFrame([['mrid1', 'meas1']], columns=['eq', 'meas'])

        # Create equipment manager.
        mgr = equipment.EquipmentManager(
            eq_dict=eq_dict, eq_meas=eq_meas, meas_mrid_col='meas',
            eq_mrid_col='eq')

        # Create forward dictionary of equipment with different state.
        eq_dict_for = {
            'mrid1': equipment.SwitchSinglePhase(
                name='switch1', mrid='mrid1', phase='A', controllable=True,
                state=0)
        }

        # Build commands.
        cmd = mgr.build_equipment_commands(eq_dict_forward=eq_dict_for)

        # Check object IDs and attributes.
        self.assertEqual(1, len(cmd['object_ids']))
        self.assertEqual(cmd['object_ids'][0], 'mrid1')
        self.assertEqual(1, len(cmd['attributes']))
        self.assertEqual(cmd['attributes'][0], 'Switch.open')

        # Ensure forward and reverse values are of the correct length.
        self.assertEqual(1, len(cmd['forward_values']))
        self.assertEqual(1, len(cmd['reverse_values']))

        # Check values based on invert.
        if invert:
            self.assertEqual(cmd['forward_values'][0], 1)
            self.assertEqual(cmd['reverse_values'][0], 0)
        else:
            self.assertEqual(cmd['forward_values'][0], 0)
            self.assertEqual(cmd['reverse_values'][0], 1)

    def test_does_invert(self):
        """Ensure states are inverted when they should be."""
        self.helper(invert=True)

    def test_does_not_invert(self):
        """Ensure states are not inverted when they should not be."""
        # Patch the switches INVERT_STATES_FOR_COMMANDS attribute and
        # run the helper.
        with patch(
                'pyvvo.equipment.SwitchSinglePhase.INVERT_STATES_FOR_COMMANDS',
                False):
            self.helper(invert=False)

    def test_error_raised_when_state_not_invertible(self):
        """Ensure error raised when it should be."""
        with patch(
                'pyvvo.equipment.SwitchSinglePhase.STATES',
                (0, 1, 2)):
            with self.assertRaisesRegex(ValueError,
                                        'Equipment has a "truthy" value for '):
                self.helper(invert=True)


# noinspection PyUnresolvedReferences
class MethodsForPQEquipmentManager:
    """Class to inherit from."""

    def test_init(self):
        """Ensure initialization occurs without errors, and that we
        get the properties expected.
        """

        self.assertIsInstance(self.mgr, equipment.EquipmentManager)
        self.assertIs(self.eq_meas, self.mgr.eq_meas)
        self.assertIs(self.eq_dict, self.mgr.eq_dict)

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

        def append_and_assert(eq_in):
            self.assertIsNotNone(eq_in.state)
            current_state.append(eq_in.state)
            self.assertIsNone(eq_in.state_old)

        equipment.loop_helper(eq_dict=self.mgr.eq_dict, func=append_and_assert)

        # Add a callback.
        m = Mock()
        self.mgr.add_callback(m)

        update_count = self.mgr.update_state(msg=self.eq_meas_msg,
                                             sim_dt=self.sim_dt)

        self.assertGreater(update_count, 0)

        # Ensure callback is called.
        m.assert_called_once()
        m.assert_called_with(self.sim_dt)

        # Ensure that the current states match the former current states
        # after update. Python now ensures that looping over the same
        # dictionary twice will loop in the same order, so we'll count
        # on that fact here.
        counter = 0
        any_changed = False

        def assert_state_matches(eq_i):
            nonlocal counter
            nonlocal any_changed
            if eq_i.state_old is not None:
                any_changed = True
                self.assertEqual(current_state[counter], eq_i.state_old)
            counter += 1

        equipment.loop_helper(eq_dict=self.mgr.eq_dict,
                              func=assert_state_matches)

        self.assertTrue(any_changed)

        # Make sure our fancy looping didn't screw something up.
        for m in self.eq_meas_msg:
            rect = utils.get_complex(r=m['magnitude'], phi=m['angle'],
                                     degrees=True)
            # Swap to load convention.
            if rect.real < 0:
                rect *= -1

            s = (rect.real, rect.imag)

            eq = self.mgr.meas_eq_map[m['measurement_mrid']]

            self.assertEqual(s, eq.state)


class PQEquipmentManagerTestCase(unittest.TestCase,
                                 MethodsForPQEquipmentManager):
    """Test the PQEquipmentManager."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.eq_meas = _df.read_pickle(_df.INVERTER_MEAS_9500)
        with open(_df.INVERTER_MEAS_MSG_9500, 'r') as f:
            cls.eq_meas_msg = json.load(f)

        # Just create a bogus datetime.
        cls.sim_dt = datetime(2019, 9, 2, 17, 8)

        cls.eq_df = _df.read_pickle(_df.INVERTERS_9500)

    def setUp(self) -> None:
        self.eq_dict = equipment.initialize_inverters(self.eq_df)

        self.mgr = \
            equipment.PQEquipmentManager(eq_dict=self.eq_dict,
                                         eq_meas=self.eq_meas,
                                         meas_mrid_col='meas_mrid',
                                         eq_mrid_col='inverter_mrid')


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
             'controllable': True, 'operable': True}

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

    def test_invert_states_for_commands(self):
        self.assertFalse(self.reg.INVERT_STATES_FOR_COMMANDS)

    def test_operable(self):
        self.assertEqual(self.inputs['operable'], self.reg.operable)

    def test_expected_state(self):
        self.assertIsNone(self.reg.expected_state)


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

    def test_invert_states_for_commands(self):
        self.assertFalse(self.cap.INVERT_STATES_FOR_COMMANDS)

    def test_operable(self):
        self.assertTrue(self.cap.operable)

    def test_expected_state(self):
        self.assertIsNone(self.cap.expected_state)


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

    def test_init_with_state(self):
        switch = equipment.SwitchSinglePhase(name='sw', mrid='a', phase='C',
                                             controllable=False, state=1)

        self.assertEqual(switch.state, 1)

    def test_init_bad_state(self):
        with self.assertRaisesRegex(ValueError, 'state must be one of'):
            equipment.SwitchSinglePhase(name='sw', mrid='a', phase='C',
                                        controllable=False, state=3)

    def test_invert_states_for_commands(self):
        self.assertTrue(self.switch.INVERT_STATES_FOR_COMMANDS)

    def test_operable(self):
        self.assertTrue(self.switch.operable)

    def test_expected_state(self):
        self.assertIsNone(self.switch.expected_state)


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
        cls.s = 100
        cls.inverter = equipment.InverterSinglePhase(
            mrid='abcd', name='my_inv', phase='s2', controllable=True, p=cls.p,
            q=cls.q, rated_s=cls.s)

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
            # noinspection PyUnusedLocal,PyTypeChecker
            inv = equipment.InverterSinglePhase(mrid='blah', name='bleh',
                                                phase='C', controllable=False,
                                                p=3.2, q='a', rated_s=10)

    def test_set_invalid_state_bad_length(self):
        inv_copy = deepcopy(self.inverter)
        with self.assertRaisesRegex(ValueError, 'state must be a two element'):
            inv_copy.state = (3, 6, 9)

    def test_set_invalid_state_non_tuple(self):
        inv_copy = deepcopy(self.inverter)
        with self.assertRaisesRegex(TypeError, 'state must be a two element'):
            inv_copy.state = [13.5, -22.7]

    def test_invert_states_for_commands(self):
        self.assertFalse(self.inverter.INVERT_STATES_FOR_COMMANDS)

    def test_setting_state_above_limit_warns(self):
        inverter = equipment.InverterSinglePhase(
            mrid='abcd', name='my_inv', phase='s2', controllable=True,
            p=self.p,
            q=self.q, rated_s=self.s)

        with self.assertLogs(logger=inverter.log, level='WARN'):
            inverter.state = (self.s, self.s)

        self.assertEqual((self.p, self.q), inverter.state_old)

        self.assertIsNone(inverter.expected_state)


class SynchronousMachineSinglePhaseTestCase(unittest.TestCase):
    """Test SynchronousMachineSinglePhase."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.p = 12.5
        cls.q = 18
        cls.s = 100
        cls.machine = equipment.SynchronousMachineSinglePhase(
            mrid='abcd', name='my_inv', phase='a', controllable=True, p=cls.p,
            q=cls.q, rated_s=cls.s)

    def test_state_expected(self):
        self.assertEqual(self.machine.state, (self.p, self.q))

    def test_p_expected(self):
        self.assertEqual(self.machine.p, self.p)

    def test_q_expected(self):
        self.assertEqual(self.machine.q, self.q)

    def test_valid_state_update(self):
        inv_copy = deepcopy(self.machine)
        inv_copy.state = (9, 3.1)
        self.assertEqual(inv_copy.state, (9, 3.1))

    def test_init_invalid_state(self):
        with self.assertRaisesRegex(ValueError, 'The contents of the state'):
            # noinspection PyUnusedLocal,PyTypeChecker
            inv = equipment.SynchronousMachineSinglePhase(
                mrid='blah', name='bleh', phase='C', controllable=False,
                p=3.2, q='a', rated_s=10)

    def test_set_invalid_state_bad_length(self):
        mach_copy = deepcopy(self.machine)
        with self.assertRaisesRegex(ValueError, 'state must be a two element'):
            mach_copy.state = (3, 6, 9)

    def test_set_invalid_state_non_tuple(self):
        mach_copy = deepcopy(self.machine)
        with self.assertRaisesRegex(TypeError, 'state must be a two element'):
            mach_copy.state = [13.5, -22.7]

    def test_invert_states_for_commands(self):
        self.assertFalse(self.machine.INVERT_STATES_FOR_COMMANDS)

    def test_setting_state_above_limit_warns(self):
        machine = equipment.SynchronousMachineSinglePhase(
            mrid='abcd', name='my_inv', phase='c', controllable=True,
            p=self.p, q=self.q, rated_s=self.s)

        with self.assertLogs(logger=machine.log, level='WARN'):
            machine.state = (self.s, self.s)

        self.assertEqual((self.p, self.q), machine.state_old)

        self.assertIsNone(machine.expected_state)


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
