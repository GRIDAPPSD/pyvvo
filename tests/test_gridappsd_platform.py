# Standard library
import os
import unittest
from unittest.mock import patch, MagicMock, Mock, create_autospec
from datetime import datetime
import logging
import re

# PyVVO + GridAPPS-D
from pyvvo import gridappsd_platform, utils
from gridappsd import GridAPPSD, topics, simulation
from pyvvo.timeseries import parse_weather
import tests.data_files as _df
from tests.models import IEEE_13

# Third-party
from stomp.exception import ConnectFailedException
import simplejson as json
import pandas as pd

# Attempt to connect to the GridAPPS-D platform.
try:
    gridappsd_platform.get_gad_object()
except ConnectFailedException:
    PLATFORM_RUNNING = False
else:
    PLATFORM_RUNNING = True

NO_CONNECTION = 'Could not connect to the GridAPPS-D platform.'


class GetGADAddressTestCase(unittest.TestCase):
    """Test get_gad_address."""

    @patch.dict(os.environ, {'platform': '1'})
    def test_get_gad_address_in_platform(self):
        self.assertEqual(('gridappsd', 61613),
                         gridappsd_platform.get_gad_address())

    @patch.dict(os.environ, {'host_ip': '192.168.0.10',
                             'GRIDAPPSD_PORT': '61613', 'platform': '0'})
    def test_get_gad_address_outside_platform(self):
        self.assertEqual(('192.168.0.10', '61613'),
                         gridappsd_platform.get_gad_address())

    def test_get_gad_address_calls_get_platform_env_var(self):
        with patch('pyvvo.gridappsd_platform.get_platform_env_var',
                   return_value='1') as mock:
            address = gridappsd_platform.get_gad_address()
            mock.assert_called_once()
            mock.assert_called_with()
            self.assertEqual(('gridappsd', 61613), address)


@unittest.skipUnless(PLATFORM_RUNNING, reason=NO_CONNECTION)
class GetGADObjectTestCase(unittest.TestCase):
    """Test get_gad_object. NOTE: Requires the GridAPPS-D platform to be
    up and running. Also, the 'platform' environment variable must be
    set.
    """

    @classmethod
    def setUpClass(cls):
        """Connect to the platform."""
        cls.gad = gridappsd_platform.get_gad_object()

    def test_get_gad_object_is_gad_object(self):
        self.assertIsInstance(self.gad, GridAPPSD)

    def test_get_gad_object_query_model_info(self):
        """Test that the query_model_info function acts as expected."""
        # Get model information from the platform.
        actual_info = self.gad.query_model_info()

        # Uncomment to update.
        # with open(MODEL_INFO, 'w') as f:
        #     json.dump(actual_info, f)

        # Load the expected result.
        with open(_df.MODEL_INFO, 'r') as f:
            expected_info = json.load(f)

        # The queries are going to have different id's, so remove those.
        actual_info.pop('id')
        expected_info.pop('id')

        self.assertDictEqual(actual_info, expected_info)


class GetPlatformEnvVarTestCase(unittest.TestCase):
    """Test the get_platform_env_var function."""

    @patch.dict(os.environ, {'PATH': '/some/path'})
    def test_get_platform_env_var_no_platform_var(self):
        os.environ.pop('platform')
        self.assertRaises(KeyError, gridappsd_platform.get_platform_env_var)

    @patch.dict(os.environ, {'platform': 'not 1 or 0'})
    def test_get_platform_env_var_bad_value(self):
        self.assertRaises(ValueError, gridappsd_platform.get_platform_env_var)

    @patch.dict(os.environ, {'platform': '1'})
    def test_get_platform_env_var_1(self):
        self.assertEqual('1', gridappsd_platform.get_platform_env_var())

    @patch.dict(os.environ, {'platform': '0'})
    def test_get_platform_env_var_0(self):
        self.assertEqual('0', gridappsd_platform.get_platform_env_var())


# Create some dummy functions for the SimOutRouter.
mock_fn_1 = Mock(return_value='yay')
mock_fn_2 = Mock(return_value=42)
mock_fn_3 = Mock(return_value='bleh')

# Create a dummy platform manager.
mock_platform_manager = create_autospec(gridappsd_platform.PlatformManager)
mock_platform_manager.gad = Mock()
mock_platform_manager.gad.subscribe = Mock()


class DummyClass:
    """Dummy class for testing the SimOutRouter."""
    def __init__(self):
        pass


class SimOutRouterTestCase(unittest.TestCase):
    """Tests for SimOutRouter class."""

    @classmethod
    def setUpClass(cls):
        """Load the file we'll be working with, initialize
        SimOutRouter."""
        with open(_df.MEASUREMENTS_13, 'r') as f:
            cls.meas = json.load(f)

        with open(_df.HEADER_13, 'r') as f:
            cls.header = json.load(f)

        # For convenience, get a reference to the measurements.
        meas = cls.meas['message']['measurements']

        # Create list of list of mrids.
        cls.all_mrids = list(meas.keys())

        # Hard code some indices for grabbing measurements.
        # Note that within each sub-list the indices ARE sorted to
        # make the testing comparison easier.
        cls.indices = [[10, 112, 114], [16, 102], [0, 1, 42]]

        # Grab mrids.
        cls.mrids = [[meas[cls.all_mrids[i]]['measurement_mrid'] for i in
                      sub_i]
                     for sub_i in cls.indices]

        # Create dummy class.
        cls.dummy_class = DummyClass()
        cls.dummy_class.my_func = MagicMock(return_value=3,
                                            side_effect=print('mocked func.'))

        # Hard-code list input for the SimOutRouter.
        cls.fn_mrid_list = [{'functions': mock_fn_1, 'mrids': cls.mrids[0]},
                            {'functions': mock_fn_2, 'mrids': cls.mrids[1],
                             'kwargs': {'param1': 'asdf'}},
                            {'functions': [cls.dummy_class.my_func, mock_fn_3],
                             'mrids': cls.mrids[2]}
                            ]

        cls.router = \
            gridappsd_platform.SimOutRouter(
                platform_manager=mock_platform_manager, sim_id='1234',
                fn_mrid_list=cls.fn_mrid_list)

    def test_filter_output_by_mrid_bad_message_type(self):
        self.assertRaises(TypeError, self.router._filter_output_by_mrid,
                          message='message', mrids=[])

    def test_filter_output_by_mrid_expected_return(self):
        # Gather expected return. This has gotten a little gnarly and
        # unreadable, but so it goes sometimes. Gotta keep moving.
        meas = self.meas['message']['measurements']
        expected = [[meas[self.mrids[i][j]] for j in range(len(sub_j))]
                    for i, sub_j in enumerate(self.indices)]

        actual = self.router._filter_output_by_mrid(message=self.meas)

        self.assertEqual(expected, actual)

        # Ensure our list of mrids didn't change.
        self.assertEqual(len(expected), len(self.router.mrids))

    def test_filter_output_by_mrid_not_all_mrids_in_meas(self):
        # Create list of mrids which is mostly bogus.
        mrids = [['abcdefg'], ['hijklmnop'], ['qrstuv', self.mrids[0][0]]]
        # Replace the mrids in self.router with our mostly bogus list.
        with patch.object(self.router, attribute='mrids', new=mrids):
            # Ensure we get a value error, use regex to ensure our
            # arithmetic is correct.
            with self.assertRaisesRegex(ValueError,
                                        'Expected measurement MRID abcdefg'):
                self.router._filter_output_by_mrid(message=self.meas)

    def test_subscribed(self):
        """Ensure that we've subscribed to the topic."""
        mock_platform_manager.gad.subscribe.assert_called_once_with(
            topic=self.router.output_topic, callback=self.router._on_message
        )

    def test_on_message_calls_filter_output_by_mrid(self):
        """Call on_message."""
        with patch.object(self.router,
                          attribute='_filter_output_by_mrid') as m:
            _ = self.router._on_message(header=self.header,
                                        message=self.meas)

        # Ensure _filter_output_by_mrid was called.
        m.assert_called_once_with(message=self.meas)

    def test_on_message_calls_methods(self):
        """Ensure that all our mock functions get called."""
        # Patch _filter_output_by_mrid as we don't need that to
        # actually be called.
        with patch.object(self.router,
                          attribute='_filter_output_by_mrid',
                          return_value=[[0], [1], [2]]) as m2:
            # Call the _on_message method.
            _ = self.router._on_message(header=self.header,
                                        message=self.meas)

        # Grab the time.
        t = utils.simulation_output_timestamp_to_dt(
            int(self.meas['message']['timestamp']))

        # Ensure each method was called appropriately.
        for idx, mock_func in enumerate(self.router.functions):
            try:
                # Simple function case.
                if self.router.kwargs[idx] is not None:
                    kwargs = self.router.kwargs[idx]
                else:
                    kwargs = {}
                mock_func.assert_called_once_with(m2.return_value[idx],
                                                  sim_dt=t, **kwargs)
            except AttributeError:
                # List of functions case.
                for f in self.router.functions[idx]:
                    f.assert_called_once_with(m2.return_value[idx], sim_dt=t)

    def test_lock(self):
        """Ensure the lock (_lock property) is working as expected"""
        # Acquire the lock
        acquired = self.router._lock.acquire(timeout=0.01)
        self.assertTrue(acquired)

        with patch('pyvvo.utils.LOCK_TIMEOUT', 0.01):
            self.assertRaises(utils.LockTimeoutError,
                              self.router.add_funcs_and_mrids, 'stuff')
            self.assertRaises(utils.LockTimeoutError,
                              self.router._on_message, 'stuff', 'thing')

        # Release the lock and call the functions again.
        self.router._lock.release()
        with patch('pyvvo.utils.LOCK_TIMEOUT', 0.01):
            # Run the function. Expect a type error since we'll just
            # pass a string.
            with self.assertRaises(TypeError):
                self.router.add_funcs_and_mrids('stuff')

            # After the call, we should be able to acquire the lock.
            acquired = self.router._lock.acquire(timeout=0.01)
            self.assertTrue(acquired)
            self.router._lock.release()

            # Run the function. Expect a type error since we'll just
            # pass a string.
            with self.assertRaises(TypeError):
                self.router._on_message('stuff', 'thing')

            # After the call, we should be able to acquire the lock.
            acquired = self.router._lock.acquire(timeout=0.01)
            self.assertTrue(acquired)
            self.router._lock.release()


@unittest.skipUnless(PLATFORM_RUNNING, reason=NO_CONNECTION)
class PlatformManagerTestCase(unittest.TestCase):
    """Test the PlatformManager. Requires the GridAPPS-D platform to
    be up and running, and the 'platform' environment variable to be
    set.
    """

    @classmethod
    def setUpClass(cls):
        """Get a PlatformManager."""
        cls.platform = gridappsd_platform.PlatformManager()

    def test_platform_manager_gad(self):
        self.assertIsInstance(self.platform.gad, GridAPPSD)

    def test_platform_manager_get_glm_patched(self):
        """Check return for get_glm, but patch the platform call."""
        # TODO: update when platform is fixed.
        m = '{"data":"model123456","responseComplete":true,"id": "bigID"}'
        platform_return = {
            'error': 'Invalid json returned',
            'header': {'stuff': 'I do not need'}, 'message': m}

        with patch.object(self.platform.gad, 'get_response',
                          return_value=platform_return) as mock:
            glm = self.platform.get_glm(model_id="someID")
            mock.assert_called_once()

        self.assertEqual(glm, '"model123456"')

    def test_platform_manager_get_glm_13_bus(self):
        """Check return for get_glm, actually calling the platform."""
        # IEEE 13 bus model.
        model_id = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
        glm = self.platform.get_glm(model_id=model_id)

        # Get expected.
        with open(IEEE_13, 'r') as f:
            expected = f.read()

        self.assertEqual(glm, expected)

    # @unittest.skip("This takes too long, and also the model converter "
    #                "is poorly written and has randomness in it.")
    # def test_platform_manager_get_glm_8500_node(self):
    #     """Check return for get_glm, actually calling the platform."""
    #     # IEEE 8500 node model.
    #     # TODO: May want to update this with the new modified 8500
    #     #   model once it has stabilized.
    #     model_id = "_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3"
    #     glm = self.platform.get_glm(model_id=model_id)
    #
    #     # Uncomment to recreate the expected return.
    #     # with open(IEEE_8500, 'w') as f:
    #     #     f.write(glm)
    #
    #     # Get expected.
    #     with open(IEEE_8500, 'r') as f:
    #         expected = f.read()
    #
    #     self.assertEqual(glm, expected)

    def test_platform_manager_get_weather_bad_start_time(self):
        self.assertRaises(TypeError, self.platform.get_weather,
                          start_time='2013-01-01 00:00:00',
                          end_time=datetime(2013, 1, 1, 0, 15))

    def test_platform_manager_get_weather_bad_end_time(self):
        self.assertRaises(TypeError, self.platform.get_weather,
                          end_time='2013-01-01 00:15:00',
                          start_time=datetime(2013, 1, 1, 0))

    def test_platform_manager_get_weather_no_data(self):
        """Try to get weather data one minute before it actually begins
        """
        # NOTE: data starts 2013-01-01 00:00:00 UTC.
        self.assertRaises(gridappsd_platform.QueryReturnEmptyError,
                          self.platform.get_weather,
                          start_time=datetime(2012, 12, 31, 23, 59),
                          end_time=datetime(2012, 12, 31, 23, 59))

    def test_platform_manager_get_weather_earliest_possible(self):
        """Get first piece of weather data available."""
        # NOTE: data starts 2013-01-01 00:00:00 UTC.
        data = self.platform.get_weather(
            start_time=datetime(2013, 1, 1, 0, 0),
            end_time=datetime(2013, 1, 1, 0, 0)
        )
        self.assertIsInstance(data, pd.DataFrame)

    def test_platform_manager_query_weather_simple(self):
        # Retrieve weather data, ensure it matches expected.
        with open(_df.WEATHER_FOR_SENSOR_DATA_9500_JSON, 'r') as f:
            expected = json.load(f)

        # Remove the 'id' key.
        del expected['id']

        # Query the platform.
        actual = self.platform._query_weather(
            start_time=_df.SENSOR_MEASUREMENT_TIME_START,
            end_time=_df.SENSOR_MEASUREMENT_TIME_END)

        # Remove 'id.'
        del actual['id']

        self.assertDictEqual(actual, expected)

    def test_platform_manager_get_weather_valid(self):
        """Ensure parse_weather is called with the expected input."""
        with patch('pyvvo.timeseries.parse_weather',
                   side_effect=parse_weather) as mock:
            _ = self.platform.get_weather(
                start_time=_df.SENSOR_MEASUREMENT_TIME_START,
                end_time=_df.SENSOR_MEASUREMENT_TIME_END)

        # Ensure parse_weather is called.
        mock.assert_called_once()

        # Get the actual data from the platform.
        actual = mock.call_args[0][0]

        # Uncomment to re-generated expected.
        # with open('weather_simple.json', 'w') as f:
        #     json.dump(actual, f)

        with open(_df.WEATHER_FOR_SENSOR_DATA_9500_JSON, 'r') as f:
            expected = json.load(f)

        # Pop the IDs from actual and expected.
        actual.pop('id')
        expected.pop('id')

        # Ensure the data provided by the platform matches.
        self.assertDictEqual(actual, expected)

    def test_platform_manager_send_command_bad_type_1(self):
        self.assertRaises(TypeError, self.platform.send_command,
                          object_ids='abcd', attributes=['a'],
                          forward_values=['b'], reverse_values=['c'],
                          sim_id='1234')

    def test_platform_manager_send_command_bad_type_2(self):
        self.assertRaises(TypeError, self.platform.send_command,
                          object_ids=[42], attributes=12,
                          forward_values=['b'], reverse_values=['c'],
                          sim_id='1234')

    def test_platform_manager_send_command_bad_type_3(self):
        self.assertRaises(TypeError, self.platform.send_command,
                          object_ids=[42], attributes=[12],
                          forward_values='b', reverse_values=['c'],
                          sim_id='1234')

    def test_platform_manager_send_command_bad_type_4(self):
        self.assertRaises(TypeError, self.platform.send_command,
                          object_ids=[42], attributes=[12],
                          forward_values=['b'], reverse_values='c',
                          sim_id='1234')

    def test_platform_manager_send_command_bad_length_1(self):
        self.assertRaises(ValueError, self.platform.send_command,
                          object_ids=[42, 420], attributes=[12],
                          forward_values=['b'], reverse_values=['c'],
                          sim_id='1234')

    def test_platform_manager_send_command_bad_length_2(self):
        self.assertRaises(ValueError, self.platform.send_command,
                          object_ids=[42], attributes=[12, 'v'],
                          forward_values=['b'], reverse_values=['c'],
                          sim_id='1234')

    def test_platform_manager_send_command_bad_length_3(self):
        self.assertRaises(ValueError, self.platform.send_command,
                          object_ids=[42], attributes=[12],
                          forward_values=['b', 12], reverse_values=['c'],
                          sim_id='1234')

    def test_platform_manager_send_command_bad_length_4(self):
        self.assertRaises(ValueError, self.platform.send_command,
                          object_ids=[42], attributes=[12],
                          forward_values=['b'], reverse_values=['c', (1,)],
                          sim_id='1234')

    def test_platform_manager_send_command_sim_id_none(self):
        self.assertRaises(ValueError, self.platform.send_command,
                          object_ids=[42], attributes=[12],
                          forward_values=['b'], reverse_values=['c'],
                          sim_id=None)

    def test_platform_manager_send_command_valid(self):
        with patch.object(self.platform.gad, 'send',
                          return_value=None) as mock:
            with self.assertLogs(logger=self.platform.log, level="INFO"):
                msg = self.platform.send_command(object_ids=[42],
                                                 attributes=[12],
                                                 forward_values=['b'],
                                                 reverse_values=['c'],
                                                 sim_id='123')

        # Hard-code expected message for regression testing.
        expected = \
            {'command': 'update',
             'input': {'simulation_id': '123',
                       'message': {
                           'timestamp': 1559334676,
                           'difference_mrid':
                               'b0b52a84-69c9-45df-b843-779bcaba7233',
                           'reverse_differences': [
                               {'object': 42,
                                'attribute': 12,
                                'value': 'c'}],
                           'forward_differences': [
                               {'object': 42,
                                'attribute': 12,
                                'value': 'b'}]}}}

        # We need to alter timestamps and MRID's.
        msg['input']['message']['timestamp'] = 10
        expected['input']['message']['timestamp'] = 10

        msg['input']['message']['difference_mrid'] = 'abcdefghijklmnop'
        expected['input']['message']['difference_mrid'] = 'abcdefghijklmnop'

        self.assertDictEqual(expected, msg)

        # Ensure we called self.gad.send.
        mock.assert_called_once()

    def test_platform_manager_send_command_empty(self):
        """If empty lists get sent in, this should be a no-op."""
        with self.assertLogs(logger=self.platform.log, level='INFO'):
            out = self.platform.send_command(object_ids=[],
                                             attributes=[], forward_values=[],
                                             reverse_values=[])

        self.assertIsNone(out)

    def test__query_simulation_output_bad_sim_id_type(self):
        with self.assertRaisesRegex(TypeError, 'simulation_id must be a str'):
            self.platform._query_simulation_output(simulation_id=1234)

    def test__query_simulation_output_bad_query_meas_type(self):
        with self.assertRaisesRegex(TypeError, 'query_measurement must be a '):
            self.platform._query_simulation_output(simulation_id='1234',
                                                   query_measurement={'a'})

    def test__query_simulation_output_bad_starttime_type(self):
        with self.assertRaisesRegex(TypeError, 'starttime must be datetime'):
            self.platform._query_simulation_output(simulation_id='1234',
                                                   starttime='noon')

    def test__query_simulation_output_bad_endtime_type(self):
        with self.assertRaisesRegex(TypeError, 'endtime must be datetime'):
            self.platform._query_simulation_output(simulation_id='1234',
                                                   endtime='2019-07-21')

    def test__query_simulation_output_bad_meas_mrid_type(self):
        with self.assertRaisesRegex(TypeError, 'measurement_mrid must be a '):
            self.platform._query_simulation_output(simulation_id='1234',
                                                   measurement_mrid=[1, 2, 3])

    def test__query_simulation_output_bad_query_meas_value(self):
        with self.assertRaisesRegex(ValueError, "query_measurement must be '"):
            self.platform._query_simulation_output(simulation_id='1234',
                                                   query_measurement='bad')

    def test__query_simulation_output_no_filters_and_defaults(self):
        with patch.object(self.platform.gad, 'get_response',
                          return_value=10) as p:
            out = self.platform._query_simulation_output(simulation_id='7')

        p.assert_called_once()
        p.assert_called_with(topic=topics.TIMESERIES,
                             message={"queryMeasurement": "simulation",
                                      "queryFilter":
                                          {"simulation_id": "7",
                                           'hasSimulationMessageType':
                                               'OUTPUT'},
                                      "responseFormat": "JSON"}, timeout=30)

        self.assertEqual(10, out)

    def test__query_simulation_output_all_filters(self):
        s = datetime(2019, 1, 1, 0)
        e = datetime(2019, 1, 1, 1)
        m = 'mrid'
        q = 'gridappsd-sensor-simulator'

        with patch.object(self.platform.gad, 'get_response',
                          return_value=10) as p1:
            with patch('pyvvo.utils.dt_to_s_from_epoch', return_value='7') \
                    as p2:
                out = self.platform._query_simulation_output(
                    simulation_id='95', query_measurement=q,
                    starttime=s, endtime=e, measurement_mrid=m
                )

        p1.assert_called_once()

        self.assertEqual(2, p2.call_count)
        self.assertEqual(10, out)

        p1.assert_called_with(topic=topics.TIMESERIES,
                              message={"queryMeasurement": q,
                                       "queryFilter":
                                           {"simulation_id": "95",
                                            "starttime": '7',
                                            'endtime': '7',
                                            'measurement_mrid': m},
                                       "responseFormat": "JSON"}, timeout=30)

    def test_platform_manager_get_simulation_output(self):
        """Ensure inputs are passed along directly, and output is
        parsed.
        """
        with patch.object(self.platform, '_query_simulation_output',
                          return_value=42) as p1:
            with patch('pyvvo.timeseries.parse_timeseries',
                       return_value=21) as p2:
                out = self.platform.get_simulation_output(
                    simulation_id=7, query_measurement=3,
                    starttime=10, endtime=16, measurement_mrid=65
                )

        p1.assert_called_once()
        p2.assert_called_once()

        p1.assert_called_with(simulation_id=7, query_measurement=3,
                              starttime=10, endtime=16, measurement_mrid=65)
        p2.assert_called_with(42)

        self.assertEqual(21, out)

    def test_run_simulation_actually_run(self):
        """Test run_simulation, and legitimately run a simulation. Note
        this is more of an integration test, and also tests
        _update_sim_complete and wait_for_simulation.
        """
        # Get a fresh manager.
        p = gridappsd_platform.PlatformManager()

        # Use the smallest model we've got.
        feeder_id = _df.FEEDER_MRID_13

        # sim and sim_complete should be None.
        self.assertIsNone(p.sim_complete)
        self.assertIsNone(p.sim)

        # Run the simulation.
        p.run_simulation(feeder_id=feeder_id,
                         start_time=datetime(2013, 1, 1, 0, 0, 0),
                         duration=5, realtime=False, random_zip=False,
                         houses=False)

        # While the simulation is still running, sim_complete should be
        # False. NOTE: This builds in the assumption that this Python
        # code will take less time than the platform does to generate
        # and run the model. Seems reasonable.
        self.assertFalse(p.sim_complete)

        # sim should now be a gridappsd.simulation.Simulation object.
        self.assertIsInstance(p.sim, simulation.Simulation)

        # Wait for the simulation to complete.
        with utils.time_limit(10):
            p.wait_for_simulation()

        # Now, sim_complete should be None.
        self.assertIsNone(p.sim_complete)

    def test_send_command_no_sim_id(self):
        """Ensure things properly blow up if there's no simulation ID to
        be found when trying to send a command.
        """
        # Get a fresh manager.
        p = gridappsd_platform.PlatformManager()
        with self.assertRaisesRegex(ValueError, 'In order to send a command,'):
            p.send_command(object_ids=['a'], attributes=['b'],
                           forward_values=[1], reverse_values=[2])


class SimulationClockTestCase(unittest.TestCase):
    """Test the SimulationClock class."""
    @classmethod
    def setUpClass(cls) -> None:
        # Read the simulation log file.
        with open(_df.SIMULATION_LOG, 'r') as f:
            cls.sim_log_list = json.load(f)

        # Extract the simulation ID from the first message.
        cls.sim_id = cls.sim_log_list[0]['message']['processId']

        # Hard code the topic. This will also help give us a regression
        # test against gridappsd-python/topics
        cls.topic = f'/topic/goss.gridappsd.simulation.log.{cls.sim_id}'
        # The SIMULATION_LOG data file maps to the start time of the 13
        # bus measurement generation simulation.
        cls.sim_start_ts = int(utils.dt_to_s_from_epoch(_df.MEAS_13_START))

    def setUp(self) -> None:
        # Initialize the SimulationClock with a Mock.
        self.gad_mock = Mock()
        # Create a fresh clock for each test.
        self.clock = gridappsd_platform.SimulationClock(
            gad=self.gad_mock, sim_id=self.sim_id,
            sim_start_ts=self.sim_start_ts)

    def test_init(self):
        """Test attributes after initialization."""
        self.assertIsInstance(self.clock.log, logging.Logger)
        self.assertEqual(self.sim_start_ts, self.clock.sim_start_ts)
        self.assertIsNone(self.clock.sim_time)
        self.assertIsNone(self.clock.time_step)
        self.assertIsNone(self.clock.msg_time)
        # We'll test the regex elsewhere.
        # Ensure the subscribe method was called.
        self.assertEqual(len(self.gad_mock.method_calls), 1)
        self.assertEqual(self.gad_mock.method_calls[0][0], 'subscribe')
        self.assertDictEqual(self.gad_mock.method_calls[0][2],
                             {'topic': self.topic,
                              'callback': self.clock._on_message})

    def test_regex(self):
        """Ensure the compiled regex for the clock works correctly."""
        self.assertEqual(
            '942',
            self.clock.regexp.match('incrementing to 942').group(1))

    def test_final_time(self):
        """Run all the messages and headers through the clock, ensure
        the final time is correct.
        """
        for d in self.sim_log_list:
            self.clock._on_message(headers=d['header'], message=d['message'])

        self.assertEqual(_df.MEAS_13_DURATION, self.clock.time_step)
        self.assertEqual(self.sim_start_ts + _df.MEAS_13_DURATION,
                         self.clock.sim_time)

        # Do some fragile hard-coding to get the last time. By looking
        # at the log messages, the last time step increment is the 2nd
        # to last. So, grab the time from that.
        last_msg = self.sim_log_list[-2]['message']
        self.assertEqual(self.clock.msg_time, last_msg['timestamp'])

    def test_bad_message(self):
        """Ensure an error is logged if given a bad message."""
        with self.assertLogs(logger=self.clock.log, level='ERROR'):
            self.clock._on_message(headers={'bleh': 2},
                                   message={'weird': 'stuff', 'bro': 'chacha'})

    def test_message_from_non_fncs_source(self):
        """Ensure clock ignores messages not from the fncs_goss_bridge.
        """
        with self.assertLogs(logger=self.clock.log, level='DEBUG') as cm:
            self.clock._on_message(
                headers={}, message={'source': 'not_fncs_goss_bridge.py',
                                     'timestamp': 123, 'logMessage': 'eh'})

        self.assertEqual(1, len(cm.records))
        # noinspection PyUnresolvedReferences
        self.assertEqual('Ignoring message from not_fncs_goss_bridge.py.',
                         cm.records[0].message)

    def test_message_does_not_match_regex(self):
        with self.assertLogs(logger=self.clock.log, level='DEBUG') as cm:
            self.clock._on_message(
                headers={}, message={'source': 'fncs_goss_bridge.py',
                                     'timestamp': 123, 'logMessage': 'eh'})

        self.assertEqual(1, len(cm.records))
        # noinspection PyUnresolvedReferences
        self.assertEqual('Ignoring message "eh."',
                         cm.records[0].message)

    def test_single_increment(self):
        """Send in a valid message and check associated attributes."""
        with self.assertLogs(logger=self.clock.log, level='DEBUG') as cm:
            self.clock._on_message(
                headers={}, message={'source': 'fncs_goss_bridge.py',
                                     'timestamp': 123,
                                     'logMessage': 'incrementing to 42'})

        self.assertEqual(1, len(cm.records))
        sim_time = self.sim_start_ts + 42

        self.assertEqual(self.clock.sim_time, sim_time)
        self.assertEqual(self.clock.msg_time, 123)
        self.assertEqual(self.clock.time_step, 42)

        # noinspection PyUnresolvedReferences
        self.assertEqual(
            f'Updated sim_time to {sim_time} and msg_time to 123.',
            cm.records[0].message)


if __name__ == '__main__':
    unittest.main()
