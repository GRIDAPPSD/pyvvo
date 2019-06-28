# Standard library
import os
import unittest
from unittest.mock import patch, MagicMock, Mock, create_autospec
from datetime import datetime

# PyVVO + GridAPPS-D
from pyvvo import gridappsd_platform
from gridappsd import GridAPPSD
from pyvvo.timeseries import parse_weather

# Third-party
from stomp.exception import ConnectFailedException
import simplejson as json

# Attempt to connect to the GridAPPS-D platform.
try:
    gridappsd_platform.get_gad_object()
except ConnectFailedException:
    PLATFORM_RUNNING = False
else:
    PLATFORM_RUNNING = True

NO_CONNECTION = 'Could not connect to the GridAPPS-D platform.'

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_INFO = os.path.join(THIS_DIR, 'query_model_info.json')
IEEE_8500 = os.path.join(THIS_DIR, 'ieee_8500.glm')
IEEE_13 = os.path.join(THIS_DIR, 'ieee_13.glm')
WEATHER = os.path.join(THIS_DIR, 'weather_simple.json')
MEASUREMENTS = os.path.join(THIS_DIR, 'simulation_measurements.json')
HEADER = os.path.join(THIS_DIR, 'simulation_measurements_header.json')


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
        with open(MODEL_INFO, 'r') as f:
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
        with open(MEASUREMENTS, 'r') as f:
            cls.meas = json.load(f)

        with open(HEADER, 'r') as f:
            cls.header = json.load(f)

        # For convenience, get a reference to the measurements.
        meas = cls.meas['message']['measurements']

        # Hard code some indices for grabbing measurements.
        # Note that within each sub-list the indices ARE sorted to
        # make the testing comparison easier.
        cls.indices = [[10, 112, 114], [16, 102], [0, 1, 42]]

        # Grab mrids.
        cls.mrids = [[meas[i]['measurement_mrid'] for i in sub_i] for sub_i in
                     cls.indices]

        # Create dummy class.
        cls.dummy_class = DummyClass()
        cls.dummy_class.my_func = MagicMock(return_value=3,
                                            side_effect=print('mocked func.'))

        # Hard-code list input for the SimOutRouter.
        cls.fn_mrid_list = [{'functions': mock_fn_1, 'mrids': cls.mrids[0]},
                            {'functions': mock_fn_2, 'mrids': cls.mrids[1]},
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
        # Gather expected return.
        meas = self.meas['message']['measurements']
        expected = [[meas[i] for i in sub_i] for sub_i in self.indices]

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
            with self.assertRaisesRegex(ValueError, '^3.*3.*'):
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

        # Ensure each method was called appropriately.
        for idx, mock_func in enumerate(self.router.functions):
            try:
                # Simple function case.
                mock_func.assert_called_once_with(m2.return_value[idx])
            except AttributeError:
                # List of functions case.
                for f in self.router.functions[idx]:
                    f.assert_called_once_with(m2.return_value[idx])


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

        # Uncomment to recreate the expected return.
        # with open(IEEE_13, 'w') as f:
        #     f.write(glm)

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
        # NOTE: data starts 2013-01-01 00:00:00 Mountain Time.
        self.assertRaises(gridappsd_platform.QueryReturnEmptyError,
                          self.platform.get_weather,
                          start_time=datetime(2013, 1, 1, 5, 59),
                          end_time=datetime(2013, 1, 1, 5, 59))

    def test_platform_manager_query_weather_simple(self):
        # Retrieve one hour of weather data, ensure it matches expected.
        with open(WEATHER, 'r') as f:
            expected = json.load(f)

        # Remove the 'id' key.
        del expected['id']

        # Query the platform. Data starts 2013-01-01 00:00:00 Mountain.
        actual = \
            self.platform._query_weather(start_time=datetime(2013, 1, 1, 6),
                                         end_time=datetime(2013, 1, 1, 6))

        # Remove 'id.'
        del actual['id']

        self.assertDictEqual(actual, expected)

    def test_platform_manager_get_weather_valid(self):
        """Ensure parse_weather is called with the expected input."""
        with patch('pyvvo.timeseries.parse_weather',
                   side_effect=parse_weather) as mock:
            _ = self.platform.get_weather(
                start_time=datetime(2013, 1, 1, 6),
                end_time=datetime(2013, 1, 1, 6))

        # Ensure parse_weather is called.
        mock.assert_called_once()

        # Get the actual data from the platform.
        actual = mock.call_args[0][0]

        # Uncomment to re-generated expected.
        # with open('weather_simple.json', 'w') as f:
        #     json.dump(actual, f)

        with open(WEATHER, 'r') as f:
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


if __name__ == '__main__':
    unittest.main()
