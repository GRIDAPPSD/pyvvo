# Standard library
import os
import unittest
from unittest.mock import patch
from datetime import datetime

# PyVVO + GridAPPS-D
from pyvvo import gridappsd_platform
from gridappsd import GridAPPSD
from pyvvo.weather import parse_weather

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
IEEE_13 = os.path.join(THIS_DIR, 'ieee_13.glm')
WEATHER = os.path.join(THIS_DIR, 'weather_simple.json')

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
        # with open('ieee_13.glm', 'w') as f:
        #     f.write(glm)

        # Get expected.
        with open(IEEE_13, 'r') as f:
            expected = f.read()

        self.assertEqual(glm, expected)

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

    def test_platform_manager_get_weather_valid(self):
        """Ensure parse_weather is called with the expected input."""
        with patch('pyvvo.weather.parse_weather', side_effect=parse_weather)\
                as mock:
            actual_df = self.platform.get_weather(
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


if __name__ == '__main__':
    unittest.main()
