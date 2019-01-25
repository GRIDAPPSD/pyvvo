# Standard library
import os
import unittest
from unittest.mock import patch

# PyVVO + GridAPPS-D
from pyvvo import platform
from gridappsd import GridAPPSD

# Third-party
from stomp.exception import ConnectFailedException
import simplejson as json


class GetGADAddressTestCase(unittest.TestCase):
    """Test get_gad_address."""

    @patch.dict(os.environ, {'platform': '1'})
    def test_get_gad_address_in_platform(self):
        self.assertEqual(('gridappsd', 61613), platform.get_gad_address())

    @patch.dict(os.environ, {'host_ip': '192.168.0.10',
                             'GRIDAPPSD_PORT': '61613', 'platform': '0'})
    def test_get_gad_address_outside_platform(self):
        self.assertEqual(('192.168.0.10', '61613'), platform.get_gad_address())

    def test_get_gad_address_calls_get_platform_env_var(self):
        with patch('pyvvo.platform.get_platform_env_var', return_value='1') \
                as mock:
            address = platform.get_gad_address()
            mock.assert_called_once()
            mock.assert_called_with()
            self.assertEqual(('gridappsd', 61613), address)


class GetGADObjectTestCase(unittest.TestCase):
    """Test get_gad_object. NOTE: Requires the GridAPPS-D platform to be
    up and running. Also, the 'platform' environment variable must be
    set.
    """

    def setUp(self):
        """Attempt to connect to the platform."""
        try:
            self.gad = platform.get_gad_object()
        except ConnectFailedException:
            # We cannot connect to the platform.
            raise unittest.SkipTest('Failed to connect to GridAPPS-D.')

    def test_get_gad_object_is_gad_object(self):
        self.assertIsInstance(self.gad, GridAPPSD)

    def test_get_gad_object_query_model_info(self):
        """Test that the query_model_info function acts as expected."""
        # Get model information from the platform.
        actual_info = self.gad.query_model_info()

        # Load the expected result.
        with open('query_model_info.json', 'r') as f:
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
        self.assertRaises(KeyError, platform.get_platform_env_var)

    @patch.dict(os.environ, {'platform': 'not 1 or 0'})
    def test_get_platform_env_var_bad_value(self):
        self.assertRaises(ValueError, platform.get_platform_env_var)

    @patch.dict(os.environ, {'platform': '1'})
    def test_get_platform_env_var_1(self):
        self.assertEqual('1', platform.get_platform_env_var())

    @patch.dict(os.environ, {'platform': '0'})
    def test_get_platform_env_var_0(self):
        self.assertEqual('0', platform.get_platform_env_var())


class PlatformManagerTestCase(unittest.TestCase):
    """Test the PlatformManager. Requires the GridAPPS-D platform to
    be up and running, and the 'platform' environment variable to be
    set.
    """
    def setUp(self):
        # Initialize a platform manager.
        self.platform = platform.PlatformManager()

    def test_platform_manager_gad(self):
        self.assertIsInstance(self.platform.gad, GridAPPSD)

    def test_platform_manager_get_glm(self):
        # TODO: update when platform is fixed.
        m = '{"data":"model123456","responseComplete":true,"id": "bigID"}'
        platform_return = {
            'error': 'Invalid json returned',
            'header': {'stuff': 'I do not need'}, 'message': m}

        with patch('gridappsd.GridAPPSD.get_response',
                   return_value=platform_return) as mock:
            glm = self.platform.get_glm(model_id="someID")
            mock.assert_called_once()

        self.assertEqual(glm, '"model123456"')



if __name__ == '__main__':
    unittest.main()
