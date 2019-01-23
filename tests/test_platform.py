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

    def test_get_gad_address_in_platform(self):
        self.assertEqual(('gridappsd', 61613),
                         platform.get_gad_address(platform='1'))

    @patch.dict(os.environ, {'host_ip': '192.168.0.10',
                             'GRIDAPPSD_PORT': '61613'})
    def test_get_gad_address_outside_platform(self):
        self.assertEqual(('192.168.0.10', '61613'),
                         platform.get_gad_address(platform='0'))

    def test_get_gad_address_bad_platform_type(self):
        self.assertRaises(TypeError, platform.get_gad_address, platform=1)

    def test_get_gad_address_bad_platform_value(self):
        self.assertRaises(ValueError, platform.get_gad_address, platform='2')


class GetGADObjectTestCase(unittest.TestCase):
    """Test get_gad_object."""

    def setUp(self):
        """Attempt to connect to the platform."""
        try:
            platform_var = os.environ['platform']
        except KeyError:
            m = ('Environment variable "platform" must be defined to connect '
                 + 'to the GridAPPS-D platform. With PyCharm, you can use the '
                 + 'EnvFile plugin to make things easier.')
            raise KeyError(m)

        try:
            self.gad = platform.get_gad_object(platform=platform_var)
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


if __name__ == '__main__':
    unittest.main()
