import unittest
import math
import cmath
from pyvvo import utils
from datetime import datetime, timezone, timedelta
import pandas as pd
import os
import numpy as np

# Handle pathing.
from tests.models import MODEL_DIR
TEST_ZIP1 = os.path.join(MODEL_DIR, 'test_zip_1.csv')
TEST_GLM2 = os.path.join(MODEL_DIR, 'test2.glm')


class TestParseComplexStr(unittest.TestCase):
    """Test utils.parse_complex_str.

    Unfortunately these tests aren't going to have very informative
    names.
    """

    def test_polar_1(self):
        num, unit = utils.parse_complex_str('+348863+13.716d VA')
        self.assertEqual(unit, 'VA')
        expected_num = 348863 * cmath.exp(1j * math.radians(13.716))
        self.assertEqual(num, expected_num)

    def test_polar_2(self):
        num, unit = utils.parse_complex_str('-12.2+13d I')
        self.assertEqual(unit, 'I')
        expected_num = -12.2 * cmath.exp(1j * math.radians(13))
        self.assertEqual(num, expected_num)

    def test_polar_3(self):
        num, unit = utils.parse_complex_str('+3.258-2.14890r kV')
        self.assertEqual(unit, 'kV')
        expected_num = 3.258 * cmath.exp(1j * -2.14890)
        self.assertEqual(num, expected_num)

    def test_polar_4(self):
        num, unit = utils.parse_complex_str('-1.5e02+12d f')
        self.assertEqual(unit, 'f')
        expected_num = -1.5e02 * cmath.exp(1j * math.radians(12))
        self.assertEqual(num, expected_num)

    def test_rect_1(self):
        num, unit = utils.parse_complex_str('-1+2j VAr')
        self.assertEqual(unit, 'VAr')
        expected_num = -1 + 1j * 2
        self.assertEqual(num, expected_num)

    def test_rect_2(self):
        num, unit = utils.parse_complex_str('+1.2e-003+1.8e-2j d')
        self.assertEqual(unit, 'd')
        expected_num = 1.2e-003 + 1j * 1.8e-2
        self.assertEqual(num, expected_num)

    def test_non_complex_num(self):
        self.assertRaises(ValueError, utils.parse_complex_str, '15')

    def test_weird_string(self):
        self.assertRaises(ValueError, utils.parse_complex_str,
                          'Look mom, a string!')

    def test_wrong_format(self):
        self.assertRaises(ValueError, utils.parse_complex_str, '1+1i')


class TestReadGLDCsv(unittest.TestCase):
    """Test utils.read_gld_csv.

    TODO: Test failures, more interesting cases?
    Maybe not worth it, read_gld_csv is really a utility for
    unit testing, and won't be used in the actual application.
    """

    @classmethod
    def setUpClass(cls):
        """Read the file"""
        cls.df = utils.read_gld_csv(TEST_ZIP1)

    def test_shape_0(self):
        self.assertEqual(self.df.shape[0], 41)

    def test_shape_1(self):
        self.assertEqual(self.df.shape[1], 5)

    def test_headings_0(self):
        self.assertEqual(self.df.columns[0], 'timestamp')

    def test_headings_end(self):
        self.assertEqual(self.df.columns[-1], 'measured_reactive_power')

    def test_values_1(self):
        self.assertAlmostEqual(self.df['measured_reactive_power'].iloc[0],
                               5.71375e-07)

    def test_values_2(self):
        val1 = self.df['measured_voltage_2'].iloc[-1]
        val2, _ = utils.parse_complex_str('+139.988-0.00164745d')
        self.assertAlmostEqual(val1.real, val2.real)
        self.assertAlmostEqual(val1.imag, val2.imag)

    def test_values_3(self):
        self.assertEqual(self.df['timestamp'].iloc[-2],
                         '2018-01-01 00:39:00 UTC')


class ListToStringTestCase(unittest.TestCase):
    """Test list_to_string.

    Keeping testing very minimal as this isn't a critical function.
    """

    def test_string_list(self):
        actual = utils.list_to_string(in_list=['A', 'b', 'C'],
                                      conjunction='and')
        self.assertEqual('A, b, and C', actual)


class GLDInstalledTestCase(unittest.TestCase):
    """Test gld_installed."""

    def test_gld_installed_simple(self):
        """Simply put, GridLAB-D should be installed in the docker
        container, so should always evaluate to True.
        """
        self.assertTrue(utils.gld_installed(env=None))

    def test_gld_installed_bad_path(self):
        """Override the path so we can't find GridLAB-D."""
        self.assertFalse(utils.gld_installed(env={'PATH': '/usr/bin'}))


@unittest.skipUnless(utils.gld_installed(),
                     reason='GridLAB-D is not installed.')
class RunGLDTestCase(unittest.TestCase):
    """Test run_gld."""

    def test_run_gld_simple(self):
        """Ensure the model runs."""
        result = utils.run_gld(TEST_GLM2)
        self.assertEqual(0, result.returncode)

    def test_run_gld_bad_model(self):
        result = utils.run_gld(os.path.join(MODEL_DIR, 'nonexistent.glm'))
        self.assertNotEqual(0, result.returncode)

    def test_run_gld_bad_dir(self):
        with self.assertRaises(FileNotFoundError):
            utils.run_gld('/some/bad/path.glm')

    def test_run_gld_bad_env(self):
        result = utils.run_gld(TEST_GLM2, env={'PATH': '/usr/bin'})
        self.assertNotEqual(0, result.returncode)


class DTToUSFromEpochTestCase(unittest.TestCase):
    """Test dt_to_us_from_epoch"""

    def test_datetime_to_microseconds_from_epoch_bad_type(self):
        self.assertRaises(AttributeError,
                          utils.dt_to_us_from_epoch,
                          '2012-01-01 00:00:00')

    def test_datetime_to_microseconds_from_epoch_1(self):
        # Source: https://www.epochconverter.com/
        self.assertEqual("1356998400000000",
                         utils.dt_to_us_from_epoch(datetime(2013, 1, 1)))


class DtToSFromEpochTestCase(unittest.TestCase):
    """Test dt_to_s_from_epoch"""
    def test_one(self):
        dt = datetime(2019, 7, 26, 14, 43, 10)
        self.assertEqual("1564152190",
                         utils.dt_to_s_from_epoch(dt))


# noinspection PyShadowingBuiltins,PyMethodMayBeStatic
class MapDataFrameColumnsTestCase(unittest.TestCase):
    """Test map_dataframe_columns."""

    def test_map_dataframe_columns_map_bad_type(self):
        map = 10
        df = pd.DataFrame({'one': [1, 2, 3], 'two': ['true', 'false', 'true']})
        cols = ['two']
        self.assertRaises(TypeError, utils.map_dataframe_columns,
                          map=map, df=df, cols=cols)

    def test_map_dataframe_columns_df_bad_type(self):
        map = {'true': True, 'false': False}
        df = {'one': [1, 2, 3], 'two': ['true', 'false', 'true']}
        cols = ['two']
        self.assertRaises(TypeError, utils.map_dataframe_columns,
                          map=map, df=df, cols=cols)

    def test_map_dataframe_columns_cols_bad_type(self):
        map = {'true': True, 'false': False}
        df = pd.DataFrame({'one': [1, 2, 3], 'two': ['true', 'false', 'true']})
        cols = 'two'
        self.assertRaises(TypeError, utils.map_dataframe_columns,
                          map=map, df=df, cols=cols)

    def test_map_dataframe_columns_cols_bad_value(self):
        map = {'true': True, 'false': False}
        df = pd.DataFrame({'one': [1, 2, 3], 'two': ['true', 'false', 'true']})
        cols = ['tow']
        with self.assertLogs(utils.LOG, level='WARN'):
            utils.map_dataframe_columns(map=map, df=df, cols=cols)

    def test_map_dataframe_expected_return(self):
        map = {'true': True, 'false': False}
        df = pd.DataFrame({'one': [1, 2, 3], 'two': ['true', 'false', 'true']})
        cols = ['two']

        expected = pd.DataFrame({'one': [1, 2, 3], 'two': [True, False, True]})
        actual = utils.map_dataframe_columns(map=map, df=df, cols=cols)

        pd.testing.assert_frame_equal(actual, expected)


class PlatformHeaderTimestampToDTTestCase(unittest.TestCase):
    """Test platform_header_timestamp_to_dt."""

    def test_int(self):
        actual = utils.platform_header_timestamp_to_dt(1559770227*1000)
        expected = datetime(2019, 6, 5, 21, 30, 27, tzinfo=timezone.utc)
        diff = actual - expected
        self.assertEqual(diff.seconds, timedelta(seconds=0).seconds)

    def test_float(self):
        actual = utils.platform_header_timestamp_to_dt(1559770227.2*1000)
        expected = datetime(2019, 6, 5, 21, 30, 27, 2000, tzinfo=timezone.utc)
        diff = actual - expected
        self.assertEqual(diff.seconds, timedelta(seconds=0).seconds)

    def test_string(self):
        with self.assertRaises(TypeError):
            utils.platform_header_timestamp_to_dt(str(1559770227.2 * 1000))


class SimulationOutputTimestampToDTTestCase(unittest.TestCase):
    """Test simulation_output_timestamp_to_dt."""
    def test_int(self):
        actual = utils.simulation_output_timestamp_to_dt(1559770227)
        expected = datetime(2019, 6, 5, 21, 30, 27, tzinfo=timezone.utc)
        diff = actual - expected
        self.assertEqual(diff.seconds, timedelta(seconds=0).seconds)

    def test_float(self):
        actual = utils.simulation_output_timestamp_to_dt(1559770227.2)
        expected = datetime(2019, 6, 5, 21, 30, 27, 2000, tzinfo=timezone.utc)
        diff = actual - expected
        self.assertEqual(diff.seconds, timedelta(seconds=0).seconds)

    def test_string(self):
        with self.assertRaises(TypeError):
            utils.simulation_output_timestamp_to_dt(str(1559770227.2))


class PowerFactorTestCase(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_pf(self):
        a = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        actual = utils.power_factor(a)
        expected = np.array([0.7071067811865475, -0.7071067811865475,
                             0.7071067811865475, -0.7071067811865475])
        np.testing.assert_allclose(actual, expected)


class GetComplex(unittest.TestCase):
    def test_scalar_degrees(self):
        expected = 1 + 1j*1
        actual = utils.get_complex(r=2**0.5, phi=45, degrees=True)
        # noinspection PyTypeChecker
        self.assertTrue(np.isclose(actual, expected))

    # noinspection PyTypeChecker
    def test_scalar_radians(self):
        expected = 2 + 1j * 2
        actual = utils.get_complex(r=abs(expected), phi=np.angle(expected),
                                   degrees=False)
        self.assertTrue(np.isclose(expected, actual))

    # noinspection PyMethodMayBeStatic
    def test_array_degrees(self):
        expected = np.random.rand(10) + 1j * np.random.rand(10)
        actual = utils.get_complex(r=np.abs(expected),
                                   phi=np.angle(expected, deg=True),
                                   degrees=True)
        np.testing.assert_allclose(actual, expected)

    # noinspection PyMethodMayBeStatic
    def test_array_radians(self):
        expected = np.random.rand(10) + 1j * np.random.rand(10)
        actual = utils.get_complex(r=np.abs(expected),
                                   phi=np.angle(expected, deg=False),
                                   degrees=False)
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
