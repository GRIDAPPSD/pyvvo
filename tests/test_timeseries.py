# Standard library.
import unittest
from datetime import datetime
import copy

# Third-party installed.
import simplejson as json
import pandas as pd

# pyvvo
from pyvvo import timeseries
import tests.data_files as _df

# class ParseTimeseriesMeasurementsTestCase(unittest.TestCase):
#     """Test parse_timeseries, and pass it simulation measurements."""
#
#     @classmethod
#     def setUpClass(cls):
#         # Load measurement data
#         with open(MEASUREMENTS, 'r') as f:
#             cls.data = json.load(f)
#
#     def test_one(self):
#
#         parsed_data = timeseries.parse_timeseries(data=self.data)
#         self.assertTrue(False)


class ParseWeatherTestCase(unittest.TestCase):
    """Test parse_weather"""

    @classmethod
    def setUpClass(cls):
        # Load the simple weather data dictionary.
        with open(_df.WEATHER, 'r') as f:
            cls.weather_simple = json.load(f)

        # Create the expected DataFrame.
        # Weather data starts at 2013-01-01 00:00:00 Mountain Time.
        dt = datetime(2013, 1, 1, 6)
        dt_index = pd.to_datetime([dt], box=True, utc=True)

        # Expected data.
        temp = 13.316
        ghi = -0.0376761524

        # Create expected DataFrame.
        cls.weather_simple_expected = \
            pd.DataFrame([[temp, ghi]], index=dt_index,
                         columns=['temperature', 'ghi'])

        # Create an entry with two rows.
        cls.weather_two = copy.deepcopy(cls.weather_simple)

        row = copy.deepcopy(
                cls.weather_two['data']['measurements'][0]['points'][0])

        cls.weather_two['data']['measurements'][0]['points'].append(row)

        # Create the expected return for the two-row case.
        dt_index_two = pd.to_datetime([dt, dt], box=True, utc=True)
        cls.weather_two_expected = \
            pd.DataFrame([[temp, ghi], [temp, ghi]], index=dt_index_two,
                         columns=['temperature', 'ghi'])

    def test_parse_weather_bad_input_type(self):
        self.assertRaises(TypeError, timeseries.parse_weather, 10)

    def test_parse_weather_bad_input_value(self):
        """This will raise a TypeError rather than a KeyError."""
        self.assertRaises(TypeError, timeseries.parse_weather,
                          {'data': {'measurements': [1, 2, 3]}})

    # noinspection PyMethodMayBeStatic
    def test_parse_weather_weather_simple_json(self):
        """Load weather_simple.json, ensure result matches."""

        # Parse the data.
        actual = timeseries.parse_weather(self.weather_simple)

        # Ensure actual and expected results are equal.
        pd.testing.assert_frame_equal(actual, self.weather_simple_expected)

    def test_parse_weather_no_temperature(self):
        # Get a copy of the data.
        data = copy.deepcopy(self.weather_simple)

        # Remove temperature
        del data['data']['measurements'][0]['points'][0]['row']['entry'][5]

        self.assertRaises(ValueError, timeseries.parse_weather, data)

    def test_parse_weather_no_ghi(self):
        # Get a copy of the data.
        data = copy.deepcopy(self.weather_simple)

        # Remove ghi
        del data['data']['measurements'][0]['points'][0]['row']['entry'][8]

        self.assertRaises(ValueError, timeseries.parse_weather, data)

    def test_parse_weather_no_time(self):
        # Get a copy of the data.
        data = copy.deepcopy(self.weather_simple)

        # Remove time
        del data['data']['measurements'][0]['points'][0]['row']['entry'][10]

        self.assertRaises(ValueError, timeseries.parse_weather, data)

    def test_parse_weather_data_no_wind(self):
        """We aren't using wind speed."""
        # Get a copy of the data.
        data = copy.deepcopy(self.weather_simple)

        # Remove average wind speed
        del data['data']['measurements'][0]['points'][0]['row']['entry'][1]

        # We should not get a ValueError.
        # noinspection PyBroadException
        try:
            timeseries.parse_weather(data)
        except Exception:
            m = ('An exception was raised when parsing simple weather without '
                 'AvgWindSpeed')
            self.fail(m)

    def test_parse_weather_two(self):
        """Check the two-row case."""
        actual = timeseries.parse_weather(self.weather_two)
        pd.testing.assert_frame_equal(actual, self.weather_two_expected)

    def test_parse_weather_two_missing_temperature(self):
        """Delete one temperature entry from the two-row case."""
        data = copy.deepcopy(self.weather_two)

        # Remove a temperature entry.
        del data['data']['measurements'][0]['points'][0]['row']['entry'][5]

        self.assertRaises(ValueError, timeseries.parse_weather, data)

    def test_parse_weather_two_missing_ghi(self):
        """Delete one ghi entry from the two-row case."""
        data = copy.deepcopy(self.weather_two)

        # Remove a ghi entry.
        del data['data']['measurements'][0]['points'][1]['row']['entry'][8]

        self.assertRaises(ValueError, timeseries.parse_weather, data)

    def test_parse_weather_two_missing_time(self):
        """Delete one time entry from the two-row case."""
        data = copy.deepcopy(self.weather_two)

        # Remove a time entry.
        del data['data']['measurements'][0]['points'][1]['row']['entry'][10]

        self.assertRaises(ValueError, timeseries.parse_weather, data)


class ResampleWeatherTestCase(unittest.TestCase):
    """Test resample_weather function."""

    @classmethod
    def setUpClass(cls):
        """Create a DataFrame for resampling."""
        dt_index = pd.date_range(start=datetime(2019, 1, 1, 0, 1), periods=15,
                                 freq='1Min')

        # Create a temperature array with an average of 2.
        temp = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

        # Create ghi array with an average of 3.
        ghi = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]

        # Create DataFrame.
        cls.weather_data = pd.DataFrame({'temperature': temp, 'ghi': ghi},
                                        index=dt_index)

        # Create expected data.
        dt_index_2 = pd.date_range(start=datetime(2019, 1, 1, 0, 15), periods=1,
                                   freq='15Min')
        cls.expected_data = pd.DataFrame({'temperature': [2], 'ghi': [3]},
                                         index=dt_index_2)

    def test_resample_weather_bad_weather_data_type(self):
        self.assertRaises(TypeError, timeseries.resample_weather,
                          weather_data={'temp': [1, 2, 3]}, interval=15,
                          interval_unit='Min')

    def test_resample_weather_bad_interval_type(self):
        self.assertRaises(TypeError, timeseries.resample_weather,
                          weather_data=pd.DataFrame(), interval=15.1,
                          interval_unit='Min')

    def test_resample_weather_bad_interval_unit(self):
        self.assertRaises(TypeError, timeseries.resample_weather,
                          weather_data=pd.DataFrame(), interval=15,
                          interval_unit=[1, 2, 3])

    def test_resample_weather_15_min(self):
        """Resample at 15 minute intervals."""
        actual = timeseries.resample_weather(weather_data=self.weather_data,
                                             interval=15, interval_unit='Min')

        pd.testing.assert_frame_equal(actual, self.expected_data)


class FixGHITestCase(unittest.TestCase):
    """Test fix_ghi"""

    def test_fix_ghi_bad_weather_data_type(self):
        self.assertRaises(TypeError, timeseries.fix_ghi, weather_data=pd.Series())

    # noinspection PyMethodMayBeStatic
    def test_fix_ghi(self):
        """Simple test."""
        temperature = [1, -1, 10]
        ghi_neg = [0.0001, -0.2, 7]
        ghi_pos = [0.0001, 0, 7]
        df_original = pd.DataFrame({'temperature': temperature,
                                    'ghi': ghi_neg})
        df_expected = pd.DataFrame({'temperature': temperature,
                                    'ghi': ghi_pos})

        df_actual = timeseries.fix_ghi(df_original)

        pd.testing.assert_frame_equal(df_expected, df_actual)


if __name__ == '__main__':
    unittest.main()
