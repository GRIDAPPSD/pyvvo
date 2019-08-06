# Standard library.
import unittest
from datetime import datetime, timezone, time
import copy

# Third-party installed.
import simplejson as json
import numpy as np
import pandas as pd

# pyvvo
from pyvvo import timeseries
import tests.data_files as _df


class ParseTimeseriesMeasurementsTestCase(unittest.TestCase):
    """Test parse_timeseries, and pass it simulation measurements which
    have been created via querying the timeseries database after a
    simulation.
    """

    @classmethod
    def setUpClass(cls):
        # Load measurement data
        with open(_df.ALL_MEAS_13, 'r') as f:
            cls.meas_13 = json.load(f)

        with open(_df.E_CONS_MEAS_9500, 'r') as f:
            cls.e_cons_meas_9500 = json.load(f)

        # HARD-CODING INBOUND:
        # This starttime must match what's in data_files.py where these
        # json files are generated.
        cls.starttime = datetime(2013, 1, 14, 0, 0, tzinfo=timezone.utc)
        # The stoptime is the starttime plus the duration.
        cls.stoptime = datetime(2013, 1, 14, 0, 0, 20, tzinfo=timezone.utc)

    def test_13(self):
        # Parse the data.
        parsed_data = timeseries.parse_timeseries(data=self.meas_13)

        # We should get back as many rows as there are "points"
        self.assertEqual(
            len(self.meas_13['data']['measurements'][0]['points']),
            parsed_data.shape[0]
        )

        # For the 13 node "all measurements" we'll be getting back
        # mixed types which results in columns which are not used for
        # all the data. Ensure we have NaN's.
        self.assertTrue(
            parsed_data.isna().any().any()
        )

        # Ensure we have the expected columns.
        self.assertListEqual(
            ['hasSimulationMessageType', 'measurement_mrid', 'angle',
             'magnitude', 'simulation_id', 'value'],
            parsed_data.columns.to_list()
        )

        # The NA signature should be identical for angle and magnitude.
        angle_na = parsed_data['angle'].isna()
        mag_na = parsed_data['magnitude'].isna()

        pd.testing.assert_series_equal(angle_na, mag_na,
                                       check_names=False)

        # The NA signature should be exactly opposite for
        # angle/magnitude and value.
        value_na = parsed_data['value'].isna()
        xor = np.logical_xor(angle_na.values, value_na.values)
        self.assertTrue(xor.all())

        # Ensure all our times are in bounds.
        self.assertTrue((parsed_data.index >= self.starttime).all())
        self.assertTrue((parsed_data.index <= self.stoptime).all())

        # Ensure all columns which should be numeric are numeric.
        for c in parsed_data.columns.to_list():
            if c in timeseries.NUMERIC_COLS:
                self.assertEqual(np.dtype('float'), parsed_data[c].dtype)

    def test_9500(self):
        parsed_data = timeseries.parse_timeseries(data=self.e_cons_meas_9500)
        # For this data, we have 6 measurements because the platform
        # outputs data every 3 seconds, and we ran a 20 second
        # simulation.
        self.assertEqual(6, parsed_data.shape[0])

        # With just one measurement, we shouldn't get any nans.
        self.assertFalse(parsed_data.isna().any().any())

        # Ensure we have the expected columns.
        self.assertListEqual(
            ['hasSimulationMessageType', 'measurement_mrid', 'angle',
             'magnitude', 'simulation_id'],
            parsed_data.columns.to_list()
        )

        # Ensure all our times are in bounds.
        self.assertTrue((parsed_data.index >= self.starttime).all())
        self.assertTrue((parsed_data.index <= self.stoptime).all())

        # Ensure all columns which should be numeric are numeric.
        for c in parsed_data.columns.to_list():
            if c in timeseries.NUMERIC_COLS:
                self.assertEqual(np.dtype('float'), parsed_data[c].dtype)

    def test_sensor_service_9500(self):
        # Read in data.
        data = []
        for file in _df.SENSOR_MEAS_LIST:
            with open(file, 'r') as f:
                data.append(json.load(f))

        # Parse each dictionary and do some rudimentary tests. Mainly,
        # we're happy if it just parses.
        for d in data:
            parsed = timeseries.parse_timeseries(data=d)

            # Ensure it's a DataFrame.
            self.assertIsInstance(parsed, pd.DataFrame)

            # Ensure we're indexed by time.
            self.assertEqual('time', parsed.index.name)
            self.assertIsInstance(parsed.index[0], datetime)

            # Ensure we have all the expected columns.
            self.assertListEqual(
                ['instance_id', 'hasSimulationMessageType', 'measurement_mrid',
                 'angle', 'magnitude', 'simulation_id'],
                parsed.columns.to_list()
            )

            # Ensure angle and magnitude are floats.
            self.assertEqual(np.dtype('float'), parsed.dtypes['angle'])
            self.assertEqual(np.dtype('float'), parsed.dtypes['magnitude'])

            # The rest of our columns should be objects.
            cols = copy.copy(parsed.columns.to_list())
            cols.remove('angle')
            cols.remove('magnitude')

            for c in cols:
                self.assertEqual(np.dtype('O'), parsed.dtypes[c])

            # TODO: Once
            #   https://github.com/GRIDAPPSD/gridappsd-forum/issues/21#issue-475728176
            #   is addressed, add a length assertion. In data_files.py,
            #   the simulation duration was 300 seconds, and we're
            #   expecting aggregation every 30 seconds, so we should
            #   get 10 records.
            # self.assertEqual(10, parsed.shape[0])


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
        dt_index = pd.to_datetime([dt], utc=True)

        # Expected data.
        temp = 13.316
        ghi = -0.0376761524

        # Create expected DataFrame.
        cls.weather_simple_expected = \
            pd.DataFrame([[temp, ghi]], index=dt_index,
                         columns=['temperature', 'ghi'])

        cls.weather_simple_expected.index.name = 'time'

        # Create an entry with two rows.
        cls.weather_two = copy.deepcopy(cls.weather_simple)

        row = copy.deepcopy(
                cls.weather_two['data']['measurements'][0]['points'][0])

        cls.weather_two['data']['measurements'][0]['points'].append(row)

        # Create the expected return for the two-row case.
        dt_index_two = pd.to_datetime([dt, dt], utc=True)
        cls.weather_two_expected = \
            pd.DataFrame([[temp, ghi], [temp, ghi]], index=dt_index_two,
                         columns=['temperature', 'ghi'])
        cls.weather_two_expected.index.name = 'time'

    def test_parse_weather_bad_input_type(self):
        self.assertRaises(TypeError, timeseries.parse_weather, 10)

    def test_parse_weather_bad_input_value(self):
        """This will raise an AssertionError."""
        self.assertRaises(AssertionError, timeseries.parse_weather,
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

        self.assertRaises(KeyError, timeseries.parse_weather, data)

    def test_parse_weather_no_ghi(self):
        # Get a copy of the data.
        data = copy.deepcopy(self.weather_simple)

        # Remove ghi
        del data['data']['measurements'][0]['points'][0]['row']['entry'][8]

        self.assertRaises(KeyError, timeseries.parse_weather, data)

    def test_parse_weather_no_time(self):
        # Get a copy of the data.
        data = copy.deepcopy(self.weather_simple)

        # Remove time
        del data['data']['measurements'][0]['points'][0]['row']['entry'][10]

        self.assertRaises(KeyError, timeseries.parse_weather, data)

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

        actual = timeseries.parse_weather(data)

        # We'll have a NaN.
        self.assertTrue(actual['temperature'].isna().any())

    def test_parse_weather_two_missing_ghi(self):
        """Delete one ghi entry from the two-row case."""
        data = copy.deepcopy(self.weather_two)

        # Remove a ghi entry.
        del data['data']['measurements'][0]['points'][1]['row']['entry'][8]

        actual = timeseries.parse_weather(data)

        # We'll have a NaN.
        self.assertTrue(actual['ghi'].isna().any())

    def test_parse_weather_two_missing_time(self):
        """Delete one time entry from the two-row case."""
        data = copy.deepcopy(self.weather_two)

        # Remove a time entry.
        del data['data']['measurements'][0]['points'][1]['row']['entry'][10]

        actual = timeseries.parse_weather(data)

        # We'll have a NaN in the Index.
        self.assertTrue(actual.index.isna().any())


class ResampleTimeSeriesTestCase(unittest.TestCase):
    """Test resample_timeseries function."""

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
        self.assertRaises(TypeError, timeseries.resample_timeseries,
                          weather_data={'temp': [1, 2, 3]},
                          interval_str='15Min')

    def test_resample_weather_15_min_downsample(self):
        """Resample at 15 minute intervals."""
        actual = timeseries.resample_timeseries(ts=self.weather_data,
                                                interval_str='15Min')

        pd.testing.assert_frame_equal(actual, self.expected_data)

    # noinspection PyMethodMayBeStatic
    def test_resample_upsample(self):
        a = [1, 2, 3, 4, 5]
        b = [11, 12, 13, 14, 15]
        dt_index = pd.date_range(start=datetime(2019, 1, 1, 0, 1), periods=5,
                                 freq='2Min')

        to_resample = pd.DataFrame({'a': a, 'b': b}, index=dt_index)
        a2 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        b2 = [11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15]
        r_index = pd.date_range(start=datetime(2019, 1, 1, 0, 1), periods=9,
                                freq='1Min')
        expected = pd.DataFrame({'a': a2, 'b': b2}, index=r_index)

        actual = timeseries.resample_timeseries(ts=to_resample,
                                                interval_str='1Min')

        pd.testing.assert_frame_equal(actual, expected)

    def test_resample_same_freq(self):
        a = [1, 2, 3, 4, 5]
        i = pd.date_range(start=datetime(2019, 1, 1, 0, 1), periods=5,
                          freq='1Min')
        s = pd.Series(a, index=i)
        with self.assertLogs(logger=timeseries.LOG, level='WARNING'):
            s2 = timeseries.resample_timeseries(s, '1Min')

        self.assertIs(s, s2)

    def test_bad_method(self):
        with self.assertRaisesRegex(ValueError, 'method must be either'):
            timeseries.resample_timeseries(ts=pd.Series([1, 2, 3]),
                                           interval_str='20Min',
                                           method='bad_method')


class UpOrDownSampleTestCase(unittest.TestCase):
    def test_upsample(self):
        self.assertEqual(
            'upsample',
            timeseries.up_or_down_sample(orig_interval='10Min',
                                         new_interval='1Min')
        )

    def test_downsample(self):
        self.assertEqual(
            'downsample',
            timeseries.up_or_down_sample(orig_interval='1Min',
                                         new_interval='10Min')
        )

    def test_none(self):
        self.assertIsNone(timeseries.up_or_down_sample(orig_interval='1Min',
                                                       new_interval='T'))


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


class FilterByTimeAndFilterByDayTestCase(unittest.TestCase):
    """Test filter_by_time, filter_by_day_of_week, filter_by_weekday,
    and filter_by_weekend.
    """

    @classmethod
    def setUpClass(cls):
        """Read our two week weather file."""
        cls.w = pd.read_csv(_df.WEATHER_TWO_WEEK, index_col='time',
                            parse_dates=True)

    def test_filter_hour_and_a_half(self):
        """Filter such that we get times from 11:30 a.m. to 12:30 p.m.
        """
        out = timeseries.filter_by_time(data=self.w, t_start=time(11, 30),
                                        t_end=time(12, 30))

        # Ensure all hours are in range.
        self.assertTrue((out.index.time <= time(12, 30)).all())
        self.assertTrue((out.index.time >= time(11, 30)).all())

    def test_filter_by_day_of_week(self):
        """Simple test """
        out = timeseries._filter_by_day_of_week(data=self.w,
                                                day_start=3, day_end=5)

        self.assertTrue((out.index.dayofweek >= 3).all())
        self.assertTrue((out.index.dayofweek <= 5).all())

    def test_filter_by_weekdays(self):
        out = timeseries.filter_by_weekday(data=self.w)

        self.assertTrue((out.index.dayofweek >= 0).all())
        self.assertTrue((out.index.dayofweek <= 4).all())

    def test_filter_by_weekend(self):
        out = timeseries.filter_by_weekend(data=self.w)

        self.assertTrue((out.index.dayofweek >= 5).all())
        self.assertTrue((out.index.dayofweek <= 6).all())


if __name__ == '__main__':
    unittest.main()
