"""Module for handling timeseries data from the platform.
https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#timeseries-api
"""
import datetime

import numpy as np
import pandas as pd
import logging

# Setup log.
LOG = logging.getLogger(__name__)


def parse_timeseries(data, index_by_time=True):
    """Helper to parse platform timeseries data.

    :param data: dictionary with results from calling the timeseries
        API (either for weather data or simulation data). Ultimately,
        this is a return from gridappsd.GridAPPSD.get_response.
    :param index_by_time: Boolean, whether or not to return data
        indexed by time. You may want to set this to false if the
        data has multiple entries for a single time.

    :returns: pandas DataFrame representing the data. Data types in
        NUMERIC_COLS will be cast to np.float. Note NaNs may be present.
    """
    # Ensure data is a dictionary.
    if not isinstance(data, dict):
        raise TypeError('data must be a dictionary!')

    # Simply use data['data'] to create a DataFrame.
    df = pd.DataFrame(data['data'])

    # Get the timestamps as Datetime-esque objects. Note that Proven
    # returns timestamps as seconds from the epoch, UTC. I think the
    # source of the timestamps is the simulation itself.
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True, origin='unix')

    # Set the time index and return
    if index_by_time:
        return df.set_index(keys='time', drop=True, inplace=False)
    else:
        return df


def parse_weather(data):
    """Helper to parse the ridiculous platform weather data return.

    For the journal paper, we used "solar_flux" from GridLAB-D. It seems
    simplest here to use the "global" irradiance, which is the sum of
    direct and diffuse irradiation.

    :param data: dictionary passed directly from
        gridappsd_platform.PlatformManager.get_weather

    :returns:
    """
    # Start by parsing into a DataFrame.
    df = parse_timeseries(data)

    # Just return a renamed version of the columns we care about.
    return df[['TowerDryBulbTemp', 'GlobalCM22']].rename(
        columns={'TowerDryBulbTemp': 'temperature', 'GlobalCM22': 'ghi'})


def resample_timeseries(ts, interval_str, method=None):
    """Resample timeseries, either up-sampling or down-sampling. For
    up-sampling, linear interpolation will be used, and means will be
    used for down-sampling.

    :param ts: DataFrame or Series indexed by time.
    :param interval_str: String representing the new desired interval.
        Docs:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        Examples:
            '15Min'
            '3S'
    :param method: String, either 'upsample' or 'downsample'. Only use
        this parameter (make it None) if ts does not have a regular
        interval, making it difficult to infer if we're upsampling or
        downsampling.
    """
    # https://stackoverflow.com/questions/24635721/how-to-compare-frequencies-sampling-rates-in-pandas
    if not isinstance(ts, (pd.DataFrame, pd.Series)):
        raise TypeError('weather_data must be a pandas DataFrame or Series.')

    if not isinstance(interval_str, str):
        raise TypeError('interval_str must be a string!')

    # If not given 'method', infer based on frequencies.
    if method is None:
        # Compare the interval of our input data and the desired interval.
        ts_freq = pd.infer_freq(ts.index)

        if ts_freq is None:
            # TODO: Find better exception.
            raise UserWarning('Could not infer frequency from timeseries!')

        # Determine if we're upsampling or downsampling.
        method = up_or_down_sample(orig_interval=ts_freq,
                                   new_interval=interval_str)

        if method is None:
            # Do nothing and return.
            LOG.warning('The given timeseries and interval_str have the same '
                        'frequency, so no resampling was performed.')
            return ts
    else:
        # If given 'method', ensure it's valid.
        if method not in ['upsample', 'downsample']:
            raise ValueError("method must be either 'upsample' or"
                             " 'downsample'")

    # Now that we've determined our method, perform the resampling
    # and return.
    if method == 'upsample':
        return ts.resample(interval_str, closed='right',
                           label='right').interpolate('time')
    elif method == 'downsample':
        return ts.resample(interval_str, closed='right',
                           label='right').mean()

    # Nothing else to see here.


# noinspection PyUnresolvedReferences
def up_or_down_sample(orig_interval, new_interval):
    """Helper to determine if upsampling or downsampling needs
    performed given two frequency strings.

    :param orig_interval: str, representing a Pandas dateoffset.
    :param new_interval: str, representing a Pandas dateoffset.

    Docs:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    Examples: '1Min', '10S'

    :returns: 'upsample', 'downsample', or None (if the intervals are
        equal)
    """
    o_o = pd.tseries.frequencies.to_offset(orig_interval)
    n_o = pd.tseries.frequencies.to_offset(new_interval)
    if o_o > n_o:
        return 'upsample'
    elif o_o < n_o:
        return 'downsample'
    else:
        return None


def fix_ghi(weather_data):
    """The weather data can have negative GHI values, which is not
    sensible. Zero them out.

    :param weather_data: DataFrame from calling parse_weather. Must have
        a 'ghi' column.
    """
    if not isinstance(weather_data, pd.DataFrame):
        raise TypeError('weather_data must be a pandas DataFrame.')

    # Zero-out negative GHI.
    weather_data.loc[weather_data['ghi'] < 0, 'ghi'] = 0

    return weather_data


def filter_by_time(data, t_start, t_end):
    """Filter data to get specific time intervals. Specifically, the
    data will be filtered such that:
    t_start <= data.index.time <= t_end

    :param data: Pandas DataFrame or Timeseries indexed by time.
    :param t_start: datetime.time object for the start of filtering.
    :param t_end: datetime.time object for the end of filtering.
    """
    # Quick type-checking.
    if (not isinstance(t_start, datetime.time)) \
            or (not isinstance(t_end, datetime.time)):
        raise ValueError('t_start and t_end must be datetime.time objects!')

    # Return the filtered data set.
    return data.loc[(data.index.time >= t_start)
                    & (data.index.time <= t_end), :]


def is_weekday(dt):
    """Indicate if datetime object represents a weekday (True) or not
    (False).

    :param dt: datetime.datetime like object.

    :returns: True if dt represents a weekday, False if dt represents a
        weekend.
    """
    dow = dt.weekday()

    if (dow == 5) or (dow == 6):
        return False
    elif (dow >= 0) and (dow <= 4):
        return True
    else:
        raise ValueError('day_of_week must be >= 0 and <= 6!')


def filter_by_weekday(data):
    """Wrapper to call filter_by_day_of week to get weekdays.

    :param data: pandas DataFrame or Timeseries indexed by time.

    :returns: the subset of data which occurs on weekdays.
    """
    return _filter_by_day_of_week(data=data, day_start=0, day_end=4)


def filter_by_weekend(data):
    """Wrapper to call filter_by_day_of_week to get weekend days.

    :param data: pandas DataFrame or Timeseries indexed by time.

    :returns: the subset of data which occurs on weekends.
    """
    return _filter_by_day_of_week(data=data, day_start=5, day_end=6)


def _filter_by_day_of_week(data, day_start, day_end):
    """Filter data by day of week to get:
     day_start <= data.index.dayofweek <= day_end

     It follows from the above that day_start should be <= day_end.

     Note from the Pandas documentation that Monday = 0, Sunday = 6:
     https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html

     :param data: Pandas DataFrame or Series, index by time.
     :param day_start: Integer, representing the starting day.
     :param day_end: Integer, representing the ending day.

     NOTE: This method is "private" as inputs won't be checked. In most
     cases you'll want to sipmly use filter_by_weekday or
     filter_by_weekend.
     """
    dow = data.index.dayofweek
    return data.loc[(dow >= day_start) & (dow <= day_end), :]
