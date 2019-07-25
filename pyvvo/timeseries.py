"""Module for handling timeseries data from the platform.
https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#timeseries-api
"""
import numpy as np
import pandas as pd
import logging

# Setup log.
LOG = logging.getLogger(__name__)

# List of numeric columns.
# https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#timeseries-api
NUMERIC_COLS = ['GlobalCM22', 'DirectCH1', 'Diffuse', 'TowerDryBulbTemp',
                'TowerRH', 'AvgWindSpeed', 'AvgWindDirection',
                'angle', 'magnitude', 'value']


def parse_timeseries(data):
    """Helper to parse platform timeseries data.

    :param data: dictionary with results from calling the timeseries
        API (either for weather data or simulation data). Ultimately,
        this is a return from gridappsd.GridAPPSD.get_response.
    """
    # Ensure data is a dictionary.
    if not isinstance(data, dict):
        raise TypeError('data must be a dictionary!')

    # The data formatting/data model returned from the platform's time
    # series database is pretty ridiculous. There's all sorts of
    # unnecessary nesting. Since this nesting is present, we have a lot
    # of safety checks we need to do. To keep the code clean/short, I'm
    # just going to use asserts.
    assert 'data' in data.keys()
    assert 'measurements' in data['data'].keys()
    assert len(data['data']) == 1
    assert len(data['data']['measurements']) == 1
    assert 'name' in data['data']['measurements'][0].keys()
    assert 'points' in data['data']['measurements'][0].keys()
    assert isinstance(data['data']['measurements'][0]['points'], list)

    # Alrighty, there's our initial checks. Now move on.

    # Initialize our data list. This is eventually going to be used to
    # create a Pandas DataFrame.
    dl = []
    # Loop over the "points," which contain the data we need.
    for point in data['data']['measurements'][0]['points']:
        # We're expecting a dictionary with a single entry.
        assert len(point) == 1

        # Extract the row.
        row = point['row']

        # Again, we're expecting a dictionary with a single entry.
        assert len(row) == 1

        # Extract the "entry"
        entry = row['entry']

        # Create a dictionary for this entry.
        this_entry = {}

        # Loop over the values in entry.
        for d in entry:
            # We should have just 'key' and 'value' keys.
            assert len(d) == 2

            # Use the key as well, a key, and the value as well, a
            # value. The fact that I have to do this seems crazy.
            this_entry[d['key']] = d['value']

        # Append to our list.
        dl.append(this_entry)

    # Create our DataFrame.
    df = pd.DataFrame(dl)

    # Get the timestamps as Datetime-esque objects. Note that Proven
    # returns timestamps as seconds from the epoch, UTC. I think the
    # source of the timestamps is the simulation itself.
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True, origin='unix',
                                box=True)

    # Set the time index.
    df.set_index(keys='time', drop=True, inplace=True)

    # Get dictionary of numeric types for this data.
    dtype_dict = {key: np.float for key in df.columns.to_list()
                  if key in NUMERIC_COLS}

    # Return the DataFrame with numeric types properly casted.
    return df.astype(dtype=dtype_dict, copy=False, errors="raise")


def parse_weather(data):
    """Helper to parse the ridiculous platform weather data return.

    For the journal paper, we used "solar_flux" from GridLAB-D. It seems
    simplest here to use the "global" irradiance, which is the sum of
    direct and diffuse irradiation.

    :param data: dictionary passed directly from
        gridappsd_platform.PlatformManager.get_weather
    """
    # Ensure data is a dictionary. We won't check its integrity, and
    # let a KeyError get raised if it's incorrectly formatted.
    if not isinstance(data, dict):
        raise TypeError('data must be a dictionary!')

    # Initialize dictionary (which we'll later convert to a DataFrame)
    wd = {'temperature': [], 'ghi': []}
    t = []

    # Loop over the "rows."
    for row in data['data']['measurements'][0]['points']:
        # Loop over all the measurements, since they aren't properly
        # keyed.
        for meas_dict in row['row']['entry']:
            # Grab type and value of measurement.
            meas_type = meas_dict['key']
            meas_value = meas_dict['value']

            if meas_type == 'TowerDryBulbTemp':
                wd['temperature'].append(float(meas_value))
            elif meas_type == 'GlobalCM22':
                wd['ghi'].append(float(meas_value))
            elif meas_type == 'time':
                # Use an integer for time since we don't care about
                # fractions of a second for this application.
                t.append(round(float(meas_value)))

    # If any of these are empty, raise an exception.
    if len(t) == 0:
        raise ValueError('data did not have time!')

    if len(wd['temperature']) == 0:
        raise ValueError('data did not have TowerDryBulbTemp!')

    if len(wd['ghi']) == 0:
        raise ValueError('data did not have GlobalCM22!')

    # Ensure the lengths are the same.
    if not (len(t) == len(wd['temperature']) == len(wd['ghi'])):
        m = "data did not contain the same number of entries for "\
            "'TowerDryBulbTemp,' 'GlobalCM22', or 'time!'"
        raise ValueError(m)

    # Note that Proven returns time in seconds despite requiring
    # microseconds as input.
    t_index = pd.to_datetime(t, unit='s', utc=True, origin='unix',
                             box=True)
    # Convert to pandas DataFrame
    df_weather = pd.DataFrame(wd, index=t_index)
    return df_weather


def resample_weather(weather_data, interval, interval_unit):
    """Resample weather data.

    :param weather_data: DataFrame result from calling parse_weather.
    :param interval: Integer for resampling, e.g. 15
    :param interval_unit: One of the "offset aliases" for frequencies
        in pandas, e.g. "Min":
        http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    """
    if not isinstance(weather_data, pd.DataFrame):
        raise TypeError('weather_data must be a pandas DataFrame.')

    if not isinstance(interval, int):
        raise TypeError('interval must be an integer!')

    if not isinstance(interval_unit, str):
        raise TypeError('interval_unit must be a string!')

    # Perform the resampling.
    return weather_data.resample('{}{}'.format(interval, interval_unit),
                                 closed='right', label='right').mean()


def fix_ghi(weather_data):
    """The weather data can have negative GHI values, which is not
    sensible. Zero them out.

    :param weather_data: DataFrame from calling parse_weather. Must have
        a 'ghi' column.
    """
    if not isinstance(weather_data, pd.DataFrame):
        raise TypeError('weather_data must be a pandas DataFrame.')

    # Zero-out negative GHI.
    weather_data['ghi'][weather_data['ghi'] < 0] = 0

    return weather_data

