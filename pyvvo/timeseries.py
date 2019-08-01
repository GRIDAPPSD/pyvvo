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

    :returns pandas DataFrame representing the data. Data types in
        NUMERIC_COLS will be cast to np.float. Note NaNs may be present.
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
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True, origin='unix')

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

    :returns:
    """
    # Start by parsing into a DataFrame.
    df = parse_timeseries(data)

    # Just return a renamed version of the columns we care about.
    return df[['TowerDryBulbTemp', 'GlobalCM22']].rename(
        columns={'TowerDryBulbTemp': 'temperature', 'GlobalCM22': 'ghi'})


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
    weather_data.loc[weather_data['ghi'] < 0, 'ghi'] = 0

    return weather_data
