"""Module for handling timeseries data from the platform.
https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#timeseries-api
"""

import pandas as pd
import logging

# Setup log.
LOG = logging.getLogger(__name__)


def parse_timeseries(data):
    """Helper to parse platform timeseries data.

    :param data: dictionary with results from calling the timeseries
        API (either for weather data or simulation data). Ultimately,
        this is a return from gridappsd.GridAPPSD.get_response. It is
        assumed that data['data'] is not None, as that check is done
        at a higher level. See gridappsd_platform.py.
    """
    # Ensure data is a dictionary. We won't check its integrity, and
    # let a KeyError get raised if it's incorrectly formatted.
    if not isinstance(data, dict):
        raise TypeError('data must be a dictionary!')

    # Initialize dictionary for flattening the interesting return
    # from the platform.
    flat_dict = {}

    # The measurements can come back with mixed types. However, I'm not
    # going to support that. Query filters should be used to avoid these
    # situations.
    #
    # Grab the 'keys' for the very first entry. Yes, this depth and
    # hard-coding are insane.
    for entry in data['data']['measurements'][0]['points'][0]['row']['entry']:
        flat_dict[entry['key']] = []

    # Loop over the "rows."
    for row in data['data']['measurements'][0]['points']:
        # Keep track of which keys have been accessed - we need to
        # ensure consistency.
        keys = list(flat_dict.keys())

        # Loop over all the measurements, since they aren't properly
        # keyed.
        for meas_dict in row['row']['entry']:
            # Grab type and value of measurement.
            meas_type = meas_dict['key']
            meas_value = meas_dict['value']

            # Place the measurement in the dictionary.
            try:
                # Attempt to append to the list.
                flat_dict[meas_type].append(meas_value)
            except KeyError:
                # We have inconsistent data.
                m = ('The data is inconsistent! Found field {} which was '
                     + 'not present in the first entry/row!').format(meas_type)
                raise ValueError(m) from None

            # Remove the meas_type from the keys list.
            keys.remove(meas_type)

        # Ensure we used up all the keys.
        if len(keys) != 0:
            m = ('The data is inconsistent! Found row which is missing '
                 + 'the following fields: {}').format(keys)
            raise ValueError(m)

    # Return a DataFrame.
    return pd.DataFrame(flat_dict)

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

    # Note that Proven returns
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

