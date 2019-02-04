"""Module for handling weather data."""
import pandas as pd


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


# def adjust_weather(data, interval, interval_unit):
#     """Resample weather data, zero out negative GHI.
#     data should be DataFrame from parse_weather
#     interval: e.g. 15
#     interval_unit: e.g. "Min"
#     """
#     # Get 15-minute average. Since we want historic data leading up to
#     # the time in our interval, use 'left' options
#     weather = data.resample('{}{}'.format(interval, interval_unit),
#                             closed='right', label='right').mean()
#
#     # Zero-out negative GHI.
#     weather['ghi'][weather['ghi'] < 0] = 0
#
#     return weather

