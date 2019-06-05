"""Miscellaneous utility functions for pyvvo"""
import re
import math
import cmath
import subprocess
import logging
from datetime import datetime, timezone
import pandas as pd

# Setup log.
LOG = logging.getLogger(__name__)

# Regular expressions for complex number parsing (from GridLAB-D).
RECT_EXP = re.compile(r'[+-]*([0-9])+(\.)*([0-9])*(e[+-]*([0-9])+)*[+-]'
                      + r'([0-9])+(\.)*([0-9])*(e[+-]([0-9])+)*j')
FIRST_EXP = re.compile(r'[+-]*([0-9])+(\.)*([0-9])*(e[+-]*([0-9])+)*')
SECOND_EXP = re.compile(r'[+-]*([0-9])+(\.)*([0-9])*(e[+-]*([0-9])+)*[dr]')


def parse_complex_str(s):
    """Parse a string representing a complex number.

    Specifically designed to work with various types of output from
    GridLAB-D.

    Raises a ValueError if string cannot be cast to a complex number.

    :param s: string representing a complex number. Examples:
        +12.34-1.2j VA
        +15-20d V
        +12-3.14r I

    :returns complex number, unit associated with it.
    """
    # Return a ValueError if the input is not a string.
    if not isinstance(s, str):
        raise ValueError('The input to parse_complex_str must be a string.')

    # First, strip whitespace, then split on whitespace to strip off the
    # unit
    t = s.strip().split()
    # Grab complex number part of the string.
    c = t[0]

    # Attempt to grab the unit.
    try:
        u = t[1]
    except IndexError:
        # There's no unit.
        u = None

    # Take action depending on whether or not the complex number is
    # already in rectangular form.
    if RECT_EXP.fullmatch(c):
        # If it's already in rectangular form, there's not much to do.
        n = complex(c)
    else:
        # Extract the first and second terms
        mag_match = FIRST_EXP.match(c)
        phase_match = SECOND_EXP.search(c)

        # If the number doesn't fit the form, raise exception.
        if (not mag_match) or (not phase_match):
            raise ValueError(('Inputs to getComplex must have a sign defined '
                              + 'for both components.\n'
                              + 'Decimals are optional.\n'
                              + 'Number must end in j, d, or r.'))

        # Grab the groups of matches
        mag_float = float(mag_match.group())
        phase_str = phase_match.group()

        # Extract the unit and phase from the phase string
        phase_unit = phase_str[-1]
        phase_float = float(phase_str[:-1])
        # If the unit is degrees, convert to radians
        if phase_unit == 'd':
            phase_float = math.radians(phase_float)

        # Convert to complex.
        n = (mag_float * cmath.exp(1j * phase_float))

    return n, u


def read_gld_csv(f):
    """Read a .csv file from a GridLAB-D recorder into a DataFrame.

    NOTE: No time parsing/indexing will be attempted, as this isn't
    presently needed.
    """
    # Read the file
    df = pd.read_csv(f, skiprows=8)

    # Rename the '# timestamp' column. Pretty hard-coded, but oh well.
    df.rename(columns={'# timestamp': 'timestamp'}, inplace=True)

    # Remove leading whitespace from columns. Unfortunately, the
    # GridLAB-D output is inconsistent with spaces, which makes pandas
    # unhappy.
    df.rename(mapper=str.strip, inplace=True, axis=1)

    # Loop over the columns, and attempt to convert the value to a
    # complex number. If we get a ValueError, we won't convert.
    for c in df.columns:
        # Grab the first element.
        item = df.iloc[0][c]

        try:
            parse_complex_str(item)
        except ValueError:
            # Move to the next column - this string can't be converted.
            continue

        # Create a Series with the complex numbers. Nobody is claiming
        # this is efficient: it doesn't have to be. pyvvo primarily uses
        # the MySQL recorders (which don't have this problem). We're
        # just using these .csv files for unit tests.
        s = pd.Series(0+1j*0, index=df.index)

        # Loop over the items in this column.
        for ind, item in df[c].iteritems():

            # Place the parsed complex number in the series.
            s.loc[ind] = parse_complex_str(item)[0]

        # Replace this column.
        df[c] = s

    return df


def list_to_string(in_list, conjunction):
    """Simple helper for formatting lists contaings strings as strings.

    This is intended for simple lists that contain strings. Input will
    not be checked.

    :param in_list: List to be converted to a string.
    :param conjunction: String - conjunction to be used (e.g. and, or).
    """
    return ", ".join(in_list[:-1]) + ", {} {}".format(conjunction, in_list[-1])


def gld_installed(env=None):
    """Test if GridLAB-D is installed or not."""
    # Attempt to run GridLAB-D.
    result = subprocess.run("gridlabd --version", shell=True,
                            stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE, env=env)

    LOG.debug('"gridlabd --version" result:\n{}'.format(result.stdout))

    if result.returncode == 0:
        return True
    else:
        return False


def run_gld(model_path, env=None):
    """Helper to run a GRIDLAB-D model. Returns True for success, False
    for failure.

    If needed, run options can be added in the future.

    :param model_path: path (preferably full path) to GridLAB-D model.
    :param env: used to override the environment for subprocess. Leave
        this as None.
    """
    result = subprocess.run("gridlabd {}".format(model_path), shell=True,
                            stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                            env=env)

    if result.returncode == 0:
        # TODO: Do we want to log GridLAB-D output? It would really clog
        #   up the main log, so we might want a different handler.
        LOG.debug('GridLAB-D model {} ran successfully.'.format(model_path))
        return True
    else:
        m = ('GridLAB-D model {} failed to run.\n\tstdout:{}\n\t'
             + 'stderr:{}').format(model_path, result.stdout, result.stderr)
        LOG.error(m)
        return False


def dt_to_us_from_epoch(dt):
    """Convert datetime.datetime object to microseconds since the epoch.

    :param dt: datetime.datetime object.

    :returns: microseconds since the epoch as a string.
    """
    return '{:.0f}'.format(dt.timestamp() * 1e6)


def platform_header_timestamp_to_dt(timestamp):
    """Convert timestamp (milliseconds from epoch) to datetime object.
    This is specifically built for reading the 'timestamp' field of the
    header which comes in from the GridAPPS-D platform.

    :param timestamp: Integer or float. Milliseconds since
        1970-01-01 00:00:00.000. Assumed to be in UTC.

    :returns dt: timezone aware (UTC) datetime.datetime object.
    """
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc)


def simulation_output_timestamp_to_dt(timestamp):
    """Convert timestamp (seconds from epoch) to datetime object.
    This is specifically built for reading the 'timestamp' field of the
    message object which comes from the GridAPPS-D simulator output.

    :param timestamp: Integer or float. Seconds since
        1970-01-01 00:00:00.000. Assumed to be in UTC.

    :returns: dt: timezone aware (UTC) datetime.datetime object.
    """
    return datetime.fromtimestamp(timestamp, timezone.utc)


# noinspection PyShadowingBuiltins
def map_dataframe_columns(map, df, cols):
    """Helper to apply a map to specified columns in a pandas DataFrame.

    :param map: valid input to pandas.Series.map.
    :param df: pandas DataFrame.
    :param cols: list of columns in 'df' to apply 'map' to.
    """
    # Check inputs (but allow pandas to check the map).
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df input must be a pandas DataFrame.')

    if not isinstance(cols, list):
        raise TypeError('cols input must be a list.')

    for col in cols:
        try:
            df[col] = df[col].map(map)
        except KeyError:
            # If we're trying to map a column which doesn't exist,
            # warn.
            LOG.warning('Column {} does not exist in DataFrame.'.format(col))

    return df
