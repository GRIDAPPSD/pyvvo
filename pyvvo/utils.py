"""Miscellaneous utility functions for pyvvo"""
import re
import math
import cmath

import pandas as pd

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
