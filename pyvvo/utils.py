"""Miscellaneous utility functions for pyvvo"""
import re
import math
import cmath

# Regular expressions for complex number parsing (from GridLAB-D).
RECT_EXP = re.compile(r'[+-]*([0-9])+(\.)*([0-9])*(e[+-]*([0-9])+)*[+-]'
                      + r'([0-9])+(\.)*([0-9])*(e[+-]([0-9])+)*j')
FIRST_EXP = re.compile(r'[+-]*([0-9])+(\.)*([0-9])*(e[+-]*([0-9])+)*')
SECOND_EXP = re.compile(r'[+-]*([0-9])+(\.)*([0-9])*(e[+-]*([0-9])+)*[dr]')


def parse_complex_str(s):
    """Parse a string representing a complex number.

    Specifically designed to work with various types of output from
    GridLAB-D.

    :param s: string representing a complex number. Examples:
        +12.34-1.2j VA
        +15-20d V
        +12-3.14r I

    :returns complex number, unit associated with it.
    """
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
        mag_float = float(FIRST_EXP.match(c).group())
        phase_str = SECOND_EXP.search(c).group()
        # If the number doesn't fit the form, raise exception.
        if (not mag_float) or (not phase_str):
            raise ValueError(('Inputs to getComplex must have a sign defined '
                              + 'for both components.\nNo space is allowed, '
                              + 'except between the number and the unit.\n'
                              + 'Decimals are optional.\n'
                              + 'Number must end in j, d, or r.'))
        # Extract the unit and phase from the phase string
        phase_unit = phase_str[-1]
        phase_float = float(phase_str[:-1])
        # If the unit is degrees, convert to radians
        if phase_unit == 'd':
            phase_float = math.radians(phase_float)

        # Convert to complex.
        n = (mag_float * cmath.exp(1j * phase_float))

    return n, u

