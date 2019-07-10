"""Module for managing testing data files.

TODO: There's a lot of nasty hard-coding in here (copy + paste type
    stuff) that could be cleaned up.
"""
# Standard library.
import os
import pickle
from uuid import UUID

# PyVVO.
from pyvvo import sparql

# Third-party.
import pandas as pd

# Hard-code 8500 node feeder MRID.
FEEDER_MRID_8500 = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3'

# 13 node:
FEEDER_MRID_13 = '_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62'

# 123 node:
FEEDER_MRID_123 = '_C1C3E687-6FFD-C753-582B-632A27E28507'

# We'll be mocking some query returns.
MOCK_RETURN = pd.DataFrame({'name': ['thing1', 'thing2'],
                            'prop': ['prop1', 'prop2']})

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'data')
CAPACITORS_8500 = os.path.join(DATA_DIR, 'query_capacitors_8500.csv')
REGULATORS_8500 = os.path.join(DATA_DIR, 'query_regulators_8500.csv')
REG_MEAS_8500 = os.path.join(DATA_DIR, 'query_reg_meas_8500.csv')
CAP_MEAS_8500 = os.path.join(DATA_DIR, 'query_cap_meas_8500.csv')
LOAD_MEAS_8500 = os.path.join(DATA_DIR, 'query_load_measurements_8500.csv')
SUBSTATION_8500 = os.path.join(DATA_DIR, 'query_substation_source_8500.csv')
SWITCHES_8500 = os.path.join(DATA_DIR, 'query_switches_8500.csv')
SWITCH_MEAS_8500 = os.path.join(DATA_DIR, 'query_switch_meas_8500.csv')

# 13 bus outputs for smaller/quicker/more manual tests.
CAPACITORS_13 = os.path.join(DATA_DIR, 'query_capacitors_13.csv')
REGULATORS_13 = os.path.join(DATA_DIR, 'query_regulators_13.csv')
REG_MEAS_13 = os.path.join(DATA_DIR, 'query_reg_meas_13.csv')
CAP_MEAS_13 = os.path.join(DATA_DIR, 'query_cap_meas_13.csv')
LOAD_MEAS_13 = os.path.join(DATA_DIR, 'query_load_measurements_13.csv')
SUBSTATION_13 = os.path.join(DATA_DIR, 'query_substation_source_13.csv')
SWITCHES_13 = os.path.join(DATA_DIR, 'query_switches_13.csv')
SWITCH_MEAS_13 = os.path.join(DATA_DIR, 'query_switch_meas_13.csv')

# 123 bus for "medium" tests
CAPACITORS_123 = os.path.join(DATA_DIR, 'query_capacitors_123.csv')
REGULATORS_123 = os.path.join(DATA_DIR, 'query_regulators_123.csv')
REG_MEAS_123 = os.path.join(DATA_DIR, 'query_reg_meas_123.csv')
CAP_MEAS_123 = os.path.join(DATA_DIR, 'query_cap_meas_123.csv')
LOAD_MEAS_123 = os.path.join(DATA_DIR, 'query_load_measurements_123.csv')
SUBSTATION_123 = os.path.join(DATA_DIR, 'query_substation_source_123.csv')
SWITCHES_123 = os.path.join(DATA_DIR, 'query_switches_123.csv')
SWITCH_MEAS_123 = os.path.join(DATA_DIR, 'query_switch_meas_123.csv')

# Misc json files.
REG_MEAS_MSG_8500 = os.path.join(DATA_DIR, 'reg_meas_message_8500.json')
CAP_MEAS_MSG_8500 = os.path.join(DATA_DIR, 'cap_meas_message_8500.json')

MEASUREMENTS_13 = os.path.join(DATA_DIR, 'simulation_measurements_13.json')
HEADER_13 = os.path.join(DATA_DIR, 'simulation_measurements_header_13.json')

WEATHER = os.path.join(DATA_DIR, 'weather_simple.json')

MODEL_INFO = os.path.join(DATA_DIR, 'query_model_info.json')


def read_pickle(csv_file):
    """Helper to read a pickle file corresponding to a csv file."""
    with open(csv_file.replace('.csv', '.pickle'), 'rb') as f:
        p = pickle.load(f)

    return p


def to_file(df, csv_file):
    """Helper to write a DataFrame both to csv and pickle."""
    df.to_csv(csv_file, index=False)
    with open(csv_file.replace('.csv', '.pickle'), 'wb') as f:
        pickle.dump(df, f)


def gen_expected_results():
    """Helper to generate expected results. Uncomment in the "main"
    section. This function is a bit gross and not particularly
    maintainable, it'll do.
    """
    s1 = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID_8500)

    # Create list of lists to run everything. The third tuple element
    # is used to truncate DataFrames (no reason to save 4000 entries
    # to file for testing)
    a1 = [
        (s1.query_capacitors, CAPACITORS_8500),
        (s1.query_regulators, REGULATORS_8500),
        (s1.query_rtc_measurements, REG_MEAS_8500),
        (s1.query_capacitor_measurements, CAP_MEAS_8500),
        (s1.query_load_measurements, LOAD_MEAS_8500),
        (s1.query_substation_source, SUBSTATION_8500),
        (s1.query_switches, SWITCHES_8500),
        # The v2019.06.0 version of the platform does not have discrete
        # position measurements.
        # (s1.query_switch_measurements, SWITCH_MEAS)
    ]

    s2 = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID_13)
    a2 = [
        (s2.query_capacitors, CAPACITORS_13),
        (s2.query_regulators, REGULATORS_13),
        (s2.query_rtc_measurements, REG_MEAS_13),
        (s2.query_capacitor_measurements, CAP_MEAS_13),
        (s2.query_load_measurements, LOAD_MEAS_13),
        (s2.query_substation_source, SUBSTATION_13),
        (s2.query_switches, SWITCHES_13),
        # The v2019.06.0 version of the platform does not have discrete
        # position measurements.
        # (s2.query_switch_measurements, SWITCH_MEAS_13)
    ]

    s3 = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID_123)
    a3 = [
        (s3.query_capacitors, CAPACITORS_123),
        (s3.query_regulators, REGULATORS_123),
        (s3.query_rtc_measurements, REG_MEAS_123),
        (s3.query_capacitor_measurements, CAP_MEAS_123),
        (s3.query_load_measurements, LOAD_MEAS_123),
        (s3.query_substation_source, SUBSTATION_123),
        (s3.query_switches, SWITCHES_123),
        # The v2019.06.0 version of the platform does not have discrete
        # position measurements.
        # (s3.query_switch_measurements, SWITCH_MEAS_123)
    ]

    for a in [a1, a2, a3]:
        for b in a:
            # Run function.
            actual_full = b[0]()

            # Truncate if necessary.
            try:
                actual = actual_full.iloc[0:b[2]]
            except IndexError:
                actual = actual_full

            # If changing column names, etc, you'll need to write to
            # file here. Otherwise, to just update MRIDs the file gets
            # written at the end of the loop.
            to_file(actual, b[1])

            # Read file.
            expected = read_pickle(b[1])

            # Ensure frames match except for MRIDs.
            ensure_frame_equal_except_mrid(actual, expected)

            # Write new file.
            # to_csv(actual, b[1])


def ensure_frame_equal_except_mrid(left, right):
    """Helper to ensure DataFrames are equal except for MRIDs.

    When the platform blazegraph docker container gets updated, the
    MRIDs of things can change.
    """
    # Filter columns.
    left_non_id_cols, left_id_cols = get_non_id_cols(left)
    right_non_id_cols, right_id_cols = get_non_id_cols(right)

    # Ensure all the "non id" columns match exactly.
    pd.testing.assert_frame_equal(left[left_non_id_cols],
                                  right[right_non_id_cols])

    # Ensure everything else is a properly formatted UUID (MRID in the
    # CIM).
    for df in [left[left_id_cols], right[right_id_cols]]:
        for c in list(df.columns):
            for v in df[c].values:
                # GridAPPS-D prefixes all of the UUIDs with an
                # underscore.
                assert v[0] == '_'
                UUID(v[1:])


def get_non_id_cols(df):
    """Assume all MRID related columns end with 'id.' This is a bit
    fragile.
    """
    non_id_cols = []
    id_cols = []
    for c in list(df.columns.values):
        if c.endswith('id'):
            id_cols.append(c)
        else:
            non_id_cols.append(c)

    return non_id_cols, id_cols


if __name__ == '__main__':
    gen_expected_results()
