"""Module for managing testing data files.

TODO: There's a lot of nasty hard-coding in here (copy + paste type
    stuff) that could be cleaned up.
"""
# Standard library.
import os
import pickle
from uuid import UUID
from datetime import datetime
import time
from unittest.mock import patch

# PyVVO.
from pyvvo import sparql, gridappsd_platform, timeseries, load_model

# Third-party.
import numpy as np
import pandas as pd
import simplejson as json
from gridappsd import topics

# Hard-code 8500 node feeder MRID.
FEEDER_MRID_8500 = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3'

# 13 node:
FEEDER_MRID_13 = '_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62'

# 123 node:
FEEDER_MRID_123 = '_C1C3E687-6FFD-C753-582B-632A27E28507'

# Modified 8500 node.
FEEDER_MRID_9500 = '_AAE94E4A-2465-6F5E-37B1-3E72183A4E44'

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
LOAD_NOM_V_8500 = os.path.join(DATA_DIR,
                               'query_load_nominal_voltage_8500.csv')
LOAD_MEAS_8500 = os.path.join(DATA_DIR, 'query_load_measurements_8500.csv')
SUBSTATION_8500 = os.path.join(DATA_DIR, 'query_substation_source_8500.csv')
SWITCHES_8500 = os.path.join(DATA_DIR, 'query_switches_8500.csv')
SWITCH_MEAS_8500 = os.path.join(DATA_DIR, 'query_switch_meas_8500.csv')

# 13 bus outputs for smaller/quicker/more manual tests.
CAPACITORS_13 = os.path.join(DATA_DIR, 'query_capacitors_13.csv')
REGULATORS_13 = os.path.join(DATA_DIR, 'query_regulators_13.csv')
REG_MEAS_13 = os.path.join(DATA_DIR, 'query_reg_meas_13.csv')
CAP_MEAS_13 = os.path.join(DATA_DIR, 'query_cap_meas_13.csv')
LOAD_NOM_V_13 = os.path.join(DATA_DIR, 'query_load_nominal_voltage_13.csv')
LOAD_MEAS_13 = os.path.join(DATA_DIR, 'query_load_measurements_13.csv')
SUBSTATION_13 = os.path.join(DATA_DIR, 'query_substation_source_13.csv')
SWITCHES_13 = os.path.join(DATA_DIR, 'query_switches_13.csv')
SWITCH_MEAS_13 = os.path.join(DATA_DIR, 'query_switch_meas_13.csv')

# 123 bus for "medium" tests
CAPACITORS_123 = os.path.join(DATA_DIR, 'query_capacitors_123.csv')
REGULATORS_123 = os.path.join(DATA_DIR, 'query_regulators_123.csv')
REG_MEAS_123 = os.path.join(DATA_DIR, 'query_reg_meas_123.csv')
CAP_MEAS_123 = os.path.join(DATA_DIR, 'query_cap_meas_123.csv')
LOAD_NOM_V_123 = os.path.join(DATA_DIR, 'query_load_nominal_voltage_123.csv')
LOAD_MEAS_123 = os.path.join(DATA_DIR, 'query_load_measurements_123.csv')
SUBSTATION_123 = os.path.join(DATA_DIR, 'query_substation_source_123.csv')
SWITCHES_123 = os.path.join(DATA_DIR, 'query_switches_123.csv')
SWITCH_MEAS_123 = os.path.join(DATA_DIR, 'query_switch_meas_123.csv')

# New model.
CAPACITORS_9500 = os.path.join(DATA_DIR, 'query_capacitors_9500.csv')
REGULATORS_9500 = os.path.join(DATA_DIR, 'query_regulators_9500.csv')
REG_MEAS_9500 = os.path.join(DATA_DIR, 'query_reg_meas_9500.csv')
CAP_MEAS_9500 = os.path.join(DATA_DIR, 'query_cap_meas_9500.csv')
LOAD_NOM_V_9500 = os.path.join(DATA_DIR,
                               'query_load_nominal_voltage_9500.csv')
LOAD_MEAS_9500 = os.path.join(DATA_DIR, 'query_load_measurements_9500.csv')
SUBSTATION_9500 = os.path.join(DATA_DIR, 'query_substation_source_9500.csv')
SWITCHES_9500 = os.path.join(DATA_DIR, 'query_switches_9500.csv')
SWITCH_MEAS_9500 = os.path.join(DATA_DIR, 'query_switch_meas_9500.csv')
INVERTERS_9500 = os.path.join(DATA_DIR, 'query_inverters_9500.csv')
INVERTER_MEAS_9500 = os.path.join(DATA_DIR,
                                  'query_inverter_measurements_9500.csv')

# Misc json files.
REG_MEAS_MSG_9500 = os.path.join(DATA_DIR, 'reg_meas_message_9500.json')
CAP_MEAS_MSG_9500 = os.path.join(DATA_DIR, 'cap_meas_message_9500.json')
SWITCH_MEAS_MSG_9500 = os.path.join(DATA_DIR, 'switch_meas_message_9500.json')

MEASUREMENTS_13 = os.path.join(DATA_DIR, 'simulation_measurements_13.json')
HEADER_13 = os.path.join(DATA_DIR, 'simulation_measurements_header_13.json')

MODEL_INFO = os.path.join(DATA_DIR, 'query_model_info.json')

E_CONS_MEAS_9500 =\
    os.path.join(DATA_DIR, 'energy_consumer_measurements_9500.json')
ALL_MEAS_13 = os.path.join(DATA_DIR, 'all_measurements_13.json')

SENSOR_MEASUREMENT_TIME_START = datetime(2013, 1, 14, 0, 0)
SENSOR_MEASUREMENT_TIME_END = datetime(2013, 1, 14, 0, 5)

SENSOR_MEASUREMENT_BASE = 'sensor_measurements_9500'
SENSOR_MEAS_9500_0 = \
    os.path.join(DATA_DIR, SENSOR_MEASUREMENT_BASE + '_0.json')
SENSOR_MEAS_9500_1 = \
    os.path.join(DATA_DIR, SENSOR_MEASUREMENT_BASE + '_1.json')
SENSOR_MEAS_9500_2 = \
    os.path.join(DATA_DIR, SENSOR_MEASUREMENT_BASE + '_2.json')
SENSOR_MEAS_9500_3 = \
    os.path.join(DATA_DIR, SENSOR_MEASUREMENT_BASE + '_3.json')
SENSOR_MEAS_LIST = [SENSOR_MEAS_9500_0, SENSOR_MEAS_9500_1, SENSOR_MEAS_9500_2,
                    SENSOR_MEAS_9500_3]

PARSED_SENSOR_BASE = 'parsed_sensor_measurements_9500'
PARSED_SENSOR_9500_0 = \
    os.path.join(DATA_DIR, PARSED_SENSOR_BASE + '_0.csv')
PARSED_SENSOR_9500_1 = \
    os.path.join(DATA_DIR, PARSED_SENSOR_BASE + '_1.csv')
PARSED_SENSOR_9500_2 = \
    os.path.join(DATA_DIR, PARSED_SENSOR_BASE + '_2.csv')
PARSED_SENSOR_9500_3 = \
    os.path.join(DATA_DIR, PARSED_SENSOR_BASE + '_3.csv')
PARSED_SENSOR_LIST = [PARSED_SENSOR_9500_0, PARSED_SENSOR_9500_1,
                      PARSED_SENSOR_9500_2, PARSED_SENSOR_9500_3]

PARSED_SENSOR_VPQ = os.path.join(DATA_DIR, 'parsed_sensor_vpq_9500.csv')

WEATHER_FOR_SENSOR_DATA_9500_JSON = \
    os.path.join(DATA_DIR, 'weather_for_sensors_data_9500.json')
WEATHER_FOR_SENSOR_DATA_9500 = \
    os.path.join(DATA_DIR, 'weather_for_sensors_data_9500.csv')

# The two-week weather is parsed and re-sampled into 15 minute
# intervals.
WEATHER_TWO_WEEK = os.path.join(DATA_DIR, 'weather_two_week.csv')
WEATHER_TWO_WEEK_START = datetime(2013, 7, 14, 0, 0)
WEATHER_TWO_WEEK_END = datetime(2013, 7, 28, 0, 0)

# Simple, single entry weather file.
WEATHER_SIMPLE_JSON = os.path.join(DATA_DIR, 'weather_simple.json')
WEATHER_SIMPLE_START = datetime(2013, 1, 1, 6)
WEATHER_SIMPLE_END = WEATHER_SIMPLE_START


def read_pickle(csv_file):
    """Helper to read a pickle file corresponding to a csv file."""
    with open(csv_file.replace('.csv', '.pickle'), 'rb') as f:
        p = pickle.load(f)

    return p


def to_file(df, csv_file, index=False):
    """Helper to write a DataFrame both to csv and pickle."""
    df.to_csv(csv_file, index=index)
    with open(csv_file.replace('.csv', '.pickle'), 'wb') as f:
        pickle.dump(df, f)


# noinspection DuplicatedCode
def gen_expected_sparql_results():
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
        (s1.query_load_nominal_voltage, LOAD_NOM_V_8500),
        (s1.query_rtc_measurements, REG_MEAS_8500),
        (s1.query_capacitor_measurements, CAP_MEAS_8500),
        (s1.query_load_measurements, LOAD_MEAS_8500),
        (s1.query_substation_source, SUBSTATION_8500),
        (s1.query_switches, SWITCHES_8500),
        (s1.query_switch_measurements, SWITCH_MEAS_8500)
    ]

    s2 = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID_13)
    a2 = [
        (s2.query_capacitors, CAPACITORS_13),
        (s2.query_regulators, REGULATORS_13),
        (s2.query_rtc_measurements, REG_MEAS_13),
        (s2.query_capacitor_measurements, CAP_MEAS_13),
        (s2.query_load_nominal_voltage, LOAD_NOM_V_13),
        (s2.query_load_measurements, LOAD_MEAS_13),
        (s2.query_substation_source, SUBSTATION_13),
        (s2.query_switches, SWITCHES_13),
        (s2.query_switch_measurements, SWITCH_MEAS_13)
    ]

    s3 = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID_123)
    a3 = [
        (s3.query_capacitors, CAPACITORS_123),
        (s3.query_regulators, REGULATORS_123),
        (s3.query_rtc_measurements, REG_MEAS_123),
        (s3.query_capacitor_measurements, CAP_MEAS_123),
        (s3.query_load_nominal_voltage, LOAD_NOM_V_123),
        (s3.query_load_measurements, LOAD_MEAS_123),
        (s3.query_substation_source, SUBSTATION_123),
        (s3.query_switches, SWITCHES_123),
        (s3.query_switch_measurements, SWITCH_MEAS_123)
    ]

    s4 = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID_9500)
    a4 = [
        (s4.query_capacitors, CAPACITORS_9500),
        (s4.query_regulators, REGULATORS_9500),
        (s4.query_rtc_measurements, REG_MEAS_9500),
        (s4.query_capacitor_measurements, CAP_MEAS_9500),
        (s4.query_load_nominal_voltage, LOAD_NOM_V_9500),
        (s4.query_load_measurements, LOAD_MEAS_9500),
        (s4.query_substation_source, SUBSTATION_9500),
        (s4.query_switches, SWITCHES_9500),
        (s4.query_switch_measurements, SWITCH_MEAS_9500),
        (s4.query_inverters, INVERTERS_9500),
        (s4.query_inverter_measurements, INVERTER_MEAS_9500)
    ]

    for a in [a1, a2, a3, a4]:
        for b in a:
            # Run function.
            actual_full = b[0]()

            # Truncate if necessary.
            try:
                actual = actual_full.iloc[0:b[2]]
            except IndexError:
                actual = actual_full

            # If changing column names, creating a new file, etc.,
            # you'll need to write to file here. Otherwise, to just
            # update MRIDs the file gets written at the end of the loop.
            # to_file(actual, b[1])

            # Read file.
            expected = read_pickle(b[1])

            # Ensure frames match except for MRIDs.
            ensure_frame_equal_except_mrid(actual, expected)

            # Write new file.
            to_file(actual, b[1])


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
                # For the inverters query, there are optional "phase"
                # objects which will have a NaN MRID if the "phase"
                # object is not present. So, we'll simply continue here
                # if a NaN pops up.
                # Hopefully this doesn't bite me later...
                if not isinstance(v, str):
                    assert np.isnan(v)
                    continue

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


def generate_all_measurements_13():
    """Generate the files ALL_MEAS_13, MEASUREMENTS_13, and HEADER_13.
    Note that data for ALL_MEAS_13 comes from the time series database,
    while MEASUREMENTS_13 and HEADER_13 come straight from the platform.
    """
    # Define a helper for writing our measurements and headers to file.
    def meas_header_to_file(header, message):
        _dict_to_json(data=header, fname=HEADER_13)
        _dict_to_json(data=message, fname=MEASUREMENTS_13)

    # Get a platform manager, start the simulation.
    platform = gridappsd_platform.PlatformManager()
    starttime = datetime(2013, 1, 14, 0, 0)
    sim_id = platform.run_simulation(feeder_id=FEEDER_MRID_13,
                                     start_time=starttime,
                                     duration=20, realtime=False)

    # Subscribe to simulation output so we can write the header +
    # message to file. Note this is not particularly efficient as the
    # files will be overwritten a few times before the simulation ends.
    # Who cares? Not me :) Just gotta make rapid progress.
    platform.gad.subscribe(topic=topics.simulation_output_topic(sim_id),
                           callback=meas_header_to_file)

    # Wait for simulation completion.
    platform.wait_for_simulation()

    # Get the measurements.
    # noinspection PyProtectedMember
    data = platform._query_simulation_output(simulation_id=sim_id)

    # Write to file.
    _dict_to_json(fname=ALL_MEAS_13,
                  data=data)


def generate_energy_consumer_measurements_9500():
    """Generate energy_consumer_measurements_9500.json"""
    platform = gridappsd_platform.PlatformManager()
    starttime = datetime(2013, 1, 14, 0, 0)
    sim_id = platform.run_simulation(feeder_id=FEEDER_MRID_9500,
                                     start_time=starttime,
                                     duration=20, realtime=False)

    # Get load measurement data. Save time by reading the csv.
    load_meas = pd.read_csv(LOAD_MEAS_9500)
    # Extract the first entry.
    mrid = load_meas.iloc[0]['id']

    # Wait for simulation completion.
    platform.wait_for_simulation()

    # TODO: Remove this time.sleep when
    #  https://github.com/GRIDAPPSD/gridappsd-forum/issues/24#issue-487936782
    #  has been addressed.
    time.sleep(10)

    # Get the measurements.
    # noinspection PyProtectedMember
    data = platform._query_simulation_output(simulation_id=sim_id,
                                             measurement_mrid=mrid)

    # Write to file.
    _dict_to_json(fname=E_CONS_MEAS_9500,
                  data=data)


def _dict_to_json(data, fname, sim_dt=None):
    """Helper to dump a dictionary to file. the sim_dt argument is there
     because we use this function with the SimOutRouter.
     """
    with open(os.path.join(DATA_DIR, fname), 'w') as f:
        json.dump(data, f, indent=2)


def generate_cap_reg_switch_meas_message_9500():
    """Generate cap_meas_message_9500.json and reg_meas_message_9500.json
    """

    # Load up the capacitor data.
    caps = pd.read_csv(CAP_MEAS_9500)
    cap_mrids = caps['state_meas_mrid'].tolist()
    # Load up regulator data.
    regs = pd.read_csv(REG_MEAS_9500)
    reg_mrids = regs['pos_meas_mrid'].tolist()
    # Load up switch data.
    switches = pd.read_csv(SWITCH_MEAS_9500)
    switch_mrids = switches['state_meas_mrid'].tolist()

    # Initialize fn_mrid_list for a SimOutRouter.
    fn_mrid_list = [{'functions': _dict_to_json, 'mrids': cap_mrids,
                     'kwargs': {'fname': CAP_MEAS_MSG_9500}},
                    {'functions': _dict_to_json, 'mrids': reg_mrids,
                     'kwargs': {'fname': REG_MEAS_MSG_9500}},
                    {'functions': _dict_to_json, 'mrids': switch_mrids,
                     'kwargs': {'fname': SWITCH_MEAS_MSG_9500}}
                    ]

    platform = gridappsd_platform.PlatformManager()
    starttime = datetime(2013, 1, 14, 0, 0)
    sim_id = platform.run_simulation(feeder_id=FEEDER_MRID_9500,
                                     start_time=starttime,
                                     duration=5, realtime=False)

    # Create a SimOutRouter to save the measurements.
    # noinspection PyUnusedLocal
    router = gridappsd_platform.SimOutRouter(platform_manager=platform,
                                             sim_id=sim_id,
                                             fn_mrid_list=fn_mrid_list)

    # Wait for simulation completion.
    platform.wait_for_simulation()


def generate_model_info():
    """Generate 'query_model_info.json.
    """
    platform = gridappsd_platform.PlatformManager()

    info = platform.gad.query_model_info()

    _dict_to_json(data=info, fname=MODEL_INFO)


def _get_9500_meas_data_for_one_node():
    """Helper to get all the measurement MRIDs for a single node in the
    9500 node model.
    """
    # Get load measurement data. Save time by reading the csv.
    load_meas = pd.read_csv(LOAD_MEAS_9500)
    # Get all the measurements associated with a single node.
    node = load_meas.loc[0, 'node']
    node_mask = load_meas['node'] == node
    node_rows = load_meas[node_mask]
    assert node_rows.shape[0] == 4

    return node_rows


def generate_sensor_service_measurements_9500():
    """NOTE: THIS ONE WON'T JUST WORK OUT OF THE BOX, SINCE EXTENSIVE
    PLATFORM CONFIGURATION IS NECESSARY.

    To get it to work, you must ensure the sample application has the
    "gridappsd-sensor-simulator" listed in its "prereqs" field.

    This method is also going to produce different results depending
    on how the sensor service is configured.
    """
    meas_data = _get_9500_meas_data_for_one_node()

    platform = gridappsd_platform.PlatformManager()
    time_diff = SENSOR_MEASUREMENT_TIME_END - SENSOR_MEASUREMENT_TIME_START
    # TODO: Use houses when
    #   https://github.com/GRIDAPPSD/gridappsd-forum/issues/26#issue-487939149
    #   is resolved.
    sim_id = platform.run_simulation(feeder_id=FEEDER_MRID_9500,
                                     start_time=SENSOR_MEASUREMENT_TIME_START,
                                     duration=time_diff.seconds,
                                     realtime=False,
                                     applications=[{'name': 'pyvvo'}],
                                     random_zip=False, houses=False)

    # Wait for simulation completion.
    platform.wait_for_simulation()

    # TODO: Remove this time.sleep when
    #  https://github.com/GRIDAPPSD/gridappsd-forum/issues/24#issue-487936782
    #  has been addressed.
    time.sleep(180)

    # Get output for all our MRIDs.
    for idx, meas_mrid in enumerate(meas_data['id'].values):
        # noinspection PyProtectedMember
        out = platform._query_simulation_output(
            simulation_id=sim_id, measurement_mrid=meas_mrid,
            query_measurement='gridappsd-sensor-simulator')

        # Save the output to file.
        _dict_to_json(data=out, fname=SENSOR_MEAS_LIST[idx])


def generate_parsed_sensor_service_measurements_9500():
    """Given the data generated by
    generate_sensor_service_measurements_9500, parse them and save the
    results.
    """
    for idx, file in enumerate(SENSOR_MEAS_LIST):
        with open(file, 'r') as f:
            data = json.load(f)

        parsed_data = timeseries.parse_timeseries(data)
        to_file(parsed_data, PARSED_SENSOR_LIST[idx])


def generate_vpq_for_parsed_sensor_service_measurements_9500():
    """Given the data generated by
    generate_parsed_sensor_service_measurements_9500, get and save a
    DataFrame with v, p, and q values.
    """
    # Read files.
    data = []
    for file in PARSED_SENSOR_LIST:
        data.append(read_pickle(file))

    # Load up related measurement data.
    meas_data = _get_9500_meas_data_for_one_node()
    meas_data.rename(columns={'id': 'meas_mrid', 'type': 'meas_type'},
                     inplace=True)

    # We need to do some patching to make this work without needing to
    # call the platform.
    s = 'pyvvo.gridappsd_platform.PlatformManager.get_simulation_output'
    with patch(s, side_effect=data):
        out = load_model.get_data_for_load(
            sim_id='1', meas_data=meas_data)

    to_file(out, PARSED_SENSOR_VPQ)


def generate_weather_for_sensor_data_9500():
    """Get weather data that lines up with the sensor data.
    """
    p = gridappsd_platform.PlatformManager()
    # Start with the json file.
    # noinspection PyProtectedMember
    j = p._query_weather(start_time=SENSOR_MEASUREMENT_TIME_START,
                         end_time=SENSOR_MEASUREMENT_TIME_END)
    _dict_to_json(j, WEATHER_FOR_SENSOR_DATA_9500_JSON)
    d = p.get_weather(start_time=SENSOR_MEASUREMENT_TIME_START,
                      end_time=SENSOR_MEASUREMENT_TIME_END)
    to_file(d, WEATHER_FOR_SENSOR_DATA_9500)


def generate_weather_two_week():
    """Generate two weeks worth of data in 15 minute intervals."""
    p = gridappsd_platform.PlatformManager()
    d = p.get_weather(start_time=WEATHER_TWO_WEEK_START,
                      end_time=WEATHER_TWO_WEEK_END)
    to_file(d.resample('15Min', closed='right', label='right').mean(),
            WEATHER_TWO_WEEK, index=True)


def generate_weather_simple():
    """Generate simple single-entry weather data."""
    p = gridappsd_platform.PlatformManager()
    # Start with the json file.
    # noinspection PyProtectedMember
    j = p._query_weather(start_time=WEATHER_SIMPLE_START,
                         end_time=WEATHER_SIMPLE_END)
    _dict_to_json(j, WEATHER_SIMPLE_JSON)


if __name__ == '__main__':
    gen_expected_sparql_results()
    generate_all_measurements_13()
    generate_energy_consumer_measurements_9500()
    generate_cap_reg_switch_meas_message_9500()
    generate_model_info()
    # TODO: Run these after talking to Poorva.
    generate_sensor_service_measurements_9500()
    generate_parsed_sensor_service_measurements_9500()
    generate_vpq_for_parsed_sensor_service_measurements_9500()
    generate_weather_for_sensor_data_9500()
    # RUN TO HERE
    generate_weather_two_week()
    generate_weather_simple()

    print("All done. Don't forget to update file permissions:")
    print("chown -R thay838:thay838 ~/git/pyvvo")
