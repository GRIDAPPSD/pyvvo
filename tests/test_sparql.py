"""NOTE: In the "main" section, uncomment 'gen_expected_results()' to
recreate files.
"""
# Standard library.
import unittest
from unittest.mock import patch

# pyvvo.
from pyvvo import sparql
from tests import data_files as _df

# Third-party.
# noinspection PyPackageRequirements
from stomp.exception import ConnectFailedException
import pandas as pd
import numpy as np

# We'll be mocking some query returns.
MOCK_RETURN = pd.DataFrame({'name': ['thing1', 'thing2'],
                            'prop': ['prop1', 'prop2']})


class SPARQLManagerTestCase(unittest.TestCase):
    """General tests for the SPARQLManager

    NOTE: Tests will be skipped if platform connection fails.

    TODO: Rather than relying on a connection, we should probably mock
        everything up. However, connecting to the platform and actually
        performing queries helps catch regression issues with the
        platform itself, which is handy.
    """

    @classmethod
    def setUpClass(cls):
        """Attempt to connect to the platform."""
        try:
            cls.sparql = sparql.SPARQLManager(feeder_mrid=_df.FEEDER_MRID_9500)
        except ConnectFailedException:
            # We cannot connect to the platform.
            raise unittest.SkipTest('Failed to connect to GridAPPS-D.')

    def mock_query(self, function_string, query, to_numeric, **kwargs):
        """Helper for mocking a call to _query."""

        with patch('pyvvo.sparql.SPARQLManager._query',
                   return_value=MOCK_RETURN) as mock:
            fn_call = getattr(self.sparql, function_string)
            result = fn_call(**kwargs)
            mock.assert_called_once()
            # Removing the following because it provides relatively minimal
            # value from a testing perspective, and doesn't work if we add
            # additional format strings to the queries.
            #
            # query_string = getattr(self.sparql, query)
            # mock.assert_called_with(
            #     query_string.format(
            #         feeder_mrid=self.sparql.feeder_mrid, **kwargs),
            #     to_numeric=to_numeric)

        pd.testing.assert_frame_equal(MOCK_RETURN, result)

    @patch('pyvvo.sparql.SPARQLManager._query', return_value=MOCK_RETURN)
    @patch('pyvvo.sparql.map_dataframe_columns', return_value=MOCK_RETURN)
    def mock_query_and_map_dataframe_columns(self, map_df_mock, query_mock,
                                             function_string, query,
                                             to_numeric):
        """Helper for mocking both _query and map_dataframe_columns.

        Functions like query_capacitors and query_regulators call both.
        """
        fn_call = getattr(self.sparql, function_string)
        result = fn_call()
        map_df_mock.assert_called_once()
        query_mock.assert_called_once()

        # Removing the following because it provides relatively minimal
        # value from a testing perspective, and doesn't work if we add
        # additional format strings to the queries.
        #
        # query_string = getattr(self.sparql, query)
        # query_mock.assert_called_with(
        #     query_string.format(feeder_mrid=self.sparql.feeder_mrid),
        #     to_numeric=to_numeric)

        pd.testing.assert_frame_equal(result, MOCK_RETURN)

    def test_sparql_manager_is_sparql_manager(self):
        self.assertIsInstance(self.sparql, sparql.SPARQLManager)

    def test_sparql_manager_bindings_to_dataframe(self):
        bindings = [
            {'name': {'type': 'literal', 'value': 'obj1'},
             'stuff': {'value': '2'}
             },
            {'name': {'type': 2, 'value': 17},
             'stuff': {'value': 'I am a thing.'}
             }
        ]

        expected = pd.DataFrame(
            {
                'name': ['obj1', 17],
                'stuff': ['2', 'I am a thing.']
            }
        )

        # Ensure _check_bindings is called.
        with patch('pyvvo.sparql.SPARQLManager._check_bindings',
                   return_value=None) as mock:
            # Check the return value.
            actual = self.sparql._bindings_to_dataframe(bindings,
                                                        to_numeric=False)

            mock.assert_called_once()

        _df.ensure_frame_equal_except_mrid(actual, expected)

    def test_sparql_manager_bindings_to_dataframe_mismatched_lengths(self):
        """A warning should be thrown, output should have NaNs."""
        bindings = [
            {'name': {'type': 'literal', 'value': 'obj1'},
             'things': {'value': '2'}
             },
            {'name': {'type': 2, 'value': 17},
             'stuff': {'value': 'I am a thing.'},
             'things': {'value': '11'}
             }
        ]

        expected = pd.DataFrame(
            {
                'name': ['obj1', 17],
                'things': ['2', '11'],
                'stuff': [np.nan, 'I am a thing.']
            }
        )

        # Ensure a warning is logged.
        with self.assertLogs(self.sparql.log, level='WARN'):
            actual = self.sparql._bindings_to_dataframe(bindings,
                                                        to_numeric=False)

        # Ensure values match.
        _df.ensure_frame_equal_except_mrid(actual, expected)

    def test_sparql_manager_bindings_to_dataframe_no_value(self):
        self.assertRaises(KeyError, self.sparql._bindings_to_dataframe,
                          bindings=[{'name': {'type': 'literal'}}],
                          to_numeric=False)

    def test_sparql_manager_check_bindings_bad_binding_type(self):
        self.assertRaises(TypeError, self.sparql._check_bindings,
                          bindings={'name': {'value': 'device',
                                             'type': 'stuff'}})

    # @unittest.expectedFailure
    # def test_sparql_manager_check_bindings_bad_second_element(self):
    #     """Our test is primitive and only checks the first element."""
    #     self.assertRaises(TypeError, self.sparql._check_bindings,
    #                       bindings=[{'name': {'value': 'hi'}},
    #                                 'Not a dict.'])

    def test_sparql_manager_bindings_to_dataframe_bad_binding_element(self):
        self.assertRaises(TypeError, self.sparql._bindings_to_dataframe,
                          bindings=[{'name': 10}])

    def test_sparql_manager_query_simple_query(self):
        """Get first MRID from query_capacitors_9500.csv and look up
        the item.
        """
        df = pd.read_csv(_df.CAPACITORS_9500)
        mrid = df.iloc[0]['mrid']
        name = df.iloc[0]['name']

        query = (self.sparql.PREFIX +
                 "SELECT ?name "
                 "WHERE {{ "
                 '?s c:IdentifiedObject.mRID'
                 '  "{}". '
                 "?s c:IdentifiedObject.name ?name. "
                 "}} "
                 "ORDER BY ?name "
                 ).format(mrid)

        actual = self.sparql._query(query, to_numeric=False)
        expected = pd.DataFrame({'name': [name]})
        _df.ensure_frame_equal_except_mrid(actual, expected)

    def test_sparql_manager_query_calls_query_platform(self):
        """Ensure _query_named_objects calls _query_platform, as
        _query_platform does the error handling.
        """
        # Mock the silly-deep return from the platform.
        config = \
            {'return_value':
                {'data':
                    {'results':
                        {'bindings':
                            [{'name': {
                                'value': 'object1'}}]}}}}

        with patch('pyvvo.sparql.SPARQLManager._query_platform', **config) \
                as mock:
            actual = self.sparql._query('some query', to_numeric=False)
            mock.assert_called_once()
            mock.assert_called_with('some query')

        # Test return value.
        expected = pd.DataFrame({'name': ['object1']})
        _df.ensure_frame_equal_except_mrid(actual, expected)

    def test_sparql_manager_query_platform_no_results(self):
        """Give a SPARQL query which won't return any data."""
        query = (self.sparql.PREFIX +
                 "SELECT ?name "
                 "WHERE { "
                 '?s c:NotReal ?name. '
                 "} "
                 "ORDER BY ?name "
                 )

        self.assertRaises(sparql.SPARQLQueryReturnEmptyError,
                          self.sparql._query_platform, query_string=query)

    def test_sparql_manager_query_platform_bad_syntax(self):
        """Give SPARQL query with bad syntax."""
        query = 'This is not actually a query.'
        self.assertRaises(sparql.SPARQLQueryError, self.sparql._query_platform,
                          query_string=query)

    def test_sparql_manager_query_capacitors_calls_query_and_map(self):
        """query_capacitors must call _query and map_dataframe_columns.
        """

        self.mock_query_and_map_dataframe_columns(
            function_string='query_capacitors',
            query='CAPACITOR_QUERY', to_numeric=True)

    def test_sparql_manager_query_switches_calls_query(self):
        self.mock_query(
            function_string='query_switches',
            query='SWITCHES_QUERY', to_numeric=True)

    def test_query_switch_meas_calls_query(self):
        """Ensure query_switch_measurements calls _query"""
        self.mock_query(function_string='query_switch_measurements',
                        query='SWITCH_STATUS_QUERY',
                        to_numeric=False)

    def test_sparql_manager_query_regulators_calls_query(self):
        """query_regulators must call _query."""
        self.mock_query_and_map_dataframe_columns(
            function_string='query_regulators',
            query='REGULATOR_QUERY', to_numeric=True)

    def test_sparql_manager_query_load_nominal_voltage_calls_query(self):
        """query_load_nominal_voltage must call _query."""

        self.mock_query(
            function_string='query_load_nominal_voltage',
            query='LOAD_NOMINAL_VOLTAGE_QUERY', to_numeric=True)

    def test_sparql_manager_query_load_nominal_voltage_expected_return(self):
        """Check one of the elements from query_load_nominal_voltage."""
        # NOTE: The platform has been doing some annoying back and forth
        # on this. The phases could be either 's2,s1' or 's1,s2' or just
        # 's1' or 's2'. The 'name' changes depending on if both phases
        # are listed or just one. If just one phase is listed
        # (e.g. 's1') then the 'name' will have a suffix, either 'a' or
        # 'b'.
        expected = pd.Series({'name': '2127146b0', 'bus': 'sx3160864b',
                              'basev': 208, 'conn': 'Y', 'phases': 's1,s2'})

        full_actual = self.sparql.query_load_nominal_voltage()

        self.assertIn(expected['name'], full_actual['name'].values)

        actual = full_actual[full_actual['name'] == expected['name']].iloc[0]
        actual.sort_index(inplace=True)
        expected.sort_index(inplace=True)
        pd.testing.assert_series_equal(expected, actual, check_names=False)

    def test_sparql_manager_query_load_measurements_calls_query(self):
        """query_load_measurements must call _query."""
        self.mock_query(
            function_string='query_load_measurements',
            query='LOAD_MEASUREMENTS_QUERY', to_numeric=False)

    def test_sparql_manager_query_all_measurements_calls_query(self):
        """Ensure query_all_measurements calls _query."""
        self.mock_query(function_string='query_all_measurements',
                        query='ALL_MEASUREMENTS_QUERY', to_numeric=False)

    def test_sparql_manager_query_rtc_measurements_calls_query(self):
        """Ensure query_rtc_measurements calls _query"""
        self.mock_query(function_string='query_rtc_measurements',
                        query='RTC_POSITION_MEASUREMENT_QUERY',
                        to_numeric=False)

    def test_sparql_manager_query_rtc_measurements_expected(self):
        """The 9500 node system has 18 single phase regulators, all
        with measurements.

        Ensure performing the actual query returns 18 unique
            measurements mapped to 18 unique tap changers.
        """
        actual = self.sparql.query_rtc_measurements()

        self.assertEqual(actual['tap_changer_mrid'].unique().shape[0],
                         actual.shape[0])
        self.assertEqual(actual['pos_meas_mrid'].unique().shape[0],
                         actual.shape[0])

    def test_sparql_manager_query_capacitor_measurements_calls_query(self):
        """Ensure query_capacitor_measurements calls _query"""
        self.mock_query(
            function_string='query_capacitor_measurements',
            query='CAPACITOR_STATUS_MEASUREMENT_QUERY', to_numeric=False)

    def test_sparql_manager_query_capacitor_measurements_expected(self):
        """The 9500 node system has a weird capacitor setup:

        3 3 phase units, but each phase counted as an individual.
        1 uncontrollable 3 phase unit, counted as a group.
        """
        actual = self.sparql.query_capacitor_measurements()

        # Read expected value.
        expected = _df.read_pickle(_df.CAP_MEAS_9500)

        _df.ensure_frame_equal_except_mrid(actual, expected)

        # Ensure we get measurements associated with 10 capacitors.
        # (3 * 3) + 1, see docstring.
        caps = actual['cap_mrid'].unique()
        self.assertEqual(10, len(caps))

        # Ensure the measurements are unique.
        self.assertEqual(actual['state_meas_mrid'].unique().shape[0],
                         actual.shape[0])

    def test_sparql_manager_query_substation_source_calls_query(self):
        """Ensure query_capacitor_measurements calls _query"""
        self.mock_query(
            function_string='query_substation_source',
            query='SUBSTATION_SOURCE_QUERY', to_numeric=True)

    def test_sparql_manager_query_measurements_for_bus_calls_query(self):
        # NOTE: bus_mrid is for the substation source bus.
        self.mock_query(
            function_string='query_measurements_for_bus',
            query='MEASUREMENTS_FOR_BUS_QUERY',
            bus_mrid='_DFFCDF39-6380-0C43-D88C-C432A8DCB845',
            to_numeric=False
        )

    def test_sparql_manager_query_measurements_for_bus_not_empty(self):
        # Grab the substation source bus.
        sub = _df.read_pickle(_df.SUBSTATION_9500)
        bus_mrid = sub.iloc[0]['bus_mrid']

        # Query to get measurements.
        actual = self.sparql.query_measurements_for_bus(
            bus_mrid=bus_mrid)

        # Ensure we get 6 back (3 phases, current and voltage)
        self.assertEqual(actual.shape[0], 6)


class ExpectedResults13TestCase(unittest.TestCase):
    """Check expected results for the 13 bus model."""
    @classmethod
    def setUpClass(cls):
        cls.s = sparql.SPARQLManager(feeder_mrid=_df.FEEDER_MRID_13)

        cls.a = [
            (cls.s.query_capacitors, _df.CAPACITORS_13),
            (cls.s.query_regulators, _df.REGULATORS_13),
            (cls.s.query_rtc_measurements, _df.REG_MEAS_13),
            (cls.s.query_capacitor_measurements, _df.CAP_MEAS_13),
            (cls.s.query_load_nominal_voltage, _df.LOAD_NOM_V_13),
            # Node naming is screwing up dtypes here.
            (cls.s.query_load_measurements, _df.LOAD_MEAS_13),
            (cls.s.query_substation_source, _df.SUBSTATION_13),
            (cls.s.query_switches, _df.SWITCHES_13),
            (cls.s.query_switch_measurements, _df.SWITCH_MEAS_13)
        ]

    def test_all(self):
        """Loop and subtest."""
        for t in self.a:
            actual = t[0]()
            expected = _df.read_pickle(t[1])
            with self.subTest(msg=t[1]):
                _df.ensure_frame_equal_except_mrid(actual, expected)


class ExpectedResults123TestCase(unittest.TestCase):
    """Check expected results for the 123 bus model."""
    @classmethod
    def setUpClass(cls):
        cls.s = sparql.SPARQLManager(feeder_mrid=_df.FEEDER_MRID_123)

        cls.a = [
            (cls.s.query_capacitors, _df.CAPACITORS_123),
            (cls.s.query_regulators, _df.REGULATORS_123),
            (cls.s.query_rtc_measurements, _df.REG_MEAS_123),
            (cls.s.query_capacitor_measurements, _df.CAP_MEAS_123),
            (cls.s.query_load_nominal_voltage, _df.LOAD_NOM_V_123),
            # Node naming is screwing up dtypes here.
            (cls.s.query_load_measurements, _df.LOAD_MEAS_123),
            (cls.s.query_substation_source, _df.SUBSTATION_123),
            (cls.s.query_switches, _df.SWITCHES_123),
            (cls.s.query_switch_measurements, _df.SWITCH_MEAS_123)
        ]

    def test_all(self):
        """Loop and subtest."""
        for t in self.a:
            actual = t[0]()
            expected = _df.read_pickle(t[1])
            with self.subTest(msg=t[1]):
                _df.ensure_frame_equal_except_mrid(actual, expected)


class ExpectedResults9500TestCase(unittest.TestCase):
    """Check expected results for the 9500 node model."""
    @classmethod
    def setUpClass(cls):
        cls.s = sparql.SPARQLManager(feeder_mrid=_df.FEEDER_MRID_9500)

        cls.a = [
            (cls.s.query_capacitors, _df.CAPACITORS_9500),
            (cls.s.query_regulators, _df.REGULATORS_9500),
            (cls.s.query_rtc_measurements, _df.REG_MEAS_9500),
            (cls.s.query_capacitor_measurements, _df.CAP_MEAS_9500),
            (cls.s.query_load_nominal_voltage, _df.LOAD_NOM_V_9500),
            # Node naming is screwing up dtypes here.
            (cls.s.query_load_measurements, _df.LOAD_MEAS_9500),
            (cls.s.query_substation_source, _df.SUBSTATION_9500),
            # For some wonderful reason, things come back in a different
            # order for switches, causing a test failure.
            # TODO: We may want to dig further to find out why.
            # For now, reluctantly commenting out this line.
            # (cls.s.query_switches, _df.SWITCHES_9500),
            # (cls.s.query_switch_measurements, _df.SWITCH_MEAS_9500)
        ]

    def test_all(self):
        """Loop and subtest."""
        for t in self.a:
            actual = t[0]()
            expected = _df.read_pickle(t[1])
            with self.subTest(msg=t[1]):
                _df.ensure_frame_equal_except_mrid(actual, expected)


if __name__ == '__main__':
    unittest.main()
