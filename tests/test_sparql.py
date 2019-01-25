# Standard library.
import unittest
from unittest.mock import patch
import os

# pyvvo.
from pyvvo import sparql

# Third-party.
import simplejson as json
from stomp.exception import ConnectFailedException

# Hard-code 8500 node feeder MRID.
FEEDER_MRID = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3'


class SPARQLManagerTestCase(unittest.TestCase):
    """General tests for the SPARQLManager

    NOTE: Tests will be skipped if platform connection fails.

    TODO: Rather than relying on a connection, we should probably mock
        everything up. However, connecting to the platform and actually
        performing queries helps catch regression issues with the
        platform itself, which is handy.
    """

    def setUp(self):
        """Attempt to connect to the platform."""
        try:
            self.sparql = sparql.SPARQLManager(feeder_mrid=FEEDER_MRID)
        except ConnectFailedException:
            # We cannot connect to the platform.
            raise unittest.SkipTest('Failed to connect to GridAPPS-D.')

    def test_sparql_manager_is_sparql_manager(self):
        self.assertIsInstance(self.sparql, sparql.SPARQLManager)

    def test_sparql_manager_bindings_to_dict_valid_input(self):
        bindings = [
            {'name': {'type': 'literal', 'value': 'obj1'},
             'stuff': {'value': '2'}
             },
            {'things': {'type': 2, 'value': 17},
             'name': {'value': 'I am a thing.'}
             }
        ]

        expected_value = {'obj1': {'name': 'obj1', 'stuff': '2'},
                          'I am a thing.': {'things': 17,
                                            'name': 'I am a thing.'}}

        self.assertEqual(self.sparql._bindings_to_dict(bindings),
                         expected_value)

    def test_sparql_manager_bindings_to_dict_no_name(self):
        self.assertRaises(KeyError, self.sparql._bindings_to_dict,
                          bindings=[{'I have no name': 'agh'}])

    def test_sparql_manager_bindings_to_dict_no_value(self):
        self.assertRaises(KeyError, self.sparql._bindings_to_dict,
                          bindings=[{'name': {'type': 'literal'}}])

    def test_sparql_manager_bindings_to_dict_bad_binding_type(self):
        self.assertRaises(TypeError, self.sparql._bindings_to_dict,
                          bindings={'name': {'value': 'device',
                                             'type': 'stuff'}})

    def test_sparql_manager_bindings_to_dict_bad_binding_element_type(self):
        self.assertRaises(TypeError, self.sparql._bindings_to_dict,
                          bindings=[{'name': 10}])

    def test_sparql_manager_query_named_objects_simple_query(self):
        """Hard-coded MRID for 8500 node capacitor."""
        query = (self.sparql.PREFIX +
                 "SELECT ?name "
                 "WHERE { "
                 '?s c:IdentifiedObject.mRID'
                 '  "_A5866105-A527-F682-C982-69807C0E088B". '
                 "?s c:IdentifiedObject.name ?name. "
                 "} "
                 "ORDER BY ?name "
                 )
        result = self.sparql.query_named_objects(query)

        self.assertDictEqual(result, {'capbank0a': {'name': 'capbank0a'}})

    def test_sparql_manager_query_named_objects_calls_query_data(self):
        """Ensure query_named_objects calls query_data, as query_data
        does the error handling.
        """
        # Mock the silly-deep return from the platform.
        config = \
            {'return_value':
                {'data':
                    {'results':
                        {'bindings':
                            [{'name': {
                                'value': 'object1'}}]}}}}

        with patch('pyvvo.sparql.SPARQLManager.query_data', **config) as mock:
            result = self.sparql.query_named_objects('some query')
            mock.assert_called_once()
            mock.assert_called_with('some query')
            self.assertDictEqual(result, {'object1': {'name': 'object1'}})

    def test_sparql_manager_query_data_no_results(self):
        """Give a SPARQL query which won't return any data."""
        query = (self.sparql.PREFIX +
                 "SELECT ?name "
                 "WHERE { "
                 '?s c:NotReal ?name. '
                 "} "
                 "ORDER BY ?name "
                 )

        self.assertRaises(sparql.SPARQLQueryReturnEmptyError,
                          self.sparql.query_data, query_string=query)

    def test_sparql_manager_query_data_bad_syntax(self):
        """Give SPARQL query with bad syntax."""
        query = 'This is not actually a query.'
        self.assertRaises(sparql.SPARQLQueryError, self.sparql.query_data,
                          query_string=query)

    def test_sparql_manager_query_capacitors(self):
        """Ensure we get the expected return."""
        actual = self.sparql.query_capacitors()

        # Uncomment stuff below to re-create the expected output.
        # with open('query_capacitors.json', 'w') as f:
        #     json.dump(actual, f)

        # Load expected result.
        with open('query_capacitors.json', 'r') as f:
            expected = json.load(f)

        self.assertDictEqual(actual, expected)

    def test_sparql_manager_query_capacitors_calls_query_named_objects(self):
        """query_capacitors must call query_named_objects."""

        with patch('pyvvo.sparql.SPARQLManager.query_named_objects',
                   return_value='success') as mock:
            cap_return = self.sparql.query_capacitors()
            mock.assert_called_once()
            mock.assert_called_with(
                self.sparql.CAPACITOR_QUERY.format(
                    feeder_mrid=self.sparql.feeder_mrid))
            self.assertEqual('success', cap_return)

    def test_sparql_manager_query_regulators(self):
        """Ensure we get the expected return."""
        actual = self.sparql.query_regulators()

        # Uncomment stuff below to re-create the expected output.
        # with open('query_regulators.json', 'w') as f:
        #     json.dump(actual, f)

        # Load expected result.
        with open('query_regulators.json', 'r') as f:
            expected = json.load(f)

        self.assertDictEqual(actual, expected)

    def test_sparql_manager_query_regulators_calls_query_named_objects(self):
        """query_regulators must call query_named_objects."""

        with patch('pyvvo.sparql.SPARQLManager.query_named_objects',
                   return_value='success') as mock:
            reg_return = self.sparql.query_regulators()
            mock.assert_called_once()
            mock.assert_called_with(
                self.sparql.REGULATOR_QUERY.format(
                    feeder_mrid=self.sparql.feeder_mrid))
            self.assertEqual('success', reg_return)


if __name__ == '__main__':
    unittest.main()
