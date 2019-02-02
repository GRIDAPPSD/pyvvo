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

        # Ensure _check_bindings is called.
        with patch('pyvvo.sparql.SPARQLManager._check_bindings',
                   return_value=None) as mock:
            # Check the return value.
            self.assertEqual(self.sparql._bindings_to_dict(bindings),
                             expected_value)

            mock.assert_called_once()

    def test_sparql_manager_check_bindings_no_name(self):
        self.assertRaises(KeyError, self.sparql._check_bindings,
                          bindings=[{'I have no name': 'agh'}])

    def test_sparql_manager_bindings_to_dict_no_value(self):
        self.assertRaises(KeyError, self.sparql._bindings_to_dict,
                          bindings=[{'name': {'type': 'literal'}}])

    def test_sparql_manager_check_bindings_bad_binding_type(self):
        self.assertRaises(TypeError, self.sparql._check_bindings,
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
                    feeder_mrid=self.sparql.feeder_mrid), one_to_many=False)
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
                    feeder_mrid=self.sparql.feeder_mrid), one_to_many=True)
            self.assertEqual('success', reg_return)

    def test_sparql_manager_bindings_to_dict_of_lists_valid_input(self):
        bindings = [
            {'name': {'type': 'literal', 'value': 'obj1'},
             'stuff': {'value': '2'}
             },
            {'things': {'type': 2, 'value': 17},
             'name': {'value': 'obj1'}
             },
            {'name': {'type': 'stuff', 'value': 'obj2'},
             'property': {'value': 'yay'}}
        ]

        expected_value = {'obj1': [{'name': 'obj1', 'stuff': '2'},
                                   {'name': 'obj1', 'things': 17}],
                          'obj2': [{'name': 'obj2', 'property': 'yay'}]
                          }

        # Ensure _check_bindings is called.
        with patch('pyvvo.sparql.SPARQLManager._check_bindings',
                   return_value=None) as mock:
            # Check the return value.
            self.assertEqual(self.sparql._bindings_to_dict_of_lists(bindings),
                             expected_value)

            mock.assert_called_once()

    def test_sparql_manager_bindings_to_dict_of_lists_bad_mapping(self):
        bindings = [{'name': {'value': 'object'},
                     'attribute': 7}]

        # This will raise a TypeError since:
        # "'int' object is not subscriptable"
        self.assertRaises(TypeError, self.sparql._bindings_to_dict_of_lists,
                          bindings=bindings)

    def test_sparql_manager_bindings_to_dict_of_lists_no_value(self):
        bindings = [{'name': {'value': 'object'},
                     'attribute': {'type': 'attribute'}}]

        # This will raise a KeyError since there's no 'value' key.
        self.assertRaises(KeyError, self.sparql._bindings_to_dict_of_lists,
                          bindings=bindings)

    def test_sparql_manager_query_load_nominal_voltage_calls_qno(self):
        """query_load_nominal_voltage must call query_named_objects."""

        with patch('pyvvo.sparql.SPARQLManager.query_named_objects',
                   return_value='success') as mock:
            load_nom_v = self.sparql.query_load_nominal_voltage()
            mock.assert_called_once()
            mock.assert_called_with(
                self.sparql.LOAD_NOMINAL_VOLTAGE_QUERY.format(
                    feeder_mrid=self.sparql.feeder_mrid), one_to_many=False)
            self.assertEqual('success', load_nom_v)

    def test_sparql_manager_query_load_nominal_voltage_expected_return(self):
        """Check one of the elements from query_load_nominal_voltage."""
        expected = {'name': '2127146b0', 'bus': 'sx3160864b',
                    'basev': '208', 'conn': 'Y', 'phases': 's1,s2'}

        full_actual = self.sparql.query_load_measurements()

        self.assertIn(expected['name'], full_actual)

        self.assertDictEqual(full_actual[expected['name']], expected)

    def test_sparql_manager_query_load_measurements_calls_qno(self):
        """query_load_measurements must call query_named_objects."""

        with patch('pyvvo.sparql.SPARQLManager.query_named_objects',
                   return_value='success') as mock:
            load_meas = self.sparql.query_load_measurements()
            mock.assert_called_once()
            mock.assert_called_with(
                self.sparql.LOAD_MEASUREMENTS_QUERY.format(
                    feeder_mrid=self.sparql.feeder_mrid), one_to_many=True)
            self.assertEqual('success', load_meas)

    def test_sparql_manager_query_load_measurements_expected_return(self):
        """Check on of the elements from query_load_measurements."""
        expected = [{'class': 'Analog', 'type': 'PNV',
                     'name': 'EnergyConsumer_21395720c0',
                     'node': 'sx2860492c', 'phases': 's1',
                     'load': '21395720c0',
                     'eqid': '_40DAA2E0-A34E-0807-6879-6D908E586EF4',
                     'trmid': '_9E36102E-7888-11F3-D887-20E7764FFCA4',
                     'id': '_88bcd540-33e0-444b-8393-1ea955bc72f4'},
                    {'class': 'Analog', 'type': 'PNV',
                     'name': 'EnergyConsumer_21395720c0',
                     'node': 'sx2860492c', 'phases': 's2',
                     'load': '21395720c0',
                     'eqid': '_40DAA2E0-A34E-0807-6879-6D908E586EF4',
                     'trmid': '_9E36102E-7888-11F3-D887-20E7764FFCA4',
                     'id': '_a77b6a44-4960-43cb-bd11-10e7c93a0b61'},
                    {'class': 'Analog', 'type': 'VA',
                     'name': 'EnergyConsumer_21395720c0',
                     'node': 'sx2860492c', 'phases': 's1',
                     'load': '21395720c0',
                     'eqid': '_40DAA2E0-A34E-0807-6879-6D908E586EF4',
                     'trmid': '_9E36102E-7888-11F3-D887-20E7764FFCA4',
                     'id': '_41ce341e-e55e-4560-809c-67df0c85c27b'},
                    {'class': 'Analog', 'type': 'VA',
                     'name': 'EnergyConsumer_21395720c0',
                     'node': 'sx2860492c', 'phases': 's2',
                     'load': '21395720c0',
                     'eqid': '_40DAA2E0-A34E-0807-6879-6D908E586EF4',
                     'trmid': '_9E36102E-7888-11F3-D887-20E7764FFCA4',
                     'id': '_dbf70967-46e1-4c5e-bc83-e514f01a2142'}]

        full_actual = self.sparql.query_load_measurements()

        self.assertIn(expected[0]['name'], full_actual)

        self.assertEqual(expected, full_actual[expected[0]['name']])


if __name__ == '__main__':
    unittest.main()
