# Standard library imports
import unittest
from unittest.mock import patch
from datetime import datetime
import os
import logging

# Import module to test
from pyvvo import glm, db
from pyvvo.utils import gld_installed, run_gld
from tests.models import MODEL_DIR, IEEE_13, IEEE_8500

# Setup log.
LOG = logging.getLogger(__name__)

# Define our test files.
TEST_FILE = os.path.join(MODEL_DIR, 'test.glm')
TEST_FILE2 = os.path.join(MODEL_DIR, 'test2.glm')
TEST_FILE3 = os.path.join(MODEL_DIR, 'test3.glm')
TEST_FILE4 = os.path.join(MODEL_DIR, 'test4.glm')
EXPECTED4 = os.path.join(MODEL_DIR, 'test4_expected.glm')

# See if we have database inputs defined.
DB_ENVIRON_PRESENT = db.db_env_defined()


class TestParseFile(unittest.TestCase):
    """Test parsing a test file."""

    @classmethod
    def setUpClass(cls):
        # Call parse
        cls._parsed_tokens = glm.parse(TEST_FILE, True)

    def test_parse_returns_dict(self):
        self.assertIsInstance(self._parsed_tokens, dict)

    def test_parse_dict_length_is_21(self):
        # Recorders are no longer removed.
        self.assertTrue(len(self._parsed_tokens) == 21)

    def test_parse_item_1_has_omftype(self):
        self.assertIn('omftype', self._parsed_tokens[1])

    def test_parse_item_1_has_argument_mysql(self):
        self.assertEqual('mysql', self._parsed_tokens[1]['argument'])

    def test_parse_item_2_has_object(self):
        self.assertIn('object', self._parsed_tokens[2])

    def test_parse_item_2_is_database(self):
        self.assertEqual('database', self._parsed_tokens[2]['object'])

    def test_parse_item_10_is_load(self):
        self.assertEqual('load', self._parsed_tokens[10]['object'])

    def test_parse_item_16_has_phases_ABCN(self):
        self.assertEqual('ABCN', self._parsed_tokens[16]['phases'])

    def test_parse_item_16_has_power_fraction_A_080(self):
        self.assertEqual('0.8', self._parsed_tokens[16]['power_fraction_A'])

    def test_parse_item_4_starttime_is_correct(self):
        self.assertEqual("'2001-01-01 00:00:00'",
                         self._parsed_tokens[4]['starttime'])


class TestParseFile2(unittest.TestCase):
    """Second class for testing parsing. Adding this after the regular
    expression for tokenizing was updated to allow for parameter
    expansions. See:
        http://gridlab-d.shoutwiki.com/wiki/Creating_GLM_Files
    """
    @classmethod
    def setUpClass(cls):
        cls.parsed_tokens = glm.parse(TEST_FILE2, True)

    def test_parse2_dict_length_is_4(self):
        self.assertEqual(len(self.parsed_tokens), 4)

    def test_parse2_define(self):
        self.assertEqual(self.parsed_tokens[0]['omftype'], '#define')

    def test_parse2_define_value(self):
        self.assertEqual(self.parsed_tokens[0]['argument'], 'VSOURCE=66400')

    def test_parse2_substation_voltage(self):
        self.assertEqual(self.parsed_tokens[3]['positive_sequence_voltage'],
                         '${VSOURCE}')


class TestGLMManager(unittest.TestCase):
    """Test the GLMManager class.

    This simply calls 'parse' to get its model_dict, which is tested in
    the TestParseFile class, so no need to heavily inspect the model_dict.

    NOTE: During this test, we'll add items.
    """

    @classmethod
    def setUpClass(cls):
        # Get a GLMManager object
        cls._GLMManager = glm.GLMManager(TEST_FILE, True)

    '''
    def test_prepend_key_is_neg_1(self):
        # NOTE: since these tests are not run in order, this could fail.
        self.assertEqual(-1, self._GLMManager.prepend_key)

    def test_append_key_is_18(self):
        # NOTE: since these tests are not run in order, this could fail.
        self.assertEqual(18, self._GLMManager.append_key)
    '''

    def test_clock_in_map(self):
        self.assertTrue(len(self._GLMManager.model_map['clock']) > 0)

    def test_clock_maps_correctly(self):
        # Assert is ensures we're accessing the same place in memory (I think)
        self.assertIs(self._GLMManager.model_dict[4],
                      self._GLMManager.model_map['clock'][1])

    def test_clock_map_key_correct(self):
        self.assertEqual(4, self._GLMManager.model_map['clock'][0])

    def test_powerflow_map_key_correct(self):
        self.assertEqual(3,
                         self._GLMManager.model_map['module']['powerflow'][0])

    def test_power_powerflow_maps_correctly(self):
        self.assertIs(self._GLMManager.model_dict[3],
                      self._GLMManager.model_map['module']['powerflow'][1])

    def test_meter_1_key_correct(self):
        self.assertEqual(11, self._GLMManager.model_map['object']['meter'][
            'meter_1'][0])

    def test_meter_1_maps_correctly(self):
        self.assertIs(self._GLMManager.model_dict[11],
                      self._GLMManager.model_map['object']['meter'][
                          'meter_1'][1])

    def test_add_named_recorder(self):
        # Build dictionary for recorder.
        r = {'object': 'recorder', 'group': '"groupid=meter_group"',
             'interval': 30, 'name': 'test_recorder'}

        # Get value of the append_key.
        k = self._GLMManager.append_key

        # Add recorder.
        self._GLMManager.add_item(r)

        # Ensure its in the map.
        self.assertIs(r, self._GLMManager.model_map['object']['recorder'][
            'test_recorder'][1])

        # Ensure its in the model (at the end)
        self.assertIs(r, self._GLMManager.model_dict[k])

        # Ensure 'interval' property is there and is a string.
        self.assertEqual('30', self._GLMManager.model_dict[k]['interval'])

        # Ensure the key was incremented.
        self.assertEqual(k+1, self._GLMManager.append_key)

    def test_add_unnamed_recorder(self):
        # Build dictionary for recorder.
        r = {'object': 'recorder', 'group': '"groupid=meter_group"',
             'interval': '30'}

        # Get value of the append_key.
        k = self._GLMManager.append_key

        # Add recorder.
        self._GLMManager.add_item(r)

        # Ensure its in the map and the key is correct.
        in_map = False
        for item in self._GLMManager.model_map['object_unnamed']:
            in_map = in_map or ((item[1] is r) and (item[0] == k))

        self.assertTrue(in_map)

        # Ensure its in the model (at the end)
        self.assertIs(r, self._GLMManager.model_dict[k])

        # Check the group property.
        self.assertEqual(r['group'], self._GLMManager.model_dict[k]['group'])

        # Ensure the key was incremented.
        self.assertEqual(k+1, self._GLMManager.append_key)

    def test_add_set_randomseed(self):
        # Note this should go through add_item to hit _add_non_object
        obj = {'#set': 'randomseed=42'}

        # Get prepend key
        k = self._GLMManager.prepend_key

        # Add it.
        self._GLMManager.add_item(obj)

        # Ensure its in the model_dict
        self.assertIs(obj, self._GLMManager.model_dict[k])

        # Ensure the prepend key was updated.
        self.assertEqual(k-1, self._GLMManager.prepend_key)

    def test_add_clock_to_map_fails(self):
        # This model already has a clock.
        # TODO: should probably get a barebones model and test making a
        # bunch of additions.
        self.assertRaises(glm.ItemExistsError, self._GLMManager.add_item,
                          {'clock': 'clock'})

    def test_add_nonexistent_item_type_fails(self):
        self.assertRaises(TypeError, self._GLMManager.add_item,
                          {'foo': 'bar', 'baz': 42})

    def test_update_append_key(self):
        # Get original value.
        k = self._GLMManager.append_key

        # Increment.
        self._GLMManager._update_append_key()

        # Test.
        self.assertEqual(k+1, self._GLMManager.append_key)

    def test_update_prepend_key(self):
        # Get original value.
        k = self._GLMManager.prepend_key

        # Increment.
        self._GLMManager._update_prepend_key()

        # Test.
        self.assertEqual(k - 1, self._GLMManager.prepend_key)

    def test_modify_object_nonexistent(self):
        # Try to find object that doesn't exist.
        item = {'object': 'foo', 'name': 'bar'}

        self.assertRaises(KeyError, self._GLMManager.modify_item, item)

    def test_modify_powerflow(self):
        # Change solver to FBS.
        item = {'module': 'powerflow', 'solver_method': 'FBS'}

        # Modify.
        self._GLMManager.modify_item(item)

        # Ensure model_dict has been appropriately changed.
        self.assertEqual('FBS',
                         self._GLMManager.model_dict[3]['solver_method'])

        # Ensure that our mapping is still good:
        # Check key.
        self.assertEqual(3,
                         self._GLMManager.model_map['module']['powerflow'][0])
        # Check map.
        self.assertIs(self._GLMManager.model_dict[3],
                      self._GLMManager.model_map['module']['powerflow'][1])

    def test_modify_mysql(self):
        # Interesting test, because 'parse' maps mysql as an 'omftype'
        item = {'module': 'mysql', 'port': 3306}

        # Modify
        self._GLMManager.modify_item(item)

        # Ensure model_dict has been changed.
        self.assertEqual('3306', self._GLMManager.model_dict[1]['port'])

        # Ensure mapping is still good:
        # Check key.
        self.assertEqual(1, self._GLMManager.model_map['module']['mysql'][0])
        # Map.
        self.assertIs(self._GLMManager.model_dict[1],
                      self._GLMManager.model_map['module']['mysql'][1])

    def test_modify_clock(self):
        item = {'clock': 'clock', 'timezone': 'EST+5EDT'}

        # Modify.
        self._GLMManager.modify_item(item)

        # Ensure model_dict has been appropriately changed.
        self.assertEqual('EST+5EDT',
                         self._GLMManager.model_dict[4]['timezone'])

        # Ensure that our mapping is still good:
        # Check key.
        self.assertEqual(4, self._GLMManager.model_map['clock'][0])
        # Check map.
        self.assertIs(self._GLMManager.model_dict[4],
                      self._GLMManager.model_map['clock'][1])

    def test_modify_load(self):
        item = {'object': 'load', 'name': 'load_3', 'base_power_A': '120000',
                'power_pf_B': 1.0, 'nominal_voltage': 7300,
                'groupid': 'new_load_group'}

        # Modify.
        self._GLMManager.modify_item(item)

        # Check model_dict. Note values get cast to strings.
        self.assertEqual(item['base_power_A'],
                         self._GLMManager.model_dict[16]['base_power_A'])
        self.assertEqual(str(item['power_pf_B']),
                         self._GLMManager.model_dict[16]['power_pf_B'])
        self.assertEqual(str(item['nominal_voltage']),
                         self._GLMManager.model_dict[16]['nominal_voltage'])
        self.assertEqual(item['groupid'],
                         self._GLMManager.model_dict[16]['groupid'])

        # Check mapping. Key first.
        self.assertEqual(16, self._GLMManager.model_map['object']['load'][
            'load_3'][0])
        # Dict mapping.
        self.assertIs(self._GLMManager.model_dict[16],
                      self._GLMManager.model_map['object']['load'][
                          'load_3'][1])

    def test_modify_bad_item_type(self):
        item = {'#set': 'minimum_timestep=30.0'}

        self.assertRaises(TypeError, self._GLMManager.modify_item, item)

    def test_remove_properties_from_ol_3(self):
        item = {'object': 'overhead_line', 'name': 'ol_3'}
        property_list = ['phases', 'length']

        # Remove items.
        self._GLMManager.remove_properties_from_item(item, property_list)

        # Ensure they're gone in map.
        obj = self._GLMManager._lookup_object(object_type='overhead_line',
                                              object_name='ol_3')
        try:
            obj['phases']
        except KeyError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, 'phases not successfully removed from '
                                   'model_map.')

        try:
            obj['length']
        except KeyError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, 'length not successfully removed. from '
                                   'model_map')

        # Ensure they're gone in the model.
        obj = self._GLMManager.model_dict[15]

        try:
            obj['phases']
        except KeyError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, 'phases not successfully removed from '
                                   'model_dict.')

        try:
            obj['length']
        except KeyError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, 'length not successfully removed. from '
                                   'model_dict')

        # Ensure 'to' is still present.
        try:
            obj['to']
        except KeyError:
            self.assertTrue(False, 'to inadvertently removed from the object!')
        else:
            self.assertTrue(True)

    def test_remove_properties_from_clock(self):
        item = {'clock': 'clock'}
        property_list = ['stoptime']

        self._GLMManager.remove_properties_from_item(item, property_list)

        # Ensure stoptime is gone in the map.
        clock = self._GLMManager._lookup_clock()

        try:
            clock['stoptime']
        except KeyError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, 'stoptime not successfully removed from '
                                   'the clock in the model_map.')

        # Ensure its gone in the model
        try:
            self._GLMManager.model_dict[4]['stoptime']
        except KeyError:
            self.assertTrue(True)
        else:
            self.assertTrue(False, 'stoptime not successfully removed from '
                                   'the clock in the model_dict.')

        # Ensure we still have the starttime
        try:
            clock['starttime']
        except KeyError:
            self.assertTrue(False, 'starttime inadvertently removed from the '
                                   'clock in the model_map')
        else:
            self.assertTrue(True)

        try:
            self._GLMManager.model_dict[4]['starttime']
        except KeyError:
            self.assertTrue(False, 'starttime inadvertently removed from the '
                                   'clock in the model_dict')
        else:
            self.assertTrue(True)

    def test_object_present_bad_type(self):
        self.assertRaises(TypeError, self._GLMManager.object_type_present, 10)

    def test_object_present_not_there(self):
        """This model doesn't have inverters."""
        self.assertFalse(self._GLMManager.object_type_present('inverter'))

    def test_object_present_there(self):
        """This model has line_configurations."""
        self.assertTrue(
            self._GLMManager.object_type_present('line_configuration'))

    def test_module_present_bad_type(self):
        self.assertRaises(TypeError, self._GLMManager.module_present,
                          {'not': 'a string'})

    def test_module_present_powerflow(self):
        self.assertTrue(self._GLMManager.module_present('powerflow'))

    def test_module_present_generators(self):
        self.assertFalse(self._GLMManager.module_present('generators'))

    # TODO: model doesn't currently have a module which we can remove a
    # property from, because we modify the powerflow solver_method, and
    # cannot count on tests to run in order.


class TestGLMManagerRemove(unittest.TestCase):
    """Test the removal methods of the GLMManager class."""

    @classmethod
    def setUpClass(cls):
        # Get a GLMManager object
        cls._GLMManager = glm.GLMManager(TEST_FILE, True)

    def test_remove_clock(self):
        # Remove clock.
        self._GLMManager.remove_item({'clock': 'clock'})

        # Ensure it's gone in the map.
        self.assertRaises(IndexError, self._GLMManager._lookup_clock)

        # Make sure its gone in the model.
        self.assertNotIn(4, self._GLMManager.model_dict)

    def test_remove_powerflow(self):
        # Remove powerflow module
        self._GLMManager.remove_item({'module': 'powerflow'})

        # Ensure it's gone in the map.
        self.assertRaises(KeyError, self._GLMManager._lookup_module,
                          'powerflow')

        # Ensure it's gone in the model.
        self.assertNotIn(3, self._GLMManager.model_dict)

    def test_remove_mysql(self):
        # Remove the mysql module. Interesting because its parsed as 'omftype'
        self._GLMManager.remove_item({'module': 'mysql'})

        # Ensure it's gone in the map.
        self.assertRaises(KeyError, self._GLMManager._lookup_module,
                          'mysql')

        # Ensure its gone in the model.
        self.assertNotIn(1, self._GLMManager.model_dict)

    def test_remove_line_spacing_1(self):
        # Remove named object line_spacing_1
        self._GLMManager.remove_item({'object': 'line_spacing',
                                      'name': 'line_spacing_1'})

        # Ensure it's gone in the map.
        self.assertRaises(KeyError, self._GLMManager._lookup_object,
                          'line_spacing', 'line_spacing_1')

        # Ensure its gone in the model.
        self.assertNotIn(7, self._GLMManager.model_dict)

    def test_remove_unnamed_object(self):
        # Not currently allowed.
        self.assertRaises(KeyError, self._GLMManager.remove_item,
                          {'object': 'overhead_line'})


class TestGLMManagerMisc(unittest.TestCase):
    """Test functions in the GLMManager class which can't be run with
    the primary testing class. For example, the function
    get_objects_by_type can't easily be tested when other methods are
    adding or removing objects from the model.
    """

    @classmethod
    def setUpClass(cls):
        # Get a GLMManager object
        cls._GLMManager = glm.GLMManager(TEST_FILE, True)

    def test_get_object_by_type_loads(self):
        # Grab a listing of loads
        load_list = self._GLMManager.get_objects_by_type(object_type='load')

        # Ensure we have three.
        self.assertEqual(3, len(load_list))

        # Ensure all are dictionaries, and that they have a name.
        for load_dict in load_list:
            self.assertIsInstance(load_dict, dict)
            self.assertIn('name', load_dict)

    def test_get_object_by_type_recorders(self):
        # Grab a listing of loads
        recorder_list = \
            self._GLMManager.get_objects_by_type(object_type='load')

        # Ensure we have three.
        self.assertEqual(3, len(recorder_list))

        # Ensure all are dictionaries, and that they have a name.
        for recorder_dict in recorder_list:
            self.assertIsInstance(recorder_dict, dict)
            self.assertIn('name', recorder_dict)

    def test_get_object_by_type_clock(self):
        # There should be no clock in the "objects" listing.
        clock = self._GLMManager.get_objects_by_type(object_type='clock')

        self.assertIsNone(clock)

    def test_find_object_nonexistent(self):
        # Try finding a non-existent object. Should return None.
        obj = self._GLMManager.find_object(obj_type='meter',
                                           obj_name='Not there')
        self.assertIsNone(obj)

    def test_find_object_bad_type(self):
        # Try finding an object type which isn't in the model
        obj = self._GLMManager.find_object(obj_type='nonexistent',
                                           obj_name='meter_1')
        self.assertIsNone(obj)

    def test_find_object_load(self):
        # Test finding a load.
        obj = self._GLMManager.find_object(obj_type='load', obj_name='load_2')

        # Ensure it's a dictionary.
        self.assertIsInstance(obj, dict)

        # Ensure it has the 'object' property and it evaluates to 'load'
        self.assertIn('object', obj)
        self.assertEqual(obj['object'], 'load')

        # Ensure it's name evaluates to 'load_2'
        self.assertIn('name', obj)
        self.assertEqual(obj['name'], 'load_2')

    def test_get_objects_by_type_nonexistent(self):
        # Try looking up an object type which isn't present
        obj_list = self._GLMManager.get_objects_by_type('bananas')
        self.assertIsNone(obj_list)

    def test_get_objects_by_type_overhead_line(self):
        # Look up overhead lines.
        obj_list = self._GLMManager.get_objects_by_type('overhead_line')

        # Ensure we received a list.
        self.assertIsInstance(obj_list, list)

        # Ensure we received 3 elements.
        self.assertEqual(len(obj_list), 3)

        # Ensure all elements are dictionaries.
        for d in obj_list:
            with self.subTest():
                self.assertIsInstance(d, dict)

        # Ensure all objects have a 'from' and 'to'
        for d in obj_list:
            with self.subTest():
                self.assertIn('from', d)
                self.assertIn('to', d)

    def test_get_items_by_type_clock(self):
        c = self._GLMManager.get_items_by_type(item_type='clock')
        self.assertIsInstance(c, dict)
        self.assertIn('clock', c)
        self.assertIn('starttime', c)

    def test_get_items_by_type_module(self):
        m = self._GLMManager.get_items_by_type(item_type='module')
        self.assertIsInstance(m, dict)
        self.assertIn('mysql', m)
        self.assertIn('powerflow', m)
        for d in m.values():
            self.assertIsInstance(d, dict)

    def test_get_items_by_type_object_no_type(self):
        with self.assertRaisesRegex(ValueError, "If item_type is 'object'"):
            self._GLMManager.get_items_by_type(item_type='object')

    def test_get_items_by_type_object_meter(self):
        o = self._GLMManager.get_items_by_type(item_type='object',
                                               object_type='meter')
        self.assertIsInstance(o, dict)
        self.assertEqual(4, len(o))
        for name, obj in o.items():
            self.assertIsInstance(obj, dict)
            self.assertEqual(name, obj['name'])
            self.assertEqual(obj['object'], 'meter')

    def test_get_items_by_type_object_unnamed(self):
        u = self._GLMManager.get_items_by_type(item_type='object_unnamed')
        self.assertIsInstance(u, list)
        for i in u:
            self.assertIsInstance(i, dict)

        # This model has 3 recorders and a database object which are
        # unnamed.
        self.assertEqual(4, len(u))

    def test_get_items_by_type_bad_type(self):
        self.assertIsNone(
            self._GLMManager.get_items_by_type(item_type='bogus'))


class AddOrModifyClockTestCase(unittest.TestCase):
    """Test GLMManager.add_or_modify_clock."""
    @classmethod
    def setUpClass(cls):
        """Get a GLMManager. Use the simpler model for speed."""
        cls.glm = glm.GLMManager(model=TEST_FILE2, model_is_path=True)

    def test_add_or_modify_clock_bad_starttime_type(self):
        self.assertRaises(TypeError, self.glm.add_or_modify_clock,
                          starttime='2012-07-21 20:00:00')

    def test_add_or_modify_clock_bad_stoptime_type(self):
        self.assertRaises(TypeError, self.glm.add_or_modify_clock,
                          stoptime='2012-07-21 20:00:00')

    def test_add_or_modify_clock_bad_timezone(self):
        self.assertRaises(TypeError, self.glm.add_or_modify_clock,
                          timezone=-8)

    def test_add_or_modify_clock_all_inputs_None(self):
        self.assertRaises(ValueError, self.glm.add_or_modify_clock,
                          starttime=None, stoptime=None, timezone=None)

    def test_add_or_modify_clock_change_all(self):
        st = datetime(year=2012, month=1, day=1)
        et = datetime(year=2017, month=6, day=10, hour=8, minute=35,
                      second=12)
        # Timezone doesn't have to be valid... oh well.
        tz = 'Pacific'
        self.glm.add_or_modify_clock(starttime=st, stoptime=et, timezone=tz)

        # Lookup the clock item.
        actual = self.glm._lookup_clock()

        expected = {'clock': 'clock', 'starttime': "'2012-01-01 00:00:00'",
                    'stoptime': "'2017-06-10 08:35:12'", 'timezone': tz}
        self.assertDictEqual(actual, expected)

    def test_add_or_modify_clock_add_clock(self):
        # Start by removing the clock.
        self.glm.remove_item({'clock': 'clock'})

        # Cheat and simply call another test method.
        self.test_add_or_modify_clock_change_all()

    def test_add_or_modify_clock_add_clock_incomplete_inputs(self):
        # To avoid interfering with other tests, we'll create our own
        # manager here.
        glm_manager = glm.GLMManager(model=TEST_FILE2, model_is_path=True)
        # Remove the clock.
        glm_manager.remove_item({'clock': 'clock'})
        # Add new one, but don't include all inputs.
        st = datetime(year=2016, month=12, day=6)
        et = None
        tz = 'Central'
        self.assertRaises(ValueError, glm_manager.add_or_modify_clock,
                          starttime=st, stoptime=et, timezone=tz)


class AddRunComponentsBadInputsTestCase(unittest.TestCase):
    """Test add_run_components function with bad inputs."""

    @classmethod
    def setUpClass(cls):
        """Load a model."""
        cls.glm = glm.GLMManager(TEST_FILE3, model_is_path=True)

    def test_add_run_components_add_or_modify_clock_is_called(self):
        """add_or_modify_clock will handle input checking for starttime,
        stoptime, and timezone, so we need to ensure it gets called.
        """
        with patch('pyvvo.glm.GLMManager.add_or_modify_clock',
                   return_value=None) as mock:
            self.glm.add_run_components(starttime='bleh', stoptime='blah',
                                        timezone='Eastern')
            mock.assert_called_once()
            mock.assert_called_with(starttime='bleh', stoptime='blah',
                                    timezone='Eastern')

        self.assertTrue(True)

    def test_add_run_components_v_source_bad_type(self):
        self.assertRaises(ValueError, self.glm.add_run_components,
                          starttime=datetime(2012, 1, 1),
                          stoptime=datetime(2012, 1, 1, 0, 15),
                          timezone='UTC0', v_source='one thousand')

    def test_add_run_components_profiler_bad_type(self):
        self.assertRaises(TypeError, self.glm.add_run_components,
                          starttime=datetime(2012, 1, 1),
                          stoptime=datetime(2012, 1, 1, 0, 15),
                          timezone='UTC0', profiler='0')

    def test_add_run_components_profiler_bad_value(self):
        self.assertRaises(ValueError, self.glm.add_run_components,
                          starttime=datetime(2012, 1, 1),
                          stoptime=datetime(2012, 1, 1, 0, 15),
                          timezone='UTC0', profiler=2)

    def test_add_run_components_minimum_timestep_bad_type(self):
        self.assertRaises(TypeError, self.glm.add_run_components,
                          starttime=datetime(2012, 1, 1),
                          stoptime=datetime(2012, 1, 1, 0, 15),
                          timezone='UTC0', minimum_timestep=60.1)


class AddRunComponentsTestCase(unittest.TestCase):
    """Call add_run_components with no arguments, model should run.

    We already have a test ensuring add_or_modify_clock is called, so
    no need to check clock values. However, we need to ensure the clock
    is present, and also need to check the other parameters.
    """
    # Define the model we'll use.
    MODEL = TEST_FILE3

    @classmethod
    def setUpClass(cls):
        """Load model, add components."""
        cls.glm = glm.GLMManager(cls.MODEL, model_is_path=True)

        cls.out_file = 'tmp.glm'
        cls.glm.add_run_components(starttime=datetime(2012, 1, 1),
                                   stoptime=datetime(2012, 1, 1, 0, 15),
                                   timezone='UTC0', v_source=None,
                                   profiler=0, minimum_timestep=60)

        cls.glm.write_model(out_path=cls.out_file)

    @classmethod
    def tearDownClass(cls):
        # noinspection PyUnresolvedReferences
        os.remove(cls.out_file)

    @unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
    def test_add_run_components_model_runs(self):
        result = run_gld(model_path=self.out_file)

        self.assertEqual(0, result.returncode)

    def test_add_run_components_clock(self):
        """Ensure the clock is there."""
        clock = self.glm._lookup_clock()

        self.assertIn('clock', clock)

    def test_add_run_components_minimum_timestep(self):
        minimum_timestep = self.glm.model_dict[-6]

        self.assertDictEqual(minimum_timestep, {'#set': 'minimum_timestep=60'})

    def test_add_run_components_profiler(self):
        profiler = self.glm.model_dict[-5]

        self.assertDictEqual(profiler, {'#set': 'profiler=0'})

    def test_add_run_components_relax_naming_rules(self):
        rnr = self.glm.model_dict[-4]

        self.assertDictEqual(rnr, {'#set': 'relax_naming_rules=1'})

    def test_add_run_components_powerflow(self):
        pf = self.glm.model_dict[-2]

        self.assertDictEqual(pf, {'module': 'powerflow', 'solver_method': 'NR',
                                  'line_capacitance': 'TRUE'})

    def test_add_run_components_v_source(self):
        vs = self.glm.model_dict[-1]

        self.assertDictEqual(vs, {'#define': 'VSOURCE=66395.28'})

    def test_add_run_components_generators(self):
        """This model should not have the generators added."""
        self.assertFalse(self.glm.module_present('generators'))


class AddRunComponentsIEEE13NodeTestCase(AddRunComponentsTestCase):
    """Run tests in AddRunComponentsTestCase, but use IEEE 13 bus model.

    Some methods are overridden intentionally.
    """
    MODEL = IEEE_13

    def test_add_run_components_generators(self):
        """This model should have the generators added."""
        self.assertTrue(self.glm.module_present('generators'))

    def test_add_run_components_minimum_timestep(self):
        minimum_timestep = self.glm.model_dict[-7]

        self.assertDictEqual(minimum_timestep, {'#set': 'minimum_timestep=60'})

    def test_add_run_components_profiler(self):
        profiler = self.glm.model_dict[-6]

        self.assertDictEqual(profiler, {'#set': 'profiler=0'})

    def test_add_run_components_relax_naming_rules(self):
        rnr = self.glm.model_dict[-5]

        self.assertDictEqual(rnr, {'#set': 'relax_naming_rules=1'})


class NestedObjectsIEEE13TestCase(unittest.TestCase):
    """Ensure that nested objects get properly mapped."""

    # Define the model we'll use.
    MODEL = IEEE_13

    @classmethod
    def setUpClass(cls):
        """Load model, add components."""
        cls.glm = glm.GLMManager(cls.MODEL, model_is_path=True)

    def test_nested_objects_ieee_13_solar_in_map(self):
        self.assertTrue(self.glm.object_type_present('solar'))

    def test_nested_objects_ieee_13_solar_in_dict(self):
        self.assertEqual(self.glm.model_dict[13]['name'],
                         '"pv_school"')


class NestedObjectsDoubleNestTestCase(unittest.TestCase):
    """Check that double-nesting works."""

    @classmethod
    def setUpClass(cls):
        """Load, save to file."""
        cls.glm = glm.GLMManager(TEST_FILE4, model_is_path=True)
        cls.glm.write_model('tmp.glm')

    @classmethod
    def tearDownClass(cls):
        os.remove('tmp.glm')

    def test_nested_objects_double_nesting(self):

        with open('tmp.glm', 'r') as f:
            actual = f.read()

        with open(EXPECTED4, 'r') as f:
            expected = f.read()

        self.assertEqual(actual, expected)


@unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
@unittest.skipIf(not DB_ENVIRON_PRESENT,
                 reason='Database environment variables are not present.')
class RunModelWithDatabaseTestCase(unittest.TestCase):
    """This is more of an integration test than anything. Add database
    components to a model and ensure it runs.
    """

    # noinspection PyPackageRequirements
    @classmethod
    def setUpClass(cls):
        # Use the simplest model.
        cls.glm_mgr = glm.GLMManager(TEST_FILE2, model_is_path=True)

        # Add the MySQL module.
        cls.glm_mgr.add_item({'module': 'mysql'})

        # Add a database object.
        # At the time of writing (2019-07-02), the following environment
        #   variables are only defined in the docker-compose file.
        cls.glm_mgr.add_item({'object': 'database',
                              'hostname': os.environ['DB_HOST'],
                              'username': os.environ['DB_USER'],
                              'password': os.environ['DB_PASS'],
                              'port': os.environ['DB_PORT'],
                              'schema': os.environ['DB_DB']})

        cls.out_file = 'tmp_db.glm'
        cls.glm_mgr.write_model(cls.out_file)

        # It can take a while to get the database up and running with
        # PyCharm using docker-compose. Let's do a 30 second wait loop.
        _ = db.connect_loop(timeout=30, retry_interval=0.1)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.out_file)
        try:
            os.remove('gridlabd.xml')
        except FileNotFoundError:
            pass

    def test_model_runs(self):
        result = run_gld(self.out_file)
        self.assertEqual(0, result.returncode)


class AddSubstationMeter13TestCase(unittest.TestCase):
    """Primitive test case for add_substation_meter with IEEE 13 node.
    """

    @classmethod
    def setUpClass(cls):
        cls.mgr = glm.GLMManager(IEEE_13, model_is_path=True)
        cls.meter = cls.mgr.add_substation_meter()

    def test_meter_object(self):
        m = self.mgr.find_object(obj_type='meter', obj_name=self.meter)
        self.assertIsNotNone(m)
        self.assertEqual(m['name'], '"sourcebus_meter"')
        self.assertEqual(m['parent'], '"sourcebus"')

    def test_xfmr_updated(self):
        x = self.mgr.find_object(obj_type='transformer', obj_name='"xf_sub3"')
        self.assertEqual(x['from'], self.meter)


class AddSubstationMeter8500TestCase(unittest.TestCase):
    """Primitive test case for add_substation_meter with IEEE 8500 node.
    """
    @classmethod
    def setUpClass(cls):
        cls.mgr = glm.GLMManager(IEEE_8500, model_is_path=True)
        cls.meter = cls.mgr.add_substation_meter()

    def test_meter_object(self):
        m = self.mgr.find_object(obj_type='meter', obj_name=self.meter)
        self.assertIsNotNone(m)
        self.assertEqual(m['name'], '"sourcebus_meter"')
        self.assertEqual(m['parent'], '"sourcebus"')

    def test_line_updated(self):
        x = self.mgr.find_object(obj_type='overhead_line',
                                 obj_name='"line_hvmv_sub_hsb"')
        self.assertEqual(x['from'], self.meter)


class UpdateRegTapsTestCase(unittest.TestCase):
    """Test update_reg_taps"""
    @classmethod
    def setUpClass(cls):
        cls.mgr = glm.GLMManager(IEEE_13, model_is_path=True)

    def test_bad_reg_name(self):
        with self.assertRaisesRegex(ValueError, 'There is no regulator'):
            self.mgr.update_reg_taps('bad_reg', {'A': 0, 'B': 0, 'C': 0})

    def test_bad_dict_keys(self):
        with self.assertRaisesRegex(ValueError, "pos_dict's keys must be in"):
            self.mgr.update_reg_taps('"reg_Reg"', {'a': 7, 'b': 2, 'c': 9})

    def test_missing_regulator_configuration(self):
        mgr = glm.GLMManager(IEEE_13, model_is_path=True)
        mgr.remove_item({'object': 'regulator_configuration',
                         'name': '"rcon_Reg"'})
        with self.assertRaisesRegex(ValueError, 'While the regulator '):
            mgr.update_reg_taps('"reg_Reg"', {'A': 2, 'B': 3, 'C': 7})

    def test_tap_too_high(self):
        with self.assertRaisesRegex(ValueError, 'Given tap position, 17,'):
            self.mgr.update_reg_taps('"reg_Reg"', {'A': 2, 'B': 17, 'C': 7})

    def test_tap_too_low(self):
        with self.assertRaisesRegex(ValueError, 'Given tap position, -20,'):
            self.mgr.update_reg_taps('"reg_Reg"', {'A': 2, 'B': 6, 'C': -20})

    def test_non_integer_tap(self):
        with self.assertRaisesRegex(TypeError, 'Tap for phase A is not an'):
            self.mgr.update_reg_taps('"reg_Reg"', {'A': 2.0, 'B': 6, 'C': -2})

    def test_update_nonexistent_phase(self):
        # Get an independent manager.
        mgr = glm.GLMManager(IEEE_13, model_is_path=True)

        # Lookup the regulator.
        reg = mgr.find_object(obj_type='regulator', obj_name='"reg_Reg"')

        # Tweak its phases.
        reg['phases'] = 'AB'

        # Attempt to update all three phases.
        with self.assertRaisesRegex(ValueError, 'does not have phase'):
            mgr.update_reg_taps('"reg_Reg"', {'A': 2, 'B': 6, 'C': -8})

    def test_works(self):
        d = {'A': -7, 'B': 16, 'C': 0}
        self.mgr.update_reg_taps('"reg_Reg"', d)

        # Lookup and check.
        reg = self.mgr.find_object(obj_type='regulator', obj_name='"reg_Reg"')
        rc = self.mgr.find_object(obj_type='regulator_configuration',
                                  obj_name='"rcon_Reg"')

        for key, value in d.items():
            self.assertEqual(str(value), reg['tap_' + key])
            self.assertEqual(str(value), rc['tap_pos_' + key])


class UpdateCapSwitchesTestCase(unittest.TestCase):
    """Test update_cap_switches"""
    @classmethod
    def setUpClass(cls):
        cls.mgr = glm.GLMManager(IEEE_13, model_is_path=True)

    def test_bad_cap_name(self):
        with self.assertRaisesRegex(ValueError, 'There is no capacitor named'):
            self.mgr.update_cap_switches('"cap_cap3"', {'A': 'OPEN',
                                                        'B': 'CLOSED',
                                                        'C': 'CLOSED'})

    def test_bad_phase_dict(self):
        with self.assertRaisesRegex(ValueError, "phase_dict's keys must be"):
            self.mgr.update_cap_switches('"cap_cap1"', {'A': 'OPEN',
                                                        'b': 'CLOSED',
                                                        'C': 'CLOSED'})

    def test_bad_status(self):
        with self.assertRaisesRegex(ValueError, "Capacitor status must be in"):
            self.mgr.update_cap_switches('"cap_cap1"', {'A': 'OPEN',
                                                        'B': 'CLOSED',
                                                        'C': 'CLOsED'})

    def test_missing_phase(self):
        with self.assertRaisesRegex(ValueError, 'does not have phase'):
            self.mgr.update_cap_switches('"cap_cap2"', {'A': 'OPEN',
                                                        'B': 'CLOSED',
                                                        'C': 'CLOSED'})

    def test_works(self):
        d = {'A': 'OPEN', 'B': 'OPEN', 'C': 'OPEN'}
        self.mgr.update_cap_switches(cap_name='"cap_cap1"', phase_dict=d)

        # Lookup and check.
        cap = self.mgr.find_object(obj_type='capacitor', obj_name='"cap_cap1"')

        for phase, status in d.items():
            self.assertEqual(status, cap['switch' + phase])


if __name__ == '__main__':
    unittest.main()
