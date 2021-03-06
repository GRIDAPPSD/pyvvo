# Standard library imports
import unittest
from unittest.mock import patch, Mock
from datetime import datetime
import os
import logging
import re

# Import module to test
from pyvvo import glm, db
from pyvvo.utils import gld_installed, run_gld, read_gld_csv
from tests.models import MODEL_DIR, IEEE_13, IEEE_123_mod, IEEE_9500

# Installed packages.
import numpy as np
import pandas as pd

# Setup log.
LOG = logging.getLogger(__name__)

# Define our test files.
TEST_FILE = os.path.join(MODEL_DIR, 'test.glm')
TEST_FILE2 = os.path.join(MODEL_DIR, 'test2.glm')
TEST_FILE3 = os.path.join(MODEL_DIR, 'test3.glm')
TEST_FILE4 = os.path.join(MODEL_DIR, 'test4.glm')
EXPECTED4 = os.path.join(MODEL_DIR, 'test4_expected.glm')
TEST_SUBSTATION_METER = os.path.join(MODEL_DIR, 'test_substation_meter.glm')
TEST_INVERTER = os.path.join(MODEL_DIR, 'test_inverter_output.glm')
TEST_INVERTER_3_PHASE = os.path.join(MODEL_DIR,
                                     'test_three_phase_inverter_output.glm')
TEST_SWITCH_MOD = os.path.join(MODEL_DIR, 'test_switch_modifications.glm')

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

    def test_parse_dict_length_is_22(self):
        # Check out the model file - we expect 22 items.
        self.assertTrue(len(self._parsed_tokens) == 22)

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
        # Assert is ensures we're accessing the same place in memory.
        self.assertIs(self._GLMManager.model_dict[4],
                      self._GLMManager.model_map['clock'][1])

    def test_class_in_map(self):
        self.assertTrue(len(self._GLMManager.model_map['class']) > 0)

    def test_class_maps_correctly(self):
        self.assertIs(self._GLMManager.model_dict[21],
                      self._GLMManager.model_map['class']['my_class'][1])
        # Classes are tricky since they may have multiple fields like
        # "double myproperty[w]", which means dictionary keys can get
        # overridden. This is worked around by creating 'variable_types'
        # and 'variable_names' lists.
        self.assertIn('variable_types', self._GLMManager.model_dict[21])
        self.assertIn('variable_names', self._GLMManager.model_dict[21])
        # Should we also be adding something like 'variable_options'?
        # This gets really tricky...

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

    # TODO: These tests below mark that the "class" handling
    #   capabilities of the GLMManager are incomplete.
    """
    def test_modify_class(self):
        self.assertTrue(False, 'Have not yet added ability to modify a class.')

    def test_remove_class(self):
        self.assertTrue(False, 'Have not yet added ability to remove a class.')
    """

    def test_class_read_write(self):
        """Show a simple class can successfully be read and written."""
        c = """
        class my_class {
          double my_property[Hz];
          double other_stuff[miles];
        }
        """
        mgr = glm.GLMManager(model=c, model_is_path=False)
        c_out = mgr.write_model(out_path=None)

        # Ensure our strings are the same not counting white space.
        # Note this isn't a great test, but it'll do for now.
        self.assertEqual(re.sub(r"\s*", "", c), re.sub(r"\s*", "", c_out))

    def test_class_fails_with_enumeration(self):
        """Test to lock in the fact that we aren't supporting
        enumerations
        """

        c = """
        class some_silly_class {
          enumeration {OFF=0, ON=1} status;
        }
        """
        self.assertRaises(AssertionError, glm.parse, c, False)

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

        # Ensure it's in the model_dict
        self.assertIs(obj, self._GLMManager.model_dict[k])

        # Ensure the prepend key was updated.
        self.assertEqual(k-1, self._GLMManager.prepend_key)

    def test_add_clock_to_map_fails(self):
        # This model already has a clock.
        # TODO: should probably get a barebones model and test making a
        # bunch of additions.
        self.assertRaises(glm.ItemExistsError, self._GLMManager.add_item,
                          {'clock': 'clock'})

    def test_add_new_class_to_map(self):
        # Define the dictionary for the class to be added.
        cls = {'class': 'my_class_two', 'double': 'someProperty[in]'}

        # We're prepending classes - grab the key.
        k = self._GLMManager.prepend_key

        # Add it.
        self._GLMManager.add_item(cls)

        # Ensure it's in the model_dict.
        self.assertIs(cls, self._GLMManager.model_dict[k])

        # Ensure the prepend key was updated.
        self.assertEqual(k-1, self._GLMManager.prepend_key)

        # Ensure it's in the map.
        self.assertIs(cls,
                      self._GLMManager.model_map['class']['my_class_two'][1])

    def test_add_my_class_to_map_fails(self):
        # This model already has an instance of 'my_class' and thus
        # this should fail.
        self.assertRaises(glm.ItemExistsError, self._GLMManager.add_item,
                          {'class': 'my_class'})

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
        minimum_timestep = self.glm.model_dict[-7]

        self.assertDictEqual(minimum_timestep, {'#set': 'minimum_timestep=60'})

    def test_add_run_components_profiler(self):
        profiler = self.glm.model_dict[-6]

        self.assertDictEqual(profiler, {'#set': 'profiler=0'})

    def test_add_run_components_relax_naming_rules(self):
        rnr = self.glm.model_dict[-5]

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

    def test_add_run_components_reliability(self):
        """The reliability module should be present."""
        self.assertTrue(self.glm.module_present('reliability'))

    def test_add_run_components_fault_check(self):
        """Ensure the fault_check object gets added."""
        # Lookup the object by type.
        obj = self.glm.get_objects_by_type('fault_check')
        # None gets returned if it isn't present.
        self.assertIsNotNone(obj)
        # We should only have one fault_check object.
        self.assertEqual(1, len(obj))
        # Ensure all the required parameters are present.
        props = ['name', 'check_mode', 'eventgen_object', 'strictly_radial',
                 'grid_association']
        for p in props:
            self.assertIn(p, obj[0])

        # Grab the event_gen_object.
        name = obj[0]['eventgen_object']

        # Now, we should be able to find the 'eventgen' object by name.
        eventgen = self.glm.find_object('eventgen', name)
        self.assertIsNotNone(eventgen)

    def test_add_run_components_eventgen(self):
        """Ensure the eventgen object gets added."""
        # Lookup by type.
        obj = self.glm.get_objects_by_type('eventgen')
        self.assertIsNotNone(obj)
        self.assertEqual(len(obj), 1)
        self.assertIn('use_external_faults', obj[0])
        self.assertIn('name', obj[0])


class AddRunComponentsIEEE13NodeTestCase(AddRunComponentsTestCase):
    """Run tests in AddRunComponentsTestCase, but use IEEE 13 bus model.

    Some methods are overridden intentionally.
    """
    MODEL = IEEE_13

    def test_add_run_components_generators(self):
        """This model should have the generators added."""
        self.assertTrue(self.glm.module_present('generators'))

    def test_add_run_components_minimum_timestep(self):
        minimum_timestep = self.glm.model_dict[-8]

        self.assertDictEqual(minimum_timestep, {'#set': 'minimum_timestep=60'})

    def test_add_run_components_profiler(self):
        profiler = self.glm.model_dict[-7]

        self.assertDictEqual(profiler, {'#set': 'profiler=0'})

    def test_add_run_components_relax_naming_rules(self):
        rnr = self.glm.model_dict[-6]

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


class AddSubstationMeter9500TestCase(unittest.TestCase):
    """Primitive test case for add_substation_meter with IEEE 9500 node.
    """
    @classmethod
    def setUpClass(cls):
        cls.mgr = glm.GLMManager(IEEE_9500, model_is_path=True)
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


class IEEE9500RunsTestCase(unittest.TestCase):
    """Use add_run_components with the 9500 model and ensure it actually
    runs.
    """
    @classmethod
    def setUpClass(cls):
        cls.mgr = glm.GLMManager(IEEE_9500, model_is_path=True)
        cls.mgr.add_run_components(starttime=datetime(2012, 1, 1),
                                   stoptime=datetime(2012, 1, 1, 0, 15),
                                   timezone='UTC0', v_source=None,
                                   profiler=0, minimum_timestep=60)

        cls.out_file = 'test_9500.glm'
        cls.mgr.write_model(cls.out_file)

    @classmethod
    def tearDownClass(cls):
        # noinspection PyUnresolvedReferences
        os.remove(cls.out_file)

    @unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
    def test_add_run_components_model_runs(self):
        result = run_gld(model_path=self.out_file)

        self.assertEqual(0, result.returncode)


class ClearAllTriplexLoadsTestCase(unittest.TestCase):
    """Test GLMManager.clear_all_triplex_loads with TEST_FILE2, which
    does not have any triplex loads.
    """
    @classmethod
    def setUpClass(cls):
        # Get a manager with the 9500 node model.
        cls.mgr = glm.GLMManager(IEEE_9500, model_is_path=True)

        # To ensure this test doesn't give us false results, let's start
        # with some assertions right off the bat.
        tl_list = cls.mgr.get_objects_by_type(object_type='triplex_load')

        # Ensure we actually have triplex load objects.
        assert tl_list is not None

        # Ensure the first one has at least one of the properties.
        has_prop = False
        keys = tl_list[0].keys()
        for prop in glm.TRIPLEX_PARAMS:
            if prop in keys:
                has_prop = True

        assert has_prop

        # Alrighty, we shouldn't get any false positives at this point.
        # Clear out the triplex loads.
        cls.mgr.clear_all_triplex_loads()

    def test_properties_gone(self):
        """Ensure none of our triplex loads have any of the properties
        in glm.TRIPLEX_PARAMS.
        """
        # Get triplex loads from the manager.
        tl_list = self.mgr.get_objects_by_type(object_type='triplex_load')

        # Get the triplex params as a set.
        tl_params = set(glm.TRIPLEX_PARAMS)

        # Loop and assert.
        for tl in tl_list:
            params = set(tl.keys())
            self.assertTrue(params.isdisjoint(tl_params))

    def test_warns(self):
        """The TEST_FILE2 model doesn't have any triplex_loads, so we
        should get a warning.
        """
        # Get a manger.
        mgr = glm.GLMManager(TEST_FILE2, model_is_path=True)

        # Ensure we get a warning.
        with self.assertLogs(logger=mgr.log, level='WARN'):
            mgr.clear_all_triplex_loads()


class UpdateAllTriplexLoadsTestCase(unittest.TestCase):
    """Test GLMManager.update_all_triplex_loads."""
    @classmethod
    def setUpClass(cls):
        # Load up the 13 node model, which has a lone triplex load.
        cls.mgr = glm.GLMManager(IEEE_13)

        cls.mgr.update_all_triplex_loads(
            {'"ld_house"': {'bogus_param': 10}})

    def test_works(self):
        tl_list = self.mgr.get_objects_by_type(object_type='triplex_load')

        # Ensure we don't get any surprises if someone mods the model.
        self.assertEqual(1, len(tl_list))

        # Grab the triplex_load.
        tl = tl_list[0]

        self.assertIn('bogus_param', tl)
        self.assertEqual(tl['bogus_param'], str(10))

    def test_throws_key_error(self):
        with self.assertRaisesRegex(KeyError, 'The triplex_load blah was not'):
            self.mgr.update_all_triplex_loads(
                {'blah': {'stuff': 'and things'}}
            )


class RemoveAllSolarTestCase(unittest.TestCase):
    """Test the GLMManager's remove_all_solar method."""

    def test_remove_all_solar_solar_present(self):
        """Test the remove_all_solar method when there is solar in the
        model"""
        mgr = glm.GLMManager(model=EXPECTED4, model_is_path=True)
        # Start by ensuring we have solar to start with.
        solar = mgr.get_objects_by_type(object_type='solar')

        self.assertIsNotNone(solar)
        self.assertGreater(len(solar), 0)

        # Now, remove all the solar.
        with self.assertLogs(logger=mgr.log, level='INFO'):
            mgr.remove_all_solar()

        # Try to get solar.
        solar_after = mgr.get_objects_by_type(object_type='solar')
        self.assertIsNone(solar_after)

    def test_remove_all_solar_no_solar_present(self):
        """Test remove_all_solar operates correctly when the initial
        model does not have any solar in it.
        """
        mgr = glm.GLMManager(model=TEST_FILE)

        # Start by ensuring there's no solar.
        solar = mgr.get_objects_by_type(object_type='solar')
        self.assertIsNone(solar)

        # Attempt to remove the solar.
        with self.assertLogs(logger=mgr.log, level='WARNING'):
            mgr.remove_all_solar()


class IEEE123ModForYuanTestCase(unittest.TestCase):
    """Ensure we can load the IEEE 123 mod model without error."""

    @classmethod
    def setUpClass(cls):
        cls.in_file = IEEE_123_mod
        # Ensure the given model runs as is.
        # Nope, can't do this since it has #include lines.
        # result = run_gld(cls.in_file)

        # Read the model.
        cls.mgr = glm.GLMManager(model=cls.in_file, model_is_path=True)

        # Get the resulting model as a string.
        cls.out_str = cls.mgr.write_model(out_path=None)

    def test_model_class_correct(self):
        """Write model to file, check the class gets written right."""
        # Ensure the class didn't get messed up.
        expected_class_regex = \
            (r"class dummy_class \{\s+"
             r"double consensus_iterations;\s+"
             r"double theoretical_feeder_load;\s+"
             r"double wholesale_LMP;\s+"
             r"double aggregator_1_cleared_quantity;\s+"
             r"double aggregator_2_cleared_quantity;\s+"
             r"double aggregator_3_cleared_quantity;\s+"
             r"double aggregator_1_limit;\s+"
             r"double aggregator_2_limit;\s+"
             r"double aggregator_3_limit;\s+"
             r"\}"
             )

        # Start by ensuring it's present in the original model.
        with open(self.in_file, 'r') as f:
            s = f.read()

        result = re.search(expected_class_regex, s)
        self.assertIsNotNone(result)

        result = re.search(expected_class_regex, self.out_str)
        self.assertIsNotNone(result)

    def test_configuration_object_transformer(self):
        """Hard-coded test to ensure 'configuration object' syntax
        worked out correctly for a transformer.
        """
        # Extract a tranformer which has a nested configuration in the
        # original model.
        xfmr = self.mgr.find_object(obj_type='transformer',
                                    obj_name='CTTF_0_A_Meter_1')
        self.assertIsNotNone(xfmr)

        # Ensure it now references the name of a
        # transformer_configuration object.
        self.assertEqual(xfmr['configuration'],
                         'CTTF_0_A_Meter_1_configuration')

        # Extract the corresponding configuration.
        config = self.mgr.find_object(
            obj_type='transformer_configuration',
            obj_name='CTTF_0_A_Meter_1_configuration')

        self.assertIsNotNone(config)
        # There should be no 'parent' line (which gets added for other
        # types of nested objects).
        self.assertNotIn('parent', config)

    def test_triplex_line_nested_config(self):
        """Hard-coded test to ensure the "trip_line_config" got
        correctly "un-nested."
        """
        # Extract the object.
        tlc = self.mgr.find_object(obj_type='triplex_line_configuration',
                                   obj_name='trip_line_config')
        self.assertIsNotNone(tlc)

        # Check properties.
        self.assertEqual(tlc['conductor_1'],
                         'trip_line_config_conductor_1')

        self.assertEqual(tlc['conductor_2'],
                         'trip_line_config_conductor_2')

        self.assertEqual(tlc['conductor_N'],
                         'trip_line_config_conductor_N')

        # Check objects corresponding to those properties.
        for name in ['trip_line_config_conductor_1',
                     'trip_line_config_conductor_2',
                     'trip_line_config_conductor_N']:

            c = self.mgr.find_object(obj_type='triplex_line_conductor',
                                     obj_name=name)
            self.assertIsNotNone(c)
            # There should not be a parent property.
            self.assertNotIn('parent', c)

    def test_collector_group(self):
        """Ensure the unquoted syntax
        "group class=house AND groupid=noController;" is working."""
        found = False

        for entry in self.mgr.model_map['object_unnamed']:
            obj = entry[1]
            try:
                # We're looking for a collector with a group.
                if obj['object'] == 'collector':
                    if 'group' in obj:
                        if ' AND ' in obj['group']:
                            found = True
                            break
            except KeyError:
                # Nothing to see here.
                continue

        self.assertTrue(found)

# Create some expected results for the SubstationMeterTapeTestCase and
# the SubstationMeterMySQLTestCase. Note values are hard-coded based on
# what's in the TEST_SUBSTATION_METER testing file. Variables will be
# prefixed with "MT_" for "meter test"


# Recorder properties.
MT_REC_PROP = ['measured_real_energy', 'measured_real_power',
               'measured_reactive_power']

MT_REC_PROP_STR = '"{}"'.format(', '.join(MT_REC_PROP))


# One minute simulation, 5 second record interval.
MT_LEN = int(60 / 5)
# Ensure all our power values are as expected. HARD-CODING!
MT_EXPECTED_REAL = np.array([10000 * 3] * MT_LEN)
MT_EXPECTED_REACTIVE = np.array([1000 * 3] * MT_LEN)

# Now, ensure our energy is as expected. More hard coding coming
# right up. Note this is for "real" energy only.
# 10000 * 3 --> power
# 60 * 60 --> seconds in an hour
# 5 --> recorder interval in seconds
MT_ENERGY_PER_INTERVAL = 10000 * 3 / (60 * 60 / 5)
# noinspection PyTypeChecker
MT_ENERGY_P_I_ARRAY = \
    np.array([0] + [MT_ENERGY_PER_INTERVAL] * (MT_LEN - 1))
# Create a cumulative sum.
MT_ENERGY_CUM = np.cumsum(MT_ENERGY_P_I_ARRAY)


def helper_compare_data(data):
    """Helper function for the SubstationMeterTapeTestCase and
    SubstationMeterMySQLTestCase.
    """
    # Ensure we have the expected columns in our DataFrame.
    for c in MT_REC_PROP:
        assert c in data.columns.values

    # Ensure all our power values are as expected. HARD-CODING!
    np.testing.assert_array_equal(MT_EXPECTED_REAL,
                                  data['measured_real_power'].values)
    np.testing.assert_array_equal(MT_EXPECTED_REACTIVE,
                                  data['measured_reactive_power'].values)

    # Ensure energy values are as expected.
    np.testing.assert_allclose(MT_ENERGY_CUM,
                               data['measured_real_energy'].values,
                               rtol=1e-5, atol=0)


@unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
class SubstationMeterTapeTestCase(unittest.TestCase):
    """Test that we get expected results when adding a tape recorder to
    the TEST_SUBSTATION_METER file.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Create a GLMManager.
        cls.glm_mgr = glm.GLMManager(TEST_SUBSTATION_METER, model_is_path=True)

        # Define output file.
        cls.out_csv = 'sub_out.csv'

        # Add a recorder. Hard-coding to match what's in the model
        # rather than the more robust approach of programmatic
        # discovery.
        cls.glm_mgr.add_item({'object': 'recorder', 'file': cls.out_csv,
                              'name': 'substation_recorder',
                              'parent': '"sourcebus_meter"',
                              'property': MT_REC_PROP_STR,
                              'interval': 5,
                              'limit': -1})

        # Add the tape module.
        cls.glm_mgr.add_item({'module': 'tape'})

        # Write model to file.
        cls.out_model = 'test_sub_meter_tape.glm'
        cls.glm_mgr.write_model(cls.out_model)

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove the model.
        try:
            # noinspection PyUnresolvedReferences
            os.remove(cls.out_model)
        except FileNotFoundError:
            pass

        # Remove the csv file.
        try:
            # noinspection PyUnresolvedReferences
            os.remove(cls.out_csv)
        except FileNotFoundError:
            pass

    def test_run_and_expected(self):
        """Run the model, ensure results are as expected."""
        # Run model.
        result = run_gld(self.out_model)

        # Ensure success.
        self.assertEqual(result.returncode, 0)

        # Load the results.
        data = read_gld_csv(self.out_csv)

        helper_compare_data(data)


@unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
@unittest.skipIf(not DB_ENVIRON_PRESENT,
                 reason='Database environment variables are not present.')
class SubstationMeterMySQLTestCase(unittest.TestCase):
    """Test that we get expected results when adding a MySQL recorder to
    the TEST_SUBSTATION_METER file.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Create a GLMManager.
        cls.glm_mgr = glm.GLMManager(TEST_SUBSTATION_METER, model_is_path=True)

        # Add database object.
        cls.glm_mgr.add_item({'object': 'database',
                              'hostname': os.environ['DB_HOST'],
                              'username': os.environ['DB_USER'],
                              'password': os.environ['DB_PASS'],
                              'port': os.environ['DB_PORT'],
                              'schema': os.environ['DB_DB']})

        # Add a recorder. Hard-coding to match what's in the model
        # rather than the more robust approach of programmatic
        # discovery.
        cls.table = 'sub_tmp'
        cls.glm_mgr.add_item({'object': 'mysql.recorder',
                              'table': cls.table,
                              'name': 'substation_recorder',
                              'parent': '"sourcebus_meter"',
                              'property': MT_REC_PROP_STR,
                              'interval': 5,
                              'limit': -1,
                              'mode': 'w',
                              'query_buffer_limit': 20000})

        # Add the MySQL module.
        cls.glm_mgr.add_item({'module': 'mysql'})

        # Write model to file.
        cls.out_model = 'test_sub_meter_tape.glm'
        cls.glm_mgr.write_model(cls.out_model)

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove the model.
        try:
            # noinspection PyUnresolvedReferences
            os.remove(cls.out_model)
        except FileNotFoundError:
            pass

        # Drop the table.
        db_conn = db.connect_loop(timeout=1)
        result = db.execute_and_fetch_all(db_conn=db_conn,
                                          query='DROP TABLE {};'.format(
                                              cls.table))

        assert result == tuple()

    def test_run_and_expected(self):
        """Run the model, ensure results are as expected."""
        # Run model.
        result = run_gld(self.out_model)

        # Ensure success.
        self.assertEqual(result.returncode, 0)

        # Load the results.
        db_conn = db.connect_loop(timeout=1)
        data = pd.read_sql(sql='SELECT * FROM {};'.format(self.table),
                           con=db_conn)
        # Can't use read_sql_table without SQLAlchemy.
        # data = pd.read_sql_table(table_name=self.table,
        #                          con=db_conn)

        # The tape and MySQL recorders handle the stopping time
        # differently. So, we'll send in all data except the last row.

        helper_compare_data(data[0:-1])


@unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
class InverterOutputTestCase(unittest.TestCase):
    """Ensure that given a model with an inverter that lacks an explicit
    DC source and if the inverter is in CONSTANT_PQ mode, the output
    of the inverter will be the correct power.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Set up the manager and extract the inverter and recorder.
        cls.mgr = glm.GLMManager(TEST_INVERTER)
        cls.inverter = cls.mgr.find_object(obj_type='inverter',
                                           obj_name='"pv_inverter"')
        cls.recorder = cls.mgr.find_object(obj_name='inverter_recorder',
                                           obj_type='recorder')
        # the run_gld helper runs the model from its directory.
        cls.out_file = os.path.join(MODEL_DIR, cls.recorder['file'])

        # Run the model.
        result = run_gld(TEST_INVERTER)
        assert result.returncode == 0

        # Read the file.
        cls.output = read_gld_csv(cls.out_file)

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove the file the recorder writes.
        # noinspection PyUnresolvedReferences
        os.remove(cls.out_file)

    def test_inv_pq(self):
        """Ensure the inverter is in CONSTANT_PQ mode."""
        self.assertEqual(self.inverter['four_quadrant_control_mode'],
                         'CONSTANT_PQ')
        self.assertEqual(self.inverter['inverter_type'],
                         'FOUR_QUADRANT')

    def test_results_match_settings(self):
        """Ensure the inverters P_Out and Q_Out match what's in the
        output file.
        """
        p = float(self.inverter['P_Out'])
        q = float(self.inverter['Q_Out'])
        va = p + 1j*q

        self.assertTrue((self.output['P_Out'].values == p).all())
        self.assertTrue((self.output['Q_Out'].values == q).all())
        self.assertTrue((self.output['VA_Out'].values == va).all())


@unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
class InverterThreePhaseOutputTestCase(unittest.TestCase):
    """Test case for proving that inverter properties P_Out and Q_Out
    are three phase, despite not being documented. A ticket has been
    opened to improve the documentation:

    https://github.com/gridlab-d/gridlab-d/issues/1201
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.mgr = glm.GLMManager(TEST_INVERTER_3_PHASE)

        cls.inverter = cls.mgr.find_object(obj_type='inverter',
                                           obj_name='"three_phase_inv"')
        cls.recorder_inv = cls.mgr.find_object(obj_name='inverter_recorder',
                                               obj_type='recorder')
        cls.recorder_meter = cls.mgr.find_object(obj_name='meter_recorder',
                                                 obj_type='recorder')

        # the run_gld helper runs the model from its directory.
        cls.out_file_inv = os.path.join(MODEL_DIR, cls.recorder_inv['file'])
        cls.out_file_meter = os.path.join(MODEL_DIR, cls.recorder_meter['file'])

        # Run the model.
        result = run_gld(TEST_INVERTER_3_PHASE)
        assert result.returncode == 0

        # Read the results.
        cls.results_inv = read_gld_csv(cls.out_file_inv)
        cls.results_meter = read_gld_csv(cls.out_file_meter)

    # noinspection PyUnresolvedReferences
    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.out_file_inv)
        os.remove(cls.out_file_meter)

    def test_inv_pq(self):
        """Ensure the inverter is in CONSTANT_PQ mode."""
        self.assertEqual(self.inverter['four_quadrant_control_mode'],
                         'CONSTANT_PQ')
        self.assertEqual(self.inverter['inverter_type'],
                         'FOUR_QUADRANT')

    def test_inverter_output(self):
        """Ensure inverter measurements are as expected."""
        p = float(self.inverter['P_Out'])
        q = float(self.inverter['Q_Out'])
        va = p + 1j*q

        self.assertTrue((self.results_inv['P_Out'].values == p).all())
        self.assertTrue((self.results_inv['Q_Out'].values == q).all())
        self.assertTrue((self.results_inv['VA_Out'].values == va).all())

    def test_meter_output(self):
        va = float(self.inverter['P_Out']) + 1j * float(self.inverter['Q_Out'])

        for p in ['A', 'B', 'C']:
            # Show that the VA_Out is really three phase, so it should
            # be /3 for each individual phase.
            # noinspection PyTypeChecker
            np.testing.assert_allclose(
                self.results_meter[f'measured_power_{p}'].values, -va / 3,
                rtol=1e-4, atol=0
            )


class GetItemsAndObjectsAfterRemovalTestCase(unittest.TestCase):
    """Test GLMManager methods get_items_by_type and get_objects_by_type
    to ensure they return None in the edge case where the model map
    shows the type being present, but the objects have been removed.
    """
    @classmethod
    def setUpClass(cls) -> None:
        # Load model.
        cls.mgr = glm.GLMManager(TEST_FILE)

        # Get overhead lines.
        lines = cls.mgr.get_objects_by_type(object_type='overhead_line')

        assert len(lines) > 0

        # Remove the lines.
        for line in lines:
            cls.mgr.remove_item(line)

    def test_lines_in_map(self):
        self.assertIn('overhead_line', self.mgr.model_map['object'])
        self.assertEqual(0, len(self.mgr.model_map['object']['overhead_line']))

    def test_get_objects_by_type_returns_none(self):
        self.assertIsNone(self.mgr.get_objects_by_type('overhead_line'))

    def test_get_items_by_type_returns_none(self):
        self.assertIsNone(
            self.mgr.get_items_by_type(item_type='object',
                                       object_type='overhead_line'))


class SetInverterVAndITestCase(unittest.TestCase):
    """Test the GLMManager's set_inverter_v_and_i method."""
    MODEL = \
        """
        object inverter {
          name inv1;
          rated_power 25000;
        }
        object inverter {
          name inv2;
        }
        """

    def test_set_inverter_v_and_i(self):
        # Initialize manager.
        mgr = glm.GLMManager(model=self.MODEL, model_is_path=False)

        # Call method, ensuring we get a warning for the lack of
        # rated_power for inv2.
        with self.assertLogs(logger=mgr.log, level='WARNING'):
            mgr.set_inverter_v_and_i()

        # Ensure v and i are expected for inv1.
        inv1 = mgr.find_object(obj_type='inverter', obj_name='inv1')
        v_1 = float(inv1['V_In'])
        i_1 = float(inv1['I_In'])
        rp = float(inv1['rated_power'])
        self.assertEqual(rp * 1.1, v_1 * i_1)

        # Ensure v and i are expected for inv2. It has not rated power,
        # so they're both set to 10000.
        inv2 = mgr.find_object(obj_type='inverter', obj_name='inv2')
        v_2 = float(inv2['V_In'])
        i_2 = float(inv2['I_In'])
        self.assertEqual(v_2, 10000)
        self.assertEqual(i_2, 10000)


class LoopOverObjectsHelperTestCase(unittest.TestCase):
    """Test the loop_over_objects_helper method of GLMManager."""
    @classmethod
    def setUpClass(cls) -> None:
        cls.mgr = glm.GLMManager(model=TEST_FILE)

    def test_missing_object_type(self):
        with self.assertRaisesRegex(KeyError, 'The given object_type bleh'):
            self.mgr.loop_over_objects_helper(object_type='bleh', func=print)

    def test_removed_objects(self):
        mgr = glm.GLMManager(model=EXPECTED4)
        mgr.remove_all_solar()
        with self.assertRaisesRegex(KeyError, 'The given object_type solar'):
            mgr.loop_over_objects_helper(object_type='solar', func=print)

    def test_func_called(self):
        f = Mock()
        self.mgr.loop_over_objects_helper('load', f, 7, silly_arg='hello')
        # There are three loads in this model.
        self.assertEqual(f.call_count, 3)
        # Ensure the function is being called correctly.
        for ca in f.call_args_list:
            args = ca[0]
            kwargs = ca[1]
            self.assertEqual(len(args), 2)
            self.assertIsInstance(args[0], dict)
            self.assertEqual(args[1], 7)
            self.assertDictEqual(kwargs, {'silly_arg': 'hello'})

    def test_runtime_error_for_removal(self):
        mgr = glm.GLMManager(model=TEST_FILE)
        with self.assertRaises(RuntimeError):
            mgr.loop_over_objects_helper(object_type='load',
                                         func=mgr.remove_item)


class ConvertSwitchStatusToThreePhaseTestCase(unittest.TestCase):
    """Test the convert_switch_status_to_three_phase method of the
    GLMManager.
    """
    @classmethod
    def tearDownClass(cls) -> None:
        # Model written in test_model_runs_after_modifications
        os.remove('tmp.glm')

    def test_no_switches(self):
        mgr = glm.GLMManager(EXPECTED4)
        with self.assertLogs(logger=mgr.log, level='WARNING'):
            mgr.convert_switch_status_to_three_phase()

    def test_well_formed_switches(self):
        model = \
            """
            object switch {
                name switch1;
                phases ABCN;
                status OPEN;
            }
            object switch {
                name switch2;
                phases AC;
                status CLOSED;
            }
            """

        mgr = glm.GLMManager(model=model, model_is_path=False)

        with self.assertLogs(logger=mgr.log, level='INFO'):
            mgr.convert_switch_status_to_three_phase(banked=False)

        # Ensure our switches are as they should be.
        switch1 = mgr.find_object(obj_type='switch', obj_name='switch1')
        for p in ['A', 'B', 'C']:
            self.assertEqual(switch1[f'phase_{p}_state'], 'OPEN')

        self.assertNotIn('status', switch1)
        self.assertEqual(switch1['operating_mode'], 'INDIVIDUAL')

        switch2 = mgr.find_object(obj_type='switch', obj_name='switch2')
        for p in ['A', 'C']:
            self.assertEqual(switch2[f'phase_{p}_state'], 'CLOSED')

        self.assertNotIn('phase_B_state', switch2)
        self.assertNotIn('status', switch2)
        self.assertEqual(switch2['operating_mode'], 'INDIVIDUAL')

    def test_malformed_switch(self):
        model = \
            """
            object switch {
                name ms;
                phases A;
            }
            """

        mgr = glm.GLMManager(model, model_is_path=False)
        with self.assertLogs(logger=mgr.log, level='WARNING') as cm:
            mgr.convert_switch_status_to_three_phase(banked=True)

        self.assertEqual(len(cm.output), 1)
        self.assertIn('Switch ms does not have the "status" attribute',
                      cm.output[0])

        s = mgr.find_object(obj_name='ms', obj_type='switch')
        self.assertEqual(s['phase_A_state'], 'CLOSED')
        self.assertNotIn('status', s)
        self.assertNotIn('phase_B_state', s)
        self.assertNotIn('phase_C_state', s)
        self.assertEqual(s['operating_mode'], 'BANKED')

    def check_for_error(self, std):
        """Given binary stderr/stdout, ensure there is not an error."""
        self.assertFalse('error' in std.decode('utf-8').lower())

    @unittest.skipIf(not gld_installed(), reason='GridLAB-D is not installed.')
    def test_model_runs_after_modifications(self):
        # Start by ensuring the model runs before modifications.
        result1 = run_gld(model_path=TEST_SWITCH_MOD)
        self.assertEqual(result1.returncode, 0)
        self.check_for_error(result1.stderr)
        self.check_for_error(result1.stdout)

        # Modify the model.
        mgr = glm.GLMManager(model=TEST_SWITCH_MOD, model_is_path=True)
        mgr.convert_switch_status_to_three_phase(banked=False)

        # Write out to file.
        mgr.write_model(out_path='tmp.glm')

        # Run it.
        result2 = run_gld('tmp.glm')
        self.assertEqual(result2.returncode, 0)
        self.check_for_error(result2.stderr)
        self.check_for_error(result2.stdout)


if __name__ == '__main__':
    unittest.main()
