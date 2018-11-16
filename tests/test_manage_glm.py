# Standard library imports
import unittest

# Import module to test
from pyvvo import manage_glm

# Define our test file.
TEST_FILE = 'test.glm'

# TODO: probably should test "sorted_write" and ensure GridLAB-D runs.
# This can't be done until we have a Docker container with GridLAB-D
# installed and configured.


class TestParseFile(unittest.TestCase):
    """Test parsing a test file."""

    @classmethod
    def setUpClass(cls):
        # Call parse
        cls._parsed_tokens = manage_glm.parse(TEST_FILE, True)

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


class TestGLMManager(unittest.TestCase):
    """Test the GLMManager class.

    This simply calls 'parse' to get its model_dict, which is tested in
    the TestParseFile class, so no need to heavily inspect the model_dict.

    NOTE: During this test, we'll add items.
    """

    @classmethod
    def setUpClass(cls):
        # Get a GLMManager object
        cls._GLMManager = manage_glm.GLMManager(TEST_FILE, True)

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
        self.assertRaises(UserWarning, self._GLMManager.add_item,
                          {'clock': 'clock'})

    def test_add_nonexistent_item_type_fails(self):
        self.assertRaises(UserWarning, self._GLMManager.add_item,
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

        self.assertRaises(UserWarning, self._GLMManager.modify_item, item)

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

        self.assertRaises(UserWarning, self._GLMManager.modify_item,
                          item)

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

    # TODO: model doesn't currently have a module which we can remove a
    # property from, because we modify the powerflow solver_method, and
    # cannot count on tests to run in order.


class TestGLMManagerRemove(unittest.TestCase):
    """Test the removal methods of the GLMManager class."""

    @classmethod
    def setUpClass(cls):
        # Get a GLMManager object
        cls._GLMManager = manage_glm.GLMManager(TEST_FILE, True)

    def test_remove_clock(self):
        # Remove clock.
        self._GLMManager.remove_item({'clock': 'clock'})

        # Ensure it's gone in the map.
        self.assertRaises(UserWarning, self._GLMManager._lookup_clock)

        # Make sure its gone in the model.
        self.assertNotIn(4, self._GLMManager.model_dict)

    def test_remove_powerflow(self):
        # Remove powerflow module
        self._GLMManager.remove_item({'module': 'powerflow'})

        # Ensure it's gone in the map.
        self.assertRaises(UserWarning, self._GLMManager._lookup_module,
                          'powerflow')

        # Ensure it's gone in the model.
        self.assertNotIn(3, self._GLMManager.model_dict)

    def test_remove_mysql(self):
        # Remove the mysql module. Interesting because its parsed as 'omftype'
        self._GLMManager.remove_item({'module': 'mysql'})

        # Ensure it's gone in the map.
        self.assertRaises(UserWarning, self._GLMManager._lookup_module,
                          'mysql')

        # Ensure its gone in the model.
        self.assertNotIn(1, self._GLMManager.model_dict)

    def test_remove_line_spacing_1(self):
        # Remove named object line_spacing_1
        self._GLMManager.remove_item({'object': 'line_spacing',
                                      'name': 'line_spacing_1'})

        # Ensure it's gone in the map.
        self.assertRaises(UserWarning, self._GLMManager._lookup_object,
                          'line_spacing', 'line_spacing_1')

        # Ensure its gone in the model.
        self.assertNotIn(7, self._GLMManager.model_dict)

    def test_remove_unnamed_object(self):
        # Not currently allowed.
        self.assertRaises(UserWarning, self._GLMManager.remove_item,
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
        cls._GLMManager = manage_glm.GLMManager(TEST_FILE, True)

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

    def test_find_object(self):
        # Test the find_object method.
        # TODO
        self.assertTrue(False)

    def test_get_objects_of_type(self):
        # TODO
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
