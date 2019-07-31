import unittest
from unittest.mock import patch
from copy import deepcopy

from tests import data_files as _df
from tests import models
from pyvvo.glm import GLMManager
from pyvvo import load_model

import pandas as pd


class LoadModelManager9500TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.load_nom_v = pd.read_csv(_df.LOAD_NOM_V_9500)
        cls.load_meas = pd.read_csv(_df.LOAD_MEAS_9500)
        cls.glm_mgr = GLMManager(models.IEEE_9500, model_is_path=True)
        cls.load_names_glm = \
            list(
                cls.glm_mgr.get_items_by_type(item_type='object',
                                              object_type=
                                              'triplex_load').keys()
            )

    def test_successful_init(self):
        """Given data for the same model, initialization should not
        throw any errors.
        """
        # Construct the manager.
        lm = load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                         load_measurements=self.load_meas,
                                         load_names_glm=self.load_names_glm)

        # Ensure the DataFrame columns are as expected.
        expected_columns = ['meas_type', 'meas_mrid', 'load_name']
        for c in list(lm.load_df.columns):
            self.assertIn(c, expected_columns)

    def test_misaligned_load_names(self):
        """Misaligned names leads to an incorrect length after merging.
        """
        load_meas = self.load_meas.copy(deep=True)
        load_meas.loc[0, 'load'] = 'bad_name'
        with self.assertRaisesRegex(ValueError, 'The number of triplex loads'):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=load_meas,
                                        load_names_glm=self.load_names_glm)

    def test_missing_meas(self):
        """Drop a measurement and ensure we get an error."""
        with self.assertRaisesRegex(ValueError,
                                    'The number of triplex loads in load'):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=
                                        self.load_meas.drop(index=0),
                                        load_names_glm=self.load_names_glm)

    def test_duplicate_measurement(self):
        """Having more than 4 measurements per triplex load should raise
        an exception.
        """
        df = self.load_meas.append(self.load_meas.iloc[0])
        with self.assertRaisesRegex(ValueError,
                                    'Each load should have four measurements'):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=df,
                                        load_names_glm=self.load_names_glm)

    def test_mismatched_load_names(self):
        """If load names don't match up, we should get an exception."""
        load_names_glm = deepcopy(self.load_names_glm)
        load_names_glm[0] = '"ld_bad_nameb"'
        with self.assertRaisesRegex(ValueError,
                                    'The load names given in load_nominal_vo'):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=self.load_meas,
                                        load_names_glm=load_names_glm)


class LoadModelManager13TestCase(unittest.TestCase):
    """The 13 bus model has only one triplex load, but several other
    loads at different voltages.
    """
    @classmethod
    def setUpClass(cls):
        cls.load_nom_v = pd.read_csv(_df.LOAD_NOM_V_13)
        cls.load_meas = pd.read_csv(_df.LOAD_MEAS_13)
        cls.glm_mgr = GLMManager(models.IEEE_13, model_is_path=True)
        cls.load_names_glm = \
            list(
                cls.glm_mgr.get_items_by_type(item_type='object',
                                              object_type=
                                              'triplex_load').keys()
            )

    def test_warns(self):
        """Should get a warning that not all loads are triplex."""
        with self.assertLogs(level='WARNING'):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=self.load_meas,
                                        load_names_glm=self.load_names_glm)


class LoadModelManager123TestCase(unittest.TestCase):
    """The 123 load model has not triplex loads."""
    @classmethod
    def setUpClass(cls):
        cls.load_nom_v = pd.read_csv(_df.LOAD_NOM_V_123)
        cls.load_meas = pd.read_csv(_df.LOAD_MEAS_123)
        cls.glm_mgr = GLMManager(models.IEEE_123, model_is_path=True)
        # The 123 node model shouldn't have any triplex loads.
        tl = cls.glm_mgr.get_items_by_type(item_type='object',
                                           object_type='triplex_load')
        assert tl is None
        cls.load_names_glm = []

    def test_fails(self):
        """No triplex loads means no load manager."""
        with self.assertRaisesRegex(ValueError, 'load_names_glm cannot be '):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=self.load_meas,
                                        load_names_glm=self.load_names_glm)


class FixLoadNameTestCase(unittest.TestCase):

    def test_one(self):
        self.assertEqual('234098ufs',
                         load_model.fix_load_name('"ld_234098ufsa"'))

    def test_two(self):
        self.assertEqual('234098ufs',
                         load_model.fix_load_name('"ld_234098ufsb"'))

    def test_patched(self):
        with patch.object(load_model, 'TRIPLEX_LOAD_PREFIX', 'mk785'):
            self.assertEqual('abcd',
                             load_model.fix_load_name('"mk785abcdb"'))

    def test_no_starting_quote(self):
        with self.assertRaisesRegex(ValueError,
                                    'must start and end with a double quote'):
            load_model.fix_load_name('ld_abcdb"')

    def test_no_ending_quote(self):
        with self.assertRaisesRegex(ValueError,
                                    'must start and end with a double quote'):
            load_model.fix_load_name('"ld_abcdb')

    def test_no_quotes(self):
        with self.assertRaisesRegex(ValueError,
                                    'must start and end with a double quote'):
            load_model.fix_load_name('ld_abcdb')

    def test_no_prefix(self):
        p = load_model.TRIPLEX_LOAD_PREFIX
        with self.assertRaisesRegex(ValueError,
                                    'must start with {}'.format(p)):
            load_model.fix_load_name('"l_stuffa"')

    def test_no_suffix(self):
        self.assertEqual('stuff', load_model.fix_load_name('"ld_stuff"'))

    def test_bad_input(self):
        with self.assertRaises(AttributeError):
            load_model.fix_load_name(75)


if __name__ == '__main__':
    unittest.main()
