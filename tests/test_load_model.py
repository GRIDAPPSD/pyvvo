import unittest
from unittest.mock import patch, MagicMock
from copy import deepcopy
import time
import multiprocessing as mp
import threading
import queue
import simplejson as json

from tests import data_files as _df
from tests import models
from pyvvo.glm import GLMManager
from pyvvo import load_model, timeseries, zip, gridappsd_platform, sparql, \
    utils
from datetime import datetime

import numpy as np
import pandas as pd


class LoadModelManager9500TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.load_nom_v = pd.read_csv(_df.LOAD_NOM_V_9500)
        cls.load_meas = pd.read_csv(_df.LOAD_MEAS_9500)
        cls.glm_mgr = GLMManager(models.IEEE_9500, model_is_path=True)
        cls.load_names_glm = \
            list(
                cls.glm_mgr.get_items_by_type(
                    item_type='object', object_type='triplex_load').keys()
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

    def test_fit_for_all(self):
        self.assertTrue(False, "Need to finish this method.")
        #
        # # TODO: Migrate this method elsewhere to use the modified 123
        # #   node model.
        #
        # # Construct the manager.
        # lm = load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
        #                                  load_measurements=self.load_meas,
        #                                  load_names_glm=self.load_names_glm)
        #
        # # Fit for all, using the appropriate times.
        # lm.fit_for_all()

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
            load_model.LoadModelManager(
                load_nominal_voltage=self.load_nom_v,
                load_measurements=self.load_meas.drop(index=0),
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


class MockOptimizeResult:
    """Helper mock class since we can't pickle Mock/MagicMock objects.
    """
    def __init__(self, success, status, message):
        self.success = success
        self.status = status
        self.message = message


class LoadModelManager13TestCase(unittest.TestCase):
    """The 13 bus model has only one triplex load, but several other
    loads at different voltages.

    This test case will also have an initialized LoadModelManager to
    ease testing some methods like _start_processes and _stop_processes.
    """

    @classmethod
    def setUpClass(cls):
        cls.load_nom_v = pd.read_csv(_df.LOAD_NOM_V_13)
        cls.load_meas = pd.read_csv(_df.LOAD_MEAS_13)
        cls.glm_mgr = GLMManager(models.IEEE_13, model_is_path=True)
        cls.load_names_glm = \
            list(
                cls.glm_mgr.get_items_by_type(
                    item_type='object', object_type='triplex_load').keys()
            )
        # Create a manager.
        cls.load_mgr = load_model.LoadModelManager(
            load_nominal_voltage=cls.load_nom_v,
            load_measurements=cls.load_meas,
            load_names_glm=cls.load_names_glm)

    def test_warns(self):
        """Should get a warning that not all loads are triplex."""
        with self.assertLogs(level='WARNING'):
            load_model.LoadModelManager(load_nominal_voltage=self.load_nom_v,
                                        load_measurements=self.load_meas,
                                        load_names_glm=self.load_names_glm)

    def test_start_stop_processes(self):
        """Test _start_processes and stop_processes methods."""
        self.load_mgr._start_processes()

        for p in self.load_mgr.processes:
            self.assertTrue(p.is_alive())

        # Attempting to start the processes again should warn.
        with self.assertLogs(logger=self.load_mgr.log, level='WARNING'):
            self.load_mgr._start_processes()

        # Now, stop the processes.
        self.load_mgr._stop_processes()

        # The processes attribute should now be None.
        self.assertIsNone(self.load_mgr.processes)

        # Attempting to stop the processes again should warn.
        with self.assertLogs(logger=self.load_mgr.log, level='WARNING'):
            self.load_mgr._stop_processes()

    def test_logging_worker(self):
        """Put a log entry into the logging queue and ensure things
        work.
        """
        # Create a successful optimize result
        sol = MockOptimizeResult(success=True, status=42, message="Testing...")

        d = {'load_name': 'test_name', 'time': 1.4598, 'clusters': 3,
             'data_samples': 10, 'sol': sol}

        with self.assertLogs(logger=load_model.LOG, level='INFO'):
            self.load_mgr.logging_queue.put(d)
            # Sleep to let all the work happen.
            time.sleep(0.01)

        # Now, create an entry which should trigger a warning.
        sol = MockOptimizeResult(success=False, status=42, message="Testing...")

        d['sol'] = sol

        with self.assertLogs(logger=load_model.LOG, level='WARNING'):
            self.load_mgr.logging_queue.put(d)
            time.sleep(0.01)


class LoadModelManager123TestCase(unittest.TestCase):
    """The 123 load model has no triplex loads."""

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


class LoadModelManagerModified123TestCase(unittest.TestCase):
    """As of platform version v2019.08.0, there should exist a version
    of the 123 node model with triplex_loads. This will be a good model
    to run tests on since they won't take as long to run as the 9500
    node model.
    """
    def test_stuff(self):
        self.assertTrue(False, "Write these tests once the time series API"
                        " is fixed.")


class TransformDataForLoadTestCase(unittest.TestCase):
    """Test transform_data_for_load"""

    @classmethod
    def setUpClass(cls):
        # Grab the all measurement information for a single node/load.
        cls.meas_data = _df._get_9500_meas_data_for_one_node()

        # Get dates corresponding to measurements.
        cls.starttime = _df.SENSOR_MEASUREMENT_TIME_START
        cls.endtime = _df.SENSOR_MEASUREMENT_TIME_END

        # Load up the time series data.
        ts = _df.read_pickle(_df.PARSED_SENSOR_9500_ALL)

        # Combine it with the meas data map.
        grouped = load_model._drop_merge_group(map_df=cls.meas_data,
                                               data_df=ts)

        # Ensure grouped just has one element.
        assert len(grouped) == 1

        # Assign the group data.
        for _, group in grouped:
            cls.df = group

    def test_runs(self):
        """Use our testing data from the platform to ensure this works
        all the way through.
        """
        # Call the function. It should run without error.
        result = load_model.transform_data_for_load(meas_data=self.df)

        # Now, time to ensure we got back what we expected.

        # Start by ensuring the columns are correct.
        self.assertSetEqual({'p', 'q', 'v'}, set(result.columns.tolist()))

        # Now, we should have one fourth the rows of the original,
        # since we've taken the four triplex measurements and added
        # them to effectively create one measurement per time step.
        self.assertEqual(self.df.shape[0] / 4, result.shape[0])

        # Next up, ensure our index is time.
        self.assertEqual('time', result.index.name)

    # noinspection PyMethodMayBeStatic
    def test_expected(self):
        """Create expected results, create inputs by deconstructing
        them.
        """
        # Initialize expected return.
        t = [datetime(2019, 11, 27, 8), datetime(2019, 11, 27, 9)]
        expected = pd.DataFrame(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                columns=['v', 'p', 'q'], index=t)
        expected.index.name = 'time'

        # Extract v from our expected.
        v = expected['v'].values

        # Divide by 3.
        v_3 = v / 3

        v_angle = pd.Series(np.zeros_like(v_3))

        # Create type lists for PNV and VA.
        pnv_type = ['PNV', 'PNV']
        va_type = ['VA', 'VA']

        # Create DataFrame for v, which holds 1/3 of the sum.
        df1 = pd.DataFrame(data={'magnitude': v_3, 'angle': v_angle,
                                 'type': pnv_type, 'time': t})

        # Second DataFrame for v should hold 2/3 of sum.
        df2 = pd.DataFrame(data={'magnitude': 2 * v_3, 'angle': v_angle})

        # Now, create complex numbers for VA.
        va = expected['p'].values + 1j * expected['q'].values

        # Divide our va by 4.
        va_4 = va / 4

        # Create our first DataFrame for VA
        df3 = pd.DataFrame(data={'magnitude': np.abs(va_4),
                                 'angle': np.angle(va_4, deg=True),
                                 'type': va_type, 'time': t})
        # Now, our second.
        df4 = pd.DataFrame(data={'magnitude': np.abs(3 * va_4),
                                 'angle': np.angle(3 * va_4, deg=True),
                                 'type': va_type, 'time': t})

        # Concatenate all our DataFrames. Order should not matter.
        df_in = pd.concat([df1, df4, df3, df2], axis=0)

        # We're ready to call the function.
        actual = load_model.transform_data_for_load(meas_data=df_in)

        # Ensure our sorted frames match after sorting.
        pd.testing.assert_frame_equal(expected.sort_index(),
                                      actual.sort_index())


class FitForLoadTestCase(unittest.TestCase):
    """Test fit_for_load.

    TODO: This method could use some more tests...
    """

    @classmethod
    def setUpClass(cls):
        cls.load_data = _df.read_pickle(_df.PARSED_SENSOR_VPQ)
        cls.weather_data = _df.read_pickle(_df.WEATHER_FOR_SENSOR_DATA_9500)
        cls.weather_data = timeseries.fix_ghi(cls.weather_data)

    @patch.dict(load_model.CONFIG['load_model'],
                {'averaging_interval': '1Min',
                 'filtering_interval_minutes': 5})
    @patch('pyvvo.timeseries.filter_by_time',
           wraps=timeseries.filter_by_time)
    @patch('pyvvo.timeseries.filter_by_weekday',
           wraps=timeseries.filter_by_weekday)
    @patch('pyvvo.timeseries.up_or_down_sample',
           wraps=timeseries.up_or_down_sample)
    @patch('pyvvo.zip.get_best_fit_from_clustering',
           wraps=zip.get_best_fit_from_clustering)
    @patch('pyvvo.timeseries.resample_timeseries',
           wraps=timeseries.resample_timeseries)
    def test_runs(self, p_resample, p_fit, p_up_or_down, p_filter_weekday,
                  p_filter_time):
        """Ensure that with real data, it runs."""
        # Run the function. Note all the patching.
        output = load_model.fit_for_load(
            load_data=self.load_data,
            weather_data=self.weather_data)

        # Ensure patched methods were called once.
        p_resample.assert_called_once()
        p_fit.assert_called_once()
        p_up_or_down.assert_called_once()
        p_filter_weekday.assert_called_once()
        p_filter_time.assert_called_once()

        # Since our load data is 3 second (for now) and we're passing an
        # interval_str of '1Min', we should be downsampling.
        self.assertEqual('downsample', p_resample.call_args[1]['method'])

        # Ensure the interval string was passed through.
        self.assertEqual('1Min', p_resample.call_args[1]['interval_str'])

        # Ensure the output looks as expected.
        self.assertIsInstance(output, dict)
        self.assertIsInstance(output['zip_gld'], dict)
        self.assertTrue(output['sol'].success)
        self.assertEqual('Optimization terminated successfully.',
                         output['sol'].message)
        self.assertIsInstance(output['p_pred'], np.ndarray)
        self.assertIsInstance(output['q_pred'], np.ndarray)
        self.assertIn('mse_p', output)
        self.assertIn('mse_q', output)
        self.assertIn('data_len', output)
        self.assertIn('k', output)


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


class GetDataAndFitTestCase(unittest.TestCase):
    """Test get_data_and_fit. Since get_data_and_fit is a very simple
    wrapper, we'll patch the functions it calls and check that they were
    called correctly.
    """

    @patch('pyvvo.load_model.fit_for_load', return_value=17)
    @patch('pyvvo.load_model.get_data_for_load', return_value=10)
    def test_correct_calls(self, p_gdfl, p_ffl):
        """Simply call function, check patched functions used."""
        d1 = {'some': 'stuff', 'for': 'function'}
        d2 = {'more': 'things', '42': 24}

        result = load_model.get_data_and_fit(gdfl_kwargs=d1, ffl_kwargs=d2)

        p_gdfl.assert_called_once()
        p_gdfl.assert_called_with(**d1)

        p_ffl.assert_called_once()
        p_ffl.assert_called_with(load_data=10, **d2)

        self.assertEqual(result, 17)


class QueueFeederTestCase(unittest.TestCase):
    """Test queue_feeder method."""

    @classmethod
    def setUpClass(cls) -> None:
        # TODO: Switch back to files later.
        # cls.load_meas = _df.read_pickle(_df.LOAD_MEAS_9500)
        s_mgr = sparql.SPARQLManager(_df.FEEDER_MRID_9500)
        cls.load_meas = s_mgr.query_load_measurements()

    def _check_dataframe(self, df):
        # TODO: Finish this method.
        # Each DataFrame should correspond to a single piece of
        # equipment.
        self.assertEqual(1, len(df['eqid'].unique()))

        # Each DataFrame should have four unique measurements.
        self.assertEqual(4, len(df['measurement_mrid'].unique()))

        # Each DataFrame should not be indexed by time, but rather have
        # time in its columns.
        self.assertIn('time', df.columns)

        # Each DataFrame should also have magnitude and angle in the
        # columns.
        self.assertIn('magnitude', df.columns)
        self.assertIn('angle', df.columns)

        # No NaNs allowed.
        self.assertFalse(df.isna().any().any())

    def test_stuff(self):
        # Triplex loads have 4 measurements per load.
        meas_per_load = 4
        # Let's set things up to grab 1/5 of the loads up front.
        initial_n = int(self.load_meas.shape[0] / meas_per_load / 5)
        # Grab 90% of the initial size each time.
        subsequent_n = int(initial_n * 0.9)

        # Initialize queue.
        q = mp.Queue()

        # TODO: hard-coding simulation ID...
        kwargs = {'load_measurements': self.load_meas,
                  'data_queue': q, 'simulation_id': '1285593388',
                  'starttime': None, 'endtime': None,
                  'initial_n': initial_n, 'subsequent_n': subsequent_n,
                  'meas_per_load': meas_per_load}

        feeder = load_model.QueueFeeder(**kwargs)

        self.assertIsNone(feeder.starttime)
        self.assertIsNone(feeder.endtime)

        self.assertFalse(feeder.done)

        # TODO: mock calls to platform.
        t = threading.Thread(target=feeder.run)
        t.start()

        # Check our queue size after we receive the signal.
        self.assertEqual(1, feeder.signal_queue.get(timeout=10))
        self.assertEqual(initial_n, q.qsize())

        # Loader function should be called once.
        self.assertEqual(feeder.load_count, 1)

        # Pull number of objects equal to the threshold.
        get_count = 0
        for n in range(subsequent_n):
            self.assertEqual(initial_n - n, q.qsize())
            self.assertEqual(feeder.load_count, 1)
            self._check_dataframe(q.get(timeout=0.1))
            get_count += 1

        # After a bit, our queue size should go right back up.
        self.assertEqual(initial_n - subsequent_n, q.qsize())
        self.assertEqual(feeder.signal_queue.get(timeout=10), 2)

        # Pull objects out of the queue until we've gone through all of
        # them. Note that at this point we've extracted subsequent_n
        # items from the queue.
        for i in range(int(self.load_meas.shape[0] / meas_per_load)
                       - subsequent_n):
            self._check_dataframe(q.get(timeout=15))
            get_count += 1

        # Sanity check to ensure I didn't do anything dumb.
        self.assertEqual(get_count, self.load_meas.shape[0] / meas_per_load)

        # Ensure the call count is as expected.
        self.assertEqual(
            1 + np.ceil((self.load_meas.shape[0] / meas_per_load - initial_n)
                        / subsequent_n),
            feeder.load_count
        )

        # Ensure we're done.
        self.assertTrue(feeder.done)
        pass


if __name__ == '__main__':
    unittest.main()
