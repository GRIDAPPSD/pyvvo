"""Module for managing load modeling: take data from the platform,
manipulate it, then perform a ZIP fit.
"""
# Standard library:
import logging
from datetime import timedelta
import time
import multiprocessing as mp
import threading
import queue

# Third party:
import numpy as np
import pandas as pd

# pyvvo:
from pyvvo.gridappsd_platform import PlatformManager
from pyvvo import utils, timeseries, zip

LOG = logging.getLogger(__name__)

# The GridLAB-D triplex loads (from the platform) start with ld_ while
# the loads straight from CIM do not.
TRIPLEX_LOAD_PREFIX = 'ld_'
# In CIM, triplex loads come in as 208V, when they should be 120.
# This is a known issue that won't be fixed.
CIM_TRIPLEX_VOLTAGE = 208
# At some point, the platform started keeping separate EnergyConsumer
# objects for each triplex load phase. These seem to be suffixed with
# 'a' and 'b'.
CIM_TRIPLEX_SUFFIX_SET = {'a', 'b'}
# For fitting, we need to use a nominal voltage.
FIT_NOMINAL_VOLTAGE = 240

# Get the configuration. TODO: We may want to load this dynamically.
CONFIG = utils.read_config()


class QueueFeeder:

    def __init__(self, load_measurements: pd.DataFrame, simulation_id: str,
                 data_queue: mp.Queue,
                 initial_n: int, subsequent_n: int, meas_per_load: int = 4,
                 starttime=None, endtime=None,
                 ):

        """
        Class to continuously feed a queue with historic load data
        from the platform. After initialization, attach a thread to
        the object's "run" method. This class's singular purpose is to
        get load data from the platform and put it into a queue.
        However, it does so in a way that is memory friendly - we don't
        go and ask for data for all loads at once.

        :param load_measurements: DataFrame as would come from
            sparql.SPARQLManager.query_load_measurements()
        :param simulation_id:
        :param data_queue: Multiprocessing queue to put data into.
        :param initial_n: Initial number of
        :param subsequent_n:
        :param meas_per_load: Number of measurements per load. Will always
            be 4 for triplex loads (one VA and one PNV measurement per
            phase).
        :param starttime: Passed directly to
            gridappsd_platform.PlatformManager.get_simulation_output
        :param endtime: Passed directly to
            gridappsd_platform.PlatformManager.get_simulation_output
        """
        # Ensure we have the correct number of measurements per load.
        if (load_measurements.groupby('eqid').size() != meas_per_load).all():
            raise ValueError(f'Given DataFrame does not have {meas_per_load} '
                             'measurements per load!')

        # Initialize logger.
        self.log = logging.getLogger(self.__class__.__name__)

        # Simply set some attributes equal to inputs.
        self.simulation_id = simulation_id
        self.q = data_queue
        self.initial_n = initial_n
        self.subsequent_n = subsequent_n
        self.meas_per_load = meas_per_load
        self.starttime = starttime
        self.endtime = endtime

        # Our queue will change by initial_n - subsequent_n before any
        # more work is done.
        self.q_threshold = self.initial_n - subsequent_n

        # Initialize a signal queue for flagging each time _load_queue
        # is called. This is really to make testing/debugging easier.
        self.signal_queue = queue.Queue()
        self.load_count = 0

        # Initialize a platform manager.
        self.mgr = PlatformManager()

        # Sort data by energy consumer ID so we can use iloc to slice up
        # the DataFrame
        self.df = load_measurements.sort_values(by='eqid')

        # Rename 'id' to 'measurement_mrid' to make merging easier.
        self.df.rename(columns={'id': 'measurement_mrid'}, inplace=True)

        # Hang on to the columns we care about.
        self.df = self.df[['measurement_mrid', 'eqid', 'type']]

        # Number of measurements to query initially.
        self.initial_meas = self.initial_n * self.meas_per_load

        if self.initial_meas > self.df.shape[0]:
            self.log.warning(
                'The number of measurements to query for initially, '
                f'{self.initial_meas}, is > the total number of measurements, '
                f'{self.df.shape[0]}. All measurements will be extracted at '
                'once.')
            self.initial_meas = self.df.shape[0]
            self.subsequent_meas = 0
        else:
            # Number of measurements to query subsequently.
            self.subsequent_meas = self.subsequent_n * self.meas_per_load

        # Set flag for when we're done.
        self.done = False

    def run(self, timeout=15):
        # Initialize indices.
        s = 0
        e = self.initial_meas

        # Load up the queue initially.
        self._load_queue(self.df.iloc[s:e]['measurement_mrid'].tolist())

        # Compute the remaining measurements to query for.
        remaining = self.df.shape[0] - self.initial_meas

        # Nothing more to do. Job complete.
        if remaining == 0:
            return None

        # Determine how many loop iterations we need to run to fill get
        # through all the measurements.
        n = int(np.ceil(remaining / self.subsequent_meas))
        for i in range(n):
            # Update our indices for this iteration.
            s = e
            e = s + self.subsequent_meas

            # Wait until the queue size has reduced adequately.
            t = 0
            while (self.q.qsize() > self.q_threshold) and (t < timeout):
                time.sleep(0.1)
                t += 0.1

            # Raise a timeout error if necessary.
            if t >= timeout:
                raise TimeoutError('The queue did not reduce in size '
                                   f'within {timeout} seconds.')

            # Extract measurement MRIDs and load up the queue.
            # Note that Pandas will not throw an error here if we
            # exceed the index, so it'll naturally go to the end
            # without the need to try/except IndexError.
            self._load_queue(self.df.iloc[s:e]['measurement_mrid'].tolist())

        # Flag completion.
        self.done = True

    def _load_queue(self, meas_mrids):
        # Update the counter
        self.load_count += 1

        # Query platform.
        data = self.mgr.get_simulation_output(
            simulation_id=self.simulation_id,
            query_measurement='gridappsd-sensor-simulator',
            measurement_mrid=meas_mrids)

        # Drop the instance_id, hasSimulationMessageType, and simulation_id
        # columns.
        data.drop(columns=['instance_id', 'hasSimulationMessageType',
                           'simulation_id'], inplace=True)

        # Merge with our original DataFrame to map the measurements to
        # equipment.
        merged = data.merge(right=self.df, how='left', on='measurement_mrid')

        # Group by equipment ID and start filling up the queue.
        grouped = merged.groupby(by='eqid')

        # Put each group in the queue.
        for _, group in grouped:
            self.q.put(group)

        # Flag that we've put data into the queue.
        self.signal_queue.put(self.load_count)


class LoadModelManager:
    """Class for managing our load models."""

    def __init__(self, load_nominal_voltage, load_measurements,
                 load_names_glm):
        """Map and filter the given data down to a single DataFrame with
        relevant information.

        :param load_nominal_voltage: Pandas DataFrame, should come from
            sparql.SPARQLManager.query_load_nominal_voltage()
        :param load_measurements: Pandas DataFrame, should come from
            sparql.SPARQLManager.query_load_measurements()
        :param load_names_glm: List of strings of the triplex load names
            within a GridLAB-D model. The simplest way to obtain this is
            through a glm.GLMManager object ('mgr' in this example):
            list(mgr.get_items_by_type(item_type='object',
                                       object_type='triplex_load').keys())
        """
        # Initialize the log.
        self.log = logging.getLogger(self.__class__.__name__)

        # Map our inputs together, resulting in a single DataFrame.
        self._load_df = self._map_names_to_measurements(load_nominal_voltage,
                                                        load_measurements,
                                                        load_names_glm)

        # Log success.
        self.log.info('GridLAB-D load names successfully mapped to CIM '
                      'measurements.')

        # Get a PlatformManager for weather queries.
        self._platform = PlatformManager()
        self.log.info('PlatformManager initialized.')

        # Initialize queues to be used for creating load models in
        # parallel.
        self._input_queue = mp.JoinableQueue()
        self._output_queue = mp.Queue()
        self._logging_queue = mp.Queue()

        # Start the logging thread.
        self._logging_thread = \
            threading.Thread(target=_logging_worker,
                             kwargs={'logging_queue': self.logging_queue})
        self.log.info('Logging thread started.')

        self.logging_thread.start()

        # Initialize processes to be None.
        self._processes = None

        # Determine how many processes to run.
        # TODO: We'll want to make this configurable in the future.
        self._n_jobs = mp.cpu_count() - 1
        self.log.info('Initialization complete.')

    @property
    def load_df(self):
        """Pandas DataFrame with three columns: meas_type, meas_mrid,
        and load_name.
        """
        return self._load_df

    @property
    def input_queue(self):
        """Multiprocessing JoinableQueue which provides input to the
        _get_data_and_fit_worker method.
        """
        return self._input_queue

    @property
    def output_queue(self):
        """Multiprocessing Queue where output from
        _get_data_and_fit_worker is placed.
        """
        return self._output_queue

    @property
    def logging_queue(self):
        """Multiprocessing Queue where logging information from
        _get_data_and_fit_worker is placed.
        """
        return self._logging_queue

    @property
    def logging_thread(self):
        """threading.Thread object targeted at _logging_worker.
        """
        return self._logging_thread

    @property
    def processes(self):
        """list of multiprocessing.Process objects, targeted at
        _get_data_and_fit_worker.
        """
        return self._processes

    @property
    def n_jobs(self):
        """Integer number of processes/jobs to run."""
        return self._n_jobs

    @property
    def platform(self):
        """gridappsd_platform.PlatformManager object, intended for
        getting weather data.
        """
        return self._platform

    def _map_names_to_measurements(self, load_nominal_voltage,
                                   load_measurements, load_names_glm):
        """Initialization helper to create the load_df attribute.
        """

        # For starters, ensure we actually have triplex loads from the
        # GridLAB-D model.
        if len(load_names_glm) < 1:
            raise ValueError('load_names_glm cannot be empty.')

        # Start by merging the load_nominal_voltage and
        # load_measurements DataFrames.
        merged = load_measurements.merge(right=load_nominal_voltage,
                                         left_on='load', right_on='name')
        self.log.debug('load_measurements and load_nominal_voltage merged.')

        # Drop columns we don't need.
        # Notes:
        #   name_x: This is the EnergyConsumer name in CIM - who cares.
        #   phases_y: We've shown this is duplicated/unnecessary
        #   name_y: We've shown this is duplicated/unnecessary.
        merged = merged.drop(columns=['name_x', 'phases_y', 'name_y'])

        # Rename the 'phases_x' column for simplicity/clarity
        merged.rename(columns={'phases_x': 'phases'}, inplace=True)

        # For now, we're only working with 120/240V loads. Filter.
        nom_v_filter = merged['basev'] == CIM_TRIPLEX_VOLTAGE

        # Log if it's not all 208.
        if not nom_v_filter.all():
            self.log.warning('Not all given loads have a base voltage of '
                             '{:.0f}V.'.format(CIM_TRIPLEX_VOLTAGE))

        # We only care about the 208V loads. To avoid "settingwithcopy"
        # issues in Pandas, create a copy and delete our old frame.
        merged_208 = merged.loc[nom_v_filter, :].copy(deep=True)
        del merged
        self.log.debug('Merged DataFrame filtered by nominal voltage.')

        # Now, strip the last character from the load names IFF it's an
        # 'a' or a 'b'. This will ONLY strip the character if it's
        # there, just like the fix_load_name function.
        merged_208.loc[:, 'load'].replace(regex=True, inplace=True,
                                          to_replace=r'[ab]$', value='')

        # Group by load name, measurement type, and load phases to
        # ensure that we have all the expected measurements for each
        # load.
        grouped = merged_208.groupby(['load', 'type', 'phases'])

        # For starters, we should have four times the number of groups
        # as triplex loads in the model, because each load should have
        # four measurements.
        if not (len(grouped) == len(load_names_glm) * 4):
            raise ValueError('The number of triplex loads in load_nominal_'
                             'voltage/load_measurements does not match the '
                             'number of loads in load_names_glm. This could '
                             'be due to mismatched names during merging, miss'
                             'ing measurements, or a similar issue.')

        # Now, all groups should be size 1 (essentially checking that
        # for each phase of each load we have a PNV and VA measurement).
        if not (grouped.size() == 1).all():
            raise ValueError('Each load should have four measurements, but '
                             'that is not the case.')

        # Fix the load_names_glm so they match what's in the 'load'
        # column of merged_208.
        fixed_names = [n for n in map(fix_load_name, load_names_glm)]

        # Ensure the fixed_names matches the 'load' column.
        diff = set(fixed_names) ^ set(merged_208['load'])
        if not (len(diff) == 0):
            raise ValueError('The load names given in load_nominal_voltage/loa'
                             'd_measurements do not align with the load names '
                             'in load_names_glm.')
        # Log success.
        self.log.debug('All expected measurements are present, and they match '
                       'up with load_names_glm.')

        # Now that we've confirmed everything matches, we can add a
        # column to our DataFrame.
        name_df = pd.DataFrame({'fixed_name': fixed_names,
                                'load_names_glm': load_names_glm})
        final_df = merged_208.merge(right=name_df, left_on='load',
                                    right_on='fixed_name',
                                    validate='many_to_one')

        # Finally, clean up our DataFrame to keep only the things we
        # care about.
        final_df.drop(columns=['class', 'node', 'phases', 'load', 'eqid',
                               'trmid', 'bus', 'basev', 'conn',
                               'fixed_name'], inplace=True)

        # Rename columns for clarity.
        final_df.rename(columns={'type': 'meas_type', 'id': 'meas_mrid',
                                 'load_names_glm': 'load_name'}, inplace=True,
                        errors='raise'
                        )

        # The final test: ensure we don't have any NaNs.
        if final_df.isna().any().any():
            raise ValueError('Our final DataFrame has NaNs in it!')

        return final_df

    def _start_processes(self):
        """Helper to start up processes for fitting."""
        # If the processes have already been started, log and do
        # nothing.
        if self.processes is not None:
            self.log.warning('_start_processes called when the processes '
                             'attribute is not None! Doing nothing.')
            return

        # Overwrite self._processes to be a list.
        self._processes = []
        for n in range(self.n_jobs):
            # Initialize process, attaching it to the _evaluate_worker
            # method.
            p = mp.Process(target=_get_data_and_fit_worker, name=str(n),
                           kwargs={'input_queue': self.input_queue,
                                   'output_queue': self.output_queue,
                                   'logging_queue': self.logging_queue})

            # Add this process to the list.
            self.processes.append(p)
            # Start this process.
            p.start()

        # All done.
        self.log.info('Processes started.')

    def _stop_processes(self):
        """Helper to stop all our processes.

        It will be assumed that the input_queue has been cleared out.
        """
        if self.processes is None:
            self.log.warning('_stop_processes called, but self.processes '
                             'is None! Doing nothing.')
            return

        # Send in the termination signal.
        for _ in range(len(self.processes)):
            self.input_queue.put(None)

        # Wait for all the process functions to return.
        for p in self.processes:
            p.join(timeout=None)
            p.close()

        # Set the processes property to None.
        self._processes = None

        self.log.info('Processes closed.')

    # def fit_for_all(self, sim_id, starttime, endtime):
    #     """"""
    #     # Fire up processes.
    #     self._start_processes()
    #
    #     # Get weather data.
    #     weather_data = self.platform.get_weather(start_time=starttime,
    #                                              end_time=endtime)
    #
    #     # Initialize get_data_for_load arguments.
    #     gdfl_kwargs = {'sim_id': sim_id, 'starttime': starttime,
    #                    'endtime': endtime}
    #
    #     # Initialize fit_for_load arguments.
    #     # TODO: Add options for selection_data and prediction_datetime.
    #     ffl_kwargs = {'weather_data': weather_data}
    #
    #     #
    #     # Fill up queue.

def fix_load_name(n):
    """Strip quotes, remove prefix, and remove suffix from load names.

    Load names in a GridLAB-D model from the platform are quoted and
    start with a prefix when compared to the name in the CIM. Make the
    GridLAB-D model version look like what's in the CIM, but also
    remove the one character suffix (which must be either 'a' or 'b').
    The once character 'a'/'b' suffix does not have to be present.

    :param n: String, name for triplex load to be fixed.

    :returns: n without quotes at the beginning or end, the
        TRIPLEX_LOAD_PREFIX removed, and the one character ('a' or 'b')
        suffix removed.
    """
    # Ensure our string does indeed start and end with quotes.
    if not (n.startswith('"') and n.endswith('"')):
        raise ValueError('Input to fix_load_name must start and end with a '
                         'double quote.')

    # Exclude the first and last characters.
    tmp = n[1:-1]
    # Ensure the remaining string starts with TRIPLEX_LOAD_PREFIX.
    if not tmp[0:len(TRIPLEX_LOAD_PREFIX)] == TRIPLEX_LOAD_PREFIX:
        raise ValueError('Once double quotes are removed, input to fix_load'
                         '_name must start with {}.'
                         .format(TRIPLEX_LOAD_PREFIX))

    # If the last character is either 'a' or 'b', strip it. Else, do
    # nothing.
    if (tmp[-1] == 'a') or (tmp[-1] == 'b'):
        tmp = tmp[:-1]

    # Strip off the prefix and suffix and return.
    return tmp[len(TRIPLEX_LOAD_PREFIX):]


def get_data_for_load(sim_id, meas_data,
                      query_measurement='gridappsd-sensor-simulator',
                      starttime=None, endtime=None,):
    """Query sensor service output to get data for given measurement
    mrids. This is specific to triplex_loads, and at the moment DOES NOT
    generalize to three phase loads.

    NOTE 1: query_measurement, starttime, and endtime are all directly
        passed to
        gridappsd_platform.PlatformManager.get_simulation_output. For
        details on these inputs, see that method's docstring.

    NOTE 2: The given meas_mrids are assumed to be associated with the
        same load, but no integrity checks will be performed.

    :param sim_id: Simulation ID, string.
    :param meas_data: Pandas DataFrame with 4 rows (one for each
        measurement object on a triplex load) and 2 columns:
        meas_type and meas_mrid. meas_type must be PNV or VA, and there
        must be two of each type. An easy way to get this DataFrame is
        to group a LoadModelManager's load_df DataFrame by load_name.
    :param query_measurement: String, defaults to
        'gridappsd-sensor-simulator.'
    :param starttime: datetime.datetime. Filters measurements.
    :param endtime: datetime.datetime. Filters measurements.

    :returns: pandas DataFrame with three columns, 'v', 'p', and 'q'.
        Indexed by time as returned by
        gridappsd_platform.PlatformManager.get_simulation_output.
    """
    # Do some quick input checks.
    if not isinstance(meas_data, pd.DataFrame):
        raise TypeError('meas_data must be a Pandas DataFrame.')

    if meas_data.shape[0] != 4:
        raise ValueError('Since meas_data should correspond to measurements '
                         'for a single triplex_load, it should have 4 rows.')

    # Get a PlatformManager to query the time series database.
    # NOTE: This is created on demand rather than being an input to
    # make this more robust for running in parallel.
    # TODO: Is this on-demand strategy good? Or should we be passing
    #   a PlatformManager around?
    mgr = PlatformManager()

    # Initialize dictionary to hold DataFrames. It will be keyed by
    # measurement MRID.
    data = {}

    # Query the time series database.
    for m in meas_data['meas_mrid'].values:
        # Get the data for this measurement.
        d = mgr.get_simulation_output(
            simulation_id=sim_id,
            query_measurement=query_measurement,
            starttime=starttime, endtime=endtime,
            measurement_mrid=m
        )

        # There may be some bugs in pandas related to complex numbers...
        # So, we'll hack around this.
        data[m] = {'data': utils.get_complex(r=d['magnitude'].values,
                                             phi=d['angle'].values,
                                             degrees=True),
                   'idx': d.index
                   }

    # Ensure nothing funky is going on and that all our data is the
    # same shape.
    v_iter = iter(data.values())
    dict1 = next(v_iter)
    s = dict1['data'].shape
    idx = dict1['idx']

    for d in v_iter:
        assert s == d['data'].shape
        pd.testing.assert_index_equal(idx, d['idx'])

    # Initialize arrays to hold our phase to neutral measurements and
    # VA measurements. I initially had these as columns in a DataFrame,
    # but pandas gets upset about complex numbers?
    pnv = np.zeros_like(dict1['data'])
    va = np.zeros_like(dict1['data'])

    # Loop over our list of DataFrames
    for meas_mrid, d in data.items():
        # By the nature of our query, we can ensure that each DataFrame
        # only corresponds to a single measurement MRID.
        # Extract the row in meas_data corresponding to this mrid.
        row = meas_data[meas_data['meas_mrid'] == meas_mrid]

        # Ensure it really is just a row.
        assert row.shape[0] == 1

        # Extract the type.
        meas_type = row.iloc[0]['meas_type']

        if meas_type == 'PNV':
            pnv += d['data']
        elif meas_type == 'VA':
            va += d['data']
        else:
            raise ValueError('Unexpected measurement type, {}.'
                             .format(meas_type))

    # Return a DataFrame with the 'v', 'p', and 'q' columns needed for
    # ZIP fitting.
    return pd.DataFrame(data={'v': np.abs(pnv),
                              'p': va.real,
                              'q': va.imag},
                        index=idx)


def fit_for_load(load_data, weather_data, selection_data=None,
                 prediction_datetime=None):
    """Combine load and weather data, filter by time (day of week and
    time of day), and then get a ZIP model by calling
    pyvvo.zip.get_best_fit_from_clustering.

    :param load_data: Pandas DataFrame. Return from get_data_for_load.
    :param weather_data: Pandas DataFrame which has originated from
        gridappsd_platform.PlatformManager.get_weather. The data is
        already assumed to be cleaned, e.g. it has already been passed
        through timeseries.fix_ghi.
    :param selection_data: Pandas Series, used as the point in space to
        which cluster distances are measured. Passed directly to
        zip.get_best_fit_from_clustering. Note all values of the index
        must be columns in either load_data or weather_data, and voltage
        ('v') is not allowed. If None, the last 'temperature' and 'ghi'
        values after DataFrame merging and interpolation will be used.
    :param prediction_datetime: Optional. datetime.datetime object,
        representing the starting time of the interval for which the
        load model will be used to make predictions. If not provided,
        it will be inferred from the last entry in either load_data or
        weather_data, whichever has the later time.

    NOTE 1: It's assumed that load_data and weather_data were pulled
        using the same starting and ending times.

    NOTE 2: These two DataFrames will be merged, and any resulting
        NaN values will be filled by simple linear interpolation. Thus,
        it's the caller's responsibility to ensure reasonable alignment
        between the indices of the DataFrames.

    :returns: output from pyvvo.zip.get_best_fit_from_clustering.

    TODO: Should this filtering and joining be moved? It seems like it
        could be excessive to do this for every single load. Maybe it
        would be better to create one big DataFrame with all the load
        data?

        Back of the napkin:
        2000 loads * 15 minutes * 4 intervals/hour * 24 hour/day
            * 7 days/week * 2 weeks * 3 parameters * 8 bytes/parameter
            * 1 MB/1,000,000 bytes = 968 MB.

        We may not want to suck that into memory. We could "chunk" it
        somehow. Well, we'll cross that bridge later.

    """
    # Join our load_data and weather_data, fill gaps via time-based
    # linear interpolation.
    df = load_data.join(weather_data, how='outer').interpolate(method='time')

    # If the indices didn't line up, we'll backfill and forward fill
    # the rest. If this is ever necessary, it should just be for the
    # first and last rows.
    df.fillna(method='backfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Get interval string from the configuration.
    interval_str = CONFIG['load_model']['averaging_interval']

    # At this point, our df may not have an evenly spaced index. So,
    # we need to determine if we're upsampling or downsampling.
    # noinspection PyUnresolvedReferences
    f1 = pd.tseries.frequencies.to_offset(
        pd.infer_freq(weather_data.index))
    # noinspection PyUnresolvedReferences
    f2 = pd.tseries.frequencies.to_offset(pd.infer_freq(load_data.index))

    if (f1 is None) and (f2 is not None):
        min_f = f2
    elif (f1 is not None) and (f2 is None):
        min_f = f1
    elif (f1 is not None) and (f2 is not None):
        min_f = min(f1, f2)
    else:
        raise ValueError('Neither the given load_data nor weather_data '
                         'have an evenly spaced index. This makes resampling '
                         'impossible, and is not acceptable.')

    # Determine if we're upsampling or downsampling.
    method = timeseries.up_or_down_sample(orig_interval=min_f,
                                          new_interval=interval_str)

    if method is not None:
        df = timeseries.resample_timeseries(ts=df, method=method,
                                            interval_str=interval_str)

    # If not given selection_data, use the last weather values.
    if selection_data is None:
        selection_data = df.iloc[-1][['temperature', 'ghi']]

    # If not given prediction_time, infer it from the end of the index.
    if prediction_datetime is None:
        prediction_datetime = df.index[-1]

    if timeseries.is_weekday(prediction_datetime):
        df_dow = timeseries.filter_by_weekday(df)
    else:
        df_dow = timeseries.filter_by_weekend(df)

    # Filter data by time. Start by getting some time ranges.
    t = prediction_datetime.time()
    td = timedelta(minutes=CONFIG['load_model']['filtering_interval_minutes'])
    t_start = utils.add_timedelta_to_time(t=t, td=-td)
    t_end = utils.add_timedelta_to_time(t=t, td=td)
    df_dow_t = timeseries.filter_by_time(t_start=t_start, t_end=t_end,
                                         data=df_dow)

    # Now that our data's ready, let's perform the fit.
    output = zip.get_best_fit_from_clustering(
        data=df_dow_t, zip_fit_inputs={'v_n': FIT_NOMINAL_VOLTAGE},
        selection_data=selection_data
    )

    return output


# noinspection SpellCheckingInspection
def get_data_and_fit(gdfl_kwargs, ffl_kwargs):
    """Get data via get_data_for_load, perform ZIP fit via fit_for_load.

    :param gdfl_kwargs: Keyword arguments to pass to get_data_for_load.
    :param ffl_kwargs: Keyword arguments to pass to fit_for_load, except
        load_data. The load_data comes from get_data_for_load.

    :returns: result from calling fit_for_load. See that docstring for
        more details.
    """
    # Get data.
    load_data = get_data_for_load(**gdfl_kwargs)

    # Perform the fit for this load and return.
    return fit_for_load(load_data=load_data, **ffl_kwargs)


def _get_data_and_fit_worker(input_queue, output_queue, logging_queue):
    """Method designed to be used with threading/multiprocessing to run
    get_data_and_fit.

    :param input_queue: Multiprocessing.JoinableQueue instance. The
        objects in the queue should be dictionaries with two fields:
        'gdfl_kwargs' and 'ffl_kwargs'. The dictionary will simply be
        unpacked into the get_data_and_fit method. Putting None in the
        queue is used as the termination signal for the worker, causing
        this method to return.
    :param output_queue: Multiprocessing.Queue instance. The output from
        calling get_data_and_fit will be placed in the output_queue.
        Note that a 'load_name' field will also be added.
    :param logging_queue: Multiprocessing.Queue instance. Dictionaries
        with the following fields will be placed into this queue:
        - load_name: Name of the load in question.
        - time: Total time to get data and perform the ZIP fit.
        - clusters: Number of clusters used in the fitting.
        - data_samples: Number of data samples used to create the fit.
        - sol: scipy.optimize.OptimizeResult object.

    :returns: None
    """

    # Loop until the termination signal is received.
    while True:
        # Grab data from the queue.
        d = input_queue.get(block=True, timeout=None)

        # None is the termination signal.
        if d is None:
            return

        # Do the work.
        t0 = time.time()
        result = get_data_and_fit(**d)
        t1 = time.time()

        load_name = d['meas_data'].iloc[0]['load_name']
        # Dump information into the logging queue.
        logging_queue.put({'load_name': load_name,
                           'time': t1 - t0, 'clusters': result['k'],
                           'data_samples': result['data_len'],
                           'sol': result['sol']})

        # Add a load_name field to the result.
        result['load_name'] = load_name

        # Put the result in the output queue.
        output_queue.put(result)

        # Mark task as complete.
        input_queue.task_done()


def _logging_worker(logging_queue):
    """Method to do the logging for _get_data_and_fit_worker. This
    should be used with a thread.

    :param logging_queue: Multiprocessing.Queue object, which will have
        dictionaries put in it by _get_data_and_fit_worker. For a full
        description of the fields, check that function's docstring and
        code. A None input will be the termination signal.

    :returns: None
    """

    # Loop.
    while True:
        d = logging_queue.get(block=True, timeout=None)

        # None is the termination signal.
        if d is None:
            return

        # Log.
        if d['sol'].success:
            # TODO: Should this be debug instead of info?
            LOG.info('Fit for load {} complete in {:.2f} seconds'
                     .format(d['load_name'], d['time']))
        else:
            # Warn on failure.
            LOG.warning('Fit for load {} FAILED. Solver status: {}. Solver'
                        ' message: {}'.format(d['load_name'], d['sol'].status,
                                              d['sol'].message))

        # Add detailed debugging information.
        LOG.debug('Fit details for load {}:\n\tNumber of clusters: {}'
                  '\n\tNumber of data samples: {}\n\tOptimizeResult:\n{}'
                  .format(d['load_name'], d['clusters'], d['data_samples'],
                          str(d['sol'])))

        # That's all, folks.
