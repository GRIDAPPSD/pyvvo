"""Module for managing load modeling: take data from the platform,
manipulate it, then perform a ZIP fit.
"""
# Standard library:
import logging

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

        # Alrighty, we're almost done. Keep our final_df.
        self.load_df = final_df
        self.log.info('Initialization complete. GridLAB-D load names '
                      'successfully mapped to CIM measurements.')

    def get_zip_for_load(self, load_meas_data, weather_data):
        """Given measurement data from the platform, come up with a
        ZIP load model.
        """


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
        must be two of each type.
    :param query_measurement: String, defaults to
        'gridappsd-sensor-simulator.'
    :param starttime: datetime.datetime. Filters measurements.
    :param endtime: datetime.datetime. Filters measurements.

    :returns pandas DataFrame with three columns, 'v', 'p', and 'q'.
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


def fit_for_load(load_data, weather_data, interval_str=None):
    """Get data for a load, then perform the fit by calling
    pyvvo.zip.get_best_fit_from_clustering.

    :param load_data: Pandas DataFrame. Return from get_data_for_load.
    :param weather_data: Pandas DataFrame which has originated from
        gridappsd_platform.PlatformManager.get_weather. The data is
        already assumed to be cleaned, e.g. it has already been passed
        through timeseries.fix_ghi.
    :param interval_str: String for resampling the data (after
        joining). This will be passed to timeseries.resample_timeseries,
        and should be interpretable by Pandas.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        e.g. '1Min'

    NOTE 1: It's assumed that load_data and weather_data were pulled
        using the same starting and ending times.

    NOTE 2: These two DataFrames will be merged, and any resulting
        NaN values will be filled by simple linear interpolation. Thus,
        it's the caller's responsibility to ensure reasonable alignment
        between the indices of the DataFrames.

    :returns output from pyvvo.zip.get_best_fit_from_clustering.
    """
    # Join our load_data and weather_data, fill gaps via time-based
    # linear interpolation.
    df = load_data.join(weather_data, how='outer').interpolate(method='time')

    # If the indices didn't line up, we'll backfill and forward fill
    # the rest.
    df.fillna(method='backfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    if interval_str is not None:
        # At this point, our df may not have an evenly spaced index. So,
        # we need to determine if we're upsampling or downsampling.
        # noinspection PyUnresolvedReferences
        f1 = pd.tseries.frequencies.to_offset(
            pd.infer_freq(weather_data.index))
        # noinspection PyUnresolvedReferences
        f2 = pd.tseries.frequencies.to_offset(pd.infer_freq(load_data.index))
        min_f = min(f1, f2)

        # Determine if we're upsampling or downsampling.
        method = timeseries.up_or_down_sample(orig_interval=min_f,
                                              new_interval=interval_str)

        if method is not None:
            df = timeseries.resample_timeseries(ts=df, method=method,
                                                interval_str=interval_str)

    # Now that our data's ready, let's perform the fit.
    # TODO: Find good way to configure.
    # TODO: Stop hard-coding configuration.
    # TODO: BEst way to manage selection_data?
    output = zip.get_best_fit_from_clustering(
        data=df, zip_fit_inputs={'v_n': FIT_NOMINAL_VOLTAGE},
        selection_data=df.iloc[-1][['temperature', 'ghi']]
    )

    return output




