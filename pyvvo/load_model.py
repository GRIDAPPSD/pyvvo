"""Module for managing load modeling: take data from the platform,
manipulate it, then perform a ZIP fit.
"""
import logging

import pandas as pd

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


class LoadModelManager:
    """Class for managing our load models."""

    def __init__(self, load_nominal_voltage, load_measurements,
                 load_names_glm):
        """Map and filter the given data down to a single DataFrame with
        relevant information.

        :param load_nominal_voltage: Pandas DataFrame, should come from
            sparql.SparqlManager.query_load_nominal_voltage()
        :param load_measurements: Pandas DataFrame, should come from
            sparql.SparqlManager.query_load_measurements()
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
