"""Module for pyvvo's genetic algorithm.
"""
# Standard library:
import multiprocessing as mp
import threading
import os
import queue
import logging
import time
import operator
import math
import itertools
import copy
from functools import wraps
from typing import Union

# Third party:
import numpy as np
import pandas as pd
import simplejson as json
import MySQLdb

# pyvvo:
from pyvvo import db, equipment, glm, utils

# Constants.
TRIPLEX_GROUP = 'tl'
TRIPLEX_TABLE = 'triplex'
TRIPLEX_RECORDER = 'triplex_recorder'
SUBSTATION_TABLE = 'substation'
SUBSTATION_RECORDER = 'substation_recorder'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# TODO: We may want to read the config file dynamically, so that a user
#   can change the config in between runs.
CONFIG = utils.read_config()
# By default, GridLAB-D creates a time column 't' and ID column 'id'
TIME_COL = 't'
ID_COL = 'id'
# We're using the V12. This won't change, and it's way overkill to use
# the triplestore database to determine triplex voltage levels.
TRIPLEX_NOMINAL_VOLTAGE = 240
TRIPLEX_LOW_VOLTAGE = TRIPLEX_NOMINAL_VOLTAGE * CONFIG['limits']['voltage_low']
TRIPLEX_HIGH_VOLTAGE = \
    TRIPLEX_NOMINAL_VOLTAGE * CONFIG['limits']['voltage_high']
# What we ask GridLAB-D to measure.
TRIPLEX_PROPERTY_IN = 'measured_voltage_12.mag'
# What the column name is in the database.
TRIPLEX_PROPERTY_DB = TRIPLEX_PROPERTY_IN.replace('.', '_')
SUBSTATION_ENERGY = 'measured_real_energy'
SUBSTATION_REAL_POWER = 'measured_real_power'
SUBSTATION_REACTIVE_POWER = 'measured_reactive_power'
SUBSTATION_COLUMNS = {SUBSTATION_ENERGY, SUBSTATION_REAL_POWER,
                      SUBSTATION_REACTIVE_POWER, TIME_COL}
# The GridLAB-D models from the platform have prefixes on object names,
# and thus don't precisely line up with the names from the CIM.
# https://github.com/GRIDAPPSD/GOSS-GridAPPS-D/blob/v2019.10.0/services/fncsgossbridge/service/fncs_goss_bridge.py
# Use these prefixes in conjunction with cim_to_glm_name
REG_PREFIX = 'reg'
CAP_PREFIX = 'cap'
INVERTER_PV_PREFIX = 'inv_pv'
INVERTER_BAT_PREFIX = 'inv_bat'
SWITCH_PREFIX = 'swt'
# GridLAB-D outputs things in base units (e.g. Watts or Watt-hours), but
# we want to keep our costs in more typical human terms.
TO_KW_FACTOR = 1/1000
# Map capacitor states to GridLAB-D strings.
CAP_STATE_MAP = {0: "OPEN", 1: "CLOSED"}

LOG = logging.getLogger(__name__)


def map_chromosome(regulators, capacitors):
    """Given regulators and capacitors, map states onto a chromosome.

    :param regulators: dictionary as returned by
        equipment.initialize_regulators
    :param capacitors: dictionary as returned by
        equipment.initialize_capacitors

    :returns: map_out, idx, and num_eq.
        map_out is a dict keyed by equipment name. Each equipment maps
            to a dictionary keyed by phase (A, B, or C). These phase
            dictionaries have the following fields:
            - 'idx': Tuple with a pair of integers, indicating the
                indices on the chromosome for this particular piece of
                equipment. These indices can simply be used like so:
                    dict['idx'][0]:dict['idx'][1]
            - 'eq_obj': The equipment object itself. At the moment,
                these will be equipment.CapacitorSinglePhase or
                equipment.RegulatorSinglePhase objects.
            - 'range': Tuple with a pair of integers, indicating the
                (low, high) range of states the equipment can take on.
                This is for convenience, and enables the Individual
                class to be agnostic about the differences between
                equipment.

        idx is an integer representing the full length of the chromosome
            needed to represent all the equipment.

        num_eq is an integer indicating the total number of single phase
            equipment objects.
    """
    if not isinstance(regulators, dict):
        raise TypeError('regulators must be a dictionary.')

    if not isinstance(capacitors, dict):
        raise TypeError('capacitors must be a dictionary.')

    # Initialize our output
    map_out = {}

    # Track the current index in our chromosome.
    idx = 0

    def map_reg(reg_in, dict_out, idx_in):
        """Nested helper to map a regulator."""
        # If the regulator is not controllable, DO NOT MAP.
        if not reg_in.controllable:
            return dict_out, idx_in

        # Increment the counter.
        map_reg.counter += 1

        # Compute how many bits are needed to represent this
        # regulator's tap positions.
        num_bits = _reg_bin_length(reg_in)

        # Initialize dictionary for mapping. Note the current_state
        # is computed using the tap_pos and lower_taps since GridLAB-D's
        # tap range is always [-lower_taps, raise_taps]. Adding the
        # lower_taps to the current tap_pos shifts the interval to
        # [0, raise_taps + lower_taps], which we want for encoding
        # positive integers on the chromosome.
        m = {'idx': (idx_in, idx_in + num_bits), 'eq_obj': reg_in,
             'range': (0, reg_in.raise_taps + reg_in.lower_taps),
             'current_state': reg_in.tap_pos + reg_in.lower_taps}
        # Map.
        try:
            dict_out[reg_in.name][reg_in.phase] = m
        except KeyError:
            # We don't have a dictionary for this regulator yet.
            dict_out[reg_in.name] = {reg_in.phase: m}

        # Explicitly return the dictionary and the incremented index.
        return dict_out, idx_in + num_bits

    # Give the map_reg method a counter attribute.
    # https://stackoverflow.com/a/21717084/11052174
    map_reg.counter = 0

    def map_cap(cap_in, dict_out, idx_in):
        """Nested helper to map a capacitor."""
        # DO NOT MAP if not controllable.
        if not cap_in.controllable:
            return dict_out, idx_in

        # Increment the counter.
        map_cap.counter += 1

        # Initialize dictionary for mapping. Capacitors always only get
        # one bit. The state is simpler for capacitors as compared to
        # regulators - just grab the state.
        m = {'idx': (idx_in, idx_in + 1), 'eq_obj': cap_in,
             'range': (0, 1), 'current_state': cap_in.state}

        if cap_in.state not in equipment.CapacitorSinglePhase.STATES:
            LOG.warning('Capacitor {} has state {}, which is invalid. '
                        'This is almost certainly going to lead to '
                        'errors down the line.'
                        .format(cap_in, cap_in.state))

        try:
            dict_out[cap_in.name][cap_in.phase] = m
        except KeyError:
            # We don't have a dictionary for this capacitor yet.
            dict_out[cap_in.name] = {cap_in.phase: m}

        return dict_out, idx_in + 1

    # Give the map_cap method a counter attribute.
    map_cap.counter = 0

    # Loop over the regulators.
    for reg_mrid, reg_or_dict in regulators.items():

        if isinstance(reg_or_dict, equipment.RegulatorSinglePhase):
            # Map it!
            map_out, idx = map_reg(reg_or_dict, map_out, idx)

        elif isinstance(reg_or_dict, dict):
            # Loop over the phases and map.
            for reg in reg_or_dict.values():
                map_out, idx = map_reg(reg, map_out, idx)
        else:
            raise TypeError('Unexpected type.')

    # Loop over the capacitors.
    for cap_mrid, cap_or_dict in capacitors.items():
        if isinstance(cap_or_dict, equipment.CapacitorSinglePhase):
            map_out, idx = map_cap(cap_or_dict, map_out, idx)
        elif isinstance(cap_or_dict, dict):
            # Loop over phases.
            for cap in cap_or_dict.values():
                map_out, idx = map_cap(cap, map_out, idx)
        else:
            raise TypeError('Unexpected type.')

    # At this point, our idx represents the total length of the
    # chromosome. By using counters on our map functions, we easily
    # get the number of equipment.
    return map_out, idx, map_reg.counter + map_cap.counter


def _binary_array_to_scalar(a):
    """Given a numpy.ndarray which represents a binary number, compute
    its scalar representation.

    We won't be doing any input checks here, as this should only be used
    in this module. Positive numbers only.
    """
    return (a * np.power(2, np.arange(a.shape[0] - 1, -1, -1))).sum()


def cim_to_glm_name(prefix, cim_name):
    """Helper to manage the fact that we need to prefix our object names
    to make them match what's in the GridLAB-D model.

    Also, if the given name is NOT surrounded by quotes, it will be.
    It would appear that all the 'name' attributes in the GridLAB-D
    models from the platform are quoted with quotes. However, when
    querying the CIM triple-store, the names do NOT come back with
    quotes.
    """
    return '"{}_{}"'.format(prefix.replace('"', ''), cim_name.replace('"', ''))


def _int_bin_length(x):
    """Determine how many bits are needed to represent an integer."""
    # Rely on the fact that Python's "bin" method prepends the string
    # with "0b": https://docs.python.org/3/library/functions.html#bin
    return len(bin(x)[2:])


def _int_to_binary_list(n, m):
    """Private helper to convert a number to a binary list given
    the maximum value the number could possibly take on.

    :param n: integer to be converted to a binary list.
    :param m: integer representing the maximum possible value of n.
    """
    # Compute the necessary binary width to represent
    # this range.
    w = _int_bin_length(m)

    # Get the binary representation as a string.
    bs = format(n, '0{}b'.format(w))

    # Return the list representation.
    return [int(x) for x in bs]


def _reg_bin_length(reg):
    """Determine how many bits are needed to represent a regulator.

    :param reg: regulator.RegulatorSinglePhase object.

    :returns: integer representing how many bits are needed to represent
    a regulators tap positions.
    """
    # Use raise_taps and lower_taps from GridLAB-D to compute the number
    # of bits needed.
    return _int_bin_length(reg.raise_taps + reg.lower_taps)


def prep_glm_mgr(glm_mgr, starttime, stoptime):
    """Helper to get a glm.GLMManager object ready to run.

    :param glm_mgr: glm.GLMManager object, which has been instantiated
        by passing the result from a
        gridappsd_platform.PlatformManager's get_glm method.
    :param starttime: Python datetime.datetime object for simulation
        start. Do everyone a favor: convert to UTC.
    :param stoptime: See starttime, but for simulation stop.

    This method will update the glm_mgr as follows:
    1) Call glm_mgr.add_run_components, passing in starttime and
        stoptime.
        NOTE: This method has more inputs, which we could update later.
    2) Ensure all regulators are set to MANUAL control.
    3) Ensure all capacitors are set to MANUAL control.
    4) Add all triplex_load objects to a group.
    5) Add the MySQL module to the model.
    6) Add database object. The following environment variables (
        at present defined in docker-compose.yml) are necessary:
            DB_HOST: hostname of the MySQL database.
            DB_USER: User to use to connect to MySQL.
            DB_PASS: Password for DB_USER.
            DB_DB: Database to use.
            DB_PORT: Port to connect to MySQL.
    7) Add a recorder for the triplex group.
    8) Add a meter to the substation object.
    9) Add a recorder for the substation meter.
    """
    ####################################################################
    # 1)
    # Make the model runnable.
    # TODO: get the latest VSOURCE from the platform.
    glm_mgr.add_run_components(
        starttime=starttime, stoptime=stoptime,
        minimum_timestep=CONFIG['ga']['intervals']['minimum_timestep'],
        profiler=0
    )
    ####################################################################

    ####################################################################
    # 2)
    # Lookup regulator_configuration objects.
    reg_conf_objs = glm_mgr.get_objects_by_type('regulator_configuration')
    # Switch control to MANUAL.
    for reg_conf in reg_conf_objs:
        # Note that Regulators have a capital 'C' in control, while caps
        # don't. Yay for consistency.
        glm_mgr.modify_item({'object': 'regulator_configuration',
                             'name': reg_conf['name'],
                             'Control': 'MANUAL'})
    ####################################################################

    ####################################################################
    # 3)
    # Lookup capacitor configuration objects.
    cap_objs = glm_mgr.get_objects_by_type('capacitor')
    # Switch control to MANUAL.
    for cap in cap_objs:
        glm_mgr.modify_item({'object': 'capacitor',
                             'name': cap['name'],
                             'control': 'MANUAL'})
    ####################################################################

    ####################################################################
    # 4)
    # Lookup triplex_load objects.
    tl_objs = glm_mgr.get_objects_by_type('triplex_load')
    # Add the 'groupid' property to each load.
    for tl in tl_objs:
        glm_mgr.modify_item({'object': 'triplex_load',
                             'name': tl['name'],
                             'groupid': TRIPLEX_GROUP})
    ####################################################################

    ####################################################################
    # 5)
    # Add 'mysql' module.
    glm_mgr.add_item({'module': 'mysql'})
    ####################################################################

    ####################################################################
    # 6)
    # Add a 'database' object.
    glm_mgr.add_item({'object': 'database',
                      'hostname': os.environ['DB_HOST'],
                      'username': os.environ['DB_USER'],
                      'password': os.environ['DB_PASS'],
                      'port': os.environ['DB_PORT'],
                      'schema': os.environ['DB_DB']})
    ####################################################################

    ####################################################################
    # 7)
    # Add a MySQL recorder for the triplex loads.
    glm_mgr.add_item(
        # We use the mysql.group_recorder syntax to be very careful
        # avoiding collisions with the tape module.
        {'object': 'mysql.recorder',
         'table': TRIPLEX_TABLE,
         'name': TRIPLEX_RECORDER,
         'group': '"groupid={}"'.format(TRIPLEX_GROUP),
         'property': '"{}"'.format(TRIPLEX_PROPERTY_IN),
         'interval': CONFIG['ga']['intervals']['sample'],
         'limit': -1,
         'mode': 'a',
         'query_buffer_limit': CONFIG['database']['query_buffer_limit']
         })
    ####################################################################

    ####################################################################
    # 8)
    # Add a meter at the head of the feeder.
    sub_meter = glm_mgr.add_substation_meter()
    ####################################################################

    ####################################################################
    # 8)
    # Add a recorder for the new meter.
    glm_mgr.add_item(
        # We use the mysql.group_recorder syntax to be very careful
        # avoiding collisions with the tape module.
        {'object': 'mysql.recorder',
         'table': SUBSTATION_TABLE,
         'name': SUBSTATION_RECORDER,
         'parent': sub_meter,
         'property': '"{}, {}, {}"'.format(SUBSTATION_ENERGY,
                                           SUBSTATION_REAL_POWER,
                                           SUBSTATION_REACTIVE_POWER),
         'interval': CONFIG['ga']['intervals']['sample'],
         'limit': -1,
         'mode': 'a',
         'query_buffer_limit': CONFIG['database']['query_buffer_limit']
         })
    ####################################################################

    pass


class Individual:
    """Class for representing an individual in the genetic algorithm."""

    # Possible values for the special_init input to __init__
    SPECIAL_INIT_OPTIONS = (None, 'max', 'min', 'current_state')

    def __init__(self, uid, chrom_len, num_eq, chrom_map, chrom_override=None,
                 special_init=None):
        """Initialize an individual for the genetic algorithm.

        :param uid: Unique identifier for this individual. Integer.
        :param chrom_len: Length of chromosome to generate. This is the
            second return from the map_chromosome method of this module.
        :param num_eq: Total number of equipment.EquipmentSinglePhase
            objects present in the chrom_map input. This is the third
            return from the map_chromosome method of this module.
        :param chrom_map: Dictionary mapping of the chromosome. Comes
            from the first return of the map_chromosome method.
        :param chrom_override: If provided (not None), this chromosome
            is used instead of randomly generating one. Must be a
            numpy.ndarray with dtype np.bool and shape (chrom_len,).
        :param special_init: String or None (default). Flag for special
            initialization. The options are:
            - None: Default. Randomly initialize the chromosome to
                values within each equipment's valid range.
            - 'max': All equipment will be initialized to the maximum of
                their range. For regulators, this means top tap, and for
                capacitors this means closed.
            - 'min': All equipment will be initialized to the minimum of
                their range. For regulators, this means bottom tap, and
                for capacitors this means open.
            - 'current_state': All equipment will be initialized via
                their 'current_state' attribute within the chrom_map.
                Note that the current state is taken at the time of
                mapping, and could possibly change over time without
                being reflected in the map's 'current_state' attribute.

            NOTE: special_init will be ignored if chrom_override is not
                None.

        NOTE: It is expected that chrom_len, num_eq, and chrom_map come
            from the same call to map_chromosome. Thus, there will be no
            integrity checks to ensure they align.
        """
        # Input checking.
        if not isinstance(uid, int):
            raise TypeError('uid should be an integer.')

        if uid < 0:
            raise ValueError('uid must be greater than 0.')

        # Set uid (read only).
        self._uid = uid

        # Set up our log.
        self.log = logging.getLogger(self.__class__.__name__
                                     + '_{}'.format(self.uid))

        if not isinstance(chrom_len, int):
            raise TypeError('chrom_len should be an integer.')

        if chrom_len < 0:
            raise ValueError('chrom_len must be greater than 0.')

        # Set chrom_len (read only).
        self._chrom_len = chrom_len

        if not isinstance(num_eq, int):
            raise TypeError('num_eq should be an integer.')

        if num_eq < 1:
            raise ValueError('There must be at least one piece of equipment.')

        self._num_eq = num_eq

        if not isinstance(chrom_map, dict):
            raise TypeError('chrom_map must be a dictionary. It should come '
                            'from the map_chromosome method.')

        self._chrom_map = chrom_map

        # Lazily raise a ValueError if special_init is not valid.
        if special_init not in self.SPECIAL_INIT_OPTIONS:
            raise ValueError('special_init must be one of {}.'
                             .format(self.SPECIAL_INIT_OPTIONS))

        self._special_init = special_init

        # Either randomly initialize a chromosome, or use the given
        # chromosome.
        if chrom_override is None:
            # Initialize the chromosome by looping over equipment in the
            # map and drawing random valid states.
            self._chromosome = self._initialize_chromosome()
        else:
            # Warn if incompatible inputs are given, which could lead to
            # unexpected behavior.
            if special_init is not None:
                self.log.warning('The given value of special_init, {}, '
                                 'is being ignored because chrom_override '
                                 'is not None.'.format(self.special_init))

            # Check the chromosome, alter if necessary.
            self._chromosome = self._check_and_fix_chromosome(chrom_override)

        # Initialize fitness to None.
        self._fitness = None

        # Initialize penalties to None.
        self._penalties = None

    def __repr__(self):
        if self.fitness is None:
            f_str = 'None'
        else:
            f_str = '{:.2f}'.format(self.fitness)

        return 'ga.Individual, UID: {}, Fitness: {}'.format(self.uid, f_str)

    @property
    def uid(self):
        return self._uid

    @property
    def chrom_len(self):
        return self._chrom_len

    @property
    def num_eq(self):
        return self._num_eq

    @property
    def chromosome(self):
        return self._chromosome

    @property
    def chrom_map(self):
        return self._chrom_map

    @property
    def fitness(self):
        return self._fitness

    @property
    def penalties(self):
        return self._penalties

    @property
    def special_init(self):
        return self._special_init

    def _initialize_chromosome(self):
        """Helper to initialize a chromosome.

        The chromosome is initialized by looping over the chromosome
        map, and selecting a valid state for each piece of equipment.
        This state will be chosen based on the value of
        self.special_init. See __init__ for full documentation.

        The simpler method, generating a purely random array of the
        correct length of the entire chromosome, can bias regulators
        to more likely be toward the top of their range. This stems
        from the fact that it requires 6 bits to represent the number
        32. However, of the 64 possible random numbers that can be
        generated by selecting 6 random bits, 31 of the possibilities
        are actually out of range. So, we would either have to round
        overshoot down (which doesn't help the bias problem), or
        keep drawing until everything is in range. Hence, this method
        draws a valid number for each piece of equipment and circumvents
        these issues.

        NOTE/TODO: We could easily compute the tap changing/cap
            switching costs in this function - we're already looping
            over the equipment.
        """
        # Start by initializing a chromosome of 0's of the correct\
        # length.
        c = np.zeros(self.chrom_len, dtype=np.bool)
        # Loop over the map.
        for phase_dict in self.chrom_map.values():
            # Loop over the dictionary for each equipment phase.
            for eq_dict in phase_dict.values():
                # Select a valid number based on special_init.
                if self.special_init is None:
                    # Draw a random number in the given range. Note that
                    # the interval for np.random.randint is [low, high)
                    n = np.random.randint(eq_dict['range'][0],
                                          eq_dict['range'][1] + 1,
                                          None)
                elif self.special_init == 'max':
                    n = eq_dict['range'][1]
                elif self.special_init == 'min':
                    n = eq_dict['range'][0]
                elif self.special_init == 'current_state':
                    n = eq_dict['current_state']
                    # The capacitors from the CIM triplestore query
                    # don't have any state information, so we need to
                    # ensure the state is not None here.
                    if n is None:
                        raise ValueError('Equipment {} has an associated '
                                         'current_state of None.'
                                         .format(eq_dict['eq_obj']))
                else:
                    raise ValueError('Some bad programming is afoot. '
                                     'The value of self.special_init '
                                     'is not in the if/else block of '
                                     '_initialize_chromosome.')

                # Place the binary representation into the chromosome.
                c[eq_dict['idx'][0]:eq_dict['idx'][1]] = \
                    _int_to_binary_list(n=n, m=eq_dict['range'][1])

        # Return the chromosome.
        return c

    def _check_and_fix_chromosome(self, chromosome):
        """Helper method to ensure a given chromosome is acceptable.

        :param chromosome: np.ndarray, dtype np.bool, shape
            (self.chrom_len,).

        :returns: Possibly altered chromosome.
        :raises TypeError, ValueError
        """
        # Input checking.
        if not isinstance(chromosome, np.ndarray):
            raise TypeError('chromosome must be a np.ndarray instance.')

        if not np.issubdtype(chromosome.dtype, np.dtype('bool')):
            raise ValueError('chromosome must have dtype np.bool.')

        if chromosome.shape != (self.chrom_len,):
            raise ValueError('chromosome shape must match self.chrom_len.')

        # Start by assuming the chromosome does not need modified.
        mod = False

        # Loop over the map and ensure all values are in range.
        for phase_dict in self.chrom_map.values():
            for eq_dict in phase_dict.values():
                # Convert the appropriate slice from binary to an
                # integer.
                idx = eq_dict['idx']
                n = _binary_array_to_scalar(chromosome[idx[0]:idx[1]])

                # Ensure the value is in range.
                if n < eq_dict['range'][0]:
                    flag = True
                    new_n = eq_dict['range'][0]
                elif n > eq_dict['range'][1]:
                    flag = True
                    new_n = eq_dict['range'][1]
                else:
                    flag = False

                if flag:
                    # Need to modify the chromosome.
                    mod = True

                    # noinspection PyUnboundLocalVariable
                    self.log.debug('For equipment {} for Individual {}, '
                                   'resetting the state '
                                   'in the chromosome from {} to {}.'
                                   .format(eq_dict['eq_obj'], self.uid,
                                           n, new_n))

                    # Update the chromosome.
                    chromosome[eq_dict['idx'][0]:eq_dict['idx'][1]] = \
                        _int_to_binary_list(n=new_n, m=eq_dict['range'][1])

        # Do some final logging.
        if mod:
            self.log.debug("Individual {}'s chromosome has been modified "
                           "to ensure all equipment states are in range."
                           .format(self.uid))

        # Return the chromosome.
        return chromosome

    def crossover_uniform(self, other, uid1, uid2):
        """Perform a uniform crossover between self and other, returning
        two children.

        Good reference:
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm

        :param other: initialize ga.Individual object.
        :param uid1: uid for child 1.
        :param uid2: uid for child 2.
        """
        if not isinstance(other, Individual):
            raise TypeError('other must be an Individual instance.')

        # Draw a random mask.
        mask = np.random.randint(low=0, high=2, size=self.chrom_len,
                                 dtype=np.bool)

        return self._crossover(mask=mask, other=other, uid1=uid1,
                               uid2=uid2)

    def crossover_by_gene(self, other, uid1, uid2):
        """Rather than performing a per-bit crossover like in
        crossover_uniform, instead crossover by gene. In our problem,
        each gene encodes for a piece of equipment. So, randomly
        (uniform distribution) determine on a per-equipment basis what
        each new child will get.

        :param other: ga.Individual
        :param uid1: uid to give to first child of crossover.
        :param uid2: uid to give to second child of crossover.
        """
        # Initialize a mask for the entire chromosome.
        chrom_mask = np.empty(self.chrom_len, dtype=np.bool)

        # Our map represents how many pieces of equipment we're coding
        # for. Randomly draw between zero and one to pick which child
        # will get which equipment from which parent.
        gene_mask = np.random.randint(low=0, high=2, size=self.num_eq,
                                      dtype=np.bool)

        # Loop over the map and fill in the chromosome mask.
        c = 0  # c for counter
        for value in self.chrom_map.values():
            # ed for equipment dictionary
            for ed in value.values():
                chrom_mask[ed['idx'][0]:ed['idx'][1]] = gene_mask[c]
                c += 1

        # Use the _crossover helper to create two individuals.
        return self._crossover(mask=chrom_mask, other=other, uid1=uid1,
                               uid2=uid2)

    def _crossover(self, mask, other, uid1, uid2):
        """Helper function for performing crossover given a mask for the
        entire chromosome.

        :param mask: numpy boolean array the same length as
            self.chrom_len. This will be used to select traits from
            parents.
        :param other: another Individual for crossover.
        :param uid1: uid of first child.
        :param uid2: uid of second child.

        :returns: child1, child2, both are Individuals.
        """
        if mask.shape != (self.chrom_len,):
            raise ValueError('Bad map shape, {}. Expected ({},).'
                             .format(mask.shape, self.chrom_len))

        # Initialize empty arrays for the children.
        chrom1 = np.empty(self.chrom_len, dtype=np.bool)
        chrom2 = np.empty(self.chrom_len, dtype=np.bool)

        # Use array broadcasting to fill the arrays.
        chrom1[mask] = self.chromosome[mask]
        chrom1[~mask] = other.chromosome[~mask]

        chrom2[mask] = other.chromosome[mask]
        chrom2[~mask] = self.chromosome[~mask]

        # Initialize children. Note that any out of range values will
        # be truncated to the top of the allowable range.
        child1 = Individual(uid=uid1, chrom_len=self.chrom_len,
                            chrom_override=chrom1, chrom_map=self.chrom_map,
                            num_eq=self.num_eq)
        child2 = Individual(uid=uid2, chrom_len=self.chrom_len,
                            chrom_override=chrom2, chrom_map=self.chrom_map,
                            num_eq=self.num_eq)

        # All done.
        return child1, child2

    def mutate(self, mut_prob):
        """Simple bit-flipping mutation.

        :param mut_prob: Probability an individual bit is flipped.
            Should be low. On the interval [0, 1]
        """
        # Input checking.
        if (mut_prob < 0) or (mut_prob > 1):
            raise ValueError('mut_prob must be on the interval [0, 1]')

        # Draw random numbers.
        draw = np.random.random_sample(size=(self.chrom_len,))

        # Create our mask.
        mask = draw <= mut_prob

        # Flip bits.
        self._chromosome[mask] = ~self._chromosome[mask]

        # Fix any invalid values that may have occurred as a result of
        # mutation.
        self._chromosome = self._check_and_fix_chromosome(self.chromosome)

    def evaluate(self, glm_mgr, db_conn):
        """Write + run GridLAB-D model, compute costs.

        :param glm_mgr: Initialized glm.GLMManager object, which has
            already been updated via this module's prep_glm_mgr
            function. NOTE: This manager WILL BE MODIFIED, so ensure a
            copy is passed in.
        :param db_conn: Active database connection which follows
            PEP 249.
        """
        try:
            # First, update regulators and capacitors in the glm_mgr's
            # model based on this Individual's chromosome. As the
            # glm_mgr is mutable, it's updated without a return for it
            # here.
            reg_penalty, cap_penalty = \
                self._update_model_compute_costs(glm_mgr=glm_mgr)

            # Create an _Evaluator to do the work of running the model
            # and computing associated costs.
            evaluator = _Evaluator(uid=self.uid, glm_mgr=glm_mgr,
                                   db_conn=db_conn)
            penalties = evaluator.evaluate()

            # Add the regulator tap changing and capacitor switching
            # costs.
            penalties['regulator_tap'] = reg_penalty
            penalties['capacitor_switch'] = cap_penalty

            # An individual's fitness is the sum of their penalties.
            self._fitness = 0
            for p in penalties.values():
                self._fitness += p

            self._penalties = penalties
        except Exception as e:
            # Something failed. Set fitness to infinity, and penalties
            # to None.
            self._fitness = np.inf
            self._penalties = None

            # Re-raise the exception.
            raise e from None

    def _update_model_compute_costs(self, glm_mgr):
        """Helper to update a glm.GLMManager's model via this
        Individual's chromosome. Costs associated with capacitor
        switching and regulator tapping will be computed on the fly.

        :param glm_mgr: Initialized glm.GLMManager object, which has
            already been updated via this module's prep_glm_mgr
            function. NOTE: This manager WILL BE MODIFIED, so ensure a
            copy is passed in.

        :returns: reg_penalty, cap_penalty.
        """
        # Loop over the chrom_map and build up commands to update the
        # model.
        reg_penalty = 0
        cap_penalty = 0

        for obj_name, phase_dict in self.chrom_map.items():
            # Grab an arbitrary item from the phase_dict.
            obj = next(iter(phase_dict.values()))['eq_obj']
            if isinstance(obj, equipment.RegulatorSinglePhase):
                reg_penalty += \
                    self._update_reg(phase_dict=phase_dict, glm_mgr=glm_mgr)
            elif isinstance(obj, equipment.CapacitorSinglePhase):
                cap_penalty += \
                    self._update_cap(phase_dict=phase_dict, glm_mgr=glm_mgr)
            else:
                raise TypeError('Something has gone horribly wrong. The '
                                'chrom_map has an eq_obj which is not '
                                'a regulator or capacitor!')

        # Return.
        return reg_penalty, cap_penalty

    def _update_reg(self, phase_dict, glm_mgr):
        """Helper used by _update_model_compute_costs for updating a
        regulator.
        """
        # Initialize dictionary for performing updates.
        update_dict = dict()

        # Start with a penalty of 0.
        penalty = 0

        # Loop over the phases. 'sp' for 'single phase'
        for phase, sp_dict in phase_dict.items():
            # Extract the relevant chromosome bits.
            bits = self.chromosome[sp_dict['idx'][0]:sp_dict['idx'][1]]

            # Convert to a number.
            pos = _binary_array_to_scalar(bits)

            # pos is on the interval
            # [0, reg.lower_taps + reg.raise_taps], but we want it on
            # the interval [-reg.lower_taps, reg.raise_taps].
            # Subtract lower_taps to shift the interval.
            tap_pos = pos - sp_dict['eq_obj'].lower_taps
            # Casting to an int because it's required in glm.py. This
            # is going to be a numpy int64.
            update_dict[phase] = int(tap_pos)

            # The tap changing penalty is per tap, so multiply by the
            # difference.
            penalty += (abs(tap_pos - sp_dict['eq_obj'].tap_pos)
                        * CONFIG['costs']['regulator_tap'])

        # Add the prefix to the regulator name.
        # noinspection PyUnboundLocalVariable
        model_name = cim_to_glm_name(prefix=REG_PREFIX,
                                     cim_name=sp_dict['eq_obj'].name)

        # Update this regulator in the model.
        glm_mgr.update_reg_taps(model_name, update_dict)

        # Return the penalty.
        return penalty

    def _update_cap(self, phase_dict, glm_mgr):
        """Helper used by _update_model_compute_costs for updating a
        capacitor.
        """
        # Initialize dictionary for performing updates.
        update_dict = dict()

        # Start with a penalty of 0.
        penalty = 0

        # Loop over the phases. 'sp' for 'single phase'
        for phase, sp_dict in phase_dict.items():

            # Ensure this state is valid. This check really shouldn't
            # need to be here, but more checks more better.
            if sp_dict['eq_obj'].state not in \
                    equipment.CapacitorSinglePhase.STATES:
                raise ValueError('Equipment {} has invalid state attribute, '
                                 '{}.'
                                 .format(sp_dict['eq_obj'],
                                         sp_dict['eq_obj'].state))

            # Extract the relevant chromosome bit. Note this relies
            # on the fact that each capacitor only has a single bit.
            bit = self.chromosome[sp_dict['idx'][0]:sp_dict['idx'][1]][0]

            # Get the state as a string, as GridLAB-D needs.
            state_str = CAP_STATE_MAP[bit]

            # Add this state to the dictionary.
            update_dict[phase] = state_str

            # Increment the switching cost.
            penalty += (abs(bit - sp_dict['eq_obj'].state)
                        * CONFIG['costs']['capacitor_switch'])

        # Update the capacitor in the model.

        # Add the prefix to the name.
        # noinspection PyUnboundLocalVariable
        model_name = cim_to_glm_name(prefix=CAP_PREFIX,
                                     cim_name=sp_dict['eq_obj'].name)

        # Modify it.
        glm_mgr.update_cap_switches(model_name, update_dict)

        # Return the penalty.
        return penalty


class _Evaluator:
    """Helper class used by an Individual to evaluate its fitness. The
    main method is 'evaluate' - no need to call anything else.

    The intention of this class is to encapsulate everything that comes
    with actually running the GridLAB-D model and interpreting results.
    The Individual which calls this will be responsible for computing
    costs associated with changing tap positions or capacitor switching.
    """

    def __init__(self, uid, glm_mgr, db_conn):
        """Initialize an _Evaluator object. Not all inputs will be
            checked, as they'll be coming directly from an Individual.

        :param uid: Integer, uid attribute of an Individual.
        :param glm_mgr: Initialized glm.GLMManager object, which has
            already been updated via this module's prep_glm_mgr
            function. ADDITIONALLY, the Individual should have already
            updated relevant objects, e.g. regulator tap positions.
            NOTE: This manager is coming from an Individual. That
            individual should have received a COPY of a GLMManager, as
            objects will be further modified here.
        :param db_conn: Active database connection which follows
            PEP 249.
        """
        # Set up log.
        # TODO: The issue is this gets run in multiprocessing.
        # self.log = logging.getLogger(__class__.__name__ + '_{}'.format(uid))

        # Don't check uid, it's been validated by an Individual.
        self.uid = uid

        # Ensure glm_mgr is indeed a GLMManager. It's expected that it
        # has been updated via prep_glm_mgr, but we won't test for that.
        if not isinstance(glm_mgr, glm.GLMManager):
            raise TypeError('glm_mgr must be a glm.GLMManager object.')

        self.glm_mgr = glm_mgr

        # Ensure we're getting a database connection.
        if not isinstance(db_conn, MySQLdb.connection):
            raise TypeError('db_conn must be a MySQLdb Connection object.')

        self.db_conn = db_conn

        # We'll be creating tables suffixed with '_<uid>'
        self.triplex_table = TRIPLEX_TABLE + '_' + str(self.uid)
        self.substation_table = SUBSTATION_TABLE + '_' + str(self.uid)

        # Extract the clock from the glm_mgr's model, as we'll need to
        # do some time filtering. No need to check the date format, as
        # it's already been validated.
        clock = glm_mgr.get_items_by_type('clock')
        self.starttime = clock['starttime'].replace('"', '').replace("'", '')

        # Ensure we're starting with fresh tables. Truncate if they
        # exist.
        db.truncate_table(db_conn=self.db_conn, table=self.triplex_table)
        db.truncate_table(db_conn=self.db_conn, table=self.substation_table)

        # Change table names to ensure we don't have multiple
        # Individuals writing to the same table.
        self.glm_mgr.modify_item({'object': 'mysql.recorder',
                                  'name': TRIPLEX_RECORDER,
                                  'table': self.triplex_table})
        self.glm_mgr.modify_item({'object': 'mysql.recorder',
                                  'name': SUBSTATION_RECORDER,
                                  'table': self.substation_table})

    def evaluate(self):
        """This is the 'main' method of this class. Write + run
        a GridLAB-D model and compute costs associated with it.
        """
        # Write the model to file and run it.
        model = 'model_{}.glm'.format(self.uid)
        self.glm_mgr.write_model(model)

        # Run it.
        result = utils.run_gld(model)

        # Clean up the model file - it's no longer needed.
        try:
            os.remove(model)
        except FileNotFoundError:
            # We don't want everything to come crashing down if we
            # can't find the model. This happens in testing when we
            # patch things. The testing case can be worked around, but
            # it's compelling to add this safety net for a non-critical
            # procedure.
            pass

        # TODO: Best way to handle failed runs? Maybe make costs
        #  infinite? Make sure to add logging.
        # TODO:
        assert result.returncode == 0

        # Initialize our return.
        penalties = dict()

        # Get voltage penalties.
        penalties['voltage_high'] = self._high_voltage_penalty()
        penalties['voltage_low'] = self._low_voltage_penalty()

        # Pull the substation data.
        sub_data = self._get_substation_data()

        # Get power factor penalties.
        penalties['power_factor_lead'], penalties['power_factor_lag'] = \
            self._power_factor_penalty(data=sub_data)

        penalties['energy'] = self._energy_penalty(data=sub_data)

        # We're done here. Rely on the calling Individual to add tap
        # changing and capacitor switching costs.
        return penalties

    def _voltage_penalty(self, query):
        """Helper used by _low_voltage_penalty and _high_voltage_penalty

        :param query: String. Query to run.
        """
        result = db.execute_and_fetch_all(db_conn=self.db_conn,
                                          query=query)

        # We're expecting to get back a single value inside a tuple of
        # tuples.
        if len(result) != 1 or len(result[0]) != 1:
            raise ValueError('Voltage penalty queries should be formed '
                             'in a way in which only a single value is '
                             'returned.')

        # The queries in _high and _low _voltage_penalty guarantee we
        # only get one value back. However, it's a tuple of tuples.
        penalty = result[0][0]

        # Penalty can be NULL (converted to None for Python) if there
        # are no violations. Return 0 in this case.
        if penalty is None:
            return 0

        # Return the value of the penalty.
        return penalty

    def _low_voltage_penalty(self):
        """Compute low voltage penalty for triplex loads. Called by
        'evaluate'.

        NOTES:
            - The first time step is skipped as it's unreliable -
                GridLAB-D hasn't "settled" yet.
        """
        # Create the query.
        q_low = "SELECT SUM(({nom_v} - {mag_col}) * {penalty}) as penalty" \
                " FROM {table} WHERE ({mag_col} < {low_v} " \
                "AND {time_col} > '{starttime}')".format(
                    nom_v=TRIPLEX_NOMINAL_VOLTAGE, mag_col=TRIPLEX_PROPERTY_DB,
                    penalty=CONFIG['costs']['voltage_violation_low'],
                    table=self.triplex_table, low_v=TRIPLEX_LOW_VOLTAGE,
                    starttime=self.starttime, time_col=TIME_COL
                    )

        # Use the helper to execute and extract the penalty.
        return self._voltage_penalty(query=q_low)

    def _high_voltage_penalty(self):
        """Compute high voltage penalty for triplex loads. Called by
        'evaluate'.

        NOTES:
            - The first time step is skipped as it's unreliable -
                GridLAB-D hasn't "settled" yet.
        """
        q_high = "SELECT SUM(({mag_col} - {nom_v}) * {penalty}) as penalty" \
                 " FROM {table} WHERE ({mag_col} > {high_v} " \
                 "AND {time_col} > '{starttime}')".format(
                    nom_v=TRIPLEX_NOMINAL_VOLTAGE, mag_col=TRIPLEX_PROPERTY_DB,
                    penalty=CONFIG['costs']['voltage_violation_high'],
                    table=self.triplex_table, high_v=TRIPLEX_HIGH_VOLTAGE,
                    starttime=self.starttime, time_col=TIME_COL
                    )

        # Use the helper to execute and extract the penalty.
        return self._voltage_penalty(query=q_high)

    def _get_substation_data(self):
        """Helper to grab substation data, and ensure not empty.
        """

        # Grab all the substation data.
        sub_data = \
            pd.read_sql_query(sql="SELECT * FROM {} WHERE {} > '{}';".format(
                self.substation_table, TIME_COL, self.starttime),
                con=self.db_conn, index_col=ID_COL)

        # We'll be using sub_data as a sort of safety check - it cannot
        # be empty.
        if sub_data.shape[0] < 1:
            raise ValueError('No substation data was received! This likely '
                             'indicates something is wrong with the configured'
                             ' start/stop time, sample interval, and/or '
                             'minimum timestep.')

        # We're expecting three columns back, ensure we get them.
        if set(sub_data.columns) ^ SUBSTATION_COLUMNS != set():
            raise ValueError('Unexpected substation data columns. Expected: {}'
                             ', Actual: {}'.format(SUBSTATION_COLUMNS,
                                                   set(sub_data.columns)))

        return sub_data

    def _power_factor_penalty(self, data):
        """Given substation measurement data, compute the penalties for
        power factor violations. This method is called from 'evaluate'.

        :param: data: Pandas DataFrame with data from the substation
            table.

        :returns: (lead penalty, lag penalty)

        Note that the power factor costs in CONFIG are interpreted as
        per a 0.01 deviation.
        """
        # Get a complex vector of substation power.
        power = (data[SUBSTATION_REAL_POWER].values
                 + 1j * data[SUBSTATION_REACTIVE_POWER].values)

        # Get the power factor.
        pf = utils.power_factor(power)

        return self._pf_lead_penalty(pf), self._pf_lag_penalty(pf)

    @staticmethod
    def _pf_lag_penalty(pf):
        """Compute penalty for lagging power factors. Penalties are
        assessed per 0.01 deviation.
        """
        # Find all values which are both lagging and below the limit.
        lag_limit = CONFIG['limits']['power_factor_lag']
        lag_mask = (pf > 0) & (pf < lag_limit)
        # Take the difference between the limit and the values. Multiply
        # by 100 to get things in terms of a per 0.01 deviation.
        # Finally, multiply by the costs
        return ((lag_limit - pf[lag_mask]) * 100
                * CONFIG['costs']['power_factor_lag']).sum()

    @staticmethod
    def _pf_lead_penalty(pf):
        """Compute penalty for leading power factors. Penalties are
        assessed per 0.01 deviation.
        """
        # Find all values which are both leading and below the limit.
        # Leading power factors are negative, hence the use of abs.
        lead_limit = CONFIG['limits']['power_factor_lead']
        lead_mask = (pf < 0) & (np.abs(pf) < lead_limit)
        # Take the difference between the limit and the values. Multiply
        # by 100 to get things in terms of a per 0.01 deviation.
        # Finally, multiply by the costs
        return ((lead_limit - np.abs(pf[lead_mask])) * 100
                * CONFIG['costs']['power_factor_lead']).sum()

    @staticmethod
    def _energy_penalty(data):
        """Given substation data from _get_substation_data, compute the
        energy penalty. Note GridLAB-D returns are in Wh, but penalty is
        in kWh.
        """
        # Compute the energy cost. Note the energy cost in CONFIG is
        # a per kWh figure, and returns from GridLAB-D are in Wh. Also
        # note that the measured_real_energy property of a meter
        # reports accumulation - hence why we grab the last item.
        return (data.iloc[-1][SUBSTATION_ENERGY]
                * TO_KW_FACTOR * CONFIG['costs']['energy'])


def _evaluate_worker(input_queue, output_queue, logging_queue, glm_mgr):
    """'Worker' function for evaluating individuals in parallel.

    This method is designed to be used in a multi-threaded or
    multi-processing environment.

    :param input_queue: Multiprocessing.JoinableQueue instance. The
        objects in this queue are expected to only be of type
        ga.Individual. Note that we can't do an explicit type check
        on this object, so we'll instead check for the task_done
        attribute. You're asking for trouble if this is a simple
        queue.Queue object (which is multi-threading safe, but not
        multi-processing safe).

        If None is received in the queue, the process will terminate.
    :param output_queue: Multiprocessing.Queue instance. The input
        Individuals will be placed into the output queue after they've
        been evaluated.
    :param logging_queue: Multiprocessing.Queue instance for which
        dictionaries with logging information will be placed. See the
        _logging_thread function for further reference.
    :param glm_mgr: glm.GLMManager instance which will be passed along
        to the ga.Individual's evaluate method. So, read the comment
        there for more details on requirements.

    IMPORTANT NOTE ON THE glm_mgr: The glm_mgr will be re-used for each
        subsequent individual. At the time of writing (2019-07-16), this
        is just fine, because none of the updates that happen care about
        the previous value. HOWEVER, if you go and change this to use
        multi-threading instead of multi-processing, you're going to
        enter a special kind of hell. The work-around is to create a
        deepcopy for each individual.
    """
    # Ensure our input_queue is joinable.
    try:
        input_queue.task_done
    except AttributeError:
        raise TypeError('input_queue must be multiprocessing.JoinableQueue')

    # Loop forever.
    while True:
        # Grab an individual from the queue. Wait forever.
        ind = input_queue.get(block=True, timeout=None)

        # Terminate if None is received.
        if ind is None:
            # Mark the task as done so joins won't hang later.
            input_queue.task_done()
            # We're done here. Deuces.
            return

        try:
            t0 = time.time()
            # So, we now have an individual. Evaluate.
            ind.evaluate(glm_mgr=glm_mgr,
                         db_conn=db.connect_loop(timeout=10,
                                                 retry_interval=0.1))
            t1 = time.time()

            # Dump information into the logging queue.
            logging_queue.put({'uid': ind.uid, 'fitness': ind.fitness,
                               'penalties': ind.penalties,
                               'time': t1 - t0})
        except Exception as e:
            # This is intentionally broad, and is here to ensure that
            # the process attached to this method (or when this method
            # is attached to a process?) doesn't crash and burn.
            logging_queue.put({'error': e,
                               'uid': ind.uid})
        finally:
            try:
                # Put the (possibly) fully evaluated individual in the
                # output queue. Error handling in ind.evaluate will
                # ensure a failed evaluation results in a fitness of
                # infinity.
                output_queue.put(ind)
            finally:
                # Mark this task as complete. Putting this in a finally
                # block should avoid us getting in a stuck state where
                # a failure in evaluation causes us to not mark a task
                # as complete.
                input_queue.task_done()


def _logging_thread(logging_queue):
    """Function intended to be the target of a thread, used to log
    the progress of genetic algorithm fitness evaluation.

    :param logging_queue: Multiprocessing.Queue object, which will have
        dictionaries of the following format placed in it:
        {'uid': <uid, integer>, 'fitness': <fitness, float>,
        'penalties': <penalties, dictionary>}.

        In the case of an upstream exception, the dictionary will look
        like {'error': <exception object>}

        If None is received in the queue, the thread will terminate.
    """
    # Loop forever
    while True:
        # Get dictionary from the queue.
        log_dict = logging_queue.get(block=True, timeout=None)

        # Terminate on receiving None.
        if log_dict is None:
            return

        try:
            # See if we have an error.
            log_dict['error']

        except KeyError:
            # Log the individual completion.
            LOG.debug(
                'Individual {} evaluated in {:.2f} seconds. Fitness: {:.2f}.\n'
                'Penalties:\n{}'
                .format(log_dict['uid'], log_dict['time'], log_dict['fitness'],
                        json.dumps(log_dict['penalties'], indent=4))
            )

        else:
            # We have an error.
            # noinspection PyBroadException
            try:
                raise log_dict['error'] from None
            except Exception:
                LOG.exception('Individual {} encountered an error during '
                              'evaluation.'.format(log_dict['uid']))


def _progress_thread(input_queue, output_queue, log, processes: list,
                     interval=CONFIG['ga']['log_interval']):
    """Log size of input and output queues every interval seconds.

    :param input_queue: threading.Queue like object with a qsize()
        method. This queue should hold individuals waiting to be
        evaluated.
    :param output_queue: Same as input_queue, but holds individuals
        which have already been evaluated.
    :param log: logging.Logger object to log to.
    :param processes: List of processes that are running the evaluation.
        Each one will be checked to see if it's alive.
    :param interval: Time (seconds) to wait between logging calls.
    """
    while True:
        # Get Queue sizes.
        size_in = input_queue.qsize()
        size_out = output_queue.qsize()

        # Count running processes.
        alive = 0
        for p in processes:
            if p.is_alive():
                alive += 1

        log.info('Approximately {} individuals have been evaluated, {} '
                 'are in the queue, and {} are currently being evaluated.'
                 .format(size_out, size_in, alive))

        # If the input queue is empty, quit.
        if size_in == 0:
            return

        time.sleep(interval)


class Population:
    """Class for managing a population of individuals for the GA."""

    def __init__(self, regulators, capacitors, glm_mgr, starttime, stoptime):
        """
        TODO: Document params.
        """
        ################################################################
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        ################################################################
        # Set inputs as class params.
        self._regulators = regulators
        self._capacitors = capacitors
        self._glm_mgr = glm_mgr
        self._starttime = starttime
        self._stoptime = stoptime

        ################################################################
        # Map the chromosome, update the model.
        self._chrom_map, self._chrom_len, self._num_eq = \
            map_chromosome(regulators=regulators, capacitors=capacitors)

        prep_glm_mgr(glm_mgr=self.glm_mgr, starttime=self.starttime,
                     stoptime=self.stoptime)

        ################################################################
        # Initialize uid integer (to be incremented and passed to
        # individuals)
        self._uid_counter = itertools.count()

        ################################################################
        # Setup queues and lock.
        # Queue for individual evaluation.
        self._input_queue = mp.JoinableQueue()

        # Queue for retrieving individuals after evaluation.
        self._output_queue = mp.Queue()

        # Queue for logging evaluation as it proceeds.
        self._logging_queue = mp.Queue()

        # Lock used to avoid collisions when stopping the algorithm.
        self._lock = threading.Lock()
        ################################################################
        # Threads and processes.

        # Start the logging thread.
        self._logging_thread = \
            threading.Thread(target=_logging_thread,
                             kwargs={'logging_queue': self.logging_queue})

        self.logging_thread.start()

        # On to processes.
        n_jobs = CONFIG['ga']['processes']

        # Initialize processes.
        # TODO: Move this to a method like load_model.LoadModelManager.
        # TODO: Should probably have a method
        self._processes = []
        for n in range(n_jobs):
            # Initialize process, attaching it to the _evaluate_worker
            # method.
            p = mp.Process(target=_evaluate_worker, name=str(n),
                           kwargs={'input_queue': self.input_queue,
                                   'output_queue': self.output_queue,
                                   'logging_queue': self.logging_queue,
                                   'glm_mgr': self.glm_mgr})

            # Add this process to the list.
            self._processes.append(p)
            # Start this process.
            p.start()

        ################################################################
        # For convenience, create a dictionary with inputs for
        # initializing individuals. These inputs won't change from
        # individual to individual.
        self._ind_init = {'chrom_len': self.chrom_len, 'num_eq': self.num_eq,
                          'chrom_map': self.chrom_map}

        ################################################################
        # Track all chromosomes we've ever encountered.
        self._all_chromosomes = []

        # Initialize the population.
        self._population = []

        ################################################################
        # Extract constants up front for convenience.
        # Probabilities:
        self._prob_mutate_individual = CONFIG['ga']['probabilities'][
            'mutate_individual']
        self._prob_mutate_bit = CONFIG['ga']['probabilities']['mutate_bit']
        self._prob_crossover = CONFIG['ga']['probabilities']['crossover']

        # Misc.
        self._population_size = CONFIG['ga']['population_size']
        self._generations = CONFIG['ga']['generations']
        # How many of the best/elite individuals to absolutely keep for
        # each generation.
        self._top_fraction = CONFIG['ga']['top_fraction']
        # Including the best/elite individuals, fraction of total
        # population to carry over from generation to generation.
        self._total_fraction = CONFIG['ga']['total_fraction']
        self._tournament_fraction = CONFIG['ga']['tournament_fraction']

        # Compute the number of individuals to keep via elitism.
        self._top_keep = math.ceil(CONFIG['ga']['top_fraction']
                                   * self._population_size)

        # Total number of individuals to keep between generations.
        self._total_keep = math.ceil(
            CONFIG['ga']['total_fraction'] * self._population_size)

        # Compute the number of individuals to participate in each
        # tournament.
        self._tournament_size = math.ceil(self._tournament_fraction
                                          * self._population_size)

        # Logging interval for using during calls to "evaluate".
        self._log_interval = CONFIG['ga']['log_interval']

        ################################################################

    ####################################################################
    # Property definitions.
    ####################################################################
    # Convenience numbers.

    @property
    def prob_mutate_individual(self):
        """Probability an individual enters the mutation phase. Does not
        guarantee an individual is actually mutated, as mutation is
        performed bit by bit and depends on prob_mutate_bit.
        """
        return self._prob_mutate_individual

    @property
    def prob_mutate_bit(self):
        """Per bit bit-flip probability for an individual in the
        mutation phase.
        """
        return self._prob_mutate_bit

    @property
    def prob_crossover(self):
        """Given two parents, probability that they 'mate' (crossover).
        If the random draw dictates there will be no crossover,
        offspring will be mutated versions of their parents.
        """
        return self._prob_crossover

    @property
    def population_size(self):
        """Number of individuals in the population."""
        return self._population_size

    @property
    def generations(self):
        """Number of generations of the genetic algorithm to run."""
        return self._generations

    @property
    def top_fraction(self):
        """Fraction of the most fit individuals that are guaranteed to
         survive to the next generation.
         """
        return self._top_fraction

    @property
    def total_fraction(self):
        """Total fraction of individuals that survive to the next
        generation, and will be considered for crossover + mutation.
        Note this fraction includes top_fraction.
        """
        return self._total_fraction

    @property
    def tournament_fraction(self):
        """Fraction of the population to draw for each round of
        tournament selection."""
        return self._tournament_fraction

    @property
    def top_keep(self):
        """Number of individuals to keep. ceil(population_size
                                               * top_fraction)
        """
        return self._top_keep

    @property
    def total_keep(self):
        """Total number of individuals to keep. ceil(population_size
                                                     * total_fraction)
        """
        return self._total_keep

    @property
    def tournament_size(self):
        """Number of individuals per tournament.
        ceil(population_size * tournament_fraction)
        """
        return self._tournament_size

    @property
    def log_interval(self):
        """How often (seconds) to log during evaluation."""
        return self._log_interval

    ####################################################################
    # Initializer inputs.

    @property
    def regulators(self):
        """Regulator input from equipment.initialize_regulators."""
        return self._regulators

    @property
    def capacitors(self):
        """Capacitor input from equipment.initialize_capacitors."""
        return self._capacitors

    @property
    def glm_mgr(self):
        """Initialized glm.GLMManager. Will be prepped via the
        prep_glm_mgr function.
        """
        return self._glm_mgr

    @property
    def starttime(self):
        """Python datetime representing simulation start."""
        return self._starttime

    @property
    def stoptime(self):
        """Python datetime representing simulation stop."""
        return self._stoptime

    ####################################################################
    # Derived inputs.

    @property
    def chrom_map(self):
        """Dictionary output from map_chromosome."""
        return self._chrom_map

    @property
    def chrom_len(self):
        """Integer representing the length of each Individual's
        chromosome. Output from map_chromosome.
        """
        return self._chrom_len

    @property
    def num_eq(self):
        """Total number of equipment in the chrom_map. Output from
        map_chromosome.
        """
        return self._num_eq

    ####################################################################
    # Increment UIDs.

    @property
    def uid_counter(self):
        """Simple iterator for incrementing UIDs to assign to
        individuals.
        """
        return self._uid_counter

    ####################################################################
    # Queues.

    @property
    def input_queue(self):
        """Queue which initialized individuals are put into for
        evaluation in a separate process.
        """
        return self._input_queue

    @property
    def output_queue(self):
        """Queue which individuals are placed in after evaluation is
        complete.
        """
        return self._output_queue

    @property
    def logging_queue(self):
        """Queue used for logging when individuals complete their
        evaluation.
        """
        return self._logging_queue

    ####################################################################
    # Threads and processes

    @property
    def logging_thread(self):
        """Thread for loggin individual evaluation."""
        return self._logging_thread

    @property
    def processes(self):
        """List of processes for performing individual evaluation."""
        return self._processes

    @property
    def all_processes_alive(self):
        """True if all processes return True on is_alive(), else False.
        """
        return np.array([p.is_alive() for p in self.processes]).all()

    @property
    def all_processes_dead(self):
        """True if all processes return False on is_alive(), else True.
        """
        return np.array([not p.is_alive() for p in self.processes]).all()

    ####################################################################

    @property
    def ind_init(self):
        """Helper dictionary for initializing individuals."""
        return self._ind_init

    ####################################################################

    @property
    def all_chromosomes(self):
        """List of all chromosomes which have occurred over the course
        of the genetic algorithm. For the 8500 node model, evaluation
        takes on the order of 20 or so seconds for a 20 second
        simulation (lots of overhead to start simulation), so it's
        almost certainly faster to ensure we never have duplicates.
        """
        return self._all_chromosomes

    @property
    def population(self):
        """List of all current individuals."""
        return self._population

    ####################################################################
    # Private methods
    ####################################################################

    def _init_individual(self, chrom_override=None, special_init=None):
        """Helper method to initialize an individual.

        :param chrom_override: chrom_override input for Individual.
        :param special_init: special_init input for Individual.

        :raises ChromosomeAlreadyExistedError if special_init is not
            None but the resulting individual's chromosome is identical
            to one which has already existed. IMPORTANT NOTE: This
            behavior is not true for chrom_override. It is assumed that
            if the caller is overriding the chromosome they know what
            they're doing. At the time of writing, this case of using
            chrom_override is only used by crossover_and_mutate.
        """
        # Initialize the individual.
        uid = next(self.uid_counter)
        ind = Individual(uid=uid, chrom_override=chrom_override,
                         special_init=special_init, **self.ind_init)

        # If not given a chrom_override, we need to ensure this
        # Individual does not have a chromosome which has already been
        # present in the past.
        if chrom_override is None:
            # Evaluating our individuals will almost always be more
            # expensive than checking if they existed. If this becomes a
            # serious burden, we should consider refactoring to put this in
            # the parallel workers.
            c = 0
            while self._chrom_already_existed(ind.chromosome):
                # If we were given special initialization, we have a
                # problem.
                if special_init is not None:
                    raise ChromosomeAlreadyExistedError(
                        'While trying to initialize an individual with\nchrom_'
                        'override={} and special_init={},\nit was discovered '
                        'that an individual with an identical chromosome has '
                        'already existed.'.format(chrom_override,
                                                  special_init))

                # Raise a ValueError if we've tried too many times. Putting
                # this in the loop as it'll get checked less in the long
                # run than if it were outside the loop.
                if c >= 99:
                    raise ChromosomeAlreadyExistedError(
                        'After 100 attempts, we failed to initialize an '
                        'individual with a chromosome that had not already '
                        'existed. UID : {}'.format(uid))

                self.log.debug('The chromosome for individual {} had already '
                               'existed, trying again.'.format(uid))

                # Initialize the individual.
                ind = Individual(uid=uid, chrom_override=chrom_override,
                                 special_init=special_init, **self.ind_init)

                c += 1

            # At this point, we've successfully initialized a unique
            # individual. Track its chromosome.
            self._all_chromosomes.append(ind.chromosome.copy())
        else:
            # In the case of a non-None chrom_override, only track the
            # chromosome if it's unique. Since it's expensive to search
            # through the all_chromosomes list, we don't want
            # duplicates.
            if not self._chrom_already_existed(ind.chromosome):
                self._all_chromosomes.append(ind.chromosome.copy())

        # All done - return our new individual.
        return ind

    def _chrom_already_existed(self, c):
        """Helper to check if a given chromosome has ever been present
        in the population.

        :param c: chromosome from an individual.
        """
        # Loop over all the chromosomes. We may be more likely to get
        # a hit for more recent individuals, though I don't have the
        # maths to prove it.
        for chrom in reversed(self.all_chromosomes):
            # Loop over the chromosomes element by element.
            all_match = True
            for i in range(len(c)):
                if chrom[i] != c[i]:
                    # These chromosome aren't equal. Break to outer
                    # loop.
                    all_match = False
                    break

            if all_match:
                return True

        # If we're here, we didn't have any hits.
        return False

    def _get_two_parents(self):
        """Simple helper to get two parents via a tournament for
        crossover and/or mutation.

        :returns: parent1, parent2. Both are Individuals from the
        population.
        """
        # Get two parent indices via a tournament.
        parents = _tournament(population=self.population,
                              tournament_size=self.tournament_size,
                              n=2)
        # Return the parents from the population.
        return self.population[parents[0]], self.population[parents[1]]

    def _mutate(self, ind):
        """Helper to mutate an individual. If the mutation results in
        an individual with a chromosome which has already existed,
        keep mutating.

        :param ind: Individual object to mutate.

        :returns: None. The individual is mutated in place.
        :raises ChromosomeAlreadyExistedError if 100 mutation attempts
            don't get us an individual which hasn't already existed.
        """
        ind.mutate(mut_prob=self.prob_mutate_bit)

        c = 0
        while self._chrom_already_existed(ind.chromosome) and c < 99:
            ind.mutate(mut_prob=self.prob_mutate_bit)
            c += 1

        if c >= 99:
            raise ChromosomeAlreadyExistedError(
                'After {} attempted mutations, the individual with uid {} '
                'failed to create a chromosome which has not yet already '
                'existed.'.format(c + 1, ind.uid)
            )

        # All done.

    def _crossover_and_mutate(self, parent1, parent2):
        """Given two parents, perform crossover to create children, and
        possibly mutate them. This is a helper for part of the
        crossover_and_mutate method.

        :returns: child1, child2: Resultant children from the crossover
            and possible mutation.
        """
        # Create children via crossover.
        children = \
            parent1.crossover_by_gene(other=parent2,
                                      uid1=next(self.uid_counter),
                                      uid2=next(self.uid_counter))

        # Draw two random numbers and compare to the mutation
        # probability to see if each of the children will be mutated.
        m = np.random.rand(2) < self.prob_mutate_individual

        # Possibly mutate each child. If we're not mutating but their
        # chromosome has already existed, force mutation.
        for tf, ind in zip(m, children):
            # While it may look like this if/else should be one
            # if statement with an "or," the way it's written
            # now avoids an extra call to _chrom_already_existed
            # in some cases, which is good, because that can be
            # expensive.
            if tf:
                # Perform the mutation.
                self._mutate(ind=ind)
            elif self._chrom_already_existed(ind.chromosome):
                # Force mutation if this individual isn't unique.
                # This keeps the logic simpler than excluding the
                # child.
                self._mutate(ind=ind)

        return children

    def _asexual_reproduction(self, parent1, parent2):
        """Helper to perform asexual reproduction. Children are mutated
        versions of their respective parent.

        :param parent1: ga.Individual which will be used to create the
            first child.
        :param parent2: ga.Individual which will be used to create the
            second child.
        """
        children = []
        for p in [parent1, parent2]:
            # Produce a clone of the parent.
            child = self._init_individual(chrom_override=p.chromosome.copy(),
                                          special_init=None)
            # Mutate the child so that it isn't identical to the parent.
            self._mutate(child)
            # Add it to the list.
            children.append(child)

        # Done.
        return tuple(children)

    ####################################################################
    # Public methods
    ####################################################################

    def initialize_population(self):
        """Initialize and the first generation. To keep methods simple
        and modular, evaluation will not occur.

        The population will be seeded with three individuals.
        """
        if len(self.population) > 0:
            raise ValueError('initialize_population should only be '
                             'called when the population is emtpy.')

        if self.population_size < 3:
            raise ValueError('initialize_population seeds the population '
                             'with 3 individuals, thus we must always have '
                             'a population_size of at least 3.')

        # Counter for individuals in the population.
        i = 0

        # Seed the population.
        self.population.append(self._init_individual(special_init='max'))
        self.population.append(self._init_individual(special_init='min'))
        self.population.append(self._init_individual(
            special_init='current_state'))

        # We just added three individuals.
        i += 3

        # Fill the rest of the population with randomly initialized
        # individuals.
        while i < self.population_size:
            self.population.append(self._init_individual())
            i += 1

        # All done.

    def evaluate_population(self):
        """Evaluate all individuals in the population who haven't yet
        been evaluated. NOTE: This can take a while, depending on the
        model, etc.
        """
        # Throw an error if we're trying to evaluate when we can't.
        if len(self.population) != self.population_size:
            raise ValueError('evaluate_population should only be '
                             'called when the population is full.')

        # Throw an error if all of our processes aren't alive.
        # Technically this can run if just one is alive, but that would
        # cause the algorithm to run very slowly.
        if not self.all_processes_alive:
            m = 'evaluate_population called, but not all processes are alive!'
            self.log.error(m)
            raise DeadProcessError(m)

        # Initialize list of indices for individuals we're evaluating.
        idx = []

        # Put all eligible individuals in the queue.
        for i in range(self.population_size):
            if self.population[i].fitness is None:
                # Put this individual in the queue.
                self.input_queue.put(self.population[i])
                # Track its index - we need to remove it from the
                # population since we'll be retrieving the evaluated
                # version later.
                idx.append(i)

        # Start a thread to log progress.
        # TODO: are we okay with the consequences if this thread doesn't
        #   ever get properly shut down? I think so. Eventually the
        #   queues will get emptied or deleted, and the thread will die.
        t = threading.Thread(target=_progress_thread,
                             kwargs={'input_queue': self.input_queue,
                                     'output_queue': self.output_queue,
                                     'log': self.log,
                                     'processes': self.processes,
                                     'interval': self.log_interval})
        t.start()

        # Make a new version of the population sans the individuals
        # who are currently being evaluated.
        self._population = [self.population[i]
                            for i in range(self.population_size)
                            if i not in idx]

        # For multiprocessing queues, there can be a slight delay. Avoid
        # it by sleeping. This is probably unnecessary, but safety
        # first.
        time.sleep(0.05)

        # Wait for processing to finish.
        self.input_queue.join()

        # Transfer the evaluated individuals into the population.
        self._dump_queue_into_population()

        # Check to see if we were interrupted.
        if len(self.population) != self.population_size:
            self.log.warning('The length of the population does not match the '
                             'expected population size. Perhaps evaluation was'
                             ' interrupted?')

        # All done.

    @utils.wait_for_lock
    def _dump_queue_into_population(self):
        """Simple helper used by evaluate_population to put the contents
        of the output_queue into the population list. This is put into a
        helper function so it can be wrapped by wait_for_lock."""
        self._population = _dump_queue(q=self.output_queue, i=self._population)

    def natural_selection(self):
        """Trim the population via both elitism and tournaments."""
        # Sort the population in place.
        self.sort_population()
        # Perform natural selection. Start by keeping the top fraction.
        new_population = self.population[0:self.top_keep]
        self._population = self._population[self.top_keep:]

        # Now, use tournament selection to fill in the rest.
        while len(new_population) < self.total_keep:
            # Note that I'm intentionally not using self.tournament_size
            # since that's fixed for a full population.
            winner_idx = \
                _tournament(population=self._population,
                            tournament_size=math.ceil(self.tournament_fraction
                                                      * len(self._population)),
                            n=1)[0]

            # Put the winner into the new population.
            new_population.append(self._population.pop(winner_idx))

        # At this point, we can ditch this "new_population" variable -
        # we're done selecting individuals.
        self._population = new_population

        # Done.

    def crossover_and_mutate(self):
        """Replenish the population via crossover and mutation."""
        # Initialize list to hold offspring.
        offspring = []

        while len(self.population) + len(offspring) < self.population_size:
            # Get two parents from a tournament.
            parent1, parent2 = self._get_two_parents()

            # Perform crossover with some probability.
            if np.random.rand() < self.prob_crossover:
                # Call the _crossover_and_mutate helper.
                children = self._crossover_and_mutate(parent1, parent2)
            else:
                # Get mutated versions of the parents.
                children = self._asexual_reproduction(parent1, parent2)

            # Add the children to the list of offspring.
            offspring.extend(children)

        # While this could waste some effort, we want to keep our
        # population at the correct size to avoid any surprises. Call
        # it infant mortality. Genetic algorithm joke, nice.
        if len(self.population) + len(offspring) > self.population_size:
            offspring = offspring[0:-1]

        # Merge the offspring into the population.
        self._population.extend(offspring)

        # All done.

    def sort_population(self):
        """Helper to sort the population in place. After this method
        is called, the individual with the lowest fitness (the best)
        will be in position 0.
        """
        try:
            self._population.sort(key=operator.attrgetter('fitness'))
        except TypeError:
            s = ('While attempting to sort the population, a TypeError '
                 'occurred. This likely means there is an individual '
                 'with a fitness of None, which cannot be sorted. Ensure '
                 'all individuals have been evaluated, and try again.')
            raise TypeError(s) from None

    @utils.wait_for_lock
    def graceful_shutdown(self):
        """Helper to "gracefully" stop an in-progress run of
        evaluate_population. By "gracefully", I mean that no processes
        will be terminated mid-run. Instead, the queue will be emptied
        out, and then the processes terminated.

        For now, "graceful" does not mean that we can simply resume
        where we left off. If that is desired later, it'll be pretty
        similar to this method, and not too complicated.

        Note this method is intended to be run quickly and exit. It will
        not wait for the processes to finish. If the processes should
        be waited on, this should happen outside of this method.

        Check the all_processes_dead attribute to ensure everything's
        been killed.
        """
        self.log.info('Gracefully stopping genetic algorithm evaluation.')

        # Drain the input queue. This will prevent any further
        # individuals from being evaluated.
        _drain_queue(self.input_queue)

        # Drain the output queue. This will prevent extra work from
        # being done later.
        _drain_queue(self.output_queue)

        # Send in the shutdown signal to all the processes.
        for _ in range(len(self.processes)):
            self.input_queue.put_nowait(None)

        # Log.
        self.log.debug(
            'The input and output queues have been drained, and the '
            'termination signal has been sent to the processes.')

        # We're done here. Actually waiting for the processes to finish
        # should be done elsewhere - we don't want this method to wait.
        return None

    def forceful_shutdown(self):
        """This is a bit tricky due to (probably) bad design. We have
        each process attached to a target, in this case that target is
        _evaluate_worker. During the course of running, the process
        opens up a subprocess to run GridLAB-D. Currently, that's done
        with subprocess.run, which doesn't allow us to kill the
        running subprocess. I'm concerned that if we just send the
        terminate or kill signal to the process, the GridLAB-D process
        will either keep running or die in a way that causes issues
        (like bad stuff with MySQL). In the former case (GLD keeps
        running), the forceful shutdown is practically pointless. Sure,
        it'll save some queries to the database during evaluation, but
        that's not where the heavy lifting is. In the latter case (the
        GLD process is forcefully stopped), we could totally hose
        things. So, until there's a compelling reason to implement a
        forceful shutdown, let's call it good with the graceful
        shutdown.
        """
        raise NotImplementedError

    def wait_for_processes(self, timeout: Union[float, int]):
        """Wait for the processes to terminate.

        :param timeout: Time in seconds to wait for each individual
            process.

        :raises TimeoutError: if any process does not terminate within
            the given timeout.

        :returns: None
        """
        for p in self.processes:
            p.join(timeout=timeout)

            if p.exitcode is None:
                # Process has not terminated.
                raise TimeoutError('Process did not terminate within {} '
                                   'seconds.'.format(timeout))

        # All done.
        return None


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class ChromosomeAlreadyExistedError(Error):
    """Raised if a chromosome has already been existed.
    """
    pass


class DeadProcessError(Error):
    """Raised when a Population object has a dead process but shouldn't.
    """
    pass


class GAInterruptedError(Error):
    """Raised when the genetic algorithm is externally interrupted."""


def _tournament(population, tournament_size, n):
    """Helper for performing tournament selection.

    :param population: List of Individuals comprising the population.
    :param tournament_size: Integer indicating how many individuals
        participate in each tournament.
    :param n: Integer. Top n individuals from the tournament will be
        returned.

    :returns: List of indices into the population in order of fitness.
        e.g. [7, 2, 10, 0, 4]. In this example, population[7] would be
        the individual with the lowest fitness.
    """
    # Randomly draw 'tournament_size' individuals.
    challenger_indices = np.random.choice(a=np.arange(len(population)),
                                          size=tournament_size,
                                          replace=False)

    # Extract their fitnesses into an array.
    fit = np.array([population[i].fitness for i in challenger_indices])

    # Get the indices for sorting fit.
    sort_idx = np.argsort(fit)

    # Return the correct number and ordering of challenger_indices.
    return [challenger_indices[sort_idx[i]] for i in range(n)]


def _dump_queue(q, i):
    """Helper to empty a queue into a list.

    :param q: A queue.Queue like object (e.g.
        multiprocessing.JoinableQueue)
    :param i: A list object, for which items from q will be appended to.

    :returns: i. While this isn't necessary, it's explicit.
    """
    while True:
        try:
            i.append(q.get_nowait())
        except queue.Empty:
            return i


def _drain_queue(q):
    """Helper to simply clear out a queue. The items in the queue will
    be discarded. If the queue is joinable (has a task_done() method),
    it will be called for each get_nowait() call.

    :param q: A queue.Queue lik object (e.g.
        multiprocessing.JoinableQueue)

    :returns: None
    """
    while True:
        try:
            q.get(block=True, timeout=0.1)
            try:
                q.task_done()
            except AttributeError:
                pass

        except queue.Empty:
            break


def _update_equipment_with_individual(ind, regs, caps):
    """Given an individual, update the states of equipment.

    :param ind: ga.Individual object.
    :param regs: Dictionary of regulators as returned by
        equipment.initialize_regulators.
    :param caps: Dictionary of capacitors as returned by
        equipment.initialize_capacitors.

    NOTE 1: The regs and caps should have been used in the call to
        map_chromosome which provides outputs for the initialization of
        the Individual.

    NOTE 2: This method updates the state of the regs and caps. So, if
        these regs and caps are being used elsewhere, take care. You may
        want to pass a deepcopy in here.

    NOTE 3: You may say to yourself, "Wait a minute, the individual
        already has a reference to all these regulators and capacitors.
        Why not write a method for the Individual which just iterates
        over its map and directly updates the equipment?" Well, the
        short answer is "because pickling." The GA is parallelized by
        necessity. When an individual gets put in the queue to be
        evaluated, it (and all its attributes) get pickled. So the
        equipment pointers are no longer pointers, but copies after
        evaluation. In the future we may want to refactor such that the
        individuals aren't carrying around these full-blown objects
        which then get pickled, but it likely isn't worth the effort for
        a very minimal gain.

    :returns: None. regs and caps are updated in place.
    """
    # Loop over the individual's map.
    for phase_dict in ind.chrom_map.values():
        for eq_dict in phase_dict.values():
            # Get a numerical representation of the new state from the
            # chromosome.
            idx = eq_dict['idx']
            eq_obj = eq_dict['eq_obj']
            raw_new_state = \
                _binary_array_to_scalar(ind.chromosome[idx[0]:idx[1]])

            # If we're dealing with a regulator, we need to do some
            # translations.
            if isinstance(eq_obj, equipment.RegulatorSinglePhase):
                # raw_new_state is going to be on the interval
                # [0, raise_taps + lower_taps] (GridLAB-D). We need it
                # to be on the interval [low_step, high_step] (CIM).
                # Start by subtracting lower taps to shift it to the
                # interval [-lower_taps, raise_taps].
                new_state = raw_new_state - eq_obj.lower_taps
                # Now, translate it to CIM. Yes, it's bad practice to
                # call a "private" method. My bad. Maybe it should be
                # public? But I'm not doing type-checking...
                # noinspection PyProtectedMember
                new_state = \
                    equipment._tap_gld_to_cim(tap_pos=new_state,
                                              neutral_step=eq_obj.neutral_step)
                # Regulators dictionary is keyed by tap changer mrid,
                # not by regulator mrid.
                mrid = eq_obj.tap_changer_mrid

                # Update the state.
                regs[mrid].state = new_state
            elif isinstance(eq_obj, equipment.CapacitorSinglePhase):
                # Cast to a regular Python integer (from a numpy int64)
                caps[eq_obj.mrid].state = int(raw_new_state)
            else:
                raise TypeError('Unexpected equipment!')

    # And we're done!


def _clear_and_set_event(method):
    """Decorator for use with the GA class to clear() a threading.Event
    object while a method is running, and set() the event after. For
    now, this decorator is hard-coded to use the _not_running_event
    attribute of the GA object. This could be made more generic by
    following advice found
    `here <https://stackoverflow.com/a/10176276>`_.

    NOTE: An exception will be raised if the _not_running_event is
    already set at the beginning of this method.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Raise an exception if _not_running_event is not already set.
        if not self._not_running_event.is_set():
            raise RuntimeError('The _not_running_event is set when it '
                               'should not be. This could mean that "run" was '
                               'called before a previous call to "run" had '
                               'finished.')

        # Clear the event to indicate the method is running.
        self._not_running_event.clear()

        # Run the method.
        try:
            result = method(self, *args, **kwargs)
        finally:
            # Set the event to indicate the method is not running.
            self._not_running_event.set()

        return result

    return wrapper


class GA:
    """Class for managing the genetic algorithm. Use the "run" and
    "stop" methods to start/stop the algorithm."""

    def __init__(self, regulators, capacitors, starttime, stoptime,
                 stop_timeout=120.0):
        """
        :param regulators: dictionary as returned by
            equipment.initialize_regulators. Since the states of
            these objects will be altered after the algorithm has run,
            a deepcopy will be made.
        :param capacitors: dictionary as returned by
            equipment.initialize_capacitors. Since the states of
            these objects will be altered after the algorithm has run,
            a deepcopy will be made.
        :param starttime: Python datetime object for simulation start.
            For more details, see the docstring for prep_glm_mgr.
        :param stoptime: Python datetime object for simulation end. See
            prep_glm_mgr.
        :param stop_timeout: Timeout (seconds) for waiting on the
            algorithm to stop after the "stop" method is called. If this
            timeout is exceeded, future calls to "run" will do nothing.
        """
        # Set up logging.
        self.log = logging.getLogger(__class__.__name__)

        # Assign inputs as attributes. Make copies of mutable objects
        # which will be modified.
        self._regulators = copy.deepcopy(regulators)
        self._capacitors = copy.deepcopy(capacitors)
        self._starttime = starttime
        self._stoptime = stoptime
        self.stop_timeout = stop_timeout

        # Initialize attribute which will be replaced with a Population
        # object once the algorithm is running.
        self._population = None

        # Initialize attribute which will be replaced with a thread
        # while the genetic algorithm is running.
        self._run_thread = None

        # Initialize event for signaling algorithm interruption.
        # If this event is set, the GA can run. If it is not set, the
        # GA cannot run.
        self._run_event = threading.Event()
        self._run_event.set()

        # Initialize event for tracking if the genetic algorithm is
        # running. If not set, the GA is currently running. If set,
        # the GA is not running.
        self._not_running_event = threading.Event()
        self._not_running_event.set()

    @property
    def regulators(self):
        """Dictionary as returned by equipment.initialize_regulators.
        The underlying regulator object states will be modified by the
        algorithm"""
        return self._regulators

    @property
    def capacitors(self):
        """Dictionary as returned by equipment.initialize_capacitors.
        The underlying capacitor object states will be modified by the
        algorithm."""
        return self._capacitors

    @property
    def starttime(self):
        """Python datetime object for simulation start. For more
        details, see the docstring for prep_glm_mgr."""
        return self._starttime

    @property
    def stoptime(self):
        """Python datetime object for simulation end. For more
        details, see the docstring for prep_glm_mgr."""
        return self._stoptime

    @property
    def population(self):
        """ga.Population object, or None if the algorithm has not yet
        been run.
        """
        return self._population

    @property
    def run_thread(self):
        """Thread used to run the genetic algorithm."""
        return self._run_thread

    @property
    def running(self):
        """True if the genetic algorithm is running, False otherwise."""
        return not self._not_running_event.is_set()

    @property
    def run_event(self):
        """threading.Event object. If self.run_event.is_set() returns
        True, the genetic algorithm is permitted to run. If is_set()
        instead returns False, the genetic algorithm is not permitted
        to run, and will begin shutting down if it was in progress.
        """
        return self._run_event

    def run(self, glm_mgr):
        """Run the genetic algorithm. This runs the algorithm in a
        thread for easy interruption. It will return before the genetic
        algorithm is complete (as soon as the thread is started).

        Recall that this class's "running" property indicates whether
        or not the genetic algorithm is currently running.

        After the genetic algorithm finishes, self.regulators and
        self.capacitors will get updated with the settings from the
        best individual. Then, one could use an EquipmentManager's
        "build_equipment_commands" method in conjunction with this
        object's self.regulators/self.capacitors attributes in order
        to get commands ready to send into the GridAPPS-D platform.

        If you'd like to wait for the algorithm to complete, use this
        class's "wait" method after calling "run."

        :param glm_mgr: glm.GLMManager object. The model will be updated
            via this modules 'prep_glm_mgr' function, so this object
            should just be the raw model from the platform. Since the
            states of these objects will be altered after the algorithm
            has run, a deepcopy will be made.
        """
        # Copy the GLMManager.
        mgr_copy = copy.deepcopy(glm_mgr)

        # Create and start thread.
        self._run_thread = threading.Thread(target=self._run,
                                            kwargs={'glm_mgr': mgr_copy})
        self._run_thread.start()

        # That's it!
        return None

    @_clear_and_set_event
    def _run(self, glm_mgr):
        """Private method for running the genetic algorithm. Don't ever
        use this directly, use the public "run" method instead.

        This method makes extensive use of the "_run_if_set" helper
        function so that the algorithm can be interrupted as quickly
        as possible if/when "stop" is called.

        Also note this method is decorated by _clear_and_set_event. This
        is used to flag when the genetic algorithm is running by
        clearing the _not_running_event before the starting the method,
        and then setting the _not_running_event after completion (no
        matter what happens).

        Rather than returning, this function updates self.regulators
        and self.capacitors with the states from the individual which
        is determined to be best.

        :param glm_mgr: glm.GLMManager object. The model will be updated
            via this modules 'prep_glm_mgr' function, so this object
            should just be the raw model from the platform.

        :returns: None
        """
        # We'll time the algorithm runtime.
        t0 = time.time()

        # Initialize the Population object.
        self._population = Population(regulators=self.regulators,
                                      capacitors=self.capacitors,
                                      glm_mgr=glm_mgr,
                                      starttime=self.starttime,
                                      stoptime=self.stoptime)

        try:
            # Fill the population with individuals.
            self._run_if_set(self.population.initialize_population)
            self.log.debug('Population initialized.')

            # Evaluate each individual. This will take some time.
            self._run_if_set(self.population.evaluate_population)
            self.log.debug('Initial population evaluation complete.')

            # Loop over the generations to perform natural selection,
            # crossover, and mutation.
            g = 1
            while g <= CONFIG['ga']['generations']:
                self._run_if_set(self.population.natural_selection)
                self.log.debug('Natural selection for generation {} complete.'
                               .format(g))
                # The best individual will always be in position 0 after
                # natural selection.
                self._run_if_set(self._log_best_each_gen, g)

                self._run_if_set(self.population.crossover_and_mutate)
                self.log.debug('Crossover and mutation for generation {} '
                               'complete.'.format(g))
                self._run_if_set(self.population.evaluate_population)
                self.log.debug('Population evaluation for new individuals '
                               'for generation {} complete.'.format(g))
                g += 1

            # Shut down the population processes. Putting this in a
            # _run_if_set call so that we don't incidentally call this
            # method twice.
            self._run_if_set(self.population.graceful_shutdown)

            # Sort the population.
            self._run_if_set(self.population.sort_population)
            self.log.debug('Final population sorting complete.')

            # Log.
            self._run_if_set(self._log_best_overall, t0)

            # Update the regulators and capacitors with the positions
            # given by the best individual.
            self._run_if_set(_update_equipment_with_individual,
                             ind=self.population.population[0],
                             regs=self.regulators, caps=self.capacitors)
            self.log.debug('Equipment updated with settings from the best '
                           'individual.')

        except GAInterruptedError:
            # One of our many "_run_if_set" wrappers raised this,
            # indicating the algorithm was interrupted.
            #
            # Assume that the algorithm was interrupted by the stop()
            # method. In that case, we do not need to shut down the
            # population processes.
            self.log.debug('Caught GAInterruptedError, returning.')
            # Time to bounce.
            return None
        else:
            # Wait for the population processes to shut down.
            self.population.wait_for_processes(
                timeout=CONFIG['ga']['process_shutdown_timeout'])

            # Nothing to return here, as the objects themselves get
            # updated.
            return None

    def _run_if_set(self, func, *args, **kwargs):
        """Helper to run a function if self.run_event.is_set() returns
        True. This method should only be called from within the _run
        method.

        :param func: Function to run.
        :param args: Positional arguments to pass to func.
        :param kwargs: Keyword arguments to pass to func.

        :returns: Directly returns the return from running func.

        :raises GAInterruptedError: Raised if the run_event is not set.
        """
        if self.run_event.is_set():
            return func(*args, **kwargs)
        else:
            # Log.
            self.log.warning(
                'Did not run {} because the run_event is not set.'
                .format(func))

            # Raise exception.
            raise GAInterruptedError('The genetic algorithm was interrupted.')

    def _log_best_each_gen(self, gen: int):
        """Helper for logging during execution of _run. Should only be
        called from _run.

        :param gen: Generation number of the genetic algorithm.
        """
        best = self.population.population[0]
        self.log.info(
            'After generation {}, best fitness: {:.2f} from individual'
            ' {}'.format(gen, best.fitness, best.uid))

    def _log_best_overall(self, t0: float):
        """Helper for logging at the end of the genetic algorithm.
        Should only be called from _run.

        :param t0: Starting time of the genetic algorithm. Should come
            from a time.time() call.
        """
        t1 = time.time()
        best = self.population.population[0]
        self.log.info('Best overall fitness: {:.2f} from individual {}'
                      .format(best.fitness, best.uid))

        self.log.info('Total GA run time: {:.2f}'.format(t1 - t0))

    def stop(self):
        """Method to interrupt the running genetic algorithm. This
        method will immediately return after kicking off the stopping
        process. To check on whether or not the genetic algorithm is
        still running, check self.running.
        """
        if not self.running:
            self.log.warning('The "stop" method was called, but the genetic '
                             'algorithm is not running.')
        else:
            self.log.info('Stopping the genetic algorithm.')
            # Flag that we need to stop.
            self._run_event.clear()

            # Stop the population object from running.
            self.population.graceful_shutdown()

            # Start up a thread that will set the run_event once the
            # _run function has finished.
            t = threading.Thread(target=self._set_run_event_after_run)
            t.start()

        # All done.
        return None

    def _set_run_event_after_run(self):
        """Helper used by "stop" to set the _run_event after the _run
        method finishes (as signaled by setting _not_running_event).
        """
        try:
            self.wait(timeout=self.stop_timeout)
        except TimeoutError:
            # Well, the algorithm failed to stop within the specified
            # timeout. Warn.
            self.log.warning('The "run" method failed to stop within {:.2f} '
                             'seconds of calling "stop." Future calls to '
                             '"run" will do nothing until the "run_event" is '
                             'set.'.format(self.stop_timeout))
        else:
            # Success.
            self._run_event.set()

        # All done.
        return None

    def wait(self, timeout=None):
        """Wait for the "run" method to complete.

        :param timeout: Time (seconds) to wait for the "run" method to
            complete. Defaults to None (wait forever).

        :returns: None

        :raises TimeoutError:
        """
        # Wait for the _not_running_event.
        r = self._not_running_event.wait(timeout=timeout)

        # An Event object's wait method returns False if it timed out.
        if not r:
            raise TimeoutError('Genetic algorithm did not finish within the '
                               'specified {} seconds.'.format(timeout))

        return None
