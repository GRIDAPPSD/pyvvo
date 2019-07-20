"""Module for pyvvo's genetic algorithm.

TODO: Create some sort of configuration file, like ga_config.json.
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
with open(os.path.join(THIS_DIR, 'pyvvo_config.json'), 'r') as f:
    CONFIG = json.load(f)
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
# https://github.com/GRIDAPPSD/GOSS-GridAPPS-D/blob/releases/2019.06.beta/services/fncsgossbridge/service/fncs_goss_bridge.py
# Use these prefixes in conjunction with _cim_to_glm_name
REG_PREFIX = 'reg'
CAP_PREFIX = 'cap'
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


def _cim_to_glm_name(prefix, cim_name):
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

    :returns integer representing how many bits are needed to represent
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

        :returns Possibly altered chromosome.
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
            self.log.info("Individual {}'s chromosome has been modified "
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

        :returns child1, child2, both are Individuals.
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
        # First, update regulators and capacitors in the glm_mgr's model
        # based on this Individual's chromosome. As the glm_mgr is
        # mutable, it's updated without a return for it here.
        reg_penalty, cap_penalty = \
            self._update_model_compute_costs(glm_mgr=glm_mgr)

        # Create an _Evaluator to do the work of running the model and
        # computing associated costs.
        evaluator = _Evaluator(uid=self.uid, glm_mgr=glm_mgr, db_conn=db_conn)
        penalties = evaluator.evaluate()

        # Add the regulator tap changing and capacitor switching costs.
        penalties['regulator_tap'] = reg_penalty
        penalties['capacitor_switch'] = cap_penalty

        # An individual's fitness is the sum of their penalties.
        self._fitness = 0
        for p in penalties.values():
            self._fitness += p

        self._penalties = penalties

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
        model_name = _cim_to_glm_name(prefix=REG_PREFIX,
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
        model_name = _cim_to_glm_name(prefix=CAP_PREFIX,
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
            return

        t0 = time.time()
        # So, we now have an individual. Evaluate.
        ind.evaluate(glm_mgr=glm_mgr,
                     db_conn=db.connect_loop(timeout=10, retry_interval=0.1))
        t1 = time.time()

        # Dump information into the logging queue.
        logging_queue.put({'uid': ind.uid, 'fitness': ind.fitness,
                           'penalties': ind.penalties,
                           'time': t1 - t0})

        # Put the now fully evaluated individual in the output queue.
        output_queue.put(ind)

        # Mark this task as complete.
        input_queue.task_done()


def _logging_thread(logging_queue):
    """Function intended to be the target of a thread, used to log
    the progress of genetic algorithm fitness evaluation.

    :param logging_queue: Multiprocessing.Queue object, which will have
        dictionaries of the following format placed in it:
        {'uid': <uid, integer>, 'fitness': <fitness, float>,
        'penalties': <penalties, dictionary>}.

        If None is received in the queue, the thread will terminate.
    """
    # Loop forever
    while True:
        # Get dictionary from the queue.
        log_dict = logging_queue.get(block=True, timeout=None)

        # Terminate on receiving None.
        if log_dict is None:
            return

        # Log the individual completion.
        LOG.info('Individual {} evaluated in {:.2f} seconds. Fitness: {:.2f}.'
                 .format(log_dict['uid'], log_dict['time'],
                         log_dict['fitness']))

        LOG.debug("Individual {}'s penalties:\n{}"
                  .format(log_dict['uid'],
                          json.dumps(log_dict['penalties'], indent=4)))


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
        # Setup queues.
        # Queue for individual evaluation.
        self._input_queue = mp.JoinableQueue()

        # Queue for retrieving individuals after evaluation.
        self._output_queue = mp.Queue()

        # Queue for logging evaluation as it proceeds.
        self._logging_queue = mp.Queue()

        ################################################################
        # Threads and processes.

        # Start the logging thread.
        self._logging_thread = \
            threading.Thread(target=_logging_thread,
                             kwargs={'logging_queue': self.logging_queue})

        self.logging_thread.start()

        # On to processes. For now, use all but one core.
        # TODO: We'll want to make this configurable in the future.
        n_jobs = mp.cpu_count() - 1

        # Initialize processes.
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
            self.processes.append(p)
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

        :returns parent1, parent2. Both are Individuals from the
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

        :returns None. The individual is mutated in place.
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

        # Get list to dump individuals in.
        evaluated_individuals = []

        # Dump the output queue into new list.
        _dump_queue(q=self.output_queue, i=evaluated_individuals)

        # Put the individuals back in the population (recall we popped
        # them from the list earlier).
        self._population.extend(evaluated_individuals)

        # All done.

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
                            tournament_size=
                            math.ceil(self.tournament_fraction
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
            winners = _tournament(population=self.population,
                                  tournament_size=self.tournament_size, n=2)
            parent1 = self.population[winners[0]]
            parent2 = self.population[winners[1]]

            # Perform crossover with some probability.
            if np.random.rand() < self.prob_crossover:
                child1, child2 = \
                    parent1.crossover_by_gene(other=parent2,
                                              uid1=next(self.uid_counter),
                                              uid2=next(self.uid_counter))

                # Possibly mutate these individuals.
                m = np.random.rand(2) < self.prob_mutate_individual

                for tf, ind in [(m[0], child1), (m[1], child2)]:
                    # While it may look like this if/else should be one
                    # if statement with an "or," the way it's written
                    # now avoids an extra call to _chrom_already_existed
                    # in some cases, which is good, because that can be
                    # expensive.
                    if tf:
                        self._mutate(ind=ind)
                    elif self._chrom_already_existed(ind.chromosome):
                        # Force mutation if this individual isn't unique.
                        # This keeps the logic simpler than excluding the
                        # child.
                        self._mutate(ind=ind)

            else:
                children = []
                for p in [parent1, parent2]:
                    children.append(self._init_individual(
                        chrom_override=p.chromosome.copy(),
                        special_init=None))

                # Unpack the list of children.
                child1, child2 = children

                # Mutate.
                self._mutate(child1)
                self._mutate(child2)

            # Add the children to the list of offspring.
            offspring.extend([child1, child2])

        # While this could waste some effort, we want to keep our
        # population at the correct size to avoid any surprises. Call
        # it infant mortality. Genetic algorithm joke, nice.
        if len(self.population) + len(offspring) > self.population_size:
            offspring = offspring[0:-1]

        # Merge the population into the population.
        self._population.extend(offspring)

        # TODO: Remove this assert statement when test is in place.
        assert len(self.population) == self.population_size

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


class ChromosomeAlreadyExistedError(Exception):
    """Raised if a chromosome has already been existed.
    """


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

    :returns i. While this isn't necessary, it's explicit.
    """
    while True:
        try:
            i.append(q.get_nowait())
        except queue.Empty:
            return i


def main(regulators, capacitors, glm_mgr, starttime, stoptime):
    """Function to run the GA in its entirety.

    :param regulators: dictionary as returned by
        equipment.initialize_regulators
    :param capacitors: dictionary as returned by
        equipment.initialize_capacitors
    :param glm_mgr: glm.GLMManager object. Should have a run-able model,
        which is made possible by the object's add_run_components
        method.
    :param starttime: Python datetime object for simulation start. For
        more details, see the docstring for prep_glm_mgr.
    :param stoptime: Python datetime object for simulation end. See
        prep_glm_mgr.
    """
    t0 = time.time()
    pop = Population(regulators=regulators, capacitors=capacitors,
                     glm_mgr=glm_mgr, starttime=starttime,
                     stoptime=stoptime)

    pop.initialize_population()
    pop.evaluate_population()

    g = 1
    while g < CONFIG['ga']['generations']:
        pop.natural_selection()
        # The best individual will always be in position 0 after
        # natural selection.
        best = pop.population[0]
        print('After generation {}, best fitness: {:.2f} from individual {}'
              .format(g, best.fitness, best.uid))
        pop.crossover_and_mutate()
        pop.evaluate_population()
        g += 1

    # Sort the population.
    pop.sort_population()
    t1 = time.time()
    best = pop.population[0]
    print('Best overall fitness: {:.2f} from individual {}'
          .format(best.fitness, best.uid))

    print('Total GA run time: {:.2f}'.format(t1-t0))
    print('All done!')
