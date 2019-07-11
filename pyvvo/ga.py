"""Module for pyvvo's genetic algorithm.

TODO: Create some sort of configuration file, like ga_config.json.
"""
# Standard library:
import multiprocessing as mp
import os
from queue import Queue
from datetime import datetime
import logging

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


def map_chromosome(regulators, capacitors):
    """Given regulators and capacitors, map states onto a chromosome.

    :param regulators: dictionary as returned by
        equipment.initialize_regulators
    :param capacitors: dictionary as returned by
        equipment.initialize_capacitors

    :returns: Dict keyed by equipment name. Each equipment maps to a
        dictionary keyed by phase (A, B, or C). These phase dictionaries
        have the following fields:
        - 'idx': Tuple with a pair of integers, indicating the indices
            on the chromosome for this particular piece of equipment.
            These indices can simply be used like so:
                dict['idx'][0]:dict['idx'][1]
        - 'eq_obj': The equipment object itself. At the moment, these
            will be equipment.CapacitorSinglePhase or
            equipment.RegulatorSinglePhase objects.
        - 'range': Tuple with a pair of integers, indicating the
            (low, high) range of states the equipment can take on. This
            is for convenience, and enables the Individual class to be
            agnostic about the differences between equipment.
    """
    if not isinstance(regulators, dict):
        raise TypeError('regulators must be a dictionary.')

    if not isinstance(capacitors, dict):
        raise TypeError('capacitors must be a dictionary.')

    # Initialize our output
    out = {}

    # Track the current index in our chromosome.
    idx = 0

    def map_reg(reg_in, dict_out, idx_in):
        """Nested helper to map a regulator."""
        # If the regulator is not controllable, DO NOT MAP.
        if not reg_in.controllable:
            return dict_out, idx_in

        # Compute how many bits are needed to represent this
        # regulator's tap positions.
        num_bits = _reg_bin_length(reg_in)

        # Initialize dictionary for mapping.
        m = {'idx': (idx_in, idx_in + num_bits), 'eq_obj': reg_in,
             'range': (0, reg_in.raise_taps + reg_in.lower_taps)}
        # Map.
        try:
            dict_out[reg_in.name][reg_in.phase] = m
        except KeyError:
            # We don't have a dictionary for this regulator yet.
            dict_out[reg_in.name] = {reg_in.phase: m}

        # Explicitly return the dictionary and the incremented index.
        return dict_out, idx_in + num_bits

    def map_cap(cap_in, dict_out, idx_in):
        """Nested helper to map a capacitor."""
        # DO NOT MAP if not controllable.
        if not cap_in.controllable:
            return dict_out, idx_in

        # Initialize dictionary for mapping. Capacitors always only get
        # one bit.
        m = {'idx': (idx_in, idx_in + 1), 'eq_obj': cap_in,
             'range': (0, 1)}

        try:
            dict_out[cap_in.name][cap_in.phase] = m
        except KeyError:
            # We don't have a dictionary for this capacitor yet.
            dict_out[cap_in.name] = {cap_in.phase: m}

        return dict_out, idx_in + 1

    # Loop over the regulators.
    for reg_mrid, reg_or_dict in regulators.items():

        if isinstance(reg_or_dict, equipment.RegulatorSinglePhase):
            # Map it!
            out, idx = map_reg(reg_or_dict, out, idx)

        elif isinstance(reg_or_dict, dict):
            # Loop over the phases and map.
            for reg in reg_or_dict.values():
                out, idx = map_reg(reg, out, idx)

    # Loop over the capacitors.
    for cap_mrid, cap_or_dict in capacitors.items():
        if isinstance(cap_or_dict, equipment.CapacitorSinglePhase):
            out, idx = map_cap(cap_or_dict, out, idx)
        elif isinstance(cap_or_dict, dict):
            # Loop over phases.
            for cap in cap_or_dict.values():
                out, idx = map_cap(cap, out, idx)

    # At this point, our idx represents the total length of the
    # chromosome.
    return out, idx


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

    def __init__(self, uid, chrom_len, chrom_map, chrom_override=None):
        """Initialize an individual for the genetic algorithm.

        :param uid: Unique identifier for this individual. Integer.
        :param chrom_len: Length of chromosome to generate. This is the
            second return from the map_chromosome method of this module.
        :param chrom_map: Dictionary mapping of the chromosome. Comes
            from the first return of the map_chromosome method.
        :param chrom_override: If provided (not None), this chromosome
            is used instead of randomly generating one. Must be a
            numpy.ndarray with dtype np.bool and shape (chrom_len,).

        NOTE: It is expected that chrom_len and chrom_map come from the
            same call to map_chromosome. Thus, there will be no
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

        if not isinstance(chrom_map, dict):
            raise TypeError('chrom_map must be a dictionary. It should come '
                            'from the map_chromosome method.')

        self._chrom_map = chrom_map

        # Either randomly initialize a chromosome, or use the given
        # chromosome.
        if chrom_override is None:
            # Initialize the chromosome by looping over equipment in the
            # map and drawing random valid states.
            self._chromosome = self._initialize_chromosome()
        else:
            # Check the chromosome, alter if necessary.
            self._chromosome = self._check_and_fix_chromosome(chrom_override)

        # Initialize fitness to None.
        self._fitness = None

        # Initialize penalties to None.
        self._penalties = None

    @property
    def uid(self):
        return self._uid

    @property
    def chrom_len(self):
        return self._chrom_len

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

    def _initialize_chromosome(self):
        """Helper to randomly initialize a chromosome.

        The chromosome is initialized by looping over the chromosome
        map, and randomly generating a valid state for each piece of
        equipment.

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
                # Draw a random number in the given range. Note that
                # the interval for np.random.randint is [low, high)
                n = np.random.randint(eq_dict['range'][0],
                                      eq_dict['range'][1] + 1,
                                      None)

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
        mask = np.random.randint(low=0, high=2, dtype=np.bool)

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
                            chrom_override=chrom1, chrom_map=self.chrom_map)
        child2 = Individual(uid=uid2, chrom_len=self.chrom_len,
                            chrom_override=chrom2, chrom_map=self.chrom_map)

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

        return penalties

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
        if not isinstance(db_conn, MySQLdb.Connection):
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
        """This is the 'main' method of this module. Write + run
        a GridLAB-D model and compute costs associated with it.
        """
        # Write the model to file and run it.
        model = 'model_{}.glm'.format(self.uid)
        self.glm_mgr.write_model(model)

        # Run it.
        result = utils.run_gld(model)

        # TODO: Best way to handle failed runs? Maybe make costs
        #  infinite? Make sure to add logging.
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

        # Compute the energy cost. Note the energy cost in CONFIG is
        # a per kWh figure, and returns from GridLAB-D are in Wh. Also
        # note that the measured_real_energy property of a meter
        # reports accumulation - hence why we grab the last item.
        penalties['energy'] = (sub_data.iloc[-1][SUBSTATION_ENERGY]
                               * TO_KW_FACTOR * CONFIG['costs']['energy'])

        # We're done here. Rely on the calling Individual to add tap
        # changing and capacitor switching costs.
        return penalties

    def _voltage_penalty(self, query):
        """Helper used by _low_voltage_penalty and _high_voltage_penalty

        :param query: String. Query to run.
        """
        result = db.execute_and_fetch_all(db_conn=self.db_conn,
                                          query=query)

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
            - 'id' and 't' columns are GridLAB-D defaults, and are
                hard-coded.
            - The first time step is skipped as it's unreliable -
                GridLAB-D hasn't "settled" yet.
        """
        # Create the query.
        q_low = "SELECT SUM(({nom_v} - {mag_col}) * {penalty}) as penalty" \
                " FROM {table} WHERE ({mag_col} < {low_v} " \
                "AND t > '{starttime}')".format(
                    nom_v=TRIPLEX_NOMINAL_VOLTAGE, mag_col=TRIPLEX_PROPERTY_DB,
                    penalty=CONFIG['costs']['voltage_violation_low'],
                    table=self.triplex_table, low_v=TRIPLEX_LOW_VOLTAGE,
                    starttime=self.starttime
                    )

        # Use the helper to execute and extract the penalty.
        return self._voltage_penalty(query=q_low)

    def _high_voltage_penalty(self):
        """Compute high voltage penalty for triplex loads. Called by
        'evaluate'.

        NOTES:
            - 'id' and 't' columns are GridLAB-D defaults, and are
                hard-coded.
            - The first time step is skipped as it's unreliable -
                GridLAB-D hasn't "settled" yet.
        """
        q_high = "SELECT SUM(({mag_col} - {nom_v}) * {penalty}) as penalty" \
                 " FROM {table} WHERE ({mag_col} > {high_v} " \
                 "AND t > '{starttime}')".format(
                    nom_v=TRIPLEX_NOMINAL_VOLTAGE, mag_col=TRIPLEX_PROPERTY_DB,
                    penalty=CONFIG['costs']['voltage_violation_low'],
                    table=self.triplex_table, high_v=TRIPLEX_HIGH_VOLTAGE,
                    starttime=self.starttime
                    )

        # Use the helper to execute and extract the penalty.
        return self._voltage_penalty(query=q_high)

    def _get_substation_data(self):
        """Helper to grab substation data, and ensure not empty.
        """

        # Grab all the substation data. Hard-code the 'id' column.
        sub_data = \
            pd.read_sql_query(sql="SELECT * FROM {} WHERE t > '{}';".format(
                self.substation_table, self.starttime), con=self.db_conn,
                index_col='id')

        # We'll be using sub_data as a sort of safety check - it cannot
        # be empty.
        if sub_data.shape[0] < 1:
            raise ValueError('No substation data was received! This likely '
                             'indicates something is wrong with the configured'
                             ' start/stop time, sample interval, and/or '
                             'minimum timestep.')

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
        # Compute the penalty for lagging power factors. Start by
        # finding all values which are both lagging and below the limit.
        lag_limit = CONFIG['limits']['power_factor_lag']
        lag_mask = (pf > 0) & (pf < lag_limit)
        # Take the difference between the limit and the values. Multiply
        # by 100 to get things in terms of a per 0.01 deviation.
        # Finally, multiply by the costs
        return ((lag_limit - pf[lag_mask]) * 100
                * CONFIG['costs']['power_factor_lag']).sum()

    @staticmethod
    def _pf_lead_penalty(pf):
        # Compute the penalty for leading power factors. Start by
        # finding all values which are both leading and below the limit.
        # Leading power factors are negative, hence the use of abs.
        lead_limit = CONFIG['limits']['power_factor_lead']
        lead_mask = (pf < 0) & (np.abs(pf) < lead_limit)
        # Take the difference between the limit and the values. Multiply
        # by 100 to get things in terms of a per 0.01 deviation.
        # Finally, multiply by the costs
        return ((lead_limit - np.abs(pf[lead_mask])) * 100
                * CONFIG['costs']['power_factor_lead']).sum()


def main(weight_dict, glm_mgr):
    """Function to run the GA in its entirety.

    :param weight_dict: Dictionary of weights for determining an
        individual's overall fitness.
    :param glm_mgr: glm.GLMManager object. Should have a run-able model.
    """
    # Create a queue containing ID's for individuals.
    # Note that the only parallelized operation is the evaluation of
    # an individual's fitness, so we'll use a standard Queue object.
    id_q = Queue()
    for k in range(CONFIG['ga']['individuals']):
        id_q.put_nowait(k)


if __name__ == '__main__':
    main(weight_dict={'one': -1, 'two': -2, 'three': -3}, glm_mgr=None)
