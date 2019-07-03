"""Module for pyvvo's genetic algorithm.

TODO: Create some sort of configuration file, like ga_config.json.
"""
# Standard library:
import random
import multiprocessing as mp
import array
import os
from queue import Queue

# Third party:
import numpy as np

# pyvvo:
from pyvvo import equipment

# Constants.
TRIPLEX_GROUP = 'tl'


def map_chromosome(regulators, capacitors):
    """Given regulators and capacitors, map states onto a chromosome.

    :param regulators: dictionary as returned by
        equipment.initialize_regulators
    :param capacitors: dictionary as returned by
        equipment.initialize_capacitors

    The map will be keyed by name rather than MRID - this is because
    we'll be looking up objects in GridLAB-D models by name many times,
    but only commanding regulators by MRID once.

    Equipment phase will be appended to the name with an underscore.
    E.g., reg1_A.
    """
    # Initialize our output
    out = {}

    # Track the current index in our chromosome.
    idx = 0

    def map_reg(reg_in, dict_out, idx_in):
        """Nested helper to map a regulator."""
        # If the regulator is not controllable, DO NOT MAP.
        if not reg_in.controllable:
            return dict_out, idx_in

        # Create a key for this regulator
        key = equip_key(reg_in)

        # Compute how many bits are needed to represent this
        # regulator's tap positions.
        length = reg_bin_length(reg_in)

        # Map. Track MRID so we can command the regulators later.
        dict_out[key] = {'idx': (idx_in, idx_in + length),
                         'mrid': reg_in.mrid}

        return dict_out, idx_in + length

    def map_cap(cap_in, dict_out, idx_in):
        """Nested helper to map a capacitor."""
        # DO NOT MAP if not controllable.
        if not cap_in.controllable:
            return dict_out, idx_in

        # Create a key.
        key = equip_key(cap_in)

        # At the moment, we're only supporting capacitors with only one
        # switch, so we can just hard-code the length to be one.
        dict_out[key] = {'idx': (idx_in, idx_in + 1), 'mrid': cap_in.mrid}

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


def equip_key(eq):
    """Given an object which inherits from
    equipment.EquipmentSinglePhase, create a useful key."""
    return eq.name + '_' + eq.phase


def reg_bin_length(reg):
    """Determine how many bits are needed to represent a regulator.

    :param reg: regulator.RegulatorSinglePhase object.

    :returns integer representing how many bits are needed to represent
    a regulators tap positions.
    """
    # Use raise_taps and lower_taps from GridLAB-D to compute the number
    # of bits needed.
    return int_bin_length(reg.raise_taps + reg.lower_taps)


def int_bin_length(x):
    """Determine how many bits are needed to represent an integer."""
    # Rely on the fact that Python's "bin" method prepends the string
    # with "0b": https://docs.python.org/3/library/functions.html#bin
    return len(bin(x)[2:])


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
    TODO: Add recorder(s) for head(s) of feeder.
    """
    ####################################################################
    # 1)
    # Make the model runnable.
    glm_mgr.add_run_components(starttime=starttime, stoptime=stoptime)
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
         'table': 'triplex_voltage',
         'name': 'triplex_load_recorder',
         'group': '"groupid={}"'.format(TRIPLEX_GROUP),
         'property': '"measured_voltage_12.mag"',
         # TODO: Stop hard-coding.
         'interval': 60,
         'limit': -1,
         # TODO: Ensure we're being adequately careful with mode.
         'mode': 'a',
         # TODO: Stop hard-coding, put in config.
         'query_buffer_limit': 20000
         })
    ####################################################################

    # TODO: Add recorders for other things.
    pass


def evaluate(individual):
    """Evaluate the fitness of an individual.

    In our case, this means to:
    1) Translate chromosome into regulator and capacitor positions.
    2) Update a GridLAB-D model with the aforementioned positions.
    3) Run the GridLAB-D model.
    4) Use the results of the GridLAB-D model run to compute the
        individual's fitness.
    """
    pass


def main(weight_dict, glm_mgr):
    """Function to run the GA in its entirety.

    :param weight_dict: Dictionary of weights for determining an
        individual's overall fitness.
    :param glm_mgr: glm.GLMManager object. Should have a run-able model.
    """
    # TODO: Stop hard-coding the number of individuals.
    NUM_IND = 100

    # Create a queue containing ID's for individuals.
    # Note that the only parallelized operation is the evaluation of
    # an individual's fitness, so we'll use a standard Queue object.
    id_q = Queue()
    for k in range(NUM_IND):
        id_q.put_nowait(k)


if __name__ == '__main__':
    main(weight_dict={'one': -1, 'two': -2, 'three': -3}, glm_mgr=None)

