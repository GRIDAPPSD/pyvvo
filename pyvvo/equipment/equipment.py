"""Module for general equipment classes etc. that can be shared amongst
other modules.
"""
from abc import ABC, abstractmethod
import logging
from collections import deque
from pandas import DataFrame
from pyvvo import utils


class EquipmentSinglePhase(ABC):
    """Generic 'equipment' class, for e.g. capacitors and regulators."""

    PHASES = ('A', 'B', 'C')

    def __init__(self, mrid, name, phase):
        # Check inputs and assign.
        if not isinstance(mrid, str):
            raise TypeError('mrid must be a string.')
        self._mrid = mrid

        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        self._name = name

        if not isinstance(phase, str):
            raise TypeError('phase must be string.')

        # Phase is case insensitive. Cast to upper.
        u_phase = phase.upper()
        if u_phase not in self.PHASES:
            m = 'phase must be on of ' + utils.list_to_string(self.PHASES, ',')
            raise ValueError(m)

        self._phase = u_phase

        # The state_deque will be used to track the current (index 0)
        # and previous (index 1) states.
        self._state_deque = deque([None, None], 2)

    ####################################################################
    # PROPERTIES
    ####################################################################
    # noinspection PyPep8Naming
    @property
    @abstractmethod
    def STATE_CIM_PROPERTY(self):
        """STATE_CIM_PROPERTY defines the CIM attribute which
        corresponds to an object's 'state' attribute. E.g. for
        regulators, it will be TapChanger.step, or for capacitors it
        will be ShuntCompensator.sections. This property is used for
        building commands in EquipmentManager's
        "build_equipment_commands" method.
        """
        pass

    @property
    def mrid(self):
        return self._mrid

    @property
    def name(self):
        return self._name

    @property
    def phase(self):
        return self._phase

    @property
    def state(self):
        return self._state_deque[0]

    @state.setter
    def state(self, value):
        self._check_state(value)
        self._state_deque.appendleft(value)

    @property
    def state_old(self):
        return self._state_deque[-1]

    ####################################################################
    # METHODS
    ####################################################################
    @abstractmethod
    def _check_state(self, value):
        """Children of this class should institute a _check_state method
        which ensures the given state is valid. This will be called
        BEFORE setting the state.
        """
        pass


class EquipmentManager:
    """Class to keep EquipmentSinglePhase objects up to date as a
    simulation proceeds. E.g. capacitors or regulators.

    This is meant to be used in conjunction with a SimOutRouter from
    gridappsd_platform.py, which will route relevant simulation output
    to the "update_state" method.
    """

    def __init__(self, eq_dict, eq_meas, meas_mrid_col, eq_mrid_col):
        """Initialize.

        :param eq_dict: Dictionary of EquipmentSinglePhase objects
            as returned by regulator.initialize_controllable_regulators
            or capacitor.initialize_controllable_capacitors.
        :param eq_meas: Pandas DataFrame as returned by
            sparql.SPARQLManager's query_rtc_measurements method, or
            sparql.SPARQLManager's query query_capacitor_measurements
            method.
        :param meas_mrid_col: String. Column in eq_meas corresponding
            to measurement MRIDs in the eq_meas DataFrame.
        :param eq_mrid_col: String. Column in eq_meas corresponding to
            the equipment MRIDs in the eq_meas DataFrame.

        The eq_dict and eq_meas must have the same number of elements.
        """
        # Logging.
        self.log = logging.getLogger(__name__)

        # Simple type checking.
        if not isinstance(eq_dict, dict):
            raise TypeError('eq_dict must be a dictionary.')

        if not isinstance(eq_meas, DataFrame):
            raise TypeError('eq_meas must be a Pandas DataFrame.')

        if len(eq_dict) != eq_meas.shape[0]:
            raise ValueError('The number of measurements and number of '
                             'regulators do not match!')

        # Simply assign.
        self.eq_dict = eq_dict
        self.eq_meas = eq_meas

        # Create a map from measurement MRID to RegulatorSinglePhase.
        self.meas_eq_map = {}

        for row in self.eq_meas.itertuples():
            self.meas_eq_map[getattr(row, meas_mrid_col)] = \
                self.eq_dict[getattr(row, eq_mrid_col)]

        if len(self.meas_eq_map) != len(self.eq_dict):
            raise ValueError('The eq_dict and eq_meas inputs are '
                             'inconsistent as the resulting map has '
                             'a different shape.')

    def update_state(self, msg):
        """Given a message from a gridappsd_platform SimOutRouter,
        update regulator positions.

        :param msg: list passed to this method via a SimOutRouter's
            "_on_message" method.
        """
        # # Dump message for testing:
        # import json
        # with open('reg_meas_message.json', 'w') as f:
        #     json.dump(msg, f)

        # Type checking:
        if not isinstance(msg, list):
            raise TypeError('msg must be a list!')

        # Iterate over the message and update regulators.
        for m in msg:
            # Type checking:
            if not isinstance(m, dict):
                raise TypeError('Each entry in msg must be a dict!')

            # Grab mrid and value.
            meas_mrid = m['measurement_mrid']
            value = m['value']

            # Update!
            try:
                self.meas_eq_map[meas_mrid].state = value
            except KeyError:
                self.log.warning(
                    'Measurement MRID {} not present in the map!'.format(
                        meas_mrid))
            else:
                self.log.debug('Equipment {} state updated to: {}'.format(
                    str(self.meas_eq_map[meas_mrid]), value
                ))

    def build_equipment_commands(self, eq_dict_forward):
        """Function to build command for changing equipment state. The
        command itself should be sent with a
        gridappsd_platform.PlatformManager object's send_command method.

        This object's eq_dict is considered to have the current/old
        states.

        NOTE: It isn't 100% clear if this is the best home for this
        method, but hey, that's okay.

        :param eq_dict_forward: dictionary of
            equipment.EquipmentSinglePhase objects as would be passed
            to this classes' constructor. The states for these objects
            will be what the equipment in the simulation will be
            commanded to.

        :returns dictionary with keys corresponding to the send_command
            method of a gridappsd_platform.PlatformManager object.

        NOTE: eq_dict_forward and self.eq_dict should be identical
        except for object states.
        """
        # Initialize lists of parameters.
        eq_ids = []
        eq_attr = []
        eq_forward_list = []
        eq_reverse_list = []

        # Loop over equipment.
        for eq_mrid, eq_forward in eq_dict_forward.items():
            # Lookup the equipment object in self's eq_dict
            # corresponding to this piece of equipment.
            try:
                eq_reverse = self.eq_dict[eq_mrid]
            except KeyError:
                m = 'The given eq_dict_forward is not matching up with '\
                    'self.eq_dict! Ensure these eq_dicts came from the '\
                    'same model, etc.'

                raise ValueError(m) from None

            # Add the tap change mrid.
            eq_ids.append(eq_mrid)
            # Add the attribute.
            eq_attr.append(eq_forward.STATE_CIM_PROPERTY)
            # Add the forward position.
            eq_forward_list.append(eq_forward.step)
            # Grab the reverse position.
            eq_reverse_list.append(eq_reverse.step)

        return {"object_ids": eq_ids, "attributes": eq_attr,
                "forward_values": eq_forward_list,
                "reverse_values": eq_reverse_list}

# class EquipmentMultiPhase:
#     """
#     Class that essentially acts as a container for EquipmentSinglePhase
#     objects (and their derivatives)
#     """
#
#     # Allowable phases.
#     PHASES = ('A', 'B', 'C')
#
#     def __init__(self, equipment_list):
#         """Take in list of EquipmentSinglePhase objects.
#
#         :param equipment_list: list of three or less
#             EquipmentSinglePhase objects.
#         """
#
#         # Check input type.
#         if not isinstance(equipment_list, (list, tuple)):
#             raise TypeError('equipment_list must be a list or tuple.')
#
#         if (len(equipment_list) < 1) or (len(equipment_list) > 3):
#             m = 'equipment_list must meet 1 <= len(equipment_list) <= 3.'
#             raise ValueError(m)
#
#         # Initialize phases to None. We'll use case-insensitive attributes.
#         self._a = None
#         self._b = None
#         self._c = None
#
#         # Loop over the list.
#         for equipment in equipment_list:
#             if not isinstance(equipment, EquipmentSinglePhase):
#                 m = ('All items in equipment_list must be EquipmentSinglePhase'
#                      + ' objects (or derivatives).')
#                 raise TypeError(m)
#
#             # Set name attribute if necessary, ensure all
#             # EquipmentSinglePhase objects refer to the same name.
#             try:
#                 name_match = equipment.name == self.name
#             except AttributeError:
#                 # Set the name.
#                 self._name = equipment.name
#             else:
#                 # If the names don't match, raise exception.
#                 if not name_match:
#                     m = 'EquipmentSinglePhase objects do not have matching '\
#                         '"name" attributes.'
#                     raise ValueError(m)
#
#             # Set mrid attribute if necessary, ensure all
#             # EquipmentSinglePhase objects refer to the same mrid.
#             try:
#                 mrid_match = equipment.mrid == self.mrid
#             except AttributeError:
#                 # Set mrid.
#                 self._mrid = equipment.mrid
#             else:
#                 if not mrid_match:
#                     # If the mrids don't match, raise exception.
#                     m = 'EquipmentSinglePhase objects do not have matching '\
#                         '"mrid" attributes.'
#                     raise ValueError(m)
#
#             # Get this phase attribute for self.
#             attr_str = '_' + equipment.phase.lower()
#             attr = getattr(self, attr_str)
#             # Ensure this attribute has not yet been set to anything
#             # other than None.
#             if attr is not None:
#                 raise ValueError('Multiple equipments for phase {} were '
#                                  'given!'.format(equipment.phase.lower()))
#
#             setattr(self, attr_str, equipment)
#
#     def __repr__(self):
#         return '<EquipmentMultiPhase. name: {}'.format(self.name)
#
#     ####################################################################
#     # Getter methods
#     ####################################################################
#     @property
#     def name(self):
#         return self._name
#
#     @property
#     def mrid(self):
#         return self._mrid
#
#     @property
#     def a(self):
#         return self._a
#
#     @property
#     def b(self):
#         return self._b
#
#     @property
#     def c(self):
#         return self._c
#
#     # noinspection PyPep8Naming
#     @property
#     def A(self):
#         return self._a
#
#     # noinspection PyPep8Naming
#     @property
#     def B(self):
#         return self._b
#
#     # noinspection PyPep8Naming
#     @property
#     def C(self):
#         return self._c
