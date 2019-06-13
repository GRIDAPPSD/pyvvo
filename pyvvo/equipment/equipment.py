"""Module for general equipment classes etc. that can be shared amongst
other modules.
"""
from abc import ABC, abstractmethod
from collections import deque
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
