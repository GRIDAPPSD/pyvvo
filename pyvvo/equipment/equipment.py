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
