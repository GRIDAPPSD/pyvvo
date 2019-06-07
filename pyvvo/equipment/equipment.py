"""Module for general equipment classes etc. that can be shared amongst
other modules.
"""
from pyvvo import utils


class EquipmentSinglePhase:
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

    @property
    def mrid(self):
        return self._mrid

    @property
    def name(self):
        return self._name

    @property
    def phase(self):
        return self._phase
