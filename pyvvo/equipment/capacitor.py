"""Module for all things capacitors."""
import logging
from pyvvo.utils import list_to_string

# class CapacitorMultiPhase:
#     """Multi-phase capacitor - collection of single phase capacitors.
#     TODO: Add more information.
#     """
#
#     def __init__(self, name, mrid, phases):
#         """Initialize capacitor object."""
#         self.name = name
#         self.mrid = mrid
#         self.phases = phases
#


class CapacitorSinglePhase:
    """Single phase capacitor.
    Parameters will for the most part come straight out of the CIM.

    TODO: Accept all CIM parameters.
    """
    # Allowed phases (case insensitive)
    PHASES = ('A', 'B', 'C')

    # Allowed states (case insensitive)
    STATES = ('OPEN', 'CLOSED')

    def __init__(self, name, mrid, phase, state=None, name_prefix='cap_'):
        """Initialize single phase capacitor.
        TODO: Document parameters.
        """
        # Get log.
        self.log = logging.getLogger()

        # Assign the name prefix.
        self.name_prefix = name_prefix
        # Assign the name. NOTE: the name_prefix will be prepended to
        # the given name. The syntax below is slightly unintuitive.
        self.name = name

        # Assign remaining properties.
        self.mrid = mrid
        self.phase = phase
        self.state = state

        self.log.debug('CapacitorSinglePhase {} initialized.'.format(self.name))

    ####################################################################
    # PROPERTY GETTERS AND SETTERS
    ####################################################################
    @property
    def name_prefix(self):
        """Prefix prepended to the given name."""
        return self._name_prefix

    @name_prefix.setter
    def name_prefix(self, name_prefix):
        """Simply set the name_prefix. Must be a string."""
        if not isinstance(name_prefix, str):
            raise TypeError('name_prefix must be a string.')

        self._name_prefix = name_prefix

    @property
    def name(self):
        """Capacitor name corresponding to object in GridLAB-D model.

        Note that the name_prefix is prepended to the given name.
        """
        return self._name

    @name.setter
    def name(self, name):
        """Add prefix to given name, set 'name' attribute."""
        # Strings only.
        if not isinstance(name, str):
            raise TypeError('name must be a string.')

        # Add the name_prefix to the name.
        self._name = self.name_prefix + name

    @property
    def mrid(self):
        """CIM MRID for capacitor."""
        return self._mrid

    @mrid.setter
    def mrid(self, mrid):
        """mrid must be a string.
        TODO: Enforce valid uuid?
        """
        if not isinstance(mrid, str):
            raise TypeError('mrid must be a string.')

        # Assign.
        self._mrid = mrid

    @property
    def phase(self):
        """Phase which capacitor is on."""
        return self._phase

    @phase.setter
    def phase(self, phase):
        """Phase must be a string, and must be in PHASES.

         Phase is case insensitive.
         """
        if not isinstance(phase, str):
            raise TypeError('phase must be a string.')

        # Cast to upper case.
        upper_phase = phase.upper()

        # Ensure it's valid.
        if upper_phase not in self.PHASES:
            phase_str = list_to_string(self.PHASES, 'or')
            m = 'phase must be {} (case insensitive).'.format(phase_str)
            raise ValueError(m)

        # Assign.
        self._phase = upper_phase

    @property
    def state(self):
        """State capacitor is in. OPEN/CLOSED/None. None --> unknown."""
        return self._state

    @state.setter
    def state(self, state):
        """State must be None or a string, and must be in STATES.

         State is case insensitive.
         """
        # If state is None, simply set and return.
        if state is None:
            self._state = None
            return
        elif isinstance(state, str):
            # Cast to upper case.
            upper_state = state.upper()

            # Ensure it's valid.
            if upper_state not in self.STATES:
                state_str = list_to_string(self.STATES, 'or')
                m = 'state must be {} (case insensitive).'.format(state_str)
                raise ValueError(m)

            # Assign and return.
            self._state = upper_state
            return
        else:
            raise TypeError('state must None or be a string.')

        # That's it.
