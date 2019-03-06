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


def initialize_controllable_capacitors(df):
    """
    Helper to initialize controllable capacitors given a DataFrame with
    capacitor information. The DataFrame should come from
    sparql.SPARQLManager.query_capacitors.
    """
    # Filter to get controllable capacitors. Uncontrollable capacitors
    # have a np.nan value in the ctrlenabled column.
    c_cap = df[~df['ctrlenabled'].isnull()]

    # A NaN in the phase column indicates a multi-phase capacitor. I
    # have not yet seen a controllable multi-phase capacitor in the CIM,
    # so we'll cross that bridge when we get there.
    if c_cap['phs'].isnull().any():
        # TODO: handle multi-phase controllable caps.
        raise NotImplementedError('Currently single phase controllable '
                                  'capacitors are supported.')
    # Initialize return.
    out = dict()

    # Loop and initialize.
    for row in c_cap.itertuples():
        # Build a capacitor object.
        out[row.name] = \
            CapacitorSinglePhase(name=row.name, mrid=row.mrid, phase=row.phs,
                                 state=None)

    return out


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
        self.log = logging.getLogger(__name__)

        # Check and assign the name prefix.
        if not isinstance(name_prefix, str):
            raise TypeError('name_prefix must be a string.')

        self._name_prefix = name_prefix

        # Check and assign the name. NOTE: the name_prefix will be
        # prepended to the given name.
        if not isinstance(name, str):
            raise TypeError('name must be a string.')

        # Add the name_prefix to the name.
        self._name = self.name_prefix + name

        # Check and assign MRID.
        if not isinstance(mrid, str):
            raise TypeError('mrid must be a string.')

        self._mrid = mrid

        # Check and assign phase.
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

        # Check and assign state.
        if state is None:
            # If state is None, simply set.
            self._state = None
        elif isinstance(state, str):
            # Cast to upper case.
            upper_state = state.upper()

            # Ensure it's valid.
            if upper_state not in self.STATES:
                state_str = list_to_string(self.STATES, 'or')
                m = 'state must be {} (case insensitive).'.format(state_str)
                raise ValueError(m)

            # Assign.
            self._state = upper_state
        else:
            # State is a bad type.
            raise TypeError('state must None or be a string.')

        self.log.debug('CapacitorSinglePhase {} '.format(self.name)
                       + 'initialized.')

    def __repr__(self):
        return self.name

    ####################################################################
    # PROPERTY GETTERS AND SETTERS
    ####################################################################
    @property
    def name_prefix(self):
        """Prefix prepended to the given name."""
        return self._name_prefix

    @property
    def name(self):
        """Capacitor name corresponding to object in GridLAB-D model.

        Note that the name_prefix is prepended to the given name.
        """
        return self._name

    @property
    def mrid(self):
        """CIM MRID for capacitor."""
        return self._mrid

    @property
    def phase(self):
        """Phase which capacitor is on."""
        return self._phase

    @property
    def state(self):
        """State capacitor is in. OPEN/CLOSED/None. None --> unknown."""
        return self._state
