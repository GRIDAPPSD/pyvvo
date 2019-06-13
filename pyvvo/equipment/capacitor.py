"""Module for all things capacitors."""
import logging
from pyvvo.utils import list_to_string
from .equipment import EquipmentSinglePhase

# Setup log.
LOG = logging.getLogger(__name__)

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

# Attributes of RegulatingControl:
# https://gridappsd.readthedocs.io/en/latest/_images/cim_CapacitorClass.png
# These map to variable names in sparql.py.
REG_CONTROL = ['discrete', 'ctrlenabled', 'mode', 'monphs', 'deadband',
               'setpoint']


def initialize_controllable_capacitors(df):
    """
    Helper to initialize controllable capacitors given a DataFrame with
    capacitor information. The DataFrame should come from
    sparql.SPARQLManager.query_capacitors.
    """
    # Filter to get controllable capacitors. Uncontrollable capacitors
    # do not have a "RegulatingControl" object, and are thus missing
    # the following attributes:
    # discrete, ctrlenabled, mode, monphs, deadband, setpoint, and trm.
    # These can be found in sparql.py, do a search for
    # "RegulatingControl."
    rc = df.loc[:, REG_CONTROL]
    # If all of the REG_CONTROL columns are nan, this capacitor is not
    # controllable.
    c_cap = df[~rc.isnull().all(axis=1)]

    if c_cap.shape[0] == 0:
        LOG.warning('There are no controllable capacitors present!')
        return {}

    # Initialize return.
    out = dict()

    # Loop and initialize.
    for row in c_cap.itertuples():
        # Build a capacitor object.
        out[row.name] = \
            CapacitorSinglePhase(name=row.name, mrid=row.mrid, phase=row.phase,
                                 state=None, mode=row.mode)

    return out


class CapacitorSinglePhase(EquipmentSinglePhase):
    """Single phase capacitor.
    Parameters will for the most part come straight out of the CIM.

    TODO: Accept all CIM parameters.
    """

    # Allowed states (case insensitive)
    STATES = ('OPEN', 'CLOSED')

    # Allowed control modes (case insensitive). Corresponds to CIM
    # RegulatingControlModeKind. May need to add "MANUAL" option in the
    # future for GridLAB-D. Note: In CIM these are camelCase, but
    # keeping them lower to make case insensitivity simple.
    MODES = ('voltage', 'activepower', 'reactivepower', 'currentflow',
             'admittance', 'timescheduled', 'temperature', 'powerfactor')

    def __init__(self, name, mrid, phase, mode, state=None):
        """Initialize single phase capacitor.
        TODO: Document parameters.
        """
        # Get log.
        self.log = logging.getLogger(__name__)

        # Call super.
        super().__init__(mrid=mrid, name=name, phase=phase)

        # Check and assign MRID.
        if not isinstance(mrid, str):
            raise TypeError('mrid must be a string.')

        # Check and assign mode.
        if not isinstance(mode, str):
            raise TypeError('mode must be a string.')

        # Cast to lower case.
        lower_mode = mode.lower()

        # Ensure it's valid.
        if lower_mode not in self.MODES:
            mode_str = list_to_string(self.MODES, 'or')
            m = 'mode must be {} (case insensitive)'.format(mode_str)
            raise ValueError(m)

        # Assign.
        self._mode = lower_mode

        # Assign state, casting to upper case. Note that type checking
        # happens in _check_state.
        self.state = state

        self.log.debug('CapacitorSinglePhase {} '.format(self.name)
                       + 'initialized.')

    def __repr__(self):
        return self.name

    def _check_state(self, value):
        """Method required by base class, called before setting state."""
        if value is None:
            # If state is None, simply set.
            self._state = None
        elif isinstance(value, str):

            # Ensure it's valid.
            if value not in self.STATES:
                state_str = list_to_string(self.STATES, 'or')
                m = 'state must be {} (case sensitive).'.format(state_str)
                raise ValueError(m)
        else:
            # State is a bad type.
            raise TypeError('state must None or be a string.')

    ####################################################################
    # PROPERTY GETTERS AND SETTERS
    ####################################################################

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
    def mode(self):
        """Control mode."""
        return self._mode
