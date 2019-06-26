"""Module for managing switch-like objects. In the CIM, these could be
Breakers, Reclosers, or LoadBreakSwitches.

TODO: This may grow to include DistFuse, DistBreaker, DistRecloser,
    DistLoadBreakSwitch, and DistSectionaliser
"""
# Standard library:
import logging

# Third party:
import pandas as pd

# PyVVO:
from .equipment import EquipmentSinglePhase


def initialize_switches(df):
    """
    Helper to initialize switches given a DataFrame with switch
    information.

    :param df: Pandas DataFrame from sparql.SPARQLManager.query_switches
    :return: out: Dictionary, keyed by MRID, of SwitchSinglePhase
        objects. The values in the dictionary will either be
        SwitchSinglePhase or dictionaries of SwitchSinglePhase objects,
        keyed by phase.

    ASSUMPTION: If the 'phase' column value is nan, the switch is a
    three phase switch.

    TODO: Currently, I'm hard-coding all switches to not be
        controllable. At present, pyVVO won't be commanding switches,
        only listening to their statuses, so this is no big deal.
    """
    # Hard-code controllable to be False.
    df['controllable'] = False

    # Initialize output.
    out = {}

    # Loop over the DataFrame, considering only the columns we care
    # about.
    for row in df.itertuples(index=False):
        # Get the row as a dictionary. Note these methods are not
        # private, just named with an underscore to avoid name
        # conflicts.
        # https://docs.python.org/3/library/collections.html#collections.namedtuple
        # noinspection PyProtectedMember
        row_dict = row._asdict()

        # If the phase is NaN, we'll be creating three single phase
        # objects, one for each phase.
        if pd.isna(row.phase):
            out[row.mrid] = {}
            # Loop over allowable phases and create an object for
            # each one.
            for p in EquipmentSinglePhase.PHASES:
                # Overwrite the phase.
                row_dict['phase'] = p

                # Create an entry.
                out[row.mrid][p] = SwitchSinglePhase(**row_dict)
        else:
            # Simply create a CapacitorSinglePhase.
            out[row.mrid] = SwitchSinglePhase(**row_dict)

    return out


class SwitchSinglePhase(EquipmentSinglePhase):
    """Single phase switch. For now, PyVVO is simply listening to switch
    statuses, so this class is very simple and not much different from
    EquipmentSinglePhase."""

    # Allowed states. 0 for open, 1 for closed.
    STATES = (0, 1)

    STATE_CIM_PROPERTY = 'Switch.open'

    def __init__(self, name, mrid, phase, controllable):
        """See docstring for equipment.EquipmentSinglePhase for inputs.
        """
        # Get log.
        self.log = logging.getLogger(self.__class__.__name__)

        # Call parent constructor.
        super().__init__(name=name, mrid=mrid, phase=phase,
                         controllable=controllable)

    def _check_state(self, value):
        """Method required by base class, called before setting state."""
        if isinstance(value, int):

            # Ensure it's valid.
            if value not in self.STATES:
                m = 'state must be one of {}.'.format(self.STATES)
                raise ValueError(m)
        elif value is not None:
            # State is a bad type.
            raise TypeError('state must None or one of {}'.format(self.STATES))
