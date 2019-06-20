"""Module for all things capacitors."""
import logging
from pyvvo.utils import list_to_string
from .equipment import EquipmentSinglePhase
import pandas as pd
from numpy import bool_

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

# Hard-code some inputs to CapacitorSinglePhase. Note this does NOT
# include 'state.' These will be used in a way that relies on the fact
# that the CapacitorSinglePhase inputs and column names (query
# variables) from sparql.py match up.
CAP_INPUTS = ['name', 'mrid', 'phase', 'mode', 'controllable']


def initialize_capacitors(df):
    """
    Helper to initialize controllable capacitors given a DataFrame with
    capacitor information. The DataFrame should come from
    sparql.SPARQLManager.query_capacitors.

    ASSUMPTION: If the 'phase' column value is nan, the capacitor is
    a three phase cap.

    TODO: This function could certainly be re-ordered/refactored to
        be more efficient, but I'm not going to worry too much about
        it for now.
    """
    # If the 'phase' column is nan, replicate the row so we have one for
    # phases A, B, and C.
    nan_phase = df['phase'].isnull()
    df_mod = df.append([df[nan_phase]] * 2, ignore_index=True)

    # We've replicated the phases, but now need to update from nan to
    # A, B, and C. Note this may not be hyper efficient, but really,
    # how many capacitors are we going to be dealing with?
    grouped = df_mod[df_mod['phase'].isnull()].groupby(['mrid'])
    for idx in grouped.groups.values():
        # Sanity check length of group.
        if len(idx) != len(EquipmentSinglePhase.PHASES):
            raise ValueError('Something really bizarre is going on.')

        # Iterate over PHASES and update.
        for p_idx, p in enumerate(EquipmentSinglePhase.PHASES):
            df_mod.loc[idx[p_idx], 'phase'] = p

    # Filter to get controllable capacitors. Uncontrollable capacitors
    # do not have a "RegulatingControl" object, and are thus missing
    # the attributes in REG_CONTROL.
    rc = df_mod.loc[:, REG_CONTROL]
    c_mask = ~rc.isnull().all(axis=1)
    # Add a 'controllable' column.
    df_mod['controllable'] = c_mask

    # Cast any nan modes to None.
    df_mod.loc[~c_mask, 'mode'] = None

    # Initialize return.
    out = {}

    grouped = df_mod.groupby(['mrid'])

    for mrid, idx in grouped.groups.items():
        if len(idx) == 1:
            # Simply create object.
            out[mrid] = \
                CapacitorSinglePhase(**df_mod[CAP_INPUTS].loc[idx[0]].to_dict())
            # Move to next iteration of the loop.
            continue

        # If we're here, we need a dictionary.
        out[mrid] = \
            {df_mod.loc[i, 'phase']:
             CapacitorSinglePhase(**df_mod[CAP_INPUTS].loc[i].to_dict())
             for i in idx}

    return out


class CapacitorSinglePhase(EquipmentSinglePhase):
    """Single phase capacitor.
    Parameters will for the most part come straight out of the CIM.

    TODO: Accept all CIM parameters.
    """
    # The 'state' attribute corresponds to:
    STATE_CIM_PROPERTY = 'ShuntCompensator.sections'

    # Allowed states. 0 for open, 1 for closed.
    STATES = (0, 1)

    # Allowed control modes (case insensitive). Corresponds to CIM
    # RegulatingControlModeKind. May need to add "MANUAL" option in the
    # future for GridLAB-D. Note: In CIM these are camelCase, but
    # keeping them lower to make case insensitivity simple.
    # The None mode is used if a capacitor is NOT controllable.
    MODES = ('voltage', 'activepower', 'reactivepower', 'currentflow',
             'admittance', 'timescheduled', 'temperature', 'powerfactor')

    def __init__(self, name, mrid, phase, mode, controllable, state=None):
        """Initialize single phase capacitor.

        :param mrid: MRID of this capacitor. In the CIM,
            LinearShunCompensator.mRID.
        :param name: Name of this capacitor. In the CIM,
            LinearShunCompensator.name.
        :param phase: Phase of this capacitor.
        :param mode: Control mode, must be one of MODES. In the CIM,
            this is a RegulatingControl.RegulatingControlModeKind. Note
            that if None, the capacitor is assumed to be uncontrollable.
        :param controllable: Boolean. Whether or not this capacitor is
            controllable. If False, PyVVO will never attempt to send a
            command to this capacitor.
        :param state: Integer, must be in STATES attribute. 0 indicates
            open, 1 indicates closed.
        """
        # Get log.
        self.log = logging.getLogger(__name__)

        # Call super.
        super().__init__(mrid=mrid, name=name, phase=phase,
                         controllable=controllable)

        # Check and assign MRID.
        if not isinstance(mrid, str):
            raise TypeError('mrid must be a string.')

        # Check and assign mode.
        if (not isinstance(mode, str)) and (mode is not None):
            raise TypeError('mode must be a string or None.')

        # Cast to lower case.
        if mode is not None:
            lower_mode = mode.lower()

            # Ensure it's valid.
            if lower_mode not in self.MODES:
                mode_str = list_to_string(self.MODES, 'or')
                m = 'mode must be {} (case insensitive). It can also '\
                    'be None.'.format(mode_str)
                raise ValueError(m)

            # Assign.
            self._mode = lower_mode
        else:
            # The None case.
            self._mode = mode

        # Ensure 'controllable' and 'mode' do not conflict.
        if ((self.mode is None) and self.controllable) \
                or ((self.mode is not None) and (not self.controllable)):
            raise ValueError('The mode and controllable attributes of this '
                             'capacitor seem to conflict!')

        # Assign state. Note that type checking happens in _check_state.
        self.state = state

        self.log.debug('CapacitorSinglePhase {} '.format(self.name)
                       + 'initialized.')

    def __repr__(self):
        return self.name

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

    @property
    def controllable(self):
        return self._controllable
