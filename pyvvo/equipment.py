"""Module for classes which represent equipment. E.g. regulators,
capacitors, or switches.
"""
# Standard library:
from abc import ABC, abstractmethod
import logging
from collections import deque
from threading import Lock

# Third party:
import pandas as pd
import numpy as np

# PyVVO:
from pyvvo import utils

########################################################################
# CONSTANTS
########################################################################

# Setup log.
LOG = logging.getLogger(__name__)

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

########################################################################

########################################################################
# HELPER FUNCTIONS (SEE ALSO: "INITIALIZATION HELPER FUNCTIONS")
########################################################################


def _tap_cim_to_gld(step, neutral_step):
    """Convert step and step_voltage_increment in CIM terms to tap_pos
     in GridLAB-D terms.

    :param step: CIM tap position of voltage regulator. E.g., 1, 16, 20,
        or 32. CIM taps start at 0.
    :param neutral_step: CIM neutral tap position. This will likely and
        often be 16 on a 32 tap regulator.

    :returns: tap_pos: tap position as GridLAB-D would denote it. E.g.
        10 or -2.

    NOTE: This method is private to this module, so don't go calling it,
    as inputs are not checked.
    """

    return step - neutral_step


def _tap_gld_to_cim(tap_pos, neutral_step):
    """Convert tap position as GridLAB-D denotes it to step as CIM
    denotes it.

    :param tap_pos: tap position as GridLAB-D would denote it.
    :param neutral_step: CIM neutral tap position. This will likely and
        often be 16 on a 32 tap regulator.

    :returns: step: CIM tap position of the voltage regulator.

    NOTE: This method is private to this module, so don't go calling it,
    as inputs are not checked.
    """
    return tap_pos + neutral_step

########################################################################

########################################################################
# CLASSES
########################################################################


class EquipmentSinglePhase(ABC):
    """Generic 'equipment' class, for e.g. capacitors and regulators."""

    PHASES = ('A', 'B', 'C')

    def __init__(self, mrid, name, phase, controllable):
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

        # Check and assign controllable.
        if not isinstance(controllable, (bool, np.bool_)):
            raise TypeError('controllable must be a boolean.')

        self._controllable = controllable

    def __repr__(self):
        return "{}, {}, Phase {}".format(self.__class__.__name__,
                                         self.name, self.phase)

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

    @property
    def controllable(self):
        return self._controllable

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
        self.log = logging.getLogger(self.__class__.__name__)

        # Call super.
        super().__init__(mrid=mrid, name=name, phase=phase,
                         controllable=controllable)

        # Check and assign mode.
        if (not isinstance(mode, str)) and (mode is not None):
            raise TypeError('mode must be a string or None.')

        # Cast to lower case.
        if mode is not None:
            lower_mode = mode.lower()

            # Ensure it's valid.
            if lower_mode not in self.MODES:
                mode_str = utils.list_to_string(self.MODES, 'or')
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
    def mode(self):
        """Control mode."""
        return self._mode


class RegulatorSinglePhase(EquipmentSinglePhase):
    """"""
    # 'state' corresponds to:
    STATE_CIM_PROPERTY = 'TapChanger.step'

    # Control modes taken from the CIM.
    CONTROL_MODES = ('voltage', 'activePower', 'reactivePower',
                     'currentFlow', 'admittance', 'timeScheduled',
                     'temperature', 'powerFactor')

    def __init__(self, mrid, name, phase, controllable, tap_changer_mrid,
                 step_voltage_increment, control_mode, enabled, high_step,
                 low_step, neutral_step, step):
        """Take in parameters from the CIM. See Figure 7 here:
        https://gridappsd.readthedocs.io/en/latest/developer_resources/index.html#cim-documentation

        NOTE: CIM definitions were pulled from the "CIM_GridAPPS-D_RC3"
            file as viewed in Enterprise Architect.

        NOTE: See sparql.SPARQLManager.REGULATOR_QUERY to see exactly
            where the input parameters should come from.

        :param mrid: MRID of this regulator. In the CIM,
            TransformerTank.PowerTransformer.mRID.
        :param name: Name of this regulator. In the CIM,
            TransformerTank.PowerTransformer.name.
        :param phase: Phase of the TransformerTankEnd associated with
            this regulator. So, phase of the regulator.
        :param controllable: Boolean. Indicates if this regulator can be
            controlled/commanded or not. The False case would be a
            regulator that must physically be changed by a human (or a
            robot? But then is a motorized control a robot? I digress.).
        :param tap_changer_mrid: MRID of the TapChanger.
        :param step_voltage_increment: Definition from CIM: "Tap step
            increment, in percent of neutral voltage, per step
            position."
        :param control_mode: "mode" parameter from CIM RegulatingControl
            object. For GridAPPS-D, could be "voltage," "manual," and
            maybe others.
        :param enabled: True/False, whether the regulation is
            enabled. In the CIM, RegulatingControl.enabled.
        :param high_step: Definition from CIM: "Highest possible tap
            step position, advance from neutral. The attribute shall be
            greater than lowStep."
        :param low_step: Definition from CIM: "Lowest possible tap step
            position, retard from neutral."
        :param neutral_step: Definition from CIM: "The neutral tap step
            position for this winding. The attribute shall be equal or
            greater than lowStep and equal or less than highStep."
        :param step: Integer only, no float.
            CIM definition (new): "Tap changer position. Starting step
            for a steady state solution. Non integer values are allowed
            to support continuous tap variables. The reasons for
            continuous value are to support study cases where no
            discrete tap changers has yet been designed, a solutions
            where a narrow voltage band force the tap step to oscillate
            or accommodate for a continuous solution as input. The
            attribute shall be equal or greater than lowStep and equal
            or less than highStep."
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        ################################################################
        # CIM Properties
        ################################################################
        # Start with calling the super.
        super().__init__(mrid=mrid, name=name, phase=phase,
                         controllable=controllable)

        if not isinstance(tap_changer_mrid, str):
            raise TypeError('tap_changer_mrid must be a string.')

        self._tap_changer_mrid = tap_changer_mrid

        if not isinstance(step_voltage_increment, (float, np.floating)):
            raise TypeError("step_voltage_increment must be a float!")

        self._step_voltage_increment = step_voltage_increment

        if not isinstance(control_mode, str):
            raise TypeError('control_mode must be a string.')
        elif control_mode not in self.CONTROL_MODES:
            m = 'mode must be one of ' \
                + utils.list_to_string(self.CONTROL_MODES, ',')
            raise ValueError(m)

        self._control_mode = control_mode

        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError('enabled must be True or False.')

        self._enabled = enabled

        if not isinstance(high_step, (int, np.integer)):
            raise TypeError('high_step must be an integer.')

        self._high_step = high_step

        if not isinstance(low_step, (int, np.integer)):
            raise TypeError('low_step must be an integer.')

        self._low_step = low_step

        if not isinstance(neutral_step, (int, np.integer)):
            raise TypeError('neutral_step must be an integer.')

        self._neutral_step = neutral_step

        # Ensure our steps agree with each other.
        if not ((low_step <= neutral_step) and (neutral_step <= high_step)):
            raise ValueError('The following is not True: '
                             'low_step <= neutral_step <= high_step')

        # The test for step being an integer is in its setter method.
        # NOTE: setting step here ALSO sets self._tap_pos. tap_pos is
        # for GridLAB-D, while 'step' is CIM.
        self.step = step

        ################################################################
        # GridLAB-D properties
        ################################################################
        # Derive GridLAB-D properties from CIM properties.
        # http://gridlab-d.shoutwiki.com/wiki/Power_Flow_User_Guide.
        #
        # NOTE: tap_pos is a GridLAB-D parameter.
        #
        # In CIM, tap position is on interval [low_step, high_step] with
        # neutral_step being in the interval. In GridLAB-D, the
        # neutral_step is always 0, so the interval is
        # [-low_step, high_step]. Note, however, that when giving
        # GridLAB-D the low_step (parameter lower_taps), it's given as
        # a magnitude.
        self._raise_taps = high_step - neutral_step
        self._lower_taps = neutral_step - low_step

    def _check_state(self, value):
        """Ensure 'step' is valid before setting it."""
        if not isinstance(value, (int, np.integer)):
            raise TypeError('step must be an integer.')

        # Ensure we aren't getting an out of range value.
        if value < self.low_step or value > self.high_step:
            raise ValueError(
                'step must be between low_step ({}) and '
                'high_step ({}) (inclusive)'.format(self.low_step,
                                                    self.high_step))

    def _check_tap_pos(self, value):
        """Ensure tap_pos is valid before setting it."""
        if not isinstance(value, (int, np.integer)):
            raise TypeError('tap_pos must be an integer.')

        # Ensure value is valid.
        if value < -self.lower_taps or value > self.raise_taps:
            raise ValueError(
                'tap_pos must be between lower_taps ({}) and '
                'raise_taps ({}) (inclusive)'.format(-self.lower_taps,
                                                     self.raise_taps))

    ####################################################################
    # Getter methods
    ####################################################################

    # CIM attributes.
    @property
    def tap_changer_mrid(self):
        return self._tap_changer_mrid

    @property
    def step_voltage_increment(self):
        return self._step_voltage_increment

    @property
    def control_mode(self):
        return self._control_mode

    @property
    def enabled(self):
        return self._enabled

    @property
    def high_step(self):
        return self._high_step

    @property
    def low_step(self):
        return self._low_step

    @property
    def neutral_step(self):
        return self._neutral_step

    @property
    def step(self):
        """Step is mapped to _state."""
        return self.state

    @step.setter
    def step(self, value):
        """Set the step (CIM) and also the tap_pos (GLD)"""
        # This is mapped to the state property.
        # Note the setter for state is defined in the base class.
        self.state = value

    @property
    def step_old(self):
        """Step is mapped to the base class's state."""
        return self.state_old

    # GridLAB-D attributes:
    @property
    def tap_pos(self):
        """Return the GridLAB-D style tap position."""
        # NOTE: while it might be slightly more efficient to store this
        # as a property rather than recompute each time, this vastly
        # simplifies the process of using the base class to handle state
        # updates.
        # noinspection PyTypeChecker
        return _tap_cim_to_gld(step=self.step, neutral_step=self.neutral_step)

    @tap_pos.setter
    def tap_pos(self, value):
        """Set the tap_pos (GLD) and also the step (CIM)"""
        self._check_tap_pos(value)
        # Update the step. Note that the tap_pos getter uses 'step,' so
        # we're good.
        self.step = _tap_gld_to_cim(tap_pos=value,
                                    neutral_step=self.neutral_step)

    @property
    def tap_pos_old(self):
        """See comments for tap_pos"""
        if self.step_old is None:
            return None
        else:
            # noinspection PyTypeChecker
            return _tap_cim_to_gld(step=self.step_old,
                                   neutral_step=self.neutral_step)

    @property
    def raise_taps(self):
        return self._raise_taps

    @property
    def lower_taps(self):
        return self._lower_taps


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
            as returned by regulator.initialize_regulators
            or capacitor.initialize_capacitors.
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
        self.log = logging.getLogger(self.__class__.__name__)

        # Simple type checking.
        if not isinstance(eq_dict, dict):
            raise TypeError('eq_dict must be a dictionary.')

        if not isinstance(eq_meas, pd.DataFrame):
            raise TypeError('eq_meas must be a Pandas DataFrame.')

        # Simply assign.
        self.eq_dict = eq_dict
        self.eq_meas = eq_meas

        # Use a lock to avoid collisions due to threading.
        self._lock = Lock()

        # Track the simulation timestamp of the last message that came
        # in.
        self.last_time = None

        # Create a map from measurement MRID to EquipmentSinglePhase.
        self.meas_eq_map = {}

        # Loop over the equipment dictionary.
        for eq_mrid, eq_or_dict in self.eq_dict.items():
            # Extract the relevant measurements.
            meas = self.eq_meas[self.eq_meas[eq_mrid_col] == eq_mrid]

            # Ensure the length isn't 0.
            if meas.shape[0] < 1:
                raise ValueError('The eq_meas input is missing equipment '
                                 'mrid {}'.format(eq_mrid))

            if isinstance(eq_or_dict, dict):
                # Ensure the length of meas matches the length of
                # eq_or_dict
                if len(eq_or_dict) != meas.shape[0]:
                    raise ValueError('The number of measurements for '
                                     'equipment with mrid {} does not '
                                     'match the number of EquipmentSinglePhase'
                                     ' objects for this piece of '
                                     'equipment.'.format(eq_mrid))

                # Loop over the phases and map measurements to objects.
                for phase, eq in eq_or_dict.items():
                    meas_phase = meas[meas['phase'] == phase]

                    # Ensure there's just one measurement.
                    if meas_phase.shape[0] != 1:
                        raise ValueError('There is no measurement for phase '
                                         '{} for equipment with mrid '
                                         '{}'.format(phase, eq_mrid))

                    # Map it.
                    self.meas_eq_map[meas_phase[meas_mrid_col].values[0]] = eq

            elif isinstance(eq_or_dict, EquipmentSinglePhase):
                # Simple 1:1 mapping.
                if meas.shape[0] != 1:
                    raise ValueError('Received {} measurements for equipment '
                                     'with mrid {}, but expected 1.'.format(
                                      meas.shape[0], eq_mrid))

                # Map it.
                self.meas_eq_map[meas[meas_mrid_col].values[0]] = eq_or_dict

    @utils.wait_for_lock
    def lookup_eq_by_mrid_and_phase(self, mrid, phase=None):
        """Helper function to look up equipment in the eq_dict.

        :param mrid: MRID of the equipment to find.
        :param phase: Optional. Phase of the equipment to find.
        """
        # Start by just grabbing the equipment.
        eq = self.eq_dict[mrid]

        # eq could be EquipmentSinglePhase or a dictionary.
        if isinstance(eq, EquipmentSinglePhase):
            # If given a phase, ensure it matches.
            if (phase is not None) and (eq.phase != phase):
                raise ValueError('The equipment with mrid {} has phase '
                                 '{}, not {}'.format(mrid, eq.phase, phase))

            # We're done here.
            return eq

        elif isinstance(eq, dict):
            if phase is None:
                raise ValueError('The equipment with mrid {} has multiple '
                                 'EquipmentSinglePhase objects, but no phase '
                                 'was specified!')

            # Grab and return the phase entry.
            return eq[phase]

    @utils.wait_for_lock
    def update_state(self, msg, sim_dt):
        """Given a message from a gridappsd_platform SimOutRouter,
        update equipment state.

        :param msg: list passed to this method via a SimOutRouter's
            "_on_message" method.
        :param sim_dt: datetime.datetime object passed to this method
            via a SimOutRouter's "_on_message" method.
        """

        # Type checking:
        if not isinstance(msg, list):
            raise TypeError('msg must be a list!')

        # Iterate over the message and update equipment.
        for m in msg:
            # Grab mrid and value.
            meas_mrid = m['measurement_mrid']
            value = m['value']

            # Update!
            try:
                eq = self.meas_eq_map[meas_mrid]
            except KeyError:
                self.log.warning(
                    'Measurement MRID {} not present in the map!'.format(
                        meas_mrid))
            else:
                if eq.state != value:
                    eq.state = value
                    self.log.debug('Equipment {} state updated to: {}'.format(
                        str(self.meas_eq_map[meas_mrid]), value
                    ))

    @utils.wait_for_lock
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

        :returns: dictionary with keys corresponding to the send_command
            method of a gridappsd_platform.PlatformManager object.

        NOTE: eq_dict_forward and self.eq_dict should be identical
        except for object states.
        """
        # Nested helper function.
        def update(mgr, out, eq_for, eq_for_mrid):
            # If this equipment isn't controllable, don't build a
            # command.
            if not eq_for.controllable:
                return

            # Lookup the equipment object in self's eq_dict
            # corresponding to this piece of equipment.
            try:
                eq_rev = mgr.eq_dict[eq_for_mrid]
            except KeyError:
                m = 'The given eq_dict_forward is not matching up with '\
                    'self.eq_dict! Ensure these eq_dicts came from the '\
                    'same model, etc.'

                raise ValueError(m) from None

            # Only build commands if the states are different.
            if eq_for.state != eq_rev.state:
                # We need to convert from numpy data types to regular
                # Python. Helpful:
                # https://stackoverflow.com/a/11389998/11052174
                state_for = eq_for.state
                state_rev = eq_rev.state
                try:
                    state_for = state_for.item()
                except AttributeError:
                    pass

                try:
                    state_rev = state_rev.item()
                except AttributeError:
                    pass

                # Append values to output.
                out['object_ids'].append(eq_mrid)
                out['attributes'].append(eq_for.STATE_CIM_PROPERTY)
                out['forward_values'].append(state_for)
                out['reverse_values'].append(state_rev)

        # Initialize output.
        output = {"object_ids": [], "attributes": [], "forward_values": [],
                  "reverse_values": []}

        # Loop over equipment.
        for eq_mrid, eq_forward in eq_dict_forward.items():
            if isinstance(eq_forward, EquipmentSinglePhase):
                # Call helper.
                update(mgr=self, out=output, eq_for=eq_forward,
                       eq_for_mrid=eq_mrid)
            elif isinstance(eq_forward, dict):
                # Loop over the phases.
                for eq_single in eq_forward.values():
                    # Call helper.
                    update(mgr=self, out=output, eq_for=eq_single,
                           eq_for_mrid=eq_mrid)

        return output

########################################################################

########################################################################
# INITIALIZATION HELPER FUNCTIONS
########################################################################


def initialize_capacitors(df):
    """
    Helper to initialize capacitors given a DataFrame with
    capacitor information. The DataFrame should come from
    sparql.SPARQLManager.query_capacitors.

    ASSUMPTION: If the 'phase' column value is nan, the capacitor is
    a three phase cap.
    """
    # Filter to get controllable capacitors. Uncontrollable capacitors
    # do not have a "RegulatingControl" object, and are thus missing
    # the attributes in REG_CONTROL.
    try:
        rc = df.loc[:, REG_CONTROL]
    except KeyError as e:
        if e.args[0].startswith('None of [Index(') and \
                e.args[0].endswith('are in the [columns]'):
            # No capacitors are controllable.
            nan_data = np.full((df.shape[0], len(REG_CONTROL)), np.nan)
            rc = pd.DataFrame(nan_data, columns=REG_CONTROL)

            # Create the 'mode' column, and ensure it's dtype is object.
            # This is necessary to avoid passing NaN values to the
            # capacitor constructor.
            df['mode'] = pd.Series(data=None, index=df.index, dtype='object')
        else:
            raise e

    c_mask = ~rc.isnull().all(axis=1)
    # Add a 'controllable' column.
    df['controllable'] = c_mask

    # Cast any nan modes to None.
    df.loc[~c_mask, 'mode'] = None

    # Initialize return.
    out = {}

    # Loop over the DataFrame, considering only the columns we care
    # about.
    for row in df[CAP_INPUTS].itertuples(index=False):
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
                out[row.mrid][p] = CapacitorSinglePhase(**row_dict)
        else:
            # Simply create a CapacitorSinglePhase.
            out[row.mrid] = CapacitorSinglePhase(**row_dict)

    return out


def initialize_regulators(df):
    """Helper to initialize regulators given a DataFrame
    with regulator information.

    Note that this method relies on the fact that our SPARQL query for
    regulators is tightly coupled to the __init__ inputs for
    RegulatorSinglePhase objects.

    :param df: Pandas DataFrame object from
        sparql.SPARQLManager.query_regulators

    :returns: out: dictionary, keyed by tap_changer_mrid, of
        RegulatorSinglePhase objects. Note that all regulators are
        included, even if they aren't controllable.
    """
    # Rename the ltc_flag to 'controllable'. Note this isn't technically
    # correct by the CIM definition - we should probably be looking
    # instead for the presence of a TapChangerControl object. However,
    # in PyVVO's case, a regulator is only controllable if it can tap
    # under load.
    #
    # From the CIM description for TapChanger.ltcFlag: "Specifies
    # whether or not a TapChanger has load tap changing capabilities."
    df2 = df.rename({'ltc_flag': 'controllable'}, axis=1, copy=False)

    # Use dictionary comprehension to create return.
    out = {r['tap_changer_mrid']: RegulatorSinglePhase(**r)
           for r in df2.to_dict('records')}

    return out


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


def loop_helper(eq_dict, func, *args, **kwargs):
    """Loop over an equipment dictionary returned from one of the
    initialize_* functions and apply a function to each
    EquipmentSinglePhase object.

    I constantly find myself re-writing these annoying loops due to
    the poor design decision to allow the initialize_* functions to
    have either EquipmentSinglePhase objects or dictionaries as the
    values for the top level keys of their respective returns. So, this
    method hopefully mitigates that.

    :param eq_dict: Dictionary from one of this modules initialize_*
        functions (e.g. initialize_regulators).
    :param func: Function which takes two positional inputs: a single EquipmentSinglePhase
        object as input.
    """
    for eq_or_dict in eq_dict.values():
        if isinstance(eq_or_dict, dict):
            # Loop over the phases.
            for eq in eq_or_dict.values():
                func(eq, *args, **kwargs)
        elif isinstance(eq_or_dict, EquipmentSinglePhase):
            func(eq_or_dict, *args, **kwargs)
        else:
            raise TypeError('Value was not a dict or EquipmentSinglePhase.')

########################################################################
