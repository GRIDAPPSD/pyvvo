"""Module for all things regulators."""
import logging
import pyvvo.utils as utils
from .equipment import EquipmentSinglePhase
import numpy as np

LOG = logging.getLogger(__name__)


def initialize_regulators(df):
    """Helper to initialize regulators given a DataFrame
    with regulator information.

    Note that this method relies on the fact that our SPARQL query for
    regulators is tightly coupled to the __init__ inputs for
    RegulatorSinglePhase objects.

    :param df: Pandas DataFrame object from
        sparql.SPARQLManager.query_regulators

    :returns out: dictionary, keyed by tap_changer_mrid, of
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


def _tap_cim_to_gld(step, neutral_step):
    """Convert step and step_voltage_increment in CIM terms to tap_pos
     in GridLAB-D terms.

    :param step: CIM tap position of voltage regulator. E.g., 1, 16, 20,
        or 32. CIM taps start at 0.
    :param neutral_step: CIM neutral tap position. This will likely and
        often be 16 on a 32 tap regulator.

    :returns tap_pos: tap position as GridLAB-D would denote it. E.g.
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

    :returns step: CIM tap position of the voltage regulator.

    NOTE: This method is private to this module, so don't go calling it,
    as inputs are not checked.
    """
    return tap_pos + neutral_step


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
        self.log = logging.getLogger(__name__)

        ################################################################
        # CIM Properties
        ################################################################
        # Start with calling the super.
        super().__init__(mrid=mrid, name=name, phase=phase)

        # Type checking and parameter setting:
        if not isinstance(controllable, bool):
            raise TypeError('controllable must be a boolean.')

        self.controllable = controllable

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

    def __repr__(self):
        return '<RegulatorSinglePhase. name: {}, phase: {}>'.format(self.name,
                                                                    self.phase)

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
            return _tap_cim_to_gld(step=self.step_old,
                                   neutral_step=self.neutral_step)

    @property
    def raise_taps(self):
        return self._raise_taps

    @property
    def lower_taps(self):
        return self._lower_taps
