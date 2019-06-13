"""Module for all things regulators."""
import logging
from collections import deque
import pyvvo.utils as utils
from .equipment import EquipmentSinglePhase

from pandas import DataFrame
import numpy as np

LOG = logging.getLogger(__name__)


def initialize_controllable_regulators(df):
    """Helper to initialize controllable regulators given a DataFrame
    with regulator information.

    Note that this method relies on the fact that our SPARQL query for
    regulators is tightly coupled to the __init__ inputs for
    RegulatorSinglePhase objects.

    :param df: Pandas DataFrame object from
        sparql.SPARQLManager.query_regulators

    :returns out: dictionary, keyed by tap_changer_mrid, of
        RegulatorSinglePhase objects. Note that only controllable
        regulators are returned.
    """
    # Filter by ltc_flag, and then drop the ltc_flag column. From the
    # CIM description for TapChanger.ltcFlag: "Specifies whether or not
    # a TapChanger has load tap changing capabilities."
    # We don't care about regulators that can't tap under load.
    ltc_reg = df[df['ltc_flag']].drop('ltc_flag', axis=1)

    # If we're "throwing away" some regulators, mention it.
    if ltc_reg.shape[0] != df.shape[0]:
        LOG.info('{} regulator phases were discarded because their CIM '
                 'ltcFlag was false.'.format(df.shape[0] - ltc_reg.shape[0]))

    # Use dictionary comprehension to create return.
    out = {r['tap_changer_mrid']: RegulatorSinglePhase(**r)
           for r in ltc_reg.to_dict('records')}

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


class RegulatorManager:
    """Class to keep RegulatorSinglePhase objects up to date as a
    simulation proceeds.

    This is meant to be used in conjunction with a SimOutRouter from
    gridappsd_platform.py.

    TODO: initialize_controllable_regulators filters our regulators,
        but really we want ALL of them here.
    """
    def __init__(self, reg_dict, reg_meas):
        """Initialize.

        :param reg_dict: Dictionary of RegulatorSinglePhase objects
            as returned by initialize_controllable_regulators.
        :param reg_meas: Pandas DataFrame as returned by
            sparql.SPARQLManager's query_rtc_measurements method.

        The reg_dict and reg_meas must have the same number of elements.
        """
        # Logging.
        self.log = logging.getLogger(__name__)

        # Simple type checking.
        if not isinstance(reg_dict, dict):
            raise TypeError('reg_dict must be a dictionary.')

        if not isinstance(reg_meas, DataFrame):
            raise TypeError('reg_measurements must be a Pandas DataFrame.')

        if len(reg_dict) != reg_meas.shape[0]:
            raise ValueError('The number of measurements and number of '
                             'regulators do not match!')

        # Simply assign.
        self.reg_dict = reg_dict
        self.reg_meas = reg_meas

        # Create a map from measurement MRID to RegulatorSinglePhase.
        self.meas_reg_map = {}

        for row in self.reg_meas.itertuples():
            self.meas_reg_map[row.pos_meas_mrid] = \
                self.reg_dict[row.tap_changer_mrid]

        if len(self.meas_reg_map) != len(self.reg_dict):
            raise ValueError('The reg_dict and reg_meas inputs are '
                             'inconsistent as the resulting map has '
                             'a different shape.')

    def update_regs(self, msg):
        """Given a message from a gridappsd_platform SimOutRouter,
        update regulator positions.

        :param msg: list passed to this method via a SimOutRouter's
            "_on_message" method.
        """
        # # Dump message for testing:
        # import json
        # with open('reg_meas_message.json', 'w') as f:
        #     json.dump(msg, f)

        # Type checking:
        if not isinstance(msg, list):
            raise TypeError('msg must be a list!')

        # Iterate over the message and update regulators.
        for m in msg:
            # Type checking:
            if not isinstance(m, dict):
                raise TypeError('Each entry in msg must be a dict!')
            # Update!
            # TODO: For now this is tap_pos (GLD). Update to use step
            #   (CIM) when issue 754 is resolved:
            #   https://github.com/GRIDAPPSD/GOSS-GridAPPS-D/issues/754

            # Grab mrid and value.
            meas_mrid = m['measurement_mrid']
            value = m['value']

            try:
                self.meas_reg_map[meas_mrid].tap_pos = value
            except KeyError:
                self.log.warning(
                    'Measurement MRID {} not present in the map!'.format(
                        meas_mrid))
            else:
                self.log.debug('Regulator {} tap_pos updated to: {}'.format(
                    str(self.meas_reg_map[meas_mrid]), value
                ))

    def build_regulator_commands(self, reg_dict_forward):
        """Function to build command for regulator tap positions. The
        command itself should be sent with a
        gridappsd_platform.PlatformManager object's send_command method.

        This object's reg_dict is considered to have the current/old
        regulator positions.

        NOTE: It isn't 100% clear if this is the best home for this
        method, but hey, that's okay.

        :param reg_dict_forward: dictionary of
            regulator.RegulatorSinglePhase objects as would come as
            output from regulator.initialize_controllable_regulators.
            The tap positions for these objects will be what the
            regulators in the simulation will be commanded to.

        :returns dictionary with keys corresponding to the send_command
            method of a gridappsd_platform.PlatformManager object.

        NOTE: reg_dict_forward and reg_dict_reverse should have
            identical contents except for the tap positions.
        """
        # Initialize lists of parameters.
        reg_ids = []
        reg_attr = []
        reg_forward_list = []
        reg_reverse_list = []

        # Loop over regulators.
        for reg_mrid, reg_forward in reg_dict_forward.items():
            # Loop over the phases in the regulator.

            try:
                reg_reverse = self.reg_dict[reg_mrid]
            except KeyError:
                m = 'The given reg_dict_forward is not matching up with '\
                    'self.reg_dict! Ensure these reg_dicts came from the '\
                    'same model, etc.'

                raise ValueError(m) from None

            # Add the tap change mrid.
            reg_ids.append(reg_mrid)
            # Add the attribute.
            reg_attr.append('TapChanger.step')
            # Add the forward position.
            reg_forward_list.append(reg_forward.step)
            # Grab the reverse position.
            reg_reverse_list.append(reg_reverse.step)

        return {"object_ids": reg_ids, "attributes": reg_attr,
                "forward_values": reg_forward_list,
                "reverse_values": reg_reverse_list}


class RegulatorSinglePhase(EquipmentSinglePhase):
    """"""

    # Control modes taken from the CIM.
    CONTROL_MODES = ('voltage', 'activePower', 'reactivePower',
                     'currentFlow', 'admittance', 'timeScheduled',
                     'temperature', 'powerFactor')

    def __init__(self, mrid, name, phase, tap_changer_mrid,
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
