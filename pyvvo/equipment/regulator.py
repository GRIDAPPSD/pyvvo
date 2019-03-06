"""Module for all things regulators."""
import logging

import pyvvo.utils as utils

import numpy as np

LOG = logging.getLogger(__name__)


def initialize_controllable_regulators(df):
    """Helper to initialize controllable regulators given a DataFrame
    with regulator information. The DataFrame should come from
    sparql.SPARQLManager.query_regulators.
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

    # Group by mrid and name. Both are of the parent regulator - the
    # tap_changer_mrid is unique to the regulator phase.
    groups = ltc_reg.groupby(['mrid', 'name'], sort=False)

    # Initialize return.
    out = []

    # Loop over the groups and initialize regulators.
    for label in groups.groups:
        items = groups.get_group(label)
        # Create a regulator for each item.
        reg_list = []
        for idx in items.index:
            reg_list.append(RegulatorSinglePhase(**items.loc[idx].to_dict()))

        # Create three-phase regulator and append to output.
        out.append(RegulatorThreePhase(reg_list))

    return out


def _tap_cim_to_gld(step, step_voltage_increment):
    """Convert step and step_voltage_increment in CIM terms to tap_pos
     in GridLAB-D terms.

     :param step: CIM setting of voltage regulator, which is a
        multiplier of nominal voltage, e.g. 1.0125
    :param step_voltage_increment: voltage step as percent multiplier
        of nominal voltage, e.g. 0.625.

    :returns tap_pos: tap position as GridLAB-D would denote it. E.g.
        10 or -2.

    NOTE: This method is private to this module, so don't go calling it,
    as inputs are not checked.
    """

    return round((step - 1) * 100 / step_voltage_increment)


def _tap_gld_to_cim(tap_pos, step_voltage_increment, num_digits=5):
    """Convert tap position as GridLAB-D denotes it to step as CIM
    denotes it.

    :param tap_pos: tap position as GridLAB-D would denote it.
    :param step_voltage_increment: voltage step increment in CIM.
    :param num_digits: number of digits after the decimal to round
        result to.

    :returns step: CIM setting of voltage regulator, which is a
        multiplier of nominal voltage, e.g. 1.0125.

    NOTE: This method is private to this module, so don't go calling it,
    as inputs are not checked.
    """
    return round(tap_pos * step_voltage_increment / 100 + 1, num_digits)


class RegulatorThreePhase:
    """
    Class that essentially acts as a container for RegulatorSinglePhase
    objects.
    """

    def __init__(self, regulator_list):
        """Take in 3 RegulatorSinglePhase objects.

        :param regulator_list: list of three RegulatorSinglePhase
            objects.
        """
        # Setup logging.
        self.log = logging.getLogger(__name__)

        # Check input type.
        if not isinstance(regulator_list, (list, tuple)):
            raise TypeError('regulator_list must be a list or tuple.')

        if len(regulator_list) != 3:
            raise ValueError('regulator_list must have length 3.')

        # Loop over the list.
        for regulator in regulator_list:
            if not isinstance(regulator, RegulatorSinglePhase):
                m = ('All items in regulator_list must be RegulatorSinglePhase'
                     + ' objects.')
                raise TypeError(m)

            # Set name attribute if necessary, ensure all
            # RegulatorSinglePhase objects refer to the same name.
            try:
                name_match = regulator.name == self.name
            except AttributeError:
                # Set the name.
                self._name = regulator.name
            else:
                # If the names don't match, raise exception.
                if not name_match:
                    m = 'RegulatorSinglePhase objects do not have matching '\
                        '"name" attributes.'
                    raise ValueError(m)

            # Set mrid attribute if necessary, ensure all
            # RegulatorSinglePhase objects refer to the same mrid.
            try:
                mrid_match = regulator.mrid == self.mrid
            except AttributeError:
                # Set mrid.
                self._mrid = regulator.mrid
            else:
                if not mrid_match:
                    # If the mrids don't match, raise exception.
                    m = 'RegulatorSinglePhase objects do not have matching '\
                        '"mrid" attributes.'
                    raise ValueError(m)

            # Check regulator phase, and set the attribute accordingly.
            setattr(self, '_' + regulator.phase.lower(), regulator)

    def __repr__(self):
        return '<RegulatorThreePhase. name: {}'.format(self.name)

    ####################################################################
    # Getter methods
    ####################################################################
    @property
    def name(self):
        return self._name

    @property
    def mrid(self):
        return self._mrid

    # noinspection PyUnresolvedReferences
    @property
    def a(self):
        return self._a

    # noinspection PyUnresolvedReferences
    @property
    def b(self):
        return self._b

    # noinspection PyUnresolvedReferences
    @property
    def c(self):
        return self._c


class RegulatorSinglePhase:
    """"""

    # Control modes taken from the CIM.
    CONTROL_MODES = ('voltage', 'activePower', 'reactivePower',
                     'currentFlow', 'admittance', 'timeScheduled',
                     'temperature', 'powerFactor')

    # Allowable phases.
    PHASES = ('A', 'B', 'C')

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
        :param step: NOTE: The platform needs updated to conform to a
            CIM revision. For now, we'll be treating this as a floating
            point tap ratio. TODO: Update when platform is updated.
            CIM definition (new): "Tap changer position. Starting step
            for a steady state solution. Non integer values are allowed
            to support continuous tap variables. The reasons for
            continuous value are to support study cases where no
            discrete tap changers has yet been designed, a solutions
            where a narrow voltage band force the tap step to oscillate
            or accommodate for a continuous solution as input. The
            attribute shall be equal or greater than lowStep and equal
            or less than highStep.
        """
        # Setup logging.
        self.log = logging.getLogger(__name__)

        ################################################################
        # CIM Properties
        ################################################################

        # Check inputs, set CIM properties.
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

        # TODO: Update when the platform is updated.
        if not isinstance(step, (float, np.floating)):
            raise TypeError('step must be a float.')

        self._step = step

        ################################################################
        # GridLAB-D properties
        ################################################################
        # Derive GridLAB-D properties from CIM properties.
        # http://gridlab-d.shoutwiki.com/wiki/Power_Flow_User_Guide.

        # In CIM, tap position is on interval [low_step, high_step] with
        # neutral_step being in the interval. In GridLAB-D, the
        # neutral_step is always 0, so the interval is
        # [-low_step, high_step]. Note, however, that when giving
        # GridLAB-D the low_step (parameter lower_taps), it's given as
        # a magnitude.
        self._raise_taps = high_step - neutral_step
        self._lower_taps = neutral_step - low_step

        # Set the default tap position.
        # TODO: This will need updated when the handling of 'step' is
        #   fixed in the platform.
        self._tap_pos = \
            _tap_cim_to_gld(step=self.step, step_voltage_increment=
                            self.step_voltage_increment)

    def __repr__(self):
        return '<RegulatorSinglePhase. name: {}, phase: {}>'.format(self.name,
                                                                    self.phase)

    ####################################################################
    # Getter methods
    ####################################################################

    # CIM attributes.
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
        return self._step

    # GridLAB-D attributes.
    @property
    def raise_taps(self):
        return self._raise_taps

    @property
    def lower_taps(self):
        return self._lower_taps

    @property
    def tap_pos(self):
        return self._tap_pos
