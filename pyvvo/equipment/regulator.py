"""Module for all things regulators."""
import logging

import pyvvo.utils as utils

from pandas import DataFrame
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
    out = {}

    # Loop over the groups and initialize regulators.
    for label in groups.groups:
        items = groups.get_group(label)
        # Create a regulator for each item.
        reg_list = []
        for idx in items.index:
            reg_list.append(RegulatorSinglePhase(**items.loc[idx].to_dict()))

        # Create three-phase regulator and add to output.
        reg_three_phase = RegulatorMultiPhase(reg_list)
        out[reg_three_phase.name] = reg_three_phase

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
    """Class to keep RegulatorMultiPhase objects up to date as a
    simulation proceeds.

    This is meant to be used in conjunction with a SimOutRouter from
    gridappsd_platform.py.

    TODO: initialize_controllable_regulators filters our regulators,
        but really we want ALL of them here.
    """
    def __init__(self, reg_dict, reg_measurements):
        """Initialize.

        :param reg_dict: Dictionary of RegulatorMultiPhase objects
            as returned by initialize_controllable_regulators.
        :param reg_measurements: Pandas DataFrame as returned by
            sparql.SPARQLManager's query_rtc_measurements method.
        """
        # Logging.
        self.log = logging.getLogger(__name__)

        # Simple type checking.
        if not isinstance(reg_dict, dict):
            raise TypeError('reg_dict must be a dictionary.')

        if not isinstance(reg_measurements, DataFrame):
            raise TypeError('reg_measurements must be a Pandas DataFrame.')

        # Create a mapping of measurement MRIDs to RegulatorSinglePhase
        #   objects.
        self.reg_meas_map = {}

        for reg_name, reg_obj in reg_dict.items():
            # Get entries from the DataFrame which match this regulator.
            view = reg_measurements[reg_measurements['eqid'] == reg_obj.mrid]

            # If this view is empty, warn.
            if view.shape[0] == 0:
                self.log.warning("There are no measurements for regulator "
                                 "'{}'".format(reg_name))
                # Move to the next loop iteration.
                continue

            # Loop over the phases of the regulator to match up
            # measurements with phases.
            for phase in reg_obj.PHASES:
                # Grab the RegulatorSinglePhase for this phase.
                reg_single_phase = getattr(reg_obj, phase)

                # If this phase isn't present, we'll receive None.
                if reg_single_phase is None:
                    continue

                # Extract measurements which correspond to this phase.
                single_view = view[view['phases'] == phase.upper()]

                # For now, only allow one measurement.
                if single_view.shape[0] > 1:
                    raise ValueError('The RegulatorManager currently only '
                                     'supports one measurement per phase.')
                elif single_view.shape[0] == 0:
                    # Warn if we get zero.
                    self.log.warning("Phase {} of regulator '{}' does not have"
                                     " any measurements!".format(phase,
                                                                 reg_name))
                    continue

                # Map it!
                self.reg_meas_map[single_view['id'].values[0]] = \
                    reg_single_phase

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
            mrid = m['measurement_mrid']
            value = m['value']

            try:
                self.reg_meas_map[mrid].tap_pos = value
            except KeyError:
                self.log.warning('MRID {} not present in the map!'.format(
                    mrid))
            else:
                self.log.debug('Regulator {} tap_pos updated to: {}'.format(
                    str(self.reg_meas_map[mrid]), value
                ))


class RegulatorMultiPhase:
    """
    Class that essentially acts as a container for RegulatorSinglePhase
    objects.
    """

    # Allowable phases.
    PHASES = ('A', 'B', 'C')

    def __init__(self, regulator_list):
        """Take in list of RegulatorSinglePhase objects.

        :param regulator_list: list of three RegulatorSinglePhase
            objects.
        """
        # Setup logging.
        self.log = logging.getLogger(__name__)

        # Check input type.
        if not isinstance(regulator_list, (list, tuple)):
            raise TypeError('regulator_list must be a list or tuple.')

        if (len(regulator_list) < 1) or (len(regulator_list) > 3):
            m = 'regulator_list must meet 1 <= len(regulator_list) <= 3.'
            raise ValueError(m)

        # Initialize phases to None. We'll use case-insensitive attributes.
        self._a = None
        self._b = None
        self._c = None

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

            # Get this phase attribute for self.
            attr_str = '_' + regulator.phase.lower()
            attr = getattr(self, attr_str)
            # Ensure this attribute has not yet been set to anything
            # other than None.
            if attr is not None:
                raise ValueError('Multiple regulators for phase {} were '
                                 'given!'.format(regulator.phase.lower()))

            setattr(self, attr_str, regulator)

    def __repr__(self):
        return '<RegulatorMultiPhase. name: {}'.format(self.name)

    ####################################################################
    # Getter methods
    ####################################################################
    @property
    def name(self):
        return self._name

    @property
    def mrid(self):
        return self._mrid

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    # noinspection PyPep8Naming
    @property
    def A(self):
        return self._a

    # noinspection PyPep8Naming
    @property
    def B(self):
        return self._b

    # noinspection PyPep8Naming
    @property
    def C(self):
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

        # The test for step being an integer is in its setter method.
        # NOTE: setting _step here ALSO sets self._tap_pos. tap_pos is
        # for GridLAB-D, while 'step' is CIM.
        self._tap_pos = None
        self.step = step

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

    @step.setter
    def step(self, value):
        """Set the step (CIM) and also the tap_pos (GLD)"""
        if not isinstance(value, (int, np.integer)):
            raise TypeError('step must be an integer.')

        # Ensure we aren't getting an out of range value.
        if value < self.low_step or value > self.high_step:
            raise ValueError(
                'step must be between low_step ({}) and '
                'high_step ({}) (inclusive)'.format(self.low_step,
                                                    self.high_step))

        # Set step.
        self._step = value
        # Calculate and update tap_pos.
        self._tap_pos = \
            _tap_cim_to_gld(step=self.step, neutral_step=self.neutral_step)

    # GridLAB-D attributes:
    @property
    def tap_pos(self):
        return self._tap_pos

    @tap_pos.setter
    def tap_pos(self, value):
        """Set the tap_pos (GLD) and also the step (CIM)"""
        if not isinstance(value, (int, np.integer)):
            raise TypeError('tap_pos must be an integer.')

        # Ensure value is valid.
        if value < -self.lower_taps or value > self.raise_taps:
            raise ValueError(
                'tap_pos must be between lower_taps ({}) and '
                'raise_taps ({}) (inclusive)'.format(-self.lower_taps,
                                                     self.raise_taps))

        # Set new tap_pos and step.
        self._tap_pos = value
        self._step = _tap_gld_to_cim(tap_pos=value,
                                     neutral_step=self.neutral_step)

    @property
    def raise_taps(self):
        return self._raise_taps

    @property
    def lower_taps(self):
        return self._lower_taps
