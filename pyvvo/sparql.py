"""Module for querying and parsing SPARQL through GridAPPS-D"""
import logging
import inspect
from pyvvo.gridappsd_platform import get_gad_object
from pyvvo.utils import map_dataframe_columns

import pandas as pd

# Map CIM booleans (come back as strings) to Python booleans.
BOOLEAN_MAP = {'true': True, 'false': False}

# Define constants for some variable names, which then get mapped to
# DataFrame column names.
REG_MEAS_MEAS_MRID_COL = 'pos_meas_mrid'
REG_MEAS_REG_MRID_COL = 'tap_changer_mrid'
CAP_MEAS_MEAS_MRID_COL = 'state_meas_mrid'
CAP_MEAS_CAP_MRID_COL = 'cap_mrid'
SWITCH_MEAS_MEAS_MRID_COL = 'state_meas_mrid'
SWITCH_MEAS_SWITCH_MRID_COL = 'switch_mrid'
INVERTER_MEAS_MEAS_MRID_COL = 'meas_mrid'
INVERTER_MEAS_INV_MRID_COL = 'inverter_mrid'
SYNCH_MACH_MEAS_MEAS_COL = 'meas_mrid'
SYNCH_MACH_MEAS_MACH_COL = 'mach_mrid'


class SPARQLManager:
    """Class for querying and parsing SPARQL in GridAPPS-D.

    NOTE: While it's usually conventional to have class constants
    defined near the beginning of the class definition, we'll be
    defining them at the bottom since the SPARQL queries are huge.

    TODO: Update query variables to be more reflective of what they are.
        That's what you get for copy + paste... What the hell is this?

    TODO: Some query variables should probably be constants which can
        be imported by other modules. This way, we can avoid looking
        through test files or query strings, and just find a listing
        of constants at the top.
    """

    def __init__(self, feeder_mrid, timeout=30):
        """Connect to the platform.

        :param feeder_mrid: unique identifier for the feeder in
            question. Since pyvvo works on a per feeder basis, this is
            required, and all queries will be executed for the specified
            feeder.
        :param timeout: timeout for querying the blazegraph database.
        """
        # Configure log.
        self.log = logging.getLogger(self.__class__.__name__)

        # Connect to the platform.
        self.gad = get_gad_object()
        self.log.debug('Connected to GridAPPS-D.')

        # Assign feeder mrid.
        self.feeder_mrid = feeder_mrid

        # Timeout for SPARQL queries.
        self.timeout = timeout

    ####################################################################
    # QUERY METHODS
    ####################################################################

    def query_capacitors(self):
        """Get information on capacitors in the feeder."""
        # Perform the query.
        result = self._query(
            self.CAPACITOR_QUERY.format(feeder_mrid=self.feeder_mrid),
            to_numeric=True)

        # Map boolean columns.
        result = map_dataframe_columns(
            map=BOOLEAN_MAP, df=result,
            cols=['ctrlenabled', 'discrete', 'grnd'])

        self.log.info('Capacitor data obtained.')

        # Done.
        return result

    def query_regulators(self):
        """Get information on capacitors in the feeder."""
        # Perform the query.
        result = self._query(
            self.REGULATOR_QUERY.format(feeder_mrid=self.feeder_mrid),
            to_numeric=True)

        # Map boolean columns.
        result = map_dataframe_columns(
            map=BOOLEAN_MAP, df=result,
            cols=['enabled', 'ltc_flag'])
        self.log.info('Regulator data obtained.')

        # Done.
        return result

    def query_load_nominal_voltage(self):
        """Get the nominal voltage for EnergyConsumers (loads).

        The values are not altered, so division by sqrt(3) will need to
        happen later.
        """
        result = \
            self._query(
                self.LOAD_NOMINAL_VOLTAGE_QUERY.format(
                    feeder_mrid=self.feeder_mrid), to_numeric=True)
        self.log.info('Load nominal voltage obtained.')
        return result

    def query_load_measurements(self):
        """Get measurement objects attached to EnergyConsumers (loads).

        Note that each load may have multiple measurements.
        """
        result = self._query(
            self.LOAD_MEASUREMENTS_QUERY.format(feeder_mrid=self.feeder_mrid),
            to_numeric=False)
        self.log.info('Load measurements data obtained.')
        return result

    def query_all_measurements(self):
        """Query all measurements in a model."""
        result = self._query(
            self.ALL_MEASUREMENTS_QUERY.format(feeder_mrid=self.feeder_mrid),
            to_numeric=False)
        self.log.info('All measurements obtained.')
        return result

    def query_rtc_measurements(self):
        """Query measurements attached to ratio tap changers."""
        result = self._query(
            self.RTC_POSITION_MEASUREMENT_QUERY.format(
                feeder_mrid=self.feeder_mrid, reg_mrid=REG_MEAS_REG_MRID_COL,
                meas_mrid=REG_MEAS_MEAS_MRID_COL), to_numeric=False)
        self.log.info('Regulator tap position measurements obtained.')
        return result

    def query_capacitor_measurements(self):
        """Query status measurements attached to capacitors."""
        result = self._query(
            self.CAPACITOR_STATUS_MEASUREMENT_QUERY.format(
                feeder_mrid=self.feeder_mrid, cap_mrid=CAP_MEAS_CAP_MRID_COL,
                meas_mrid=CAP_MEAS_MEAS_MRID_COL), to_numeric=False)
        self.log.info('Capacitor status measurements obtained.')
        return result

    def query_substation_source(self):
        """Get the substation source information."""
        result = self._query(
            self.SUBSTATION_SOURCE_QUERY.format(feeder_mrid=self.feeder_mrid),
            to_numeric=True)
        self.log.info('Substation source information obtained.')
        return result

    def query_measurements_for_bus(self, bus_mrid):
        """Get all measurements with a specific parent bus."""
        result = self._query(
            self.MEASUREMENTS_FOR_BUS_QUERY.format(
                feeder_mrid=self.feeder_mrid, bus_mrid=bus_mrid),
            to_numeric=False)
        self.log.info(
            'Measurements associated with bus {} obtained.'.format(bus_mrid))
        return result

    def query_switches(self):
        result = self._query(self.SWITCHES_QUERY.format(
            feeder_mrid=self.feeder_mrid), to_numeric=True)
        self.log.info('Switch information obtained.')
        return result

    def query_switch_measurements(self):
        """Query status measurements attached to switches."""
        result = self._query(
            self.SWITCH_STATUS_QUERY.format(
                feeder_mrid=self.feeder_mrid,
                switch_mrid=SWITCH_MEAS_SWITCH_MRID_COL,
                meas_mrid=SWITCH_MEAS_MEAS_MRID_COL
            ), to_numeric=False
        )
        self.log.info('Switch status measurement information obtained.')
        return result

    def query_inverters(self):
        """Get inverter attributes."""
        result = self._query(self.INVERTER_QUERY.format(
            feeder_mrid=self.feeder_mrid), to_numeric=True)
        self.log.info('Inverter information obtained.')
        return result

    def query_inverter_measurements(self):
        """Get measurements associated with inverters."""
        result = self._query(self.INVERTER_MEASUREMENTS_QUERY.format(
            feeder_mrid=self.feeder_mrid), to_numeric=False)
        self.log.info('Inverter measurement information obtained.')
        return result

    def query_synchronous_machines(self):
        """Get synchronous machines. For now, assume they're generators.
        """
        result = self._query(self.SYNCH_MACH_QUERY.format(
            feeder_mrid=self.feeder_mrid), to_numeric=True)
        self.log.info('Synchronous machine information obtained.')
        return result

    def query_synchronous_machine_measurements(self):
        """Get VA measurements associated with synchronous machines."""
        result = self._query(self.SYNCH_MACH_MEAS_QUERY.format(
            feeder_mrid=self.feeder_mrid), to_numeric=False
        )
        self.log.info('Synchronous machine measurement information obtained.')
        return result

    ####################################################################
    # HELPER FUNCTIONS
    ####################################################################

    def _query(self, query_string, to_numeric):
        """Helper to perform a data query for named objects.

        NOTE: All queries MUST return an object name. If this is too
        restrictive it can be modified later.

        :param query_string: Fully formatted SPARQL query.
        :param to_numeric: True/False, whether or not to attempt to
            convert DataFrame columns to numeric values. Set to True if
            some columns will be numeric, set to False if no columns
            will be numeric.
        """
        # Perform query.
        result = self._query_platform(query_string)

        output = self._bindings_to_dataframe(
            bindings=result['data']['results']['bindings'],
            to_numeric=to_numeric)

        return output

    def _query_platform(self, query_string):
        """Wrapper to call GridAPPSD.query_data with error handling."""
        # Perform query.
        result = self.gad.query_data(query=query_string, timeout=self.timeout)

        # Check for error (bad syntax, etc.).
        try:
            result['error']
        except KeyError:
            # No problem.
            pass
        else:
            # The 'error' key exists.
            m = 'The given query resulted in an error.'
            e = SPARQLQueryError(query=query_string, message=m)
            self.log.error('The following SPARQL query resulted in an '
                           + 'error:\n    {}'.format(query_string))
            raise e

        # Check for empty query return.
        if len(result['data']['results']['bindings']) == 0:
            m = 'The given query did not return any data.'
            e = SPARQLQueryReturnEmptyError(query=query_string,
                                            message=m)
            self.log.error('The following SPARQL query did not return any '
                           + 'data:\n    {}'.format(e.query))
            raise e

        # Done.
        return result

    def _bindings_to_dataframe(self, bindings, to_numeric):
        # Check bindings.
        self._check_bindings(bindings)

        # Create list of dictionaries.
        list_of_dicts = []
        for obj in bindings:
            # Use dict comprehension to add a dictionary to the list.
            list_of_dicts.append(
                {k: v['value'] for (k, v) in obj.items()}
            )

        # Create a DataFrame, convert applicable data to numeric types.
        output = pd.DataFrame(list_of_dicts)
        if to_numeric:
            output = output.apply(pd.to_numeric, errors='ignore')

        # Replace the empty string with nan.
        output.replace(to_replace='', value=pd.np.nan, inplace=True)

        # Warn if there are NaNs.
        if output.isnull().values.any():
            # Work up the stack to one method above '_query' so we can
            # get a more detailed warning.
            s = inspect.stack()
            q = False
            fn_name = '<function not found!>'
            try:
                # Loop over frames in the stack.
                for f in s:
                    if q:
                        # The previous stack function was '_query.'
                        # Get the function name and exit the loop.
                        fn_name = f.function
                        break

                    # Check if this function is _query.
                    q = f.function == '_query'
            finally:
                del s

            m = "DataFrame from '{}' has NaN value(s)!".format(fn_name)
            self.log.warning(m)

        return output

    def _check_bindings(self, bindings):
        """Simple helper to do primitive checks on returned bindings."""
        # Ensure bindings is a list.
        if not isinstance(bindings, list):
            self.log.error(("'Bindings' is not a list:"
                            "\n    {}".format(bindings)))
            raise TypeError("The bindings input must be a list.")

        # Ensure the first element of the list is a dictionary. Looping
        # over the whole thing seems unnecessary.
        if not isinstance(bindings[0], dict):
            m = "The first bindings element is not a dict!"
            self.log.error(m)
            raise TypeError(m)

        # We were checking for a 'name' attribute, but this is no
        # longer necessary since the bindings are being put into a
        # DataFrame.

    ####################################################################
    # SPARQL QUERY TEXT
    ####################################################################

    # Define query prefix
    PREFIX = ("PREFIX r: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
              "PREFIX c: <http://iec.ch/TC57/CIM100#>\n"
              "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
              )

    # Query for getting capacitor information. Should be formatted with
    # a feeder ID: .format(feeder_mrid=feeder_mrid)
    # NOTE: For whatever reason, adding
    # "?s c:ShuntCompensator.sections ?state. " makes our query come
    # back empty. Looking at the CIM diagram, this shouldn't be the
    # case...
    CAPACITOR_QUERY = \
        (PREFIX +
         "SELECT ?name ?basev ?nomu ?bsection ?bus ?conn ?grnd ?phase "
         "?ctrlenabled ?discrete ?mode ?deadband ?setpoint ?delay "
         "?monclass ?moneq ?monbus ?monphs ?mrid ?feeder_mrid "
         "WHERE {{ "
         "?s r:type c:LinearShuntCompensator. "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?s c:Equipment.EquipmentContainer ?fdr. "
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "?s c:IdentifiedObject.name ?name. "
         "?s c:ConductingEquipment.BaseVoltage ?bv. "
         "?bv c:BaseVoltage.nominalVoltage ?basev. "
         "?s c:ShuntCompensator.nomU ?nomu. "
         "?s c:LinearShuntCompensator.bPerSection ?bsection. "
         "?s c:ShuntCompensator.phaseConnection ?connraw. "
         'bind(strafter(str(?connraw),"PhaseShuntConnectionKind.")'
         " as ?conn) "
         "?s c:ShuntCompensator.grounded ?grnd. "
         "OPTIONAL {{ "
         "?scp c:ShuntCompensatorPhase.ShuntCompensator ?s. "
         "?scp c:ShuntCompensatorPhase.phase ?phsraw. "
         'bind(strafter(str(?phsraw),"SinglePhaseKind.")'
         " as ?phase) "
         "}} "
         "OPTIONAL {{ "
         "?ctl c:RegulatingControl.RegulatingCondEq ?s. "
         "?ctl c:RegulatingControl.discrete ?discrete. "
         "?ctl c:RegulatingControl.enabled ?ctrlenabled. "
         "?ctl c:RegulatingControl.mode ?moderaw. "
         'bind(strafter(str(?moderaw),'
         '"RegulatingControlModeKind.")'
         " as ?mode) "
         "?ctl c:RegulatingControl.monitoredPhase ?monraw. "
         'bind(strafter(str(?monraw),"PhaseCode.") as ?monphs) '
         "?ctl c:RegulatingControl.targetDeadband ?deadband. "
         "?ctl c:RegulatingControl.targetValue ?setpoint. "
         "?s c:ShuntCompensator.aVRDelay ?delay. "
         "?ctl c:RegulatingControl.Terminal ?trm. "
         "?trm c:Terminal.ConductingEquipment ?eq. "
         "?eq a ?classraw. "
         'bind(strafter(str(?classraw),"CIM100#") as ?monclass) '
         "?eq c:IdentifiedObject.name ?moneq. "
         "?trm c:Terminal.ConnectivityNode ?moncn. "
         "?moncn c:IdentifiedObject.name ?monbus. "
         "}} "
         "?s c:IdentifiedObject.mRID ?mrid. "
         "?t c:Terminal.ConductingEquipment ?s. "
         "?t c:Terminal.ConnectivityNode ?cn. "
         "?cn c:IdentifiedObject.name ?bus "
         "}} "
         "ORDER by ?name "
         )

    # Query for getting regulator information. Should be formatted with
    # a feeder ID: .format(feeder_mrid=feeder_mrid)
    REGULATOR_QUERY = \
        (PREFIX +
         "SELECT ?ltc_flag ?mrid ?name ?phase "
         "?step_voltage_increment ?control_mode ?enabled ?high_step ?low_step "
         "?neutral_step ?step ?tap_changer_mrid "
         # "?rname ?initDelay ?subDelay ?tname ?wnum ?ldc ?fwdR ?fwdX ?revR "
         # "?revX ?discrete ?ctRating ?ctRatio ?ptRatio ?feeder_mrid "
         # "?monphs ?neutralU ?vlim ?vset ?vbw ?endid ?normalStep ?mode "
         # "?control_enabled "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?pxf c:Equipment.EquipmentContainer ?fdr. "
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "?tap_changer r:type c:RatioTapChanger. "
         # "?tap_changer c:IdentifiedObject.name ?rname. "
         "?tap_changer c:RatioTapChanger.TransformerEnd ?end. "
         # "?end c:TransformerEnd.endNumber ?wnum. "
         # "?end c:IdentifiedObject.mRID ?endid. "
         "OPTIONAL {{ "
         "?end c:TransformerTankEnd.phases ?phsraw. "
         'bind(strafter(str(?phsraw),"PhaseCode.") as ?phase)'
         "}} "
         "?end c:TransformerTankEnd.TransformerTank ?tank. "
         "?tank c:TransformerTank.PowerTransformer ?pxf. "
         "?pxf c:IdentifiedObject.name ?name. "
         "?pxf c:IdentifiedObject.mRID ?mrid. "
         # "?tank c:IdentifiedObject.name ?tname. "
         "?tap_changer c:RatioTapChanger.stepVoltageIncrement "
         "?step_voltage_increment. "
         # "?tap_changer c:RatioTapChanger.tculControlMode ?moderaw. "
         # 'bind(strafter(str(?moderaw),"TransformerControlMode.")'
         # " as ?mode) "
         "?tap_changer c:IdentifiedObject.mRID ?tap_changer_mrid. "
         "?tap_changer c:TapChanger.controlEnabled ?control_enabled. "
         "?tap_changer c:TapChanger.highStep ?high_step. "
         # "?tap_changer c:TapChanger.initialDelay ?initDelay. "
         "?tap_changer c:TapChanger.lowStep ?low_step. "
         "?tap_changer c:TapChanger.ltcFlag ?ltc_flag. "
         "?tap_changer c:TapChanger.neutralStep ?neutral_step. "
         # "?tap_changer c:TapChanger.neutralU ?neutralU. "
         # "?tap_changer c:TapChanger.normalStep ?normalStep. "
         "?tap_changer c:TapChanger.step ?step. "
         # "?tap_changer c:TapChanger.subsequentDelay ?subDelay. "
         "?tap_changer c:TapChanger.TapChangerControl ?ctl. "
         # "?ctl c:TapChangerControl.limitVoltage ?vlim. "
         # "?ctl c:TapChangerControl.lineDropCompensation ?ldc. "
         # "?ctl c:TapChangerControl.lineDropR ?fwdR. "
         # "?ctl c:TapChangerControl.lineDropX ?fwdX. "
         # "?ctl c:TapChangerControl.reverseLineDropR ?revR. "
         # "?ctl c:TapChangerControl.reverseLineDropX ?revX. "
         # "?ctl c:RegulatingControl.discrete ?discrete. "
         "?ctl c:RegulatingControl.enabled ?enabled. "
         "?ctl c:RegulatingControl.mode ?ctlmoderaw. "
         'bind(strafter(str(?ctlmoderaw),'
         '"RegulatingControlModeKind.") as ?control_mode) '
         # "?ctl c:RegulatingControl.monitoredPhase ?monraw. "
         # 'bind(strafter(str(?monraw),"PhaseCode.") as ?monphs) '
         # "?ctl c:RegulatingControl.targetDeadband ?vbw. "
         # "?ctl c:RegulatingControl.targetValue ?vset. "
         "?asset c:Asset.PowerSystemResources ?tap_changer. "
         # "?asset c:Asset.AssetInfo ?inf. "
         # "?inf c:TapChangerInfo.ctRating ?ctRating. "
         # "?inf c:TapChangerInfo.ctRatio ?ctRatio. "
         # "?inf c:TapChangerInfo.ptRatio ?ptRatio. "
         "}} "
         "ORDER BY ?name ?rname "
         # "?tname ?wnum "
         )

    # Query for getting the nominal voltage of EnergyConsumers.
    LOAD_NOMINAL_VOLTAGE_QUERY = \
        (PREFIX +
         r'SELECT ?name ?bus ?basev ?conn '
         '(group_concat(distinct ?phs;separator=",") as ?phases) '
         "WHERE {{ "
         "?s r:type c:EnergyConsumer. "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?s c:Equipment.EquipmentContainer ?fdr. "
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "?s c:IdentifiedObject.name ?name. "
         "?s c:ConductingEquipment.BaseVoltage ?bv. "
         "?bv c:BaseVoltage.nominalVoltage ?basev. "
         "?s c:EnergyConsumer.phaseConnection ?connraw. "
         'bind(strafter(str(?connraw),"PhaseShuntConnectionKind.") as ?conn) '
         "OPTIONAL {{ "
         "?ecp c:EnergyConsumerPhase.EnergyConsumer ?s. "
         "?ecp c:EnergyConsumerPhase.phase ?phsraw. "
         'bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) '
         "}} "
         "?t c:Terminal.ConductingEquipment ?s. "
         "?t c:Terminal.ConnectivityNode ?cn. "
         "?cn c:IdentifiedObject.name ?bus "
         "}} "
         "GROUP BY ?name ?bus ?basev ?p ?q ?conn "
         "ORDER by ?name "
         )

    # Query for getting measurement objects attached to EnergyConsumers.
    LOAD_MEASUREMENTS_QUERY = \
        (PREFIX +
         "SELECT ?class ?type ?name ?node ?phases ?load ?eqid ?trmid ?id "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr. "
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
         'UNION '
         '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
         '?s c:IdentifiedObject.name ?name . '
         '?s c:IdentifiedObject.mRID ?id . '
         '?s c:Measurement.PowerSystemResource ?eq . '
         '?s c:Measurement.Terminal ?trm . '
         '?s c:Measurement.measurementType ?type . '
         '?trm c:IdentifiedObject.mRID ?trmid. '
         '?eq c:IdentifiedObject.mRID ?eqid. '
         '?eq c:IdentifiedObject.name ?load. '
         '?eq r:type c:EnergyConsumer. '
         '?trm c:Terminal.ConnectivityNode ?cn. '
         '?cn c:IdentifiedObject.name ?node. '
         '?s c:Measurement.phases ?phsraw . '
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phases)}} '
         '}} '
         'ORDER BY ?load ?class ?type ?name '
         )

    # List all measurements with buses and equipment - DistMeasurement.
    ALL_MEASUREMENTS_QUERY = \
        (PREFIX +
         "SELECT ?class ?type ?name ?bus ?bus_mrid ?phases ?eqtype ?eqname "
         "?eqid ?trmid ?id "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
         'UNION '
         '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
         '?s c:IdentifiedObject.name ?name .'
         '?s c:IdentifiedObject.mRID ?id .'
         '?s c:Measurement.PowerSystemResource ?eq .'
         '?s c:Measurement.Terminal ?trm .'
         '?s c:Measurement.measurementType ?type .'
         '?trm c:IdentifiedObject.mRID ?trmid.'
         '?eq c:IdentifiedObject.mRID ?eqid.'
         '?eq c:IdentifiedObject.name ?eqname.'
         '?eq r:type ?typeraw.'
         'bind(strafter(str(?typeraw),"#") as ?eqtype)'
         '?trm c:Terminal.ConnectivityNode ?cn.'
         '?cn c:IdentifiedObject.name ?bus.'
         '?cn c:IdentifiedObject.mRID ?bus_mrid. '
         '?s c:Measurement.phases ?phsraw .'
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phases)}}'
         '}} ORDER BY ?class ?type ?name'
         )

    # Get measurements associated with ratio tap changers (RTC)
    RTC_POSITION_MEASUREMENT_QUERY = \
        (PREFIX +
         # We're using this query to map measurements to tap changers,
         # so all we need is the tap changer and measurement MRIDs.
         "SELECT ?{reg_mrid} ?{meas_mrid} "
         # "?name ?trmid ?bus ?eqname ?eqtype"
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         # Ensure we're only grabbing 'position' measurements.
         'VALUES ?type {{"Pos"}}'
         # Filter by feeder.
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         '{{ ?s r:type c:Discrete. }} '
         '?s c:IdentifiedObject.name ?name .'
         '?s c:IdentifiedObject.mRID ?{meas_mrid} .'
         # Get the power system resource.
         '?s c:Measurement.PowerSystemResource ?eq .'
         '?s c:Measurement.Terminal ?trm .'
         '?s c:Measurement.measurementType ?type .'

         # Trace the power system resource, ensuring it's a tap changer.
         '?eq c:IdentifiedObject.mRID ?eqid.'
         "?eq r:type c:PowerTransformer. "
         "?tap_changer r:type c:RatioTapChanger. "
         "?tap_changer c:IdentifiedObject.mRID ?{reg_mrid}. "
         "?tap_changer c:RatioTapChanger.TransformerEnd ?end. "
         "?end c:TransformerTankEnd.TransformerTank ?tank. "
         "?tank c:TransformerTank.PowerTransformer ?eq. "

         # Get the phase of the tank end.
         "?end c:TransformerTankEnd.phases ?endphsraw. "

         # Get the measurement phase.
         '?s c:Measurement.phases ?measphsraw .'

         # Only return triples where the phases match.
         "FILTER(?endphsraw = ?measphsraw) "

         '}} ORDER BY ?{reg_mrid}'
         # ' ?name'
         )

    # Get status measurements for capacitors.
    # TODO: This query could be optimized.
    CAPACITOR_STATUS_MEASUREMENT_QUERY = \
        (PREFIX +
         "SELECT ?{cap_mrid} ?{meas_mrid} ?phase "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
         # 'UNION '
         # '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
         '?s c:IdentifiedObject.name ?name .'
         '?s c:IdentifiedObject.mRID ?{meas_mrid} .'
         '?s c:Measurement.PowerSystemResource ?eq .'
         '?s c:Measurement.Terminal ?trm .'
         '?s c:Measurement.measurementType ?type .'
         '?trm c:IdentifiedObject.mRID ?trmid.'
         '?eq c:IdentifiedObject.mRID ?{cap_mrid}.'
         "?eq r:type c:LinearShuntCompensator. "
         '?eq r:type ?typeraw.'
         'bind(strafter(str(?typeraw),"#") as ?eqtype)'
         '?trm c:Terminal.ConnectivityNode ?cn.'
         '?cn c:IdentifiedObject.name ?bus.'
         '?s c:Measurement.phases ?phsraw .'
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phase)}}'
         '}} ORDER BY ?{cap_mrid}'
         )

    # substation source - DistSubstation
    SUBSTATION_SOURCE_QUERY = \
        (PREFIX +
         "SELECT ?name ?mrid ?bus ?bus_mrid ?basev ?nomv "
         # "?vmag ?vang ?r1 ?x1 ?r0 ?x0 "
         "WHERE {{ "
         "?s r:type c:EnergySource. "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?s c:Equipment.EquipmentContainer ?fdr. "
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "?s c:IdentifiedObject.name ?name. "
         "?s c:IdentifiedObject.mRID ?mrid. "
         "?s c:ConductingEquipment.BaseVoltage ?bv. "
         "?bv c:BaseVoltage.nominalVoltage ?basev. "
         "?s c:EnergySource.nominalVoltage ?nomv. "
         # "?s c:EnergySource.voltageMagnitude ?vmag. "
         # "?s c:EnergySource.voltageAngle ?vang. "
         # "?s c:EnergySource.r ?r1. "
         # "?s c:EnergySource.x ?x1. "
         # "?s c:EnergySource.r0 ?r0. "
         # "?s c:EnergySource.x0 ?x0. "
         "?t c:Terminal.ConductingEquipment ?s. "
         "?t c:Terminal.ConnectivityNode ?cn. "
         "?cn c:IdentifiedObject.name ?bus. "
         "?cn c:IdentifiedObject.mRID ?bus_mrid "
         "}} "
         "ORDER by ?name"
         )

    MEASUREMENTS_FOR_BUS_QUERY = \
        (PREFIX +
         "SELECT ?class ?type ?name ?bus ?bus_mrid ?phases ?eqtype ?eqname "
         "?eqid ?trmid ?id "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
         'UNION '
         '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
         '?s c:IdentifiedObject.name ?name .'
         '?s c:IdentifiedObject.mRID ?id .'
         '?s c:Measurement.PowerSystemResource ?eq .'
         '?s c:Measurement.Terminal ?trm .'
         '?s c:Measurement.measurementType ?type .'
         '?trm c:IdentifiedObject.mRID ?trmid.'
         '?eq c:IdentifiedObject.mRID ?eqid.'
         '?eq c:IdentifiedObject.name ?eqname.'
         '?eq r:type ?typeraw.'
         'bind(strafter(str(?typeraw),"#") as ?eqtype)'
         '?trm c:Terminal.ConnectivityNode ?cn.'
         '?cn c:IdentifiedObject.name ?bus.'
         '?cn c:IdentifiedObject.mRID "{bus_mrid}". '
         '?s c:Measurement.phases ?phsraw .'
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phases)}}'
         '}} ORDER BY ?class ?type ?name'
         )

    # TODO: Update this query to also get switch nominal state (e.g.
    #   open/closed).
    SWITCHES_QUERY = \
        (PREFIX +
         "SELECT ?name ?mrid "
         '(group_concat(distinct ?phs;separator="") as ?phase) '
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}}. '
         'VALUES ?cimraw {{c:LoadBreakSwitch c:Recloser c:Breaker}}. '
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "?s c:Equipment.EquipmentContainer ?fdr. "
         "?s r:type ?cimraw. "
         "?s c:IdentifiedObject.name ?name. "
         "?s c:IdentifiedObject.mRID ?mrid. "
         "OPTIONAL {{ "
           "?swp c:SwitchPhase.Switch ?s. "
           "?swp c:SwitchPhase.phaseSide1 ?phsraw. "
           'bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs). ' 
           "}}"
         "}} "
         "GROUP BY ?name ?mrid ?phase "
         "ORDER BY ?mrid "
         )

    SWITCH_STATUS_QUERY = \
        (PREFIX +
         "SELECT ?{switch_mrid} ?{meas_mrid} ?phase "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         "{{ ?s r:type c:Discrete. }} "
         "?s r:type ?type . "
         "?s c:IdentifiedObject.mRID ?{meas_mrid} . "
         "?s c:Measurement.PowerSystemResource ?eq . "
         "?s c:Measurement.Terminal ?trm . "
         "?s c:Measurement.phases ?phsraw . "
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phase)}} . '
         "?eq c:IdentifiedObject.mRID ?{switch_mrid} . "
         "?eq r:type c:LoadBreakSwitch . "
         "}} "
         "ORDER BY ?{switch_mrid}"
         )

    # NOTE: For whatever reason, "inverterMode," "maxQ," and "minQ" are
    # not included for the 9500 node model, and have thus been moved
    # into an OPTIONAL block.
    #
    # NOTE: Naming conventions: "inverter_" is associated with the
    # "Wires::PowerElectronicsConnection" object, and "phase_" is
    # associated with the "Wires::PowerElectronicsConnectionPhase"
    # object. Documentation:
    # https://gridappsd.readthedocs.io/en/latest/developer_resources/index.html#cim-documentation
    # noinspection PyPep8
    INVERTER_QUERY = \
        (PREFIX +
         'SELECT ?inverter_mrid ?inverter_name ?inverter_mode ?inverter_max_q '
         '?inverter_min_q ?inverter_p ?inverter_q ?inverter_rated_s '
         '?inverter_rated_u ?phase_mrid ?phase_name ?phase_p ?phase_q '
         r'(group_concat(distinct ?phs;separator="\n") as ?phases) '
         "WHERE {{ "
            'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
            "?s r:type c:PowerElectronicsConnection. "
            "?s c:Equipment.EquipmentContainer ?fdr. "
            "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
            "?s c:IdentifiedObject.mRID ?inverter_mrid. "
            "?s c:IdentifiedObject.name ?inverter_name. "
            "?s c:PowerElectronicsConnection.p ?inverter_p. "
            "?s c:PowerElectronicsConnection.q ?inverter_q. "
            "?s c:PowerElectronicsConnection.ratedS ?inverter_rated_s. "
            "?s c:PowerElectronicsConnection.ratedU ?inverter_rated_u. "
            "OPTIONAL {{ "
                "?s c:PowerElectronicsConnection.inverterMode ?inverter_mode. "
                "?s c:PowerElectronicsConnection.maxQ ?inverter_max_q. "
                "?s c:PowerElectronicsConnection.minQ ?inverter_min_q. "
            "}} "
            "OPTIONAL {{ "
                "?pecp c:PowerElectronicsConnectionPhase.PowerElectronicsConnection ?s. "
                "?pecp c:IdentifiedObject.mRID ?phase_mrid. "
                "?pecp c:IdentifiedObject.name ?phase_name. "
                "?pecp c:PowerElectronicsConnectionPhase.p ?phase_p. "
                "?pecp c:PowerElectronicsConnectionPhase.q ?phase_q. "
                "?pecp c:PowerElectronicsConnectionPhase.phase ?phsraw "
                'bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phs) '
            "}} "
         "}} "
         "GROUP BY ?inverter_mrid ?inverter_name ?inverter_rated_s "
         "?inverter_rated_u ?inverter_p ?inverter_q ?phase_mrid ?phase_name "
         "?phase_p ?phase_q ?inverter_mode ?inverter_max_q ?inverter_min_q "
         "ORDER BY ?inverter_mrid"
         )

    # NOTE: It would appear that the measurements for inverters are
    # associated with PowerElectronicsConnection objects and NOT
    # PowerElectronicsConnectionPhase objects. You can prove this by
    # replacing PowerElectronicsConnection with
    # PowerElectronicsConnectionPhase below.
    INVERTER_MEASUREMENTS_QUERY = \
        (PREFIX +
         "SELECT ?inverter_mrid ?meas_mrid ?phase ?meas_type "
         "WHERE {{ "
            'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
            # Filter to get only power measurements.
            'VALUES ?meas_type {{"VA"}} '
            '?s r:type c:Analog. '
            '?s c:Measurement.measurementType ?meas_type. '
            "?s c:IdentifiedObject.mRID ?meas_mrid. "
            "?s c:Measurement.PowerSystemResource ?eq. "
            "?s c:Measurement.Terminal ?trm. "
            "?s c:Measurement.phases ?phsraw. "
            '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phase)}} .'
            "?eq c:IdentifiedObject.mRID ?inverter_mrid. "
            "?eq r:type c:PowerElectronicsConnection. "
            "?eq c:Equipment.EquipmentContainer ?fdr. "
            "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "}} "
         "ORDER BY ?inverter_mrid ?meas_type"
         )

    # TODO: This makes no effort to determine if the machine is a
    #   generator, and doesn't check any control parameters.
    SYNCH_MACH_QUERY = PREFIX + \
        """
        
        SELECT ?mrid ?name ?rated_s ?rated_u ?p ?q ?phase
        WHERE {{
            VALUES ?feeder_mrid {{"{feeder_mrid}"}} 
            ?s r:type c:SynchronousMachine.
            ?s c:IdentifiedObject.name ?name.
            ?s c:Equipment.EquipmentContainer ?fdr.
            ?fdr c:IdentifiedObject.mRID ?feeder_mrid.
            ?s c:SynchronousMachine.ratedS ?rated_s.
            ?s c:SynchronousMachine.ratedU ?rated_u.
            ?s c:SynchronousMachine.p ?p.
            ?s c:SynchronousMachine.q ?q. 
            bind(strafter(str(?s),"#") as ?mrid).
            OPTIONAL {{
                ?smp c:SynchronousMachinePhase.SynchronousMachine ?s.
                ?smp c:SynchronousMachinePhase.phase ?phsraw.
                bind(strafter(str(?phsraw),"SinglePhaseKind.") as ?phase)
            }}
        }}
        GROUP BY ?mrid ?name ?rated_s ?rated_u ?p ?q ?phase
        ORDER BY ?mrid
        """

    # Get VA measurements only.
    SYNCH_MACH_MEAS_QUERY = PREFIX + \
        """
        SELECT ?mach_mrid ?meas_mrid ?phase ?meas_type
        WHERE {{
            VALUES ?feeder_mrid {{"{feeder_mrid}"}}
            VALUES ?meas_type {{"VA"}}
            ?s r:type c:Analog.
            ?s c:Measurement.measurementType ?meas_type.
            ?s c:IdentifiedObject.mRID ?meas_mrid.
            ?s c:Measurement.PowerSystemResource ?eq.
            ?s c:Measurement.Terminal ?trm.
            ?s c:Measurement.phases ?phsraw.
            {{bind(strafter(str(?phsraw),"PhaseCode.") as ?phase)}}.
            ?eq c:IdentifiedObject.mRID ?mach_mrid.
            ?eq r:type c:SynchronousMachine.
            ?eq c:Equipment.EquipmentContainer ?fdr.
            ?fdr c:IdentifiedObject.mRID ?feeder_mrid.
        }}
        ORDER BY ?mach_mrid ?meas_type
        """


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class SPARQLQueryReturnEmptyError(Error):
    """Raised if a SPARQL query returns empty bindings.

    Attributes:
        query -- SPARQL query that resulted in empty return.
        message -- explanation of the error.
    """

    def __init__(self, query, message):
        self.query = query.replace('\n', '\n    ')
        self.message = message


class SPARQLQueryError(Error):
    """Raised if a SPARQL query returns an error.

    Attributes:
        query -- SPARQL query that resulted in error.
        message -- explanation of the error.
    """

    def __init__(self, query, message):
        self.query = query.replace('\n', '\n    ')
        self.message = message
