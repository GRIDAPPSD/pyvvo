"""Module for querying and parsing SPARQL through GridAPPS-D"""
import logging
import inspect
from pyvvo.gridappsd_platform import get_gad_object
from pyvvo.utils import map_dataframe_columns

import pandas as pd

# Map CIM booleans (come back as strings) to Python booleans.
BOOLEAN_MAP = {'true': True, 'false': False}


class SPARQLManager:
    """Class for querying and parsing SPARQL in GridAPPS-D.

    NOTE: While it's usually conventional to have class constants
    defined near the beginning of the class definition, we'll be
    defining them at the bottom since the SPARQL queries are huge.
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
        self.log = logging.getLogger(__name__)

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
                feeder_mrid=self.feeder_mrid), to_numeric=False)
        self.log.info('Regulator tap position measurements obtained.')
        return result

    def query_capacitor_measurements(self):
        """Query status measurements attached to capacitors."""
        result = self._query(
            self.CAPACITOR_STATUS_MEASUREMENT_QUERY.format(
                feeder_mrid=self.feeder_mrid), to_numeric=False)
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
              "PREFIX c: <http://iec.ch/TC57/2012/CIM-schema-cim17#>\n"
              "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
              )

    # Query for getting capacitor information. Should be formatted with
    # a feeder ID: .format(feeder_mrid=feeder_mrid)
    CAPACITOR_QUERY = \
        (PREFIX +
         "SELECT ?name ?basev ?nomu ?bsection ?bus ?conn ?grnd ?phs "
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
         " as ?phs) "
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
         'bind(strafter(str(?classraw),"cim17#") as ?monclass) '
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
         "?tap_changer c:RatioTapChanger.stepVoltageIncrement ?step_voltage_increment. "
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

    # Get measurements associated with power transformers which have
    # discrete measurements.
    # TODO: Rather than relying on a discrete measurement, only get
    #   measurements attached to ratio tap changers.
    RTC_POSITION_MEASUREMENT_QUERY = \
        (PREFIX +
         "SELECT ?class ?type ?phases ?eqid ?id "
         # "?name ?trmid ?bus ?eqname ?eqtype"
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
         # 'UNION '
         # '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
         '?s c:IdentifiedObject.name ?name .'
         '?s c:IdentifiedObject.mRID ?id .'
         '?s c:Measurement.PowerSystemResource ?eq .'
         '?s c:Measurement.Terminal ?trm .'
         '?s c:Measurement.measurementType ?type .'
         # '?trm c:IdentifiedObject.mRID ?trmid.'
         '?eq c:IdentifiedObject.mRID ?eqid.'
         # '?eq c:IdentifiedObject.name ?eqname.'
         "?eq r:type c:PowerTransformer. "
         # Commented stuff below includes failed attempt(s) to filter
         #      by ratio tap changers only.
         # "?tap_changer r:type c:RatioTapChanger. "
         # "?tap_changer c:RatioTapChanger.TransformerEnd ?end. "
         # "?eq c:RatioTapChanger ?tap_changer. "
         # "?end c:TransformerTankEnd.TransformerTank ?tank. "
         # "?tank c:TransformerTank.PowerTransformer ?pxf. "
         # "?asset c:Asset.PowerSystemResources ?tap_changer. "
         # "?eq r:type c:RatioTapChanger. "
         # This is where the attempts to filter end. Lines commented out
         # below were commented because the information is not needed.
         # '?eq r:type ?typeraw.'
         # 'bind(strafter(str(?typeraw),"#") as ?eqtype)'
         '?trm c:Terminal.ConnectivityNode ?cn.'
         # '?cn c:IdentifiedObject.name ?bus.'
         '?s c:Measurement.phases ?phsraw .'
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phases)}}'
         '}} ORDER BY ?class ?type'
         # ' ?name'
         )

    # Get status measurements for capacitors.
    CAPACITOR_STATUS_MEASUREMENT_QUERY = \
        (PREFIX +
         "SELECT ?class ?type ?name ?bus ?phases ?eqtype ?eqname ?eqid ?trmid "
         "?id "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?eq c:Equipment.EquipmentContainer ?fdr."
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid."
         '{{ ?s r:type c:Discrete. bind ("Discrete" as ?class)}} '
         # 'UNION '
         # '{{ ?s r:type c:Analog. bind ("Analog" as ?class)}} '
         '?s c:IdentifiedObject.name ?name .'
         '?s c:IdentifiedObject.mRID ?id .'
         '?s c:Measurement.PowerSystemResource ?eq .'
         '?s c:Measurement.Terminal ?trm .'
         '?s c:Measurement.measurementType ?type .'
         '?trm c:IdentifiedObject.mRID ?trmid.'
         '?eq c:IdentifiedObject.mRID ?eqid.'
         '?eq c:IdentifiedObject.name ?eqname.'
         "?eq r:type c:LinearShuntCompensator. "
         '?eq r:type ?typeraw.'
         'bind(strafter(str(?typeraw),"#") as ?eqtype)'
         '?trm c:Terminal.ConnectivityNode ?cn.'
         '?cn c:IdentifiedObject.name ?bus.'
         '?s c:Measurement.phases ?phsraw .'
         '{{bind(strafter(str(?phsraw),"PhaseCode.") as ?phases)}}'
         '}} ORDER BY ?class ?type ?name'
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
