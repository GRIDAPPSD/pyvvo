"""Module for querying and parsing SPARQL through GridAPPS-D"""
import logging
from pyvvo.platform import get_gad_object


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

    def query_data(self, query_string):
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

    def query_named_objects(self, query_string):
        """Helper to perform a data query for named objects.

        NOTE: All queries MUST return an object name. If this is too
        restrictive it can be modified later.
        """
        # Perform query.
        result = self.query_data(query_string)

        # Get dictionary of objects.
        output = \
            self._bindings_to_dict(result['data']['results']['bindings'])

        return output

    def query_capacitors(self):
        """Get information on capacitors in the feeder."""
        # Perform the query.
        result = self.query_named_objects(
            self.CAPACITOR_QUERY.format(feeder_mrid=self.feeder_mrid))
        self.log.info('Capacitor data obtained.')

        # Done.
        return result

    def query_regulators(self):
        """Get information on capacitors in the feeder."""
        # Perform the query.
        result = self.query_named_objects(
            self.REGULATOR_QUERY.format(feeder_mrid=self.feeder_mrid))
        self.log.info('Regulator data obtained.')

        # Done.
        return result

    ####################################################################
    # HELPER FUNCTIONS
    ####################################################################

    def _bindings_to_dict(self, bindings):
        """Given a list of bindings, map them into a usable dictionary.

        The dictionary will be keyed by 'name,' so all bindings must
        have a 'name' attribute. So the upstream SPARQL query should be
        naming the name variable 'name.'
        """
        # Ensure bindings is a list.
        if not isinstance(bindings, list):
            self.log.error(("'Bindings' is not a list:"
                            "\n    {}".format(bindings)))
            raise TypeError("The bindings input must be a list.")

        # Crude check to see if we have a 'name' attribute - assuming
        # all elements will have it.
        try:
            bindings[0]['name']
        except KeyError:
            m = ("Missing 'name' attribute in bindings:\n    "
                 + "{}".format(bindings))
            self.log.error(m)
            raise KeyError("All bindings must have a 'name' attribute.")

        # Loop over the bindings and simplify to dictionary keyed by
        # object name, and only includes attribute names and values.
        output = dict()
        for obj in bindings:
            # Perform the simplification using dictionary comprehension.
            # Note we'll get a KeyError if
            output[obj['name']['value']] = \
                {k: v['value'] for (k, v) in obj.items()}

        self.log.debug("Bindings mapped into dictionary.")
        return output

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
         "SELECT ?rname ?name ?tname ?wnum ?phs ?incr ?mode ?enabled "
         "?highStep ?lowStep ?neutralStep ?normalStep ?neutralU ?step "
         "?initDelay ?subDelay ?ltc ?vlim ?vset ?vbw ?ldc ?fwdR ?fwdX "
         "?revR ?revX ?discrete ?ctl_enabled ?ctlmode ?monphs "
         "?ctRating ?ctRatio ?ptRatio ?mrid ?feeder_mrid ?rtcid ?endid "
         "WHERE {{ "
         'VALUES ?feeder_mrid {{"{feeder_mrid}"}} '
         "?pxf c:Equipment.EquipmentContainer ?fdr. "
         "?fdr c:IdentifiedObject.mRID ?feeder_mrid. "
         "?rtc r:type c:RatioTapChanger. "
         "?rtc c:IdentifiedObject.name ?rname. "
         "?rtc c:RatioTapChanger.TransformerEnd ?end. "
         "?end c:TransformerEnd.endNumber ?wnum. "
         "?end c:IdentifiedObject.mRID ?endid. "
         "OPTIONAL {{ "
         "?end c:TransformerTankEnd.phases ?phsraw. "
         'bind(strafter(str(?phsraw),"PhaseCode.") as ?phs)'
         "}} "
         "?end c:TransformerTankEnd.TransformerTank ?tank. "
         "?tank c:TransformerTank.PowerTransformer ?pxf. "
         "?pxf c:IdentifiedObject.name ?name. "
         "?pxf c:IdentifiedObject.mRID ?mrid. "
         "?tank c:IdentifiedObject.name ?tname. "
         "?rtc c:RatioTapChanger.stepVoltageIncrement ?incr. "
         "?rtc c:RatioTapChanger.tculControlMode ?moderaw. "
         'bind(strafter(str(?moderaw),"TransformerControlMode.")'
         " as ?mode) "
         "?rtc c:IdentifiedObject.mRID ?rtcid. "
         "?rtc c:TapChanger.controlEnabled ?enabled. "
         "?rtc c:TapChanger.highStep ?highStep. "
         "?rtc c:TapChanger.initialDelay ?initDelay. "
         "?rtc c:TapChanger.lowStep ?lowStep. "
         "?rtc c:TapChanger.ltcFlag ?ltc. "
         "?rtc c:TapChanger.neutralStep ?neutralStep. "
         "?rtc c:TapChanger.neutralU ?neutralU. "
         "?rtc c:TapChanger.normalStep ?normalStep. "
         "?rtc c:TapChanger.step ?step. "
         "?rtc c:TapChanger.subsequentDelay ?subDelay. "
         "?rtc c:TapChanger.TapChangerControl ?ctl. "
         "?ctl c:TapChangerControl.limitVoltage ?vlim. "
         "?ctl c:TapChangerControl.lineDropCompensation ?ldc. "
         "?ctl c:TapChangerControl.lineDropR ?fwdR. "
         "?ctl c:TapChangerControl.lineDropX ?fwdX. "
         "?ctl c:TapChangerControl.reverseLineDropR ?revR. "
         "?ctl c:TapChangerControl.reverseLineDropX ?revX. "
         "?ctl c:RegulatingControl.discrete ?discrete. "
         "?ctl c:RegulatingControl.enabled ?ctl_enabled. "
         "?ctl c:RegulatingControl.mode ?ctlmoderaw. "
         'bind(strafter(str(?ctlmoderaw),'
         '"RegulatingControlModeKind.") as ?ctlmode) '
         "?ctl c:RegulatingControl.monitoredPhase ?monraw. "
         'bind(strafter(str(?monraw),"PhaseCode.") as ?monphs) '
         "?ctl c:RegulatingControl.targetDeadband ?vbw. "
         "?ctl c:RegulatingControl.targetValue ?vset. "
         "?asset c:Asset.PowerSystemResources ?rtc. "
         "?asset c:Asset.AssetInfo ?inf. "
         "?inf c:TapChangerInfo.ctRating ?ctRating. "
         "?inf c:TapChangerInfo.ctRatio ?ctRatio. "
         "?inf c:TapChangerInfo.ptRatio ?ptRatio. "
         "}} "
         "ORDER BY ?name ?tname ?rname ?wnum "
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
        query -- SPARQL query that resulted in empty return.
        message -- explanation of the error.
    """

    def __init__(self, query, message):
        self.query = query.replace('\n', '\n    ')
        self.message = message