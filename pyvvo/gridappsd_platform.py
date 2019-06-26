"""Module for interfacing with the GridAPPS-D platform.

Note this module depends on the GridAPPS-D platform being up and
running.
"""
from gridappsd import GridAPPSD, topics
from gridappsd.difference_builder import DifferenceBuilder as DiffBuilder
from gridappsd import utils as gad_utils

import simplejson as json
import os
import logging
import re
from datetime import datetime
import copy
from pyvvo import utils
from pyvvo.utils import platform_header_timestamp_to_dt as platform_dt
from pyvvo.utils import simulation_output_timestamp_to_dt as simulation_dt
from pyvvo import timeseries

# Setup log.
LOG = logging.getLogger(__name__)

# Use this date format for logging.
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Compile regular expressions for fixing bad json return.
# TODO: remove when the platform is fixed.
REGEX_1 = re.compile(r'^\s*\{\s*"data"\s*:\s*')
REGEX_2 = re.compile(r'\s*,\s*"responseComplete".+$')


def get_platform_env_var():
    """Helper to get the 'platform' environment variable."""
    # Assign platform input.
    try:
        platform = os.environ['platform']
    except KeyError:
        m = ("Within the pyvvo Docker container, the 'platform' "
             "environment variable must be set. It should be '0' "
             "or '1' indicating whether or not the container is "
             "inside the platform's docker-compose network.")
        raise KeyError(m)
    else:
        if (platform != '0') and (platform != '1'):
            m = ("The 'platform' environment variable must be '0' or"
                 "'1.' It currently evaluates to {}").format(platform)
            raise ValueError(m)

    LOG.debug("Retrieved 'platform' environment variable.")
    return platform


def get_gad_object(**kwargs):
    """Helper to get a GridAPPSD object.

    :param kwargs: Passed directly to the GridAPPSD constructor.
    """
    # Get platform address
    address = get_gad_address()

    # TODO: handle connection failures?
    # TODO: flexibility for different username/password?
    gad = GridAPPSD(address=address, username=gad_utils.get_gridappsd_user(),
                    password=gad_utils.get_gridappsd_pass(), **kwargs)

    LOG.debug('GridAPPSD object created and connected to the platform.')
    return gad


def get_gad_address():
    """Helper to get GRIDAPPS-D address.

    The address is different depending on whether we're operating
    inside or outside the platform.

    TODO: logging
    """
    # Determine whether or not we're running in the platform.
    platform = get_platform_env_var()

    if platform == '1':
        # We're in the platform. Use the API helper.
        address = gad_utils.get_gridappsd_address()
    elif platform == '0':
        # We're not in the platform. Use the local IP address and get
        # the port from the environment.
        try:
            address = (os.environ['host_ip'], os.environ['GRIDAPPSD_PORT'])
        except KeyError:
            # Environment variables not properly set.
            m = ("If running outside the platform ('platform' input is '0'), "
                 + "the environment variables 'host_ip' and 'GRIDAPPSD_PORT' "
                 + "must be set.")
            raise KeyError(m)
    else:
        m = "The 'platform' input must be either '0' or '1.'"
        if isinstance(platform, str):
            raise ValueError(m)
        else:
            raise TypeError(m)

    LOG.debug('GridAPPS-D address obtained.')
    return address


class SimOutRouter:
    """Class for listening and routing simulation output."""

    def __init__(self, platform_manager, sim_id, fn_mrid_list):
        """

        :param platform_manager: Initialized PlatformManager object.
        :param sim_id: Simulation ID on which to listen to.
        :param fn_mrid_list: list of dictionaries of the form
            {'functions': <function>, 'mrids': [<mrid1>, <mrid2>,...]}
            where 'functions' can either be a callable or list of
            callables that will accept a list of measurements, and
            'mrids' is a list of mrids to extract from the simulation
            output, which will correspond to measurements.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Assign platform_manager.
        self.platform = platform_manager

        # Assign topic to subscribe to.
        self.output_topic = \
            topics.simulation_output_topic(simulation_id=sim_id)

        self.mrids = []
        self.functions = []
        # Combine the mrids into a list of lists, create a list of
        # functions.
        for d in fn_mrid_list:
            self.mrids.append(d['mrids'])
            self.functions.append(d['functions'])

        # Subscribe to the simulation output.
        self.platform.gad.subscribe(topic=self.output_topic,
                                    callback=self._on_message)

    def _on_message(self, header, message):
        """Callback which is hit each time a new simulation output
        message comes in.
        """
        # Log header time (time sent, I think).
        self.log.debug(
            'Received simulation output, header timestamped '
            + platform_dt(int(header['timestamp'])).strftime(DATE_FORMAT)
        )

        # Log simulation time.
        self.log.debug(
            'Simulation timestamp: '
            + simulation_dt(
                int(message['message']['timestamp'])).strftime(DATE_FORMAT))

        # Filter the message.
        result = self._filter_output_by_mrid(message=message)

        # Iterate over the result, and call each corresponding
        # function.
        for idx, output in enumerate(result):
            # If we have a list of functions:
            try:
                func_iter = iter(self.functions[idx])
            except TypeError:
                # We have a simple function, which isn't iterable. Call
                # it.
                self.functions[idx](output)
            else:
                # Call each function.
                for f in func_iter:
                    f(output)

    def _filter_output_by_mrid(self, message):
        """Given an output message from the simulator, return only the
        measurements with mrid's that we care about.

        :param message: dictionary, message from simulator subscription.
        :returns list of list of measurements from message which
            correspond to the list of lists of mrids in self.mrids.
        """
        # Simply type check for message:
        if not isinstance(message, dict):
            raise TypeError('message must be a dictionary!')

        # Ensure we have measurements.
        try:
            measurements = message['message']['measurements']
        except KeyError:
            raise ValueError('Malformed message input!')

        # If we don't have measurements, we've got a problem.
        if (measurements is None) or (len(measurements) == 0):
            raise ValueError('There are no measurements in the message!')

        # We'll return a list of lists of measurements which
        # corresponds to self.mrids.
        out = [[] for _ in self.mrids]

        # Create a copy of the inputs mrids. Creating a copy so that we can
        # remove objects and thus shrink our list as we iterate to reduce
        # searching. Since self.mrids is a list of lists, we need a
        # deep copy.
        mrids_c = copy.deepcopy(self.mrids)

        # Iterate over the message measurements.
        for meas in measurements:
            # Count empty lists.
            empty_count = 0

            # Iterate over each sub-list of mrids.
            for idx, sub_mrid_list in enumerate(mrids_c):
                # If there are no mrids in our sub list, exit this loop.
                if len(sub_mrid_list) == 0:
                    # Increment the count of empty lists.
                    empty_count += 1
                    # Move to the next iteration of this inner loop.
                    continue

                # Attempt to delete the mrid for this measurement from our list
                # of mrids.
                try:
                    sub_mrid_list.remove(meas['measurement_mrid'])
                except ValueError:
                    # We don't care about this MRID. Move to next object.
                    continue

                # If we're here, we want to keep this measurement.
                out[idx].append(meas)

            # If all our lists are empty, stop iterating and break the
            # outer loop.
            if empty_count == len(mrids_c):
                break

        # If we don't get what we wanted, raise an exception.
        # In the future, we may not want this to be an exception.
        # noinspection PyUnboundLocalVariable
        if empty_count != len(mrids_c):
            # Get a total number of missing MRIDs.
            total = 0
            for sub_list in mrids_c:
                total += len(sub_list)

            m = ('{} MRIDs from {} sub-lists were not present in the '
                 'message').format(total, len(mrids_c) - empty_count)
            raise ValueError(m)

        # All done, return.
        return out


class PlatformManager:
    """Class for interfacing with the GridAPPS-D platform API.

    Note that this class is really intended for subscribing, etc. For
    querying the Blazegraph database (with SPARQL queries), see
    sparql.py
    """

    def __init__(self, timeout=30):
        """Gather environment variables, etc.

        :param timeout: Timeout for GridAPPS-D API requests.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Assign timeout.
        self.timeout = timeout

        # Get GridAPPSD object.
        self.gad = get_gad_object()

        self.log.info('Connected to GridAPPS-D platform.')

        # Initialize property for holding a simulation ID.
        self._sim_id = None

        # # Get information on available models.
        # self.platform_model_info = self.gad_object.query_model_info()
        #
        # # Ensure the query succeeded.
        # if not self.platform_model_info['responseComplete']:
        #     # TODO: Try again?
        #     # TODO: Exception handling, etc.
        #     raise UserWarning('GridAPPS-D query failed.')
        #
        # # Assign model name.
        # self.model_name = model_name
        # # Get the ID for the given model name.
        # self.model_id = self._get_model_id(self.model_name)

        pass

    @property
    def sim_id(self):
        return self._sim_id

    @sim_id.setter
    def sim_id(self, value):
        self._sim_id = value

    def send_command(self, object_ids, attributes, forward_values,
                     reverse_values, sim_id=None):
        """Function for sending a command into a running simulation.
        This is partly a wrapper to DifferenceBuilder, but also sends
        the command into the simulation.

        Note all parameters below must be 1:1. In other words,
        object_id[3] should correspond to attribute[3].

        :param object_ids: List of mrids of objects to command.
        :param attributes: List of CIM attributes belonging to the
            objects. These attributes are what we're actually
            commanding/changing.
        :param forward_values: List of new values for attributes.
        :param reverse_values: List of old (current) values for
            attributes.
        :param sim_id: Simulation ID. If None, will attempt to use
            self.sim_id. If self.sim_id is also None, ValueError will
            be raised.
        """
        # Ensure we get lists.
        if ((not isinstance(object_ids, list))
                or (not isinstance(attributes, list))
                or (not isinstance(forward_values, list))
                or (not isinstance(reverse_values, list))):
            m = 'object_ids, attributes, forward_values, and reverse_values '\
                'must all be lists!'
            raise TypeError(m)

        # Ensure lists are the same length.
        if not (len(object_ids)
                == len(attributes)
                == len(forward_values)
                == len(reverse_values)):
            m = 'object_ids, attributes, forward_values, and reverse_values '\
                'must be the same length!'
            raise ValueError(m)

        # Ensure we have a simulation ID.
        if sim_id is None:
            sim_id = self.sim_id

        if sim_id is None:
            m = 'sim_id input is None, and so is self.sim_id. In order to '\
                'send a command, we must have a simulation ID.'
            raise ValueError(m)

        self.log.debug('Input checks complete for send_command.')

        # Initialize difference builder.
        diff_builder = DiffBuilder(simulation_id=sim_id)
        self.log.debug('DifferenceBuilder initialized.')

        # Iterate to build add differences.
        for k in range(len(object_ids)):
            diff_builder.add_difference(object_id=object_ids[k],
                                        attribute=attributes[k],
                                        forward_value=forward_values[k],
                                        reverse_value=reverse_values[k])

        # Get the message and log it.
        msg = diff_builder.get_message()
        self.log.info('Preparing to send following command: {}'.format(msg))

        # Send command to simulation.
        self.gad.send(topic=topics.simulation_input_topic(sim_id),
                      message=json.dumps(msg))

        # Return the message in case we want to examine it, audit, etc.
        # Mainly useful for testing at this point.
        return msg

    def get_glm(self, model_id):
        """Given a model ID, get a GridLAB-D (.glm) model."""
        payload = {'configurationType': 'GridLAB-D Base GLM',
                   'parameters': {'model_id': model_id}}
        response = self.gad.get_response(topic=topics.CONFIG, message=payload,
                                         timeout=self.timeout)

        self.log.info('GridLAB-D model received from platform.')

        # Fix bad json return.
        # TODO: remove when platform is fixed.
        glm = REGEX_2.sub('', REGEX_1.sub('', response['message']))
        return glm

    def _query_weather(self, start_time, end_time):
        """Private helper for querying weather data."""
        # The weather data API needs microseconds from the epoch as a
        # string. Why this inconsistency? I don't know.
        payload = {'queryMeasurement': 'weather',
                   'queryFilter': {'startTime':
                                       utils.dt_to_us_from_epoch(start_time),
                                   'endTime':
                                       utils.dt_to_us_from_epoch(end_time)},
                   'responseFormat': 'JSON'}

        topic = topics.TIMESERIES
        data = self.gad.get_response(topic=topic, message=payload,
                                     timeout=self.timeout)

        # Check to see if we actually have any data.
        if data['data'] is None:
            raise QueryReturnEmptyError(topic=topic, query=payload)

        return data

    def get_weather(self, start_time, end_time):
        """Helper for querying weather data.

        :param start_time: datetime.datetime object denoting the
            beginning time of weather data to pull.
        :param end_time: "..." end time "..."
        """
        # Check inputs:
        if (not isinstance(start_time, datetime)) \
                or (not isinstance(end_time, datetime)):
            m = 'start_time and end_time must both be datetime.datetime!'
            raise TypeError(m)

        # Query the platform to get the weather data.
        data = self._query_weather(start_time=start_time, end_time=end_time)

        # Parse the weather data.
        data_df = timeseries.parse_weather(data)

        self.log.info(
            'Weather data for {} through {} pulled and parsed.'.format(
                start_time.strftime(DATE_FORMAT),
                end_time.strftime(DATE_FORMAT)
            )
        )
        return data_df

    def get_historic_measurements(self, sim_id, mrid=None):
        """"""
        # TODO: Use topics from gridappsd-python when fixed.
        t = '/queue/goss.gridappsd.process.request.data.timeseries'
        payload = {'queryMeasurement': 'PROVEN_MEASUREMENT',
                   'queryFilter': {'simulation_id': sim_id,
                                   'measurement_mrid': mrid},
                                   #'hasSimulationMessageType': 'OUTPUT'},
                                   #'hasSimulationMessageType': 'INPUT'},
                                   # 'hasMrid': mrid},
                   'responseFormat': 'JSON'}

        return self.gad.get_response(topic=t, message=payload, timeout=30)

    def run_simulation(self):
        """Start a simulation and return the simulation ID.

        TODO: stop hard-coding, take inputs.
        """
        # Hard-code simulation request to start simulation. This was
        # obtained by copy + pasting from the terminal in the viz app.
        geo_name = "_79C9D814-3CE0-DC11-534D-BDA1AF949810"
        subgeo_name = "_A1170111-942A-6ABD-D325-C64886DC4D7D"
        # 13-node:
        # model_mrid = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
        # sim_name = "ieee13nodeckt"
        # 8500 node:
        model_mrid = "_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3"
        sim_name = "ieee8500"
        # 123-node:
        # model_mrid = '_C1C3E687-6FFD-C753-582B-632A27E28507'
        # sim_name = 'ieee123'

        sim_request = \
            {"power_system_config": {
                "GeographicalRegion_name": geo_name,
                "SubGeographicalRegion_name": subgeo_name,
                "Line_name": model_mrid},
                "application_config": {"applications": []},
                "simulation_config":
                    {"start_time": "1560369428",
                     "duration": "5",
                     "simulator": "GridLAB-D",
                     "timestep_frequency": "1000",
                     "timestep_increment": "1000",
                     "run_realtime": True,
                     "simulation_name": sim_name,
                     "power_flow_solver_method": "NR",
                     "model_creation_config": {
                         "load_scaling_factor": "1",
                         "schedule_name": "ieeezipload",
                         "z_fraction": "0", "i_fraction": "1",
                         "p_fraction": "0",
                         "randomize_zipload_fractions": False,
                         "use_houses": False},
                     "startTime": "2019-06-12 12:57:08",
                     "runInRealtime": True,
                     "simulationName": "ieee8500",
                     "modelCreationConfig": {
                         "load_scaling_factor": "1",
                         "schedule_name": "ieeezipload",
                         "z_fraction": "0", "i_fraction": "1",
                         "p_fraction": "0",
                         "randomize_zipload_fractions": False,
                         "use_houses": False}},
                "test_config": {"events": [],
                                "appId":
                                    "_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3"}}

        # Run simulation.
        sim_id = self.gad.get_response(topic=topics.REQUEST_SIMULATION,
                                       message=json.dumps(sim_request))

        return sim_id['simulationId']


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class QueryReturnEmptyError(Error):
    """Raised if a platform query for data returns empty.

    Attributes:
        topic -- Topic query was executed on.
        query -- Given query.
        message -- explanation of the error.
    """

    def __init__(self, topic, query):
        self.topic = topic
        self.query = query
        self.message = 'Query on topic {} returned no data! Query: {}'.format(
            self.topic, self.query
        )


    # def _parse_simulation_request(self, *args, **kwargs):
    #     """Parse request to start a simulation."""
    #     print('_parse_simulation_request has been called!', flush=True)
    #     pass
    #
    # def _get_model_id(self, model_name):
    #     """Given a model's name, get it's ID."""
    #     # Loop over the models until we find our model_name, and get its ID.
    #     model_id = None
    #     for model in self.platform_model_info['data']['models']:
    #         if model['modelName'] == model_name:
    #             model_id = model['modelId']
    #
    #     # Raise exception if the model_id could not be found.
    #     # TODO: Exception management.
    #     if model_id is None:
    #         m = 'Could not find the model ID for {}.'.format(model_name)
    #         raise UserWarning(m)
    #
    #     return model_id
#
#
# if __name__ == '__main__':
#     # Get host IP address. NOTE: This environment variable must be set
#     # when pyvvo is not being run from within the GridAPPS-D platform
#     # via docker-compose.
#     HOST_IP = os.environ['host_ip']
#     PLATFORM = os.environ['platform']
#
#     # Get information related to the platform.
#     PORT = os.environ['GRIDAPPSD_PORT']
#
#     # Create a PlatformManager object, which connects to the platform.
#     mgr = PlatformManager(platform=PLATFORM, model_name='ieee8500')
#
#     # Get model information.
#     info = mgr.gad.query_model_info()
#
#
#     print('stuff cause debugger is being shitty.')
#     # Get the platform status.
#     # msg = mgr.gad.get_platform_status()
#
#     # Request a simulation.
#     # sim_topic = gad_utils.REQUEST_SIMULATION
#     # print('yay')
#
#     pass
