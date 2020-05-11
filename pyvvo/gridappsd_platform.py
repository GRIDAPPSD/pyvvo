"""Module for interfacing with the GridAPPS-D platform.

Note this module depends on the GridAPPS-D platform being up and
running.
"""
from gridappsd import GridAPPSD, topics, simulation
from gridappsd.difference_builder import DifferenceBuilder as DiffBuilder
from gridappsd import utils as gad_utils

import simplejson as json
import os
import logging
import re
from datetime import datetime
import time
from threading import Lock
import weakref
from typing import List, Union

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
# noinspection RegExpRedundantEscape
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
            {'function': <function>, 'mrids': [<mrid1>, <mrid2>,...],
            'kwargs': {'dict': 'of keyword args',
                       'to': 'pass to function'}}
            where 'function' can either be a function or class method
            that will accept both a list of measurements as a positional
            argument and 'sim_dt' (of type datetime.datetime) as a
            keyword argument. The 'mrids' field is a list of mrids to
            extract from the simulation output, which will correspond to
            measurements. 'kwargs' is optional, and consists of key word
            arguments to pass to the function.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Assign platform_manager.
        self.platform = platform_manager

        # Assign topic to subscribe to.
        self.output_topic = \
            topics.simulation_output_topic(simulation_id=sim_id)

        # Initialize list for holding our mrids, functions, and kwargs.
        self.mrid_fn_kw_list = []

        # Initialize a lock
        self._lock = Lock()

        # Add the functions and mrids to our lists.
        self.add_funcs_and_mrids(fn_mrid_list=fn_mrid_list)

        # Subscribe to the simulation output.
        self.platform.gad.subscribe(topic=self.output_topic,
                                    callback=self._on_message)

    @utils.wait_for_lock
    def add_funcs_and_mrids(self, fn_mrid_list):
        """Helper to add functions and MRIDs to the router.
        Implementation detail: functions will be stored as weak
        references.

        :param fn_mrid_list: See description in __init__.
        """
        # Combine the mrids into a list of lists, create a list of
        # functions/methods stored as weak references.
        for d in fn_mrid_list:
            # Initialize new dictionary.
            nd = {'mrids': (d['mrids'])}

            try:
                # Attempt to use a WeakMethod. Will throw a type error
                # if given a regular function.
                fn = weakref.WeakMethod(d['function'])
            except TypeError:
                # Use a simple ref.
                fn = weakref.ref(d['function'])

            # Add reference to the dictionary.
            nd['fn'] = fn

            # Extract keyword arguments if present.
            try:
                kw = d['kwargs']
            except KeyError:
                # Not given kwargs, so no worries.
                kw = {}

            # Add kw args to the dictionary.
            nd['kwargs'] = kw

            # Add dictionary to the list.
            self.mrid_fn_kw_list.append(nd)

    def _prune(self):
        """Ditch elements in self.mrid_fn_kw_list which no longer have
        a strong reference to a function.
        """
        self.mrid_fn_kw_list = \
            [d for d in self.mrid_fn_kw_list if d['fn']() is not None]

    @utils.wait_for_lock
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
        sim_dt = simulation_dt(int(message['message']['timestamp']))

        self.log.debug(
            'Simulation timestamp: {}'.format(sim_dt.strftime(DATE_FORMAT)))

        # Ensure we have measurements.
        try:
            measurements = message['message']['measurements']
        except KeyError:
            raise ValueError('Malformed message input!')

        # If we don't have measurements, we've got a problem.
        if (measurements is None) or (len(measurements) == 0):
            raise ValueError('There are no measurements in the message!')

        # Get rid of functions which have lost their strong references.
        self._prune()

        # Iterate over our list and call functions.
        for d in self.mrid_fn_kw_list:
            # Create list of measurements.
            try:
                meas_list = [measurements[mrid] for mrid in d['mrids']]
            except KeyError:
                # Build up the list without a comprehension and log the
                # missing measurements.
                meas_list = []
                for mrid in d['mrids']:
                    try:
                        meas_list.append(measurements[mrid])
                    except KeyError:
                        self.log.warning(
                            'Expected measurement with MRID '
                            f'{mrid} is missing! Perhaps there is a '
                            'communication outage.')

            # Note the initial () is to extract the reference from the
            # weak reference, and then the second (<stuff>) is to
            # actually call the function.
            d['fn']()(meas_list, sim_dt=sim_dt, **d['kwargs'])


class PlatformManager:
    """Class for interfacing with the GridAPPS-D platform API.

    Note that this class is really intended for subscribing, etc. For
    querying the Blazegraph database (with SPARQL queries), see
    sparql.py
    """

    def __init__(self, timeout=60, stomp_log_level=logging.WARNING,
                 goss_log_level=logging.WARNING):
        """Connect to the GridAPPS-D platform by initializing a
        gridappsd.GridAPPSD object.

        :param timeout: Timeout for GridAPPS-D API requests.
        :param stomp_log_level: Log level for stomp.py.
        :param goss_log_level: Log level for goss.py.

        :returns: Initialized PlatformManager object.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Assign timeout.
        self.timeout = timeout

        # Get GridAPPSD object.
        self.gad = get_gad_object(stomp_log_level=stomp_log_level,
                                  goss_log_level=goss_log_level)

        self.log.info('Connected to GridAPPS-D platform.')

        # Initialize property for holding a Simulation object from
        # GridAPPS-D.
        self.sim = None

        # We'll also use a property for tracking if a simulation is
        # complete.
        self.sim_complete = None

        # For debugging, track the simulation configuration message.
        self.last_sim_config = None

    def send_command(self, object_ids, attributes, forward_values,
                     reverse_values, sim_id=None):
        """Function for sending a command into a running simulation.
        This is partly a wrapper to DifferenceBuilder, but also sends
        the command into the simulation.

        Note all list parameters below must be 1:1. In other words,
        object_ids[3] should correspond to attributes[3], etc.

        :param object_ids: List of mrids of objects to command.
        :param attributes: List of CIM attributes belonging to the
            objects. These attributes are what we're actually
            commanding/changing.
        :param forward_values: List of new values for attributes.
        :param reverse_values: List of old (current) values for
            attributes.
        :param sim_id: Simulation ID. If None, will attempt to use
            self.sim.simulation_id. If that is also None, ValueError
            will be raised.

        :returns: Dictionary representing the message that gets sent in.
            If no message will be sent (if input lists are empty),
            returns None instead.
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

        # Don't bother continuing if the lists are empty.
        if len(object_ids) == 0:
            self.log.info('send_command given empty lists, returning None.')
            return None

        # Ensure we have a simulation ID.
        if sim_id is None:
            try:
                sim_id = self.sim.simulation_id
            except AttributeError:
                m = ('sim_id input is None, and so is self.sim.simulation_id. '
                     'In order to send a command, we must have a sim_id.')
                raise ValueError(m) from None

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
        msg_str = json.dumps(msg)
        self.log.info('Preparing to send following command: {}'
                      .format(msg_str))

        # Send command to simulation.
        self.gad.send(topic=topics.simulation_input_topic(sim_id),
                      message=msg_str)

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
                                       utils.dt_to_s_from_epoch(start_time),
                                   'endTime':
                                       utils.dt_to_s_from_epoch(end_time)},
                   'responseFormat': 'JSON'}

        topic = topics.TIMESERIES
        data = self.gad.get_response(topic=topic, message=payload,
                                     timeout=self.timeout)

        # Check to see if we actually have any data.
        if (data['data'] is None) or (len(data['data']) == 0):
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

    def _query_simulation_output(self, simulation_id,
                                 query_measurement='simulation',
                                 starttime=None, endtime=None,
                                 measurement_mrid:
                                 Union[List[str], str, None] = None):
        """Get simulation/sensor service output from the platform.
        https://gridappsd.readthedocs.io/en/latest/using_gridappsd/index.html#timeseries-api

        Note this method is SPECIFICALLY for getting simulation OUTPUT,
        not INPUT.

        Also note the platform provides a couple more filters, but at
        the time of writing they are literally useless.

        :param simulation_id: Required. ID of the simulation to get
            output for.
        :param query_measurement: Optional, defaults to 'simulation.'
            String, currently valid options are 'simulation' and
            'gridappsd-sensor-simulator'. Note that technically
            'weather' is valid, but we have the "get_weather" method
            for that.
        :param starttime: Optional. Python datetime object for filtering
            data. Data is only grabbed for at or after starttime.
        :param endtime: Optional. Python datetime object for filtering
            data. Data is only grabbed at or before endtime.
        :param measurement_mrid: Optional. String. Only get measurements
            with the given measurement MRID.

        :returns: Messy nested dictionary straight from the platform.
        """
        # Check inputs.
        if not isinstance(simulation_id, str):
            raise TypeError('simulation_id must be a string.')

        if not isinstance(query_measurement, str):
            raise TypeError('query_measurement must be a string.')

        if (starttime is not None) and not isinstance(starttime, datetime):
            raise TypeError('starttime must be datetime.datetime.')

        if (endtime is not None) and not isinstance(endtime, datetime):
            raise TypeError('endtime must be datetime.datetime.')

        # if (measurement_mrid is not None) and \
        #         not isinstance(measurement_mrid, str):
        #     raise TypeError('measurement_mrid must be a string.')

        # Initialize the filter dictionary.
        filter_dict = {'simulation_id': simulation_id}

        # Add filters if given.
        if query_measurement == 'simulation':
            filter_dict['hasSimulationMessageType'] = 'OUTPUT'
        elif query_measurement != 'gridappsd-sensor-simulator':
            raise ValueError("query_measurement must be 'simulation' or "
                             "'gridappsd-sensor-simulator'")

        if starttime is not None:
            filter_dict['starttime'] = utils.dt_to_s_from_epoch(starttime)

        if endtime is not None:
            filter_dict['endtime'] = utils.dt_to_s_from_epoch(endtime)

        if measurement_mrid is not None:
            filter_dict['measurement_mrid'] = measurement_mrid

        # Construct the full payload.
        payload = {'queryMeasurement': query_measurement,
                   'queryFilter': filter_dict,
                   'responseFormat': 'JSON'}

        return self.gad.get_response(topic=topics.TIMESERIES,
                                     message=payload, timeout=30)

    def get_simulation_output(
            self, simulation_id, query_measurement='simulation',
            starttime=None, endtime=None,
            measurement_mrid: Union[List[str], str, None] = None,
            index_by_time=True):
        """Simple wrapper to call _query_simulation_output and then
        parse and return the results. See the docstring of
        _query_simulation_output for details on inputs.

        TODO: document parameters.
            
        :param index_by_time: Passed to timeseries.parse_timeseries.
        """
        # Query the timeseries database.
        data = \
            self._query_simulation_output(
                simulation_id=simulation_id,
                query_measurement=query_measurement, starttime=starttime,
                endtime=endtime, measurement_mrid=measurement_mrid)

        # Parse the result and return.
        return timeseries.parse_timeseries(data, index_by_time)

    def run_simulation(self, feeder_id, start_time, duration, realtime,
                       applications=None, random_zip=False,
                       houses=False, z_fraction=0, i_fraction=1,
                       p_fraction=0, services=None,
                       events=None):
        """Start a simulation and return the simulation ID.

        For now, this is hard-coded to the point where it'll likely only
        work with the 8500 and maybe 9500 node models.

        :param feeder_id: mrid of feeder to run simulation for.
        :param start_time: Python datetime for starting simulation.
        :param duration: Integer. Duration of simulation.
        :param realtime: Boolean. Whether or not to run the simulation
            in real time.
        :param applications: List of dictionaries of applications to
            run. Each application must at least have a 'name' field.
        :param random_zip: Boolean, whether to randomize zip loads.
        :param houses: Boolean, whether or not to use houses in the
            simulation.
        :param z_fraction: Impedance fraction for loads.
        :param i_fraction: Current fraction for loads.
        :param p_fraction: Power fraction for loads.
        :param services: List of dictionaries of services to configure.
            Check out
            https://gridappsd-sensor-simulator.readthedocs.io/en/sensor-service-configuration/.
        :param events: List of events. Each element should be a
            dictionary with the following fields:

            "message": This should contain keys "forward_differences"
                and "reverse_differences" as created by a gridappsd
                DifferenceBuilder.

            "event_type": All I know of is the string
                "ScheduledCommandEvent"

            "occuredDateTime": integer, seconds since the epoch when the
                event will be applied (I suppose?).

            "stopDateTime": same as above, but when the event will be
                undone (I suppose?).
        """
        if applications is None:
            applications = []

        if services is None:
            services = []

        if events is None:
            events = []

        # Hard-code simulation request to start simulation. This was
        # obtained by copy + pasting from the terminal in the viz app.
        geo_name = "_73C512BD-7249-4F50-50DA-D93849B89C43"
        subgeo_name = "_A1170111-942A-6ABD-D325-C64886DC4D7D"

        run_config = {
            "power_system_config": {
                "GeographicalRegion_name": geo_name,
                "SubGeographicalRegion_name": subgeo_name,
                "Line_name": feeder_id
            },
            "application_config": {
                "applications": applications
            },
            "simulation_config": {
                "start_time": utils.dt_to_s_from_epoch(start_time),
                "duration": str(duration),
                "simulator": "GridLAB-D",
                "timestep_frequency": "1000",
                "timestep_increment": "1000",
                "run_realtime": realtime,
                "simulation_name": "ieee8500",
                "power_flow_solver_method": "NR",
                "model_creation_config": {
                    "load_scaling_factor": "1",
                    "schedule_name": "ieeezipload",
                    "z_fraction": str(z_fraction),
                    "i_fraction": str(i_fraction),
                    "p_fraction": str(p_fraction),
                    "randomize_zipload_fractions": random_zip,
                    "use_houses": houses}},
            "test_config": {"events": events, "appId": ""},
            "service_configs": services
        }

        # Simulation is not complete yet.
        self.sim_complete = False

        # Create simulation object.
        self.sim = simulation.Simulation(gapps=self.gad,
                                         run_config=run_config)

        self.last_sim_config = run_config

        # Add a callback to update sim_complete.
        self.sim.add_oncomplete_callback(self._update_sim_complete)

        # Log.
        self.log.info('Starting simulation.')

        # Start the simulation.
        self.sim.start_simulation()

        # Return the simulation ID.
        return self.sim.simulation_id

    # noinspection PyUnusedLocal
    def _update_sim_complete(self, *args):
        self.sim_complete = True
        self.log.info('Simulation complete!')

    def wait_for_simulation(self):
        """Method to block until the current simulation is complete, and
        then update self.sim_complete to be None.

        NOTE: This is not robust, and will hang if a simulation
        crashes.
        """
        # If we don't have a simulation, warn and return.
        if self.sim_complete is None:
            self.log.warning('wait_for_simulation called, but sim_complete '
                             'is None! Doing nothing.')
            return

        # Use a crude while loop and sleep call to wait.
        # TODO: Update to use threading.Event
        while not self.sim_complete:
            time.sleep(0.1)

        # Update sim_complete indicating there's not an active
        # simulation.
        self.sim_complete = None

        # All done!


class SimulationClock:
    """Class for keeping track of a simulation's time as it progresses.
    """

    def __init__(self, gad: GridAPPSD, sim_id: str, sim_start_ts: int,
                 log_interval=60):
        """Initialize attributes, subscribe to the simulation log.

        :param gad: Initialized gridappsd.GridAPPSD object.
        :param sim_id: Simulation ID of the simulation to track.
        :param sim_start_ts: Simulation start timestamp in seconds since
            the epoch.
        :param log_interval: How many simulation seconds in between
            logging the current simulation time.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Simply set the simulation starting time as an attribute.
        self.sim_start_ts = sim_start_ts
        self.log_interval = log_interval

        # Use attributes for tracking the simulation time and current
        # time (as indicated by the most recent message).
        self.sim_time = None
        self.time_step = None
        self.msg_time = None
        self.last_log_time = None

        # Compile a regular expression for extracting the time from the
        # message.
        self.regexp = re.compile('(?:incrementing to )([0-9]+)')

        # Subscribe to the simulation log.
        gad.subscribe(topic=topics.simulation_log_topic(sim_id),
                      callback=self._on_message)

        self.log.info('SimulationClock configured and initialized.')

    # noinspection PyUnusedLocal
    def _on_message(self, headers: dict, message: dict):
        """Callback which will be called upon simulation log messages
        being received.

        :param headers: Message header information. This will not be
            used.
        :param message: Message information.
        """
        # Attempt to extract the fields we care about.
        try:
            source = message['source']
            timestamp = message['timestamp']
            log_msg = message['logMessage']
        except KeyError:
            self.log.error('Incoming message missing one or more of the '
                           'following fields: source, timestamp, logMessage. '
                           'Times have not been updated.')
            return None

        # We only want message from the fncs_goss_bridge.
        if source != 'fncs_goss_bridge.py':
            self.log.debug(f'Ignoring message from {source}.')
            return None

        # See if the log message is as expected.
        match = self.regexp.match(log_msg)
        if match is None:
            # We don't care about this message.
            self.log.debug(f'Ignoring message "{log_msg}."')
            return None

        # The regular expression has been designed such that group 1
        # contains the time increment.
        self.time_step = int(match.group(1))
        self.sim_time = self.sim_start_ts + self.time_step

        # Update message time.
        self.msg_time = timestamp

        # Debug logging.
        self.log.debug(f'Updated sim_time to {self.sim_time} and msg_time to '
                       f'{self.msg_time}.')

        # Info logging every self.log_interval seconds.
        if (self.last_log_time is None) or \
                ((self.sim_time - self.last_log_time) >= self.log_interval):
            self.log.info(f'Simulation time is {self.sim_time}.')
            self.last_log_time = self.sim_time


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
