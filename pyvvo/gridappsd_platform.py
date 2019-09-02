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
import copy
import time
from threading import Lock
from functools import wraps

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

# Timeout for SimOutRouter's queue (seconds)
LOCK_TIMEOUT = 60


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


def wait_for_lock(method):
    """Wrapper used by the SimOutRouter to avoid collisions due to
    multi-threading.

    https://stackoverflow.com/a/36944992/11052174
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Block.
        acquired = self._lock.acquire(blocking=True, timeout=LOCK_TIMEOUT)

        if not acquired:
            raise LockTimeoutError('Failed to acquire lock within {} seconds'
                                   .format(LOCK_TIMEOUT))

        # Execute the method
        try:
            result = method(self, *args, **kwargs)
        finally:
            # Always indicate we're done.
            self._lock.release()

        return result

    return wrapper


class SimOutRouter:
    """Class for listening and routing simulation output."""

    def __init__(self, platform_manager, sim_id, fn_mrid_list):
        """

        :param platform_manager: Initialized PlatformManager object.
        :param sim_id: Simulation ID on which to listen to.
        :param fn_mrid_list: list of dictionaries of the form
            {'functions': <function>, 'mrids': [<mrid1>, <mrid2>,...],
            'kwargs': {'dict': 'of keyword args',
                       'to': 'pass to function'}}
            where 'functions' can either be a callable or list of
            callables that will accept both a list of measurements as
            a positional argument and 'sim_dt' (of type
            datetime.datetime) as a keyword argument. The 'mrids' field
            is a list of mrids to extract from the simulation output,
            which will correspond to measurements. 'kwargs' is optional,
            and consists of key word arguments to pass to the function.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Assign platform_manager.
        self.platform = platform_manager

        # Assign topic to subscribe to.
        self.output_topic = \
            topics.simulation_output_topic(simulation_id=sim_id)

        # Initialize lists for holding our mrids, functions, and kwargs.
        self.mrids = []
        self.functions = []
        self.kwargs = []

        # Initialize a lock
        self._lock = Lock()

        # Add the functions and mrids to our lists.
        self.add_funcs_and_mrids(fn_mrid_list=fn_mrid_list)

        # Subscribe to the simulation output.
        self.platform.gad.subscribe(topic=self.output_topic,
                                    callback=self._on_message)

    @wait_for_lock
    def add_funcs_and_mrids(self, fn_mrid_list):
        """Helper to add functions and MRIDs to the router.

        :param fn_mrid_list: See description in __init__.
        """
        # Combine the mrids into a list of lists, create a list of
        # functions.
        for d in fn_mrid_list:
            self.mrids.append(d['mrids'])
            self.functions.append(d['functions'])

            try:
                self.kwargs.append(d['kwargs'])
            except KeyError:
                # Not given kwargs, so no worries.
                self.kwargs.append({})

    @wait_for_lock
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
                self.functions[idx](output, sim_dt=sim_dt, **self.kwargs[idx])
            else:
                # Call each function.
                for f in func_iter:
                    f(output, sim_dt=sim_dt, **self.kwargs[idx])

    def _filter_output_by_mrid(self, message):
        """Given an output message from the simulator, return only the
        measurements with mrid's that we care about.

        :param message: dictionary, message from simulator subscription.
        :returns: list of list of measurements from message which
            correspond to the list of lists of mrids in self.mrids.
        """
        # Simple type check for message:
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
        """Connect to the GridAPPS-D platform by initializing a
        gridappsd.GridAPPSD object.

        :param timeout: Timeout for GridAPPS-D API requests.

        :returns: Initialized PlatformManager object.
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Assign timeout.
        self.timeout = timeout

        # Get GridAPPSD object.
        self.gad = get_gad_object()

        self.log.info('Connected to GridAPPS-D platform.')

        # Initialize property for holding a Simulation object from
        # GridAPPS-D.
        self.sim = None

        # We'll also use a property for tracking if a simulation is
        # complete.
        self.sim_complete = None

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
            try:
                sim_id = self.sim.simulation_id
            except AttributeError:
                m = ('sim_id input is None, and so is self.sim.simulation_id. '
                     'In order to send a command, we must have a sim_id.')
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

    def _query_simulation_output(self, simulation_id,
                                 query_measurement='simulation',
                                 starttime=None, endtime=None,
                                 measurement_mrid=None):
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

        if (measurement_mrid is not None) and \
                not isinstance(measurement_mrid, str):
            raise TypeError('measurement_mrid must be a string.')

        # Initialize the filter dictionary.
        filter_dict = {'simulation_id': simulation_id}

        # Add filters if given.
        if query_measurement == 'simulation':
            filter_dict['hasSimulationMessageType'] = 'OUTPUT'
        elif query_measurement != 'gridappsd-sensor-simulator':
            raise ValueError("query_measurement must be 'simulation' or "
                             "'gridappsd-sensor-simulator'")

        if starttime is not None:
            filter_dict['starttime'] = utils.dt_to_us_from_epoch(starttime)

        if endtime is not None:
            filter_dict['endtime'] = utils.dt_to_us_from_epoch(endtime)

        if measurement_mrid is not None:
            filter_dict['measurement_mrid'] = measurement_mrid

        # Construct the full payload.
        payload = {'queryMeasurement': query_measurement,
                   'queryFilter': filter_dict,
                   'responseFormat': 'JSON'}

        return self.gad.get_response(topic=topics.TIMESERIES,
                                     message=payload, timeout=30)

    def get_simulation_output(self, simulation_id,
                              query_measurement='simulation', starttime=None,
                              endtime=None, measurement_mrid=None):
        """Simple wrapper to call _query_simulation_output and then
            parse and return the results. See the docstring of
            _query_simulation_output for details on inputs.
        """
        # Query the timeseries database.
        data = \
            self._query_simulation_output(
                simulation_id=simulation_id,
                query_measurement=query_measurement, starttime=starttime,
                endtime=endtime, measurement_mrid=measurement_mrid)

        # Parse the result and return.
        return timeseries.parse_timeseries(data)

    def run_simulation(self, feeder_id, start_time, duration, realtime,
                       applications=None, random_zip=False,
                       houses=False):
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
        """
        if applications is None:
            applications = []

        # Hard-code simulation request to start simulation. This was
        # obtained by copy + pasting from the terminal in the viz app.
        geo_name = "_73C512BD-7249-4F50-50DA-D93849B89C43"
        subgeo_name = "_A1170111-942A-6ABD-D325-C64886DC4D7D"

        run_config = \
            {"power_system_config": {
                "GeographicalRegion_name": geo_name,
                "SubGeographicalRegion_name": subgeo_name,
                "Line_name": feeder_id},
                "application_config": {"applications": applications},
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
                        "z_fraction": "0", "i_fraction": "1",
                        "p_fraction": "0",
                        "randomize_zipload_fractions": random_zip,
                        "use_houses": houses}},
                "test_config": {"events": [],
                                "appId": feeder_id}}

        # Simulation is not complete yet.
        self.sim_complete = False

        # Create simulation object.
        self.sim = simulation.Simulation(gapps=self.gad,
                                         run_config=run_config)

        # Add a callback to update sim_complete.
        self.sim.add_oncomplete_callback(self._update_sim_complete)

        # Log.
        self.log.info('Starting simulation.')

        # Start the simulation.
        self.sim.start_simulation()

        # Return the simulation ID.
        return self.sim.simulation_id

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
        while not self.sim_complete:
            time.sleep(0.1)

        # Update sim_complete indicating there's not an active
        # simulation.
        self.sim_complete = None

        # All done!


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


class LockTimeoutError(Error):
    """Raised if a call to a threading.Lock object's acquire method
    time out.
    """
    pass
