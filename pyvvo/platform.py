"""Module for interfacing with the GridAPPS-D platform.

Note this module depends on the GridAPPS-D platform being up and
running.
"""
from gridappsd import GridAPPSD, topics, difference_builder
from gridappsd import utils as gad_utils

import simplejson as json
import os
import logging

# Setup log.
LOG = logging.getLogger(__name__)


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

    return platform


def get_gad_object(**kwargs):
    """Helper to get a GridAPPSD object.

    :param **kwargs: Passed directly to the GridAPPSD constructor.
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


# class PlatformManager:
#     """Class for interfacing with the GridAPPS-D platform."""
#
#     def __init__(self, platform, model_name):
#         """Gather environment variables, etc.
#
#         :param platform: 1/0 whether or not application is running
#             inside the platform.
#         """
#         # Setup logging.
#         self.log = logging.getLogger(__name__)
#
#         # Assign platform input.
#         self.platform = platform
#
#         # Get GridAPPSD object.
#         self.gad = get_gad_object(self.platform)
#
#         self.log.info('Connected to GridAPPS-D platform.')
#
#         # # If running outside the platform, listen for a simulation
#         # # request.
#         # if not self.platform:
#         #     self.gad_object.subscribe(topics.LOGS,
#         #                               callback=self._parse_simulation_request)
#
#         # # Get information on available models.
#         # self.platform_model_info = self.gad_object.query_model_info()
#         #
#         # # Ensure the query succeeded.
#         # if not self.platform_model_info['responseComplete']:
#         #     # TODO: Try again?
#         #     # TODO: Exception handling, etc.
#         #     raise UserWarning('GridAPPS-D query failed.')
#         #
#         # # Assign model name.
#         # self.model_name = model_name
#         # # Get the ID for the given model name.
#         # self.model_id = self._get_model_id(self.model_name)
#
#         pass
#
#     def _parse_simulation_request(self, *args, **kwargs):
#         """Parse request to start a simulation."""
#         print('_parse_simulation_request has been called!', flush=True)
#         pass
#
#     def _get_model_id(self, model_name):
#         """Given a model's name, get it's ID."""
#         # Loop over the models until we find our model_name, and get its ID.
#         model_id = None
#         for model in self.platform_model_info['data']['models']:
#             if model['modelName'] == model_name:
#                 model_id = model['modelId']
#
#         # Raise exception if the model_id could not be found.
#         # TODO: Exception management.
#         if model_id is None:
#             m = 'Could not find the model ID for {}.'.format(model_name)
#             raise UserWarning(m)
#
#         return model_id
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
#     # Hard-code simulation request to start simulation. This was
#     # obtained by copy + pasting from the terminal in the viz app.
#     geo_name = "_24809814-4EC6-29D2-B509-7F8BFB646437"
#     subgeo_name = "_1CD7D2EE-3C91-3248-5662-A43EFEFAC224"
#     model_mrid = "_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3"
#     sim_request = \
#         {
#             "power_system_config": {
#                 "GeographicalRegion_name": geo_name,
#                 "SubGeographicalRegion_name": subgeo_name,
#                 "Line_name": model_mrid
#             },
#             "application_config": {"applications": []},
#             "simulation_config": {
#                 "start_time": "1248152400",
#                 "duration": "2",
#                 "simulator": "GridLAB-D",
#                 "timestep_frequency": "1000",
#                 "timestep_increment": "1000",
#                 "run_realtime": True,
#                 "simulation_name": "ieee8500",
#                 "power_flow_solver_method": "NR",
#                 "model_creation_config": {
#                     "load_scaling_factor": "1",
#                     "schedule_name": "ieeezipload",
#                     "z_fraction": "0",
#                     "i_fraction": "1",
#                     "p_fraction": "0",
#                     "randomize_zipload_fractions": False,
#                     "use_houses": False
#                 }
#             }
#         }
#
#     # Run simulation.
#     # sim_response = \
#     #     mgr.gad.get_response(topic=topics.REQUEST_SIMULATION,
#     #                          message=json.dumps(sim_request))
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
