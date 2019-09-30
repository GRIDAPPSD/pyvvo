"""Main module for running the pyvvo application."""
from pyvvo import sparql
from pyvvo.glm import GLMManager
from pyvvo.gridappsd_platform import PlatformManager, SimOutRouter
from pyvvo import equipment
from pyvvo import ga, utils
from gridappsd import topics
from datetime import datetime
import simplejson as json
import time
import logging

# Setup log.
LOG = logging.getLogger(__name__)


def main(sim_id, sim_request):
    # Extract the feeder_mrid from the simulation request.
    feeder_mrid = sim_request["power_system_config"]["Line_name"]

    # # Determine whether we're running inside or outside the platform.
    # PLATFORM = os.environ['platform']
    #
    # Hard-code 8500 node MRID for now.
    # feeder_mrid = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3'
    # 9500
    # feeder_mrid = '_AAE94E4A-2465-6F5E-37B1-3E72183A4E44'
    # IEEE 13 bus
    # feeder_mrid = '_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62'
    # IEEE 123 bus
    # feeder_mrid = '_C1C3E687-6FFD-C753-582B-632A27E28507'

    # Get a SPARQL manager.
    sparql_mgr = sparql.SPARQLManager(feeder_mrid=feeder_mrid)
    # houses = sparql_mgr.query_houses()

    # Get a platform manager
    platform = PlatformManager()

    # print('hooray')
    #
    # # m = platform.get_historic_measurements(sim_id=sim_id, mrid=None)
    #
    # Hard-code some dates to work with.
    starttime = datetime(2013, 1, 14, 0, 0)
    stoptime = datetime(2013, 1, 14, 0, 1)

    # sim_id = platform.run_simulation(feeder_id=feeder_mrid,
    #                                  start_time=starttime,
    #                                  duration=20, realtime=False)

    # sim_id = '1524166554'
    # ####################################################################
    # # GET PREREQUISITE DATA
    # ####################################################################
    #
    # TODO: Dispatch these jobs to threads.
    # Get regulator information.
    reg_df = sparql_mgr.query_regulators()
    reg_objects = equipment.initialize_regulators(reg_df)
    reg_meas = sparql_mgr.query_rtc_measurements()
    reg_meas_mrid = list(reg_meas[sparql.REG_MEAS_MEAS_MRID_COL])
    reg_mgr = equipment.EquipmentManager(
        eq_dict=reg_objects, eq_meas=reg_meas,
        meas_mrid_col=sparql.REG_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.REG_MEAS_REG_MRID_COL)

    # Get capacitor information.
    cap_df = sparql_mgr.query_capacitors()
    cap_objects = equipment.initialize_capacitors(cap_df)
    cap_meas = sparql_mgr.query_capacitor_measurements()
    cap_meas_mrid = list(cap_meas[sparql.CAP_MEAS_MEAS_MRID_COL])
    cap_mgr = equipment.EquipmentManager(
        eq_dict=cap_objects, eq_meas=cap_meas,
        meas_mrid_col=sparql.CAP_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.CAP_MEAS_CAP_MRID_COL)

    # Get switch information.
    switch_df = sparql_mgr.query_switches()
    switch_objects = equipment.initialize_switches(switch_df)
    # TODO: Uncomment below when the following is resolved:
    # https://github.com/GRIDAPPSD/GOSS-GridAPPS-D/issues/969

    switch_meas = sparql_mgr.query_switch_measurements()
    switch_meas_mrid = list(switch_meas[sparql.SWITCH_MEAS_MEAS_MRID_COL])
    switch_mgr = equipment.EquipmentManager(
        eq_dict=switch_objects, eq_meas=switch_meas,
        meas_mrid_col=sparql.SWITCH_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.SWITCH_MEAS_SWITCH_MRID_COL
    )

    # Get EnergyConsumer (load) data.
    load_nom_v = sparql_mgr.query_load_nominal_voltage()
    load_meas = sparql_mgr.query_load_measurements()

    meas_id = load_meas.iloc[0]['id']

    # Get substation data.
    substation = sparql_mgr.query_substation_source()
    substation_bus_meas = sparql_mgr.query_measurements_for_bus(
        bus_mrid=substation.iloc[0]['bus_mrid'])

    # Get model, make it runnable.
    model = platform.get_glm(model_id=feeder_mrid)
    glm = GLMManager(model=model, model_is_path=False)
    model_start = datetime(2013, 4, 1, 12, 0)
    model_end = datetime(2013, 4, 1, 12, 5)

    # glm.add_run_components(starttime=model_start, stoptime=model_end)
    # glm.write_model(out_path='8500.glm')
    # result = utils.run_gld('8500.glm')
    # import timeit
    # from copy import deepcopy
    # t1 = timeit.timeit("glm = GLMManager(model=model, model_is_path=False)",
    #                    globals=globals(), number=100)
    # print(t1)
    # t2 = timeit.timeit("new_glm = deepcopy(glm)", globals=globals(), number=100)
    # print(t2)
    # # Timing results:
    # # 65.02 vs. 23.90

    # # Get historic load measurement data.
    # # TODO.
    #
    # # Get historic weather.
    # weather = platform.get_weather(start_time=starttime, end_time=stoptime)
    #
    # # Get current weather.
    # # TODO - what's the way forward here? Probably just query the
    # #   database.
    #
    # # time.sleep(10)
    #
    # # Doesn't work with mrid filter...
    # # meas = platform.get_historic_measurements(
    # #     sim_id=sim_id, mrid='_fe0a57e7-573c-47a2-ba0b-23c289f39594')
    #
    # # platform.gad.unsubscribe(sub_id)
    # # print('unsubscribed!', flush=True)
    # #
    #
    # # Get list of dictionaries for routing output.
    # fn_mrid_list = [
    #     {'functions': reg_mgr.update_state, 'mrids': reg_meas_mrid},
    #     {'functions': cap_mgr.update_state, 'mrids': cap_meas_mrid},
    #     # {'functions': switch_mgr.update_state, 'mrids': switch_meas_mrid}
    # ]
    #
    # # Run a simulation.
    # sim_id = platform.run_simulation()
    #
    # # Create a SimOutRouter to listen to simulation outputs.
    # router = SimOutRouter(platform_manager=platform, sim_id=sim_id,
    #                       fn_mrid_list=fn_mrid_list)
    #
    # # # Send commands to running simulation. Command all regulators to
    # # # their maximum.
    # # reg_ids = []
    # # reg_attr = []
    # # reg_forward = []
    # # reg_reverse = []
    # #
    # # # Loop over controllable regulators.
    # # for reg_name, multi_reg in c_regs.items():
    # #     # Loop over the phases in the regulator.
    # #     for p in multi_reg.PHASES:
    # #         # Get the single phase regulator.
    # #         single_reg = getattr(multi_reg, p)
    # #
    # #         # Move along if its None.
    # #         if single_reg is None:
    # #             continue
    # #
    # #         # Add the tap change mrid.
    # #         reg_ids.append(single_reg.tap_changer_mrid)
    # #         # Add the attribute.
    # #         reg_attr.append('TapChanger.step')
    # #         # Hard-code the forward value.
    # #         reg_forward.append(16)
    # #         # Grab the reverse from the current tap_pos.
    # #         # TODO: Need general solution for going from numpy to json.
    # #         reg_reverse.append(int(single_reg.tap_pos))
    # #
    # # # Send in the command.
    # # time.sleep(10)
    # # platform.send_command(object_ids=reg_ids, attributes=reg_attr,
    # #                       forward_values=reg_forward,
    # #                       reverse_values=reg_reverse, sim_id=sim_id)
    #
    # print('hooray')
