"""Main module for running the pyvvo application."""
from pyvvo import sparql
from pyvvo.glm import GLMManager
from pyvvo.gridappsd_platform import PlatformManager, SimOutRouter
from pyvvo import equipment, ga, utils
from gridappsd import topics
from datetime import datetime, timedelta
import simplejson as json
import time
import logging

# Setup log.
LOG = logging.getLogger(__name__)


def main(sim_id, sim_request):
    LOG.debug("Simulation ID: {}".format(sim_id))
    LOG.debug("Simulation Request:\n{}".format(json.dumps(sim_request,
                                                          indent=2)))

    # Extract the feeder_mrid from the simulation request.
    feeder_mrid = sim_request["power_system_config"]["Line_name"]
    LOG.debug("Feeder MRID extracted from simulation request.")

    # Get a SPARQL manager.
    sparql_mgr = sparql.SPARQLManager(feeder_mrid=feeder_mrid)

    # Get a platform manager
    platform = PlatformManager()

    # Extract dates from the simulation request.
    start_seconds = int(sim_request["simulation_config"]["start_time"])
    duration = int(sim_request["simulation_config"]["duration"])
    LOG.debug("Simulation start time and duration extracted from simulation "
              "request.")

    # Convert times to datetime.
    # TODO: Add information indicating this is UTC.
    starttime = datetime.fromtimestamp(start_seconds)
    stoptime = starttime + timedelta(seconds=duration)
    LOG.debug("starttime: {}, stoptime: {}".format(starttime, stoptime))

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

    # Get inverter information.
    inverter_df = sparql_mgr.query_inverters()
    inverter_meas = sparql_mgr.query_inverter_measurements()
    inverter_meas_mrid = \
        list(inverter_meas[sparql.INVERTER_MEAS_MEAS_MRID_COL])
    inverter_objects = equipment.initialize_inverters(inverter_df)
    inverter_mgr = equipment.InverterEquipmentManager(
        eq_dict=inverter_objects, eq_meas=inverter_meas,
        meas_mrid_col=sparql.INVERTER_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.INVERTER_MEAS_INV_MRID_COL)

    # Get list of dictionaries for routing output.
    fn_mrid_list = [
        {'functions': reg_mgr.update_state, 'mrids': reg_meas_mrid},
        {'functions': cap_mgr.update_state, 'mrids': cap_meas_mrid},
        {'functions': switch_mgr.update_state, 'mrids': switch_meas_mrid},
        {'functions': inverter_mgr.update_state, 'mrids': inverter_meas_mrid}
    ]

    # Create a SimOutRouter to listen to simulation outputs.
    router = SimOutRouter(platform_manager=platform, sim_id=sim_id,
                          fn_mrid_list=fn_mrid_list)

    # Get EnergyConsumer (load) data.
    load_nom_v = sparql_mgr.query_load_nominal_voltage()
    load_meas = sparql_mgr.query_load_measurements()

    meas_id = load_meas.iloc[0]['id']

    # Get substation data.
    substation = sparql_mgr.query_substation_source()
    substation_bus_meas = sparql_mgr.query_measurements_for_bus(
        bus_mrid=substation.iloc[0]['bus_mrid'])

    # Get model, instantiate GLMManager.
    model = platform.get_glm(model_id=feeder_mrid)
    glm_mgr = GLMManager(model=model, model_is_path=False)

    # Remove PV and ensure inverters have a power supply.
    glm_mgr.remove_all_solar()
    glm_mgr.set_inverter_v_and_i()
    # TODO: Ensure all inverters are in the right mode and set to be
    #   online.
    # Update the inverters in the model.
    _update_inverter_state_in_glm(glm_mgr=glm_mgr, inverters=inverter_objects)

    # Run the genetic algorithm.
    # TODO: Manage times in a better way.
    ga_stop = (starttime
               + timedelta(seconds=ga.CONFIG["ga"]["intervals"]["model_run"]))
    ga_mgr = ga.GA(regulators=reg_objects, capacitors=cap_objects,
                   starttime=starttime, stoptime=ga_stop)
    ga_mgr.run(glm_mgr=glm_mgr)

    # Wait for the genetic algorithm to complete.
    ga_mgr.wait()

    # Extract equipment settings.
    reg_forward = ga_mgr.regulators
    cap_forward = ga_mgr.capacitors

    # Get the commands.
    reg_cmd = reg_mgr.build_equipment_commands(reg_forward)
    cap_cmd = cap_mgr.build_equipment_commands(cap_forward)

    # Send 'em!
    platform.send_command(sim_id=sim_id, **reg_cmd)
    platform.send_command(sim_id=sim_id, **cap_cmd)
    LOG.info('Commands sent in.')

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
    #
    # # Run a simulation.
    # sim_id = platform.run_simulation()
    #
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


def _update_inverter_state_in_glm(glm_mgr: GLMManager, inverters):
    """Given a GLMManager and InverterSinglePhase objects, update the
    inverter P and Q values in the model based on the current state of
    the InverterSinglePhase objects.

    Eventually, the genetic algorithm should be allowed to set inverter
    set points, which will make this method obsolete.

    This function feels like maybe it should go somewhere else, but at
    the same time putting it here helps keep the proper levels of
    abstraction in the other modules.

    Also note that this method wouldn't make much sense if the inverters
    were being "driven" by solar panels in PyVVO's internal model.
    However, we're stripping out all the solar objects so that the
    inverter output is simply constant. Future work should leave the
    solar objects in, and have the panels driven by weather, where the
    weather is created by some sort of forecast for the upcoming
    interval.

    :param glm_mgr: glm.GLMManager
    :param inverters: Dictionary of equipment.InverterSinglePhase
        objects and dictionaries of equipment.InverterSinglePhase
        objects as is returned by equipment.initialize_inverters.

    :returns: None. The GLMManager is updated directly.
    """
    # Loop over the inverters/dicts of inverters.
    for inv_or_dict in inverters.values():
        # Dictionary implies three phase.
        if isinstance(inv_or_dict, dict):
            # Loop over the phases and aggregate p and q.
            p = 0
            q = 0

            for phase_inv in inv_or_dict.values():
                p += phase_inv.p
                q += phase_inv.q

            # Grab the name from the last inverter.
            # noinspection PyUnboundLocalVariable
            inv_name = phase_inv.name

        elif isinstance(inv_or_dict, equipment.InverterSinglePhase):
            # Here we just have a single phase inverter.
            p = inv_or_dict.p
            q = inv_or_dict.q
            inv_name = inv_or_dict.name
        else:
            raise ValueError('Unexpected item: {}'.format(inv_or_dict))

        # We've got P, Q, and the name. What we don't know is whether
        # the inverter is associated with PV or a battery, for which the
        # platform uses different name prefixes. Start with PV, since
        # that's more common than batteries.
        name_pv = ga.cim_to_glm_name(prefix=ga.INVERTER_PV_PREFIX,
                                     cim_name=inv_name)
        inv_dict = {'object': 'inverter', 'P_Out': p, 'Q_Out': q,
                    'name': name_pv}

        try:
            glm_mgr.modify_item(inv_dict)
        except KeyError:
            # Try again with the battery prefix.
            name_bat = ga.cim_to_glm_name(
                prefix=ga.INVERTER_BAT_PREFIX, cim_name=inv_name)

            # Completely reconstruct the inverter dict since the
            # modify_item method modifies the input dictionary.
            inv_dict = {'object': 'inverter', 'P_Out': p, 'Q_Out': q,
                        'name': name_bat}

            try:
                glm_mgr.modify_item(inv_dict)
            except KeyError:
                # TODO: Should we raise an exception?
                m = ('When attempting to update the .glm with inverter power '
                     f'measurements, neither {name_pv} nor {name_bat} could '
                     'be found in the model. They have thus not been updated.')
                LOG.error(m)

    LOG.info('All inverters in the .glm have been updated with the current '
             'inverter state.')


if __name__ == '__main__':
    pass
    # Use crappy names to avoid scope name overshadowing.
    # p = PlatformManager()
    # s = datetime(2013, 1, 14, 0, 0)
    # d = 1200
    # sid = p.run_simulation(
    #     feeder_id='_AAE94E4A-2465-6F5E-37B1-3E72183A4E44',
    #     start_time=s, duration=d, realtime=True
    # )
    #
    # # Do some crude sleeping to avoid timeouts later, since the platform
    # # takes forever and a day to start a simulation.
    # time.sleep(30)
    # main(sim_id=sid, sim_request=p.last_sim_config)
