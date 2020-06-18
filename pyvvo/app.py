"""Main module for running the pyvvo application."""
from pyvvo import sparql
from pyvvo.glm import GLMManager
from pyvvo.gridappsd_platform import PlatformManager, SimOutRouter, \
    SimulationClock
from pyvvo import equipment, ga
from datetime import datetime, timedelta
try:
    import simplejson as json
except ModuleNotFoundError:
    import json
import time
import logging
import dateutil

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
    # duration = int(sim_request["simulation_config"]["duration"])
    LOG.debug("Simulation start time and duration extracted from simulation "
              "request.")

    # Initialize a simulation clock.
    clock = SimulationClock(
        gad=platform.gad, sim_id=sim_id, sim_start_ts=start_seconds,
        log_interval=ga.CONFIG['misc']['clock_log_interval'])

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
        eq_mrid_col=sparql.REG_MEAS_REG_MRID_COL, eq_type='Regulator')

    # Get capacitor information.
    cap_df = sparql_mgr.query_capacitors()
    cap_objects = equipment.initialize_capacitors(cap_df)
    cap_meas = sparql_mgr.query_capacitor_measurements()
    cap_meas_mrid = list(cap_meas[sparql.CAP_MEAS_MEAS_MRID_COL])
    cap_mgr = equipment.EquipmentManager(
        eq_dict=cap_objects, eq_meas=cap_meas,
        meas_mrid_col=sparql.CAP_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.CAP_MEAS_CAP_MRID_COL, eq_type='Capacitor')

    # Get switch information.
    switch_df = sparql_mgr.query_switches()
    switch_objects = equipment.initialize_switches(switch_df)
    switch_meas = sparql_mgr.query_switch_measurements()
    switch_meas_mrid = list(switch_meas[sparql.SWITCH_MEAS_MEAS_MRID_COL])
    switch_mgr = equipment.EquipmentManager(
        eq_dict=switch_objects, eq_meas=switch_meas,
        meas_mrid_col=sparql.SWITCH_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.SWITCH_MEAS_SWITCH_MRID_COL, eq_type='Switch')

    # Get inverter information.
    inverter_df = sparql_mgr.query_inverters()
    inverter_meas = sparql_mgr.query_inverter_measurements()
    inverter_meas_mrid = \
        list(inverter_meas[sparql.INVERTER_MEAS_MEAS_MRID_COL])
    inverter_objects = equipment.initialize_inverters(inverter_df)
    inverter_mgr = equipment.PQEquipmentManager(
        eq_dict=inverter_objects, eq_meas=inverter_meas,
        meas_mrid_col=sparql.INVERTER_MEAS_MEAS_MRID_COL,
        eq_mrid_col=sparql.INVERTER_MEAS_INV_MRID_COL, eq_type='Inverter')

    # Get synchronous machine information.
    machine_df = sparql_mgr.query_synchronous_machines()
    machine_meas = sparql_mgr.query_synchronous_machine_measurements()
    machine_meas_mrid = \
        list(machine_meas[sparql.SYNCH_MACH_MEAS_MEAS_COL])
    machine_objects = equipment.initialize_synchronous_machines(machine_df)
    machine_mgr = equipment.PQEquipmentManager(
        eq_dict=machine_objects, eq_meas=machine_meas,
        meas_mrid_col=sparql.SYNCH_MACH_MEAS_MEAS_COL,
        eq_mrid_col=sparql.SYNCH_MACH_MEAS_MACH_COL,
        eq_type='SynchronousMachine')

    # Get list of dictionaries for routing output.
    fn_mrid_list = [
        {'function': reg_mgr.update_state, 'mrids': reg_meas_mrid},
        {'function': cap_mgr.update_state, 'mrids': cap_meas_mrid},
        {'function': switch_mgr.update_state, 'mrids': switch_meas_mrid},
        {'function': inverter_mgr.update_state, 'mrids': inverter_meas_mrid},
        {'function': machine_mgr.update_state, 'mrids': machine_meas_mrid}
    ]

    # Create a SimOutRouter to listen to simulation outputs.
    # noinspection PyUnusedLocal
    router = SimOutRouter(platform_manager=platform, sim_id=sim_id,
                          fn_mrid_list=fn_mrid_list)

    # Get EnergyConsumer (load) data.
    # noinspection PyUnusedLocal
    load_nom_v = sparql_mgr.query_load_nominal_voltage()
    load_meas = sparql_mgr.query_load_measurements()

    # noinspection PyUnusedLocal
    meas_id = load_meas.iloc[0]['id']

    # Get substation data.
    substation = sparql_mgr.query_substation_source()
    # noinspection PyUnusedLocal
    substation_bus_meas = sparql_mgr.query_measurements_for_bus(
        bus_mrid=substation.iloc[0]['bus_mrid'])

    # Get model, instantiate GLMManager.
    model = platform.get_glm(model_id=feeder_mrid)
    glm_mgr = GLMManager(model=model, model_is_path=False)

    # Tweak the model (one time setup).
    _prep_glm(glm_mgr)

    # Extract the duration for which GridLAB-D models will be run in the
    # genetic algorithm.
    model_run_time = ga.CONFIG["ga"]["intervals"]["model_run"]

    # Turn down inverter logging.
    log_level = 'WARNING'
    inverter_mgr.log.setLevel(log_level)
    LOG.info(
        f'InverterManager log level changed to {log_level} to reduce output.')
    log_level = 'ERROR'
    inverter_mgr.update_equipment_log_level(level=log_level)
    LOG.info(
        f'All individual inverter log levels changed to {log_level} to reduce '
        'output.')

    # Run the genetic algorithm.
    # TODO: Manage loop exit, etc. Should exit when simulation is
    #   complete.
    iterations = 0
    while True:
        LOG.info('*'*200)
        # Update the inverter, switches, and machines in the GridLAB-D
        # model with the current states from the platform.
        _update_glm_inverters_switches_machines(
            glm_mgr, inverter_objects, switch_objects, machine_objects)

        # Get the most recent simulation time from the clock. The
        # platform operates in UTC.
        starttime = datetime.fromtimestamp(clock.sim_time,
                                           tz=dateutil.tz.tzutc())

        # Compute stop time.
        stoptime = starttime + timedelta(seconds=model_run_time)

        LOG.info('Starting genetic algorithm to compute set points for '
                 f'{starttime} through {stoptime}.')

        # Initialize manager for genetic algorithm.
        ga_mgr = ga.GA(regulators=reg_objects, capacitors=cap_objects,
                       starttime=starttime, stoptime=stoptime)

        # Create a GAStopper to ensure that the GA stops if a switch
        # opens.
        # noinspection PyUnusedLocal
        ga_stopper = GAStopper(ga_obj=ga_mgr, eq_mgr=switch_mgr,
                               eq_type='switch')

        # Start the genetic algorithm.
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
        reg_msg = platform.send_command(sim_id=sim_id, **reg_cmd)
        if reg_msg is not None:
            LOG.info('Regulator commands sent in.')

        cap_msg = platform.send_command(sim_id=sim_id, **cap_cmd)
        if cap_msg is not None:
            LOG.info('Capacitor commands sent in.')

        # Verify that the equipment was properly updated. At present,
        # the simulator emits messages every 3 simulation seconds. So,
        # using a wait_duration of 12 will wait 3 time steps. Using a
        # timeout of 5 essentially gives a 2 second grace period for
        # all the processing in between simulation time steps.
        # TODO: regulator and capacitor command verification should be
        #   done concurrently, rather than in series like this.
        # TODO: Attempt to command inoperable equipment to bring it
        #   back into the fold.
        if reg_msg is not None:
            inoperable_regs = _verify_commands(
                mgr=reg_mgr, eq_type='regulator', wait_duration=12, timeout=5)

        if cap_msg is not None:
            inoperable_caps = _verify_commands(
                mgr=cap_mgr, eq_type='capacitor', wait_duration=12, timeout=5)

        iterations += 1

        if (iterations % 5) == 0:
            LOG.warning("I'm tired! I've ran the genetic algorithm "
                        f"{iterations} times! When does it end?")


def _verify_commands(mgr: equipment.EquipmentManager, eq_type: str,
                     wait_duration=12, timeout=5):
    """Helper to verify commands and log results."""
    inoperable_eq = mgr.verify_command(wait_duration=wait_duration,
                                       timeout=timeout)

    if inoperable_eq is None:
        LOG.info(f'Commands for {eq_type}(s) have been confirmed to have '
                 'been successfully carried out in the platform.')
    else:
        state_str_list = []
        equipment.loop_helper(
            eq_dict=inoperable_eq, func=_add_state_string_to_list,
            str_list=state_str_list)
        full_str = '\n'.join(state_str_list)
        LOG.warning(f'The following {eq_type}(s) did not respond to the '
                    f'given commands!\n{full_str}')

    return inoperable_eq


def _add_state_string_to_list(eq: equipment.EquipmentSinglePhase,
                              str_list: list):
    state_str = (f'Name: {eq.name}, Actual State: {eq.state}, Expected State: '
                 + f'{eq.expected_state}.')
    str_list.append(state_str)


def _prep_glm(glm_mgr: GLMManager):
    """Perform all necessary updates to the .glm before running the
    genetic algorithm. This includes:

    - remove solar objects
    - set inverter DC sources so they operate at constant P + Q
    - set switches to have their states listed by phase

    TODO: Ensure all inverters are in the right mode and set to be
        online.

    :param glm_mgr: GLMManager to update.
    :returns: None. glm_mgr is updated in place.
    """
    glm_mgr.remove_all_solar()
    glm_mgr.set_inverter_v_and_i()
    glm_mgr.convert_switch_status_to_three_phase(banked=False)


def _update_glm_inverters_switches_machines(glm_mgr: GLMManager,
                                            inverters, switches, machines):
    _update_inverter_state_in_glm(glm_mgr, inverters)
    _update_switch_state_in_glm(glm_mgr, switches)
    _update_diesel_dg_state_in_glm(glm_mgr, machines)


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


def _update_switch_state_in_glm(glm_mgr: GLMManager, switches):
    """Update our .glm with the current state of all the switches.

    It is assumed this has been called after _prep_glm so that the
    switches in the model have been modified to have each phase's state
    explicitly enumerated.

    :param glm_mgr: GLMManager which will have its switch states
        updated.
    :param switches: Dictionary of SwitchSinglePhase objects and
        dictionaries of SwitchSinglePhase objects as would be returned
        by equipment.initialize_switches.
    """
    # Helper function.
    def add_state(switch, ud):
        if switch.state is None:
            raise ValueError(f'Switch {switch.name} has a state of None!')

        ud[f'phase_{switch.phase}_state'] = switch.GLM_STATES[switch.state]

    # Loop over the switches/dicts of switches.
    for sw_or_dict in switches.values():
        # Initialize dictionary for performing updates.
        update_dict = {'object': 'switch'}

        # Dictionary implies three phase.
        if isinstance(sw_or_dict, dict):
            # Loop over the phases.
            for sw in sw_or_dict.values():
                add_state(sw, update_dict)

        elif isinstance(sw_or_dict, equipment.SwitchSinglePhase):
            sw = sw_or_dict
            add_state(sw, update_dict)
        else:
            raise TypeError('Unexpected type from {}.'.format(sw_or_dict))

        # Get the switch name.
        # noinspection PyUnboundLocalVariable
        name = ga.cim_to_glm_name(prefix=ga.SWITCH_PREFIX, cim_name=sw.name)
        update_dict['name'] = name

        # Update!
        try:
            glm_mgr.modify_item(update_dict)
        except KeyError:
            # TODO: Should we raise an exception?
            m = (f"The switch {name} could not be found in the "
                 "model and thus its state has not been updated.")
            LOG.error(m)

    LOG.info('All switches in the .glm have been updated with current states.')


def _update_diesel_dg_state_in_glm(glm_mgr: GLMManager, machines: dict):
    """Given current state of diesel generators, update their state in
    the GridLAB-D model.

    At present, equipment.initialize_synchronous_machines assumes all
    generators are three-phase balanced, so we'll make that assumption
    here as well.

    :param glm_mgr:
    :param machines:
    :return:
    """
    # Loop over the machines.
    for mach_dict in machines.values():
        # Raise exception if our balanced three phase assumption is not
        # met.
        if not isinstance(mach_dict, dict):
            raise ValueError('Found non-dict entry in machines, but this '
                             'method assumes all entries will be dictionaries '
                             'of phases.')

        # Initialize dictionary for performing updates.
        update_dict = {'object': 'diesel_dg'}

        # Loop over the phases and extract their state.
        for phase, s_mach in mach_dict.items():
            # Create string representing the power output.
            power_string = f'{s_mach.p:.4f}{s_mach.q:+.4f}j'

            # Add it to the update dictionary.
            update_dict[f'power_out_{phase.upper()}'] = power_string

        # Get the CIM name of the object.
        # noinspection PyUnboundLocalVariable
        name = ga.cim_to_glm_name(prefix=ga.SYNCH_MACH_PREFIX,
                                  cim_name=s_mach.name)
        update_dict['name'] = name

        # Update the object in the model.
        try:
            glm_mgr.modify_item(update_dict)
        except KeyError:
            # TODO: Should we raise an exception?
            m = (f"The machine/diesel_dg {name} could not be found in the "
                 "model and thus its state has not been updated.")
            LOG.error(m)

    LOG.info('All machines/diesel_dgs in the .glm have been updated with '
             'current states.')


class GAStopper:
    """Simple class for stopping the genetic algorithm when a piece of
    equipment changes state.
    """

    def __init__(self, ga_obj: ga.GA, eq_mgr: equipment.EquipmentManager,
                 eq_type: str):
        """Register callback.

        :param ga_obj: Initialized ga.GA object.
        :param eq_mgr: Initialized equipment.EquipmentManager.
        :param eq_type: String description of equipment type used for
            logging. e.g. "switch"
        """
        # Setup logging.
        self.log = logging.getLogger(self.__class__.__name__)

        # Keep a reference to the ga_obj so we can call its stop()
        # method later.
        self.ga_obj = ga_obj

        # Track our equipment type.
        self.eq_type = eq_type

        # Register callback. Note we had to add self since adding
        # methods does not work...
        # https://stackoverflow.com/a/21941670/11052174
        eq_mgr.add_callback(self)

    def __call__(self, sim_dt: datetime):
        """Callback method which will be hit by the EquipmentManager
        when it calls its callbacks. Have to use __call__ since one
        cannot add

        :param sim_dt: Datetime representing current simulation time.
        :returns: None, but rather calls the stop() method of the
            ga.GA object.
        """
        # Log.
        self.log.info('Stopping the genetic algorithm because at least one '
                      f'{self.eq_type} changed state at simulation time '
                      f'{sim_dt}.')

        # Stop the algorithm. Note this returns nearly immediately as
        # the actual stopping occurs in a thread.
        self.ga_obj.stop()


if __name__ == '__main__':
    # Use crappy names to avoid scope name overshadowing.
    pl = PlatformManager()
    s = datetime(2013, 1, 14, 16, 0)
    d = 1200
    e_start = int(s.timestamp()) + 60
    e_stop = e_start + 1000
    # events = [{"message": {"forward_differences": [
    #     {"object": "_1B6A5DFD-9ADA-404A-83DF-C9AC89D9323C",
    #      "attribute": "Switch.open", "value": 1}], "reverse_differences": [
    #     {"object": "_1B6A5DFD-9ADA-404A-83DF-C9AC89D9323C",
    #      "attribute": "Switch.open", "value": 0}]},
    #     "event_type": "ScheduledCommandEvent",
    #     "occuredDateTime": e_start, "stopDateTime": e_stop}]

    # Reg lockout + comm outage
    events = [
        {
            "message": {
                "forward_differences": [
                    {
                        "object": "_CE091BBD-77FD-4E8D-8815-EEF79D540108",
                        "attribute": "TapChanger.step",
                        "value": 10
                    }
                ],
                "reverse_differences": [
                    {
                        "object": "_CE091BBD-77FD-4E8D-8815-EEF79D540108",
                        "attribute": "TapChanger.step",
                        "value": 5
                    }
                ]
            },
            "event_type": "ScheduledCommandEvent",
            "occuredDateTime": e_start,
            "stopDateTime": e_stop
        },
        {
            "allOutputOutage": False,
            "allInputOutage": False,
            "tag": "mc45mjk2",
            "inputList": [
                {
                    "name": "vreg3_a",
                    "type": "Regulator",
                    "mRID": [
                        "_CE091BBD-77FD-4E8D-8815-EEF79D540108"
                    ],
                    "attribute": "TapChanger.step",
                    "phases": [
                        {
                            "phaseLabel": "A",
                            "phaseIndex": 0
                        }
                    ]
                }
            ],
            "outputList": [],
            "event_type": "CommOutage",
            "occuredDateTime": e_start,
            "stopDateTime": e_stop
        }]
    sid = pl.run_simulation(
        feeder_id='_AAE94E4A-2465-6F5E-37B1-3E72183A4E44',
        start_time=s, duration=d, realtime=True,
        events=events
    )

    # Do some crude sleeping to avoid timeouts later, since the platform
    # takes forever and a day to start a simulation.
    time.sleep(25)
    main(sim_id=sid, sim_request=pl.last_sim_config)
