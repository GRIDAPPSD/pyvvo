"""Main module for running the pyvvo application."""
from pyvvo.sparql import SPARQLManager
from pyvvo.glm import GLMManager
from pyvvo.gridappsd_platform import PlatformManager, SimOutRouter
from pyvvo.equipment import capacitor, regulator
from datetime import datetime
from gridappsd import topics
from datetime import datetime
import json
import time


def callback(header, message):
    message = json.loads(message)
    with open('measurement.json', 'w') as f:
        json.dump(message, f)

    with open('measurment_header.json', 'w') as f:
        json.dump(header, f)

    t = datetime.utcfromtimestamp(int(header['timestamp'])/1000).strftime('%Y-%m-%d %H:%M:%S')
    sim_t = datetime.utcfromtimestamp(message['message']['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    print(r'Callback hit! Time: {}. Sim time: {}'.format(t, sim_t), flush=True)
    print('Message: {}'.format(json.dumps(message, indent=4)))
    pass


def callback2(header, message):
    print(r'Sensor topic hit. Header:{}\nMessage:{}\n'.format(header, message))


def callback3(header, message):
    print('Header: {}'.format(header))
    print('Message: {}'.format(message))


class Listener:
    def __init__(self):
        pass

    def on_message(self, header, message):
        with open('tmp.txt', 'a') as f:
            print('Header: {}'.format(header), flush=True, file=f)
            print('Message: {}'.format(message), flush=True, file=f)
        # print(header['subscription'])


if __name__ == '__main__':
    # # Determine whether we're running inside or outside the platform.
    # PLATFORM = os.environ['platform']
    #
    # Hard-code 8500 node MRID for now.
    # feeder_mrid = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3'
    # IEEE 13 bus
    # feeder_mrid = '_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62'
    # IEEE 123 bus
    feeder_mrid = '_C1C3E687-6FFD-C753-582B-632A27E28507'

    # Get a SPARQL manager.
    sparql = SPARQLManager(feeder_mrid=feeder_mrid)

    # Get a platform manager
    platform = PlatformManager()

    # model = platform.get_glm(model_id=feeder_mrid)
    # glm = GLMManager(model=model, model_is_path=False)

    # Test stuff for issue 808.
    # Subscribe to topic.
    # sim_id = platform.run_simulation()
    # print('Simulation started.', flush=True)
    # platform.gad.subscribe(topic=topics.fncs_output_topic(sim_id),
    #                        callback=callback)
    # print('Subscribed to simulation out.', flush=True)
    # sensor_topic = "{}.{}.{}".format(topics.BASE_SIMULATION_TOPIC, 'sensors',
    #                                  sim_id)
    # platform.gad.subscribe(topic=sensor_topic, callback=callback2)

    # time.sleep(5)
    # Get model.
    # model = platform.get_glm(model_id=feeder_mrid)
    # print('Model obtained from platform.', flush=True)

    # m = platform.get_historic_measurements(sim_id=sim_id, mrid=None)

    # import json
    #
    # with open('measurements.json', 'w') as f:
    #     json.dump(m, f)

    # Hard-code some dates to work with.
    starttime = datetime(2013, 1, 14)
    stoptime = datetime(2013, 1, 28)

    ####################################################################
    # GET PREREQUISITE DATA
    ####################################################################

    # TODO: Dispatch these jobs to threads.
    # Get regulator information.
    regs = sparql.query_regulators()
    c_regs = regulator.initialize_controllable_regulators(regs)
    reg_meas = sparql.query_rtc_measurements()

    # Get capacitor information.
    caps = sparql.query_capacitors()
    c_caps = capacitor.initialize_controllable_capacitors(caps)
    cap_meas = sparql.query_capacitor_measurements()

    # Get EnergyConsumer (load) data.
    load_nom_v = sparql.query_load_nominal_voltage()
    load_meas = sparql.query_load_measurements()

    # Get substation data.
    substation = sparql.query_substation_source()
    substation_bus_meas = sparql.query_measurements_for_bus(
        bus_mrid=substation.iloc[0]['bus_mrid'])

    # Get feeder model, initialize GLMManager.
    model = platform.get_glm(model_id=feeder_mrid)
    glm = GLMManager(model=model, model_is_path=False)

    # Get historic load measurement data.
    # TODO.

    # Get historic weather.
    weather = platform.get_weather(start_time=starttime, end_time=stoptime)

    # Get current weather.
    # TODO - what's the way forward here? Probably just query the
    #   database.

    # Run a simulation.
    sim_id = platform.run_simulation()

    # Listen to the log.
    # TODO: This topic is incorrect in gridappsd-python - update when
    #   fixed.
    # TODO: need a marker for historic simulations.
    # sub_id = platform.gad.subscribe(
    #     topic='/topic/goss.gridappsd.simulation.log.{}'.format(sim_id),
    #     callback=Listener()
    # )

    # time.sleep(10)

    # Doesn't work with mrid filter...
    # meas = platform.get_historic_measurements(
    #     sim_id=sim_id, mrid='_fe0a57e7-573c-47a2-ba0b-23c289f39594')

    # platform.gad.unsubscribe(sub_id)
    # print('unsubscribed!', flush=True)
    #

    # Put information on measurements into list of dictionaries so that
    # we can listen to simulation outputs.
    # TODO: Add load measurements after proof of concept.
    fn_mrid_list = [{'function': print, 'mrids': list(reg_meas['id'])}]
    # fn_mrid_list = []
    # for df in [reg_meas, cap_meas, substation_bus_meas]:
    #     fn_mrid_list.append({'function': print, 'mrids': list(df['id'])})
    #
    # sim_id = platform.run_simulation()
    # # print('Simulation started.', flush=True)
    #
    # Create a SimOutRouter to listen to simulation outputs.
    router = SimOutRouter(platform_manager=platform, sim_id=sim_id,
                          fn_mrid_list=fn_mrid_list)

    # Send commands to running simulation. Command all regulators to
    # their maximum.
    reg_ids = []
    reg_attr = []
    reg_forward = []
    reg_reverse = []

    # Loop over controllable regulators.
    for reg_name, multi_reg in c_regs.items():
        # Loop over the phases in the regulator.
        for p in multi_reg.PHASES:
            # Get the single phase regulator.
            single_reg = getattr(multi_reg, p)

            # Move along if its None.
            if single_reg is None:
                continue

            # Add the tap change mrid.
            reg_ids.append(single_reg.tap_changer_mrid)
            # Add the attribute.
            reg_attr.append('TapChanger.Step')
            # Hard-code the forward value.
            reg_forward.append(16)
            # Grab the reverse from the current tap_pos.
            # TODO: Need general solution for going from numpy to json.
            reg_reverse.append(int(single_reg.tap_pos))

    # Send in the command.
    time.sleep(10)
    platform.send_command(object_ids=reg_ids, attributes=reg_attr,
                          forward_values=reg_forward,
                          reverse_values=reg_reverse, sim_id=sim_id)

    pass
