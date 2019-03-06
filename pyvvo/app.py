"""Main module for running the pyvvo application."""
from pyvvo.sparql import SPARQLManager
from pyvvo.glm import GLMManager
from pyvvo.gridappsd_platform import PlatformManager
from pyvvo.equipment import capacitor, regulator
from datetime import datetime

if __name__ == '__main__':
    # # Determine whether we're running inside or outside the platform.
    # PLATFORM = os.environ['platform']
    #
    # Hard-code 8500 node MRID for now.
    feeder_mrid = '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3'
    # IEEE 13 bus
    # feeder_mrid = '_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62'

    # Get a SPARQL manager.
    sparql = SPARQLManager(feeder_mrid=feeder_mrid)

    # Get a platform manager
    platform = PlatformManager()

    # m = platform.get_historic_measurements(sim_id='1330667293', mrid=None)

    # Hard-code some dates to work with.
    starttime = datetime(2013, 1, 14)
    stoptime = datetime(2013, 1, 21)

    ####################################################################
    # GET PREREQUISITE DATA
    ####################################################################

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

    # # Run a simulation.
    # sim_id = platform.run_simulation()

    # Send commands to running simulation.

    pass
