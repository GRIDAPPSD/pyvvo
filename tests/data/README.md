# data
This directory is used to hold data files for testing.

## Files

### cap_meas_message_8500.json
File with measurement outputs from the GridAPPS-D platform specifically
for capacitors in the 8500 node model. At present, recreating this file
isn't super trivial. TODO: Document more. See also
reg_meas_message_8500.json

### query_cap_meas_8500.csv
Created in test_sparql.py, corresponds to capacitor measurements in 
the IEEE 8500 model.

### query_capacitors_8500.csv
Created in test_sparql.py, corresponds to capacitors in IEEE 8500 model.

### query_load_measurements_8500.csv
Created in test_sparql.py, corresponds to regulators in IEEE 8500 model.

### query_model_info.json
Created in test_gridappsd_platform.py. Corresponds to output from 
calling the GridAPPS-D platform to request model info.

### query_reg_meas_8500.csv
Created in test_sparql.py, corresponds to regulator measurements in the
IEEE 8500 model.

### query_regulators_8500.csv
Created in test_sparql.py, corresponds to regulators in the IEEE 8500
model.

### query_substation_source_8500.csv
Created in test_sparql.py, corresponds to substation source bus
information in the IEEE 8500 model.

### query_switch_meas_8500.csv
Created in test_sparql.py, corresponds to discrete position measurements
for switches in the 8500 model.

### query_switches_8500.csv
Created in test_sparql.py, corresponds to switch objects in the 8500 
model.

### README.md
This file.

### reg_meas_message_8500.json
Similarly to cap_meas_message_8500.json, this corresponds to regulator
measurement output from the platform.

### simulation_measurements_13.json
Created by running the IEEE 13 node model in the platform, subscribing 
to the simulation output, and writing the first response to file.

### simulation_measurements_header_13.json
Created by running the IEEE 13 node model in the platform, subscribing
to the simulation output, and writing the first header to file.

### weather_simple.json
Created by querying the platform for weather data for
2013-01-01 00:00:00 Mountain Time.