# models
This directory is used to hold models (at this point, just GridLAB-D),
supplemental model files (e.g. voltage.player), and model outputs.

## Files

### ieee_13.glm
Created in test_gridappsd_platform.py. GridLAB-D model for the IEEE 13
bus model as provided by the platform.

### ieee_8500.glm
Created by pulling the 8500 node model from the GridAPPS-D platform and
writing to file. There's a commented out test in
test_gridappsd_platform.py that could create this, but it's commented 
out for a reason.

### README.md
This file.

### test.glm
Used by test_glm.py. Originally taken from 
https://github.com/gridlab-d/gridlab-d/blob/release/RC4.1/mysql/autotest/test_mysql_group_recorder_1.glm
and annotated. 

### test2.glm
Silly simple and runnable model. Just has powerflow, a clock, and a 
substation.

### test3.glm
Just a substation object, used to test making a model runnable.

### test4.glm
Non-runnable model for testing object recursion.

### test4_expected.glm
Flattened version of test4.glm, used to ensure glm.py's recursion 
properly flattens nested objects.

### test_zip.glm
Model with a variety of ZIP loads. Used to ensure zip.py behaves in the
same way as GridLAB-D for several cases. 

### test_zip_1.csv
One of the output files from running test_zip.glm. 

### voltage.player
Player file used by test_zip.glm to ensure node voltages are easily
discernible and reproducible.
