"""PyVVO is a Volt-VAR optimization (VVO) application for GridAPPS-D.

Author: Brandon Thayer (Pacific Northwest National Laboratory)

It is completely "Dockerized," and also dependent on the GridAPPS-D
platform itself.

In short, PyVVO pulls historic "smart meter" data in order to create
predictive, voltage-dependent load models. These load models are then
layered onto a feeder model (in GridLAB-D format). A genetic algorithm
then runs the feeder model with many different device configurations
to determine what control settings are best. At the moment, PyVVO only
controls regulators and capacitors (future work should include
distributed generation resources such a photovoltaic
inverters).

Many of PyVVO's Python modules are likely very useful for other
GridAPPS-D applications. For instance, sparql.py contains many useful
queries for obtaining information from the CIM triple-store database,
gridappsd_platform.py provides objects for interfacing with the
GridAPPS-D platform itself, glm.py is useful for managing GridLAB-D
models, and zip.py has code for creating ZIP load models from
measurement data.

This file (__init__.py) does some application-wide logging
configuration. It uses log_config.json.
"""
import logging
import logging.handlers

try:
    import simplejson as json
except ModuleNotFoundError:
    import json

import os
from logging.config import dictConfig

# Load the logging configuration file.
config_file = os.path.join(os.path.dirname(__file__), "log_config.json")
with open(config_file, 'r') as f:
    config_dict = json.load(f)

# Grab the level - we'll use the same level for all logs and handlers.
log_level = getattr(logging, config_dict.pop('level').upper())

# Set the handlers to the appropriate level, determine if we need to
# rotate any logs.
need_rotate = []
for h_name, h in config_dict['handlers'].items():
    # Set level.
    h['level'] = log_level
    # Check if we need to perform a log rotation.
    try:
        if os.path.isfile(h['filename']):
            need_rotate.append(h_name)
    except KeyError:
        # Not all handlers have a filename.
        pass

# Set root logger level.
config_dict['loggers']['']['level'] = log_level

# Configure logging.
dictConfig(config_dict)

# Get Logger.
log = logging.getLogger()
# Unfortunately, this message will get put into an old log that gets
# rotated. However, it shows up in the console, so no big deal.
log.debug('Root logger configured in {}.'.format(__file__))

# Loop through handlers, and rotate any that are RotatingFileHandlers.
if len(need_rotate) > 0:
    for h in log.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler) \
                and (h.name in need_rotate):
            h.doRollover()
            log.debug('Old log, {}, rotated.'.format(h.baseFilename))

