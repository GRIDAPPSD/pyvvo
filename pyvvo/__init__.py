"""PyVVO is a Volt-VAR optimization (VVO) application for GridAPPS-D.

TODO: documentation.

Setup logging application-wide.
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

