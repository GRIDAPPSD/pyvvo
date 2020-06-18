"""Module for running PyVVO.
"""
import argparse
try:
    import simplejson as json
except ModuleNotFoundError:
    import json
import logging

from pyvvo import app

# Setup log.
LOG = logging.getLogger(__name__)


def _main():
    # Log and collect inputs.
    LOG.info("Starting PyVVO.")
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_id", help="Simulation ID")
    parser.add_argument("request", help="Simulation request")
    opts = parser.parse_args()

    # Parse the simulation request. Not sure why the .replace is
    # needed, but it was done that way in the sample application.
    sim_request = json.loads(opts.request.replace("\'", ""))

    LOG.info("Simulation ID and request received. Initializing application.")
    app.main(sim_id=opts.sim_id, sim_request=sim_request)


if __name__ == '__main__':
    _main()
