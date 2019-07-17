"""Helper for pulling and saving GridLAB-D models from the
GridAPPS-D platform.
"""
import os

from pyvvo import gridappsd_platform

# Handle pathing.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(THIS_DIR, 'models')

# Dictionary of model MRIDs to file names. Note the 9500 node isn't
# truly from IEEE (it's a modified version of the 8500), but better to
# keep consistent + simple naming conventions.
MODELS = {
    '_4F76A5F9-271D-9EB8-5E31-AA362D86F2C3': 'ieee_8500',
    '_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62': 'ieee_13',
    '_C1C3E687-6FFD-C753-582B-632A27E28507': 'ieee_123',
    '_AAE94E4A-2465-6F5E-37B1-3E72183A4E44': 'ieee_9500'
}

# For convenience, create constants which other testing modules can
# import.
IEEE_8500 = os.path.join(MODEL_DIR, 'ieee_8500.glm')
IEEE_13 = os.path.join(MODEL_DIR, 'ieee_13.glm')
IEEE_123 = os.path.join(MODEL_DIR, 'ieee_123.glm')
IEEE_9500 = os.path.join(MODEL_DIR, 'ieee_9500.glm')

if __name__ == '__main__':
    platform = gridappsd_platform.PlatformManager()
    for mrid, model in MODELS.items():
        # Pull the model from the platform.
        model_str = platform.get_glm(model_id=mrid)

        with open(os.path.join(MODEL_DIR, model + '.glm'), 'w') as f:
            f.write(model_str)

    # That's it.
