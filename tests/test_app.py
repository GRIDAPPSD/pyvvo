# Standard library imports
import unittest

from pyvvo import app, equipment, glm
from tests.models import IEEE_9500
import tests.data_files as _df


class UpdateInverterStateInGLMTestCase(unittest.TestCase):
    """Test _update_inverter_state_in_glm."""

    def test_9500(self):
        """Just ensure the method runs for the 9500 node model."""
        # Read inverter data.
        inverter_df = _df.read_pickle(_df.INVERTERS_9500)
        inverters = equipment.initialize_inverters(inverter_df)

        # Initialize a GLMManager.
        glm_mgr = glm.GLMManager(model=IEEE_9500, model_is_path=True)

        with self.assertLogs(app.LOG, 'INFO') as cm:
            app._update_inverter_state_in_glm(glm_mgr=glm_mgr,
                                              inverters=inverters)

        # Ensure no errors were logged.
        self.assertEqual(1, len(cm.output))

    def test_expected_behavior(self):
        """Use a simple model and contrived inverter objects to verify
        updates are working as expected.
        """
        name_1_glm = '"inv_pv_inv_1"'
        name_1_cim = 'inv_1'
        name_2_glm = '"inv_bat_inv_2"'
        name_2_cim = 'inv_2'

        # Make a simple model with a pair of inverters.
        model = \
            """
            object inverter {{
              name {};
              P_Out 30.2;
              Q_Out 1.1;
            }}
            object inverter {{
              name {};
              P_Out 1200.75;
              Q_Out 0.0;
            }}
            """.format(name_1_glm, name_2_glm)

        # Create GLM Manager.
        glm_mgr = glm.GLMManager(model, model_is_path=False)

        # Create some inverters.
        p1 = 45.3
        q1 = 16.1
        p2 = 19.3
        q2 = 3.5

        inv_dict = {
            'mrid1': equipment.InverterSinglePhase(
                mrid='mrid1', name=name_1_cim, phase='S2',
                controllable=True, p=p1, q=q1),
            'mrid2': {
                'A': equipment.InverterSinglePhase(
                    mrid='mrid2', name=name_2_cim, phase='A',
                    controllable=True, p=p2, q=q2),
                'B': equipment.InverterSinglePhase(
                    mrid='mrid2', name=name_2_cim, phase='B',
                    controllable=True, p=p2, q=q2),
                'C': equipment.InverterSinglePhase(
                    mrid='mrid2', name=name_2_cim, phase='C',
                    controllable=True, p=p2, q=q2),
            }
        }

        # Run the update.
        app._update_inverter_state_in_glm(glm_mgr=glm_mgr, inverters=inv_dict)

        # Check inverter 1.
        inv1 = glm_mgr.find_object(obj_type='inverter', obj_name=name_1_glm)
        self.assertEqual(p1, float(inv1['P_Out']))
        self.assertEqual(q1, float(inv1['Q_Out']))

        # Check inverter 2.
        inv2 = glm_mgr.find_object(obj_type='inverter', obj_name=name_2_glm)
        self.assertEqual(p2*3, float(inv2['P_Out']))
        self.assertEqual(q2*3, float(inv2['Q_Out']))

    def test_cannot_find_inverter(self):
        """Ensure we get an error level log if an inverter cannot be
        found.
        """
        # Create single inverter model.
        model = \
            """
            object inverter {
                name some_inverter;
            }
            """

        # Create GLMManager.
        glm_mgr = glm.GLMManager(model, model_is_path=False)

        # Create an inverter.
        inv_dict = {
            'mrid': equipment.InverterSinglePhase(
                mrid='mrid', name='no_match', phase='s2', controllable=True,
                p=12, q=3
            )
        }

        # Ensure we get an error level log.
        with self.assertLogs(logger=app.LOG, level='ERROR'):
            app._update_inverter_state_in_glm(glm_mgr=glm_mgr,
                                              inverters=inv_dict)


if __name__ == '__main__':
    unittest.main()
