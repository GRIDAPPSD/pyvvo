# Standard library imports
import unittest

from pyvvo import app, equipment, glm
from tests.models import IEEE_9500
import tests.data_files as _df

import random

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


class UpdateSwitchStateInGLMTestCase(unittest.TestCase):
    """Test _update_switch_state_in_glm"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.switch_df = _df.read_pickle(_df.SWITCHES_9500)

    def test_9500(self):
        """Ensure the method runs without error when the switches are
        properly initialized with states.
        """
        def update_switch(switch):
            switch.state = random.choice(switch.STATES)

        # Initialize switches.
        switches = equipment.initialize_switches(self.switch_df)

        # Initialize a GLMManager.
        glm_mgr = glm.GLMManager(model=IEEE_9500, model_is_path=True)

        # The switches currently have invalid states.
        with self.assertRaisesRegex(ValueError,
                                    'Switch .+ has a state of None'):
            app._update_switch_state_in_glm(glm_mgr=glm_mgr,
                                            switches=switches)

        # Ensure all switches have valid states.
        equipment.loop_helper(switches, update_switch)

        # Update states in the mdoel and ensure we don't get errors.
        with self.assertLogs(app.LOG, 'INFO') as cm:
            app._update_switch_state_in_glm(glm_mgr=glm_mgr,
                                            switches=switches)

        # Ensure no errors were logged.
        self.assertEqual(1, len(cm.output))

    def test_expected_behavior(self):
        """Use simple switches and model to ensure the behavior is
        correct.
        """
        model = \
            """
            object switch {
                name "swt_switch1";
                phases ABC;
                phase_A_state OPEN;
                phase_B_state CLOSED;
                phase_C_state OPEN;
            }
            object switch {
                name "swt_switch2";
                phases B;
                phase_B_state OPEN;
            }
            """

        mgr = glm.GLMManager(model=model, model_is_path=False)

        switches = {
            'mrid1': {
                # Will toggle
                'A': equipment.SwitchSinglePhase(
                    name='switch1', phase='A', mrid='mrid1',
                    controllable=True, state=1),
                # Will toggle
                'B': equipment.SwitchSinglePhase(
                    name='switch1', phase='B', mrid='mrid1',
                    controllable=True, state=0),
                # Won't toggle.
                'C': equipment.SwitchSinglePhase(
                    name='switch1', phase='C', mrid='mrid1',
                    controllable=True, state=0)
            },
            # Will toggle
            'mrid2': equipment.SwitchSinglePhase(
                name='switch2', phase='B', mrid='mrid2',
                controllable=True, state=1)
        }

        # Perform the update.
        app._update_switch_state_in_glm(glm_mgr=mgr, switches=switches)

        # Check the switches.
        switch1 = mgr.find_object(obj_type='switch', obj_name='switch1')
        switch2 = mgr.find_object(obj_type='switch', obj_name='switch2')

        self.assertEqual(switch1['phase_A_state'], 'CLOSED')
        self.assertEqual(switch1['phase_B_state'], 'OPEN')
        self.assertEqual(switch1['phase_C_state'], 'OPEN')

        self.assertEqual(switch2['phase_B_state'], 'CLOSED')

    def test_missing_switch(self):
        model = \
            """
            object switch {
                name who_cares;
            }
            """

        mgr = glm.GLMManager(model=model, model_is_path=False)

        switches = {'mrid': equipment.SwitchSinglePhase(
            name='mismatch', mrid='eh', phase='C', controllable=False,
            state=0
        )}

        with self.assertLogs(logger=app.LOG, level='ERROR'):
            app._update_switch_state_in_glm(mgr, switches)


if __name__ == '__main__':
    unittest.main()
