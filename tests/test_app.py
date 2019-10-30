# Standard library imports
import unittest
from datetime import datetime
import simplejson as json
import os
import random

from pyvvo import app, db, equipment, ga, glm, sparql, utils
from tests.models import IEEE_9500
import tests.data_files as _df

# See if we have database inputs defined.
DB_ENVIRON_PRESENT = db.db_env_defined()


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


def update_switch(switch):
    """Helper for randomly updating a switch's state."""
    switch.state = random.choice(switch.STATES)


class UpdateSwitchStateInGLMTestCase(unittest.TestCase):
    """Test _update_switch_state_in_glm"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.switch_df = _df.read_pickle(_df.SWITCHES_9500)

    def test_9500(self):
        """Ensure the method runs without error when the switches are
        properly initialized with states.
        """

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
        switch1 = mgr.find_object(obj_type='switch', obj_name='"swt_switch1"')
        switch2 = mgr.find_object(obj_type='switch', obj_name='"swt_switch2"')

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


@unittest.skipIf(not utils.gld_installed(),
                 reason='GridLAB-D is not installed.')
@unittest.skipIf(not DB_ENVIRON_PRESENT,
                 reason='Database environment variables are not present.')
class AllGLMModificationsRunTestCase(unittest.TestCase):
    """Test running the 9500 node model after making the modifications
    made in app.py as well as ga.py.
    """

    @classmethod
    def setUpClass(cls):
        # Initialize inverters and switches.
        inverter_df = _df.read_pickle(_df.INVERTERS_9500)
        inverters = equipment.initialize_inverters(inverter_df)
        inverter_meas = _df.read_pickle(_df.INVERTER_MEAS_9500)
        inverter_mgr = equipment.InverterEquipmentManager(
            eq_dict=inverters, eq_meas=inverter_meas,
            eq_mrid_col=sparql.INVERTER_MEAS_INV_MRID_COL,
            meas_mrid_col=sparql.INVERTER_MEAS_MEAS_MRID_COL)

        switch_df = _df.read_pickle(_df.SWITCHES_9500)
        switches = equipment.initialize_switches(switch_df)
        switch_meas = _df.read_pickle(_df.SWITCH_MEAS_9500)
        switch_mgr = equipment.EquipmentManager(
            eq_dict=switches, eq_meas=switch_meas,
            eq_mrid_col=sparql.SWITCH_MEAS_SWITCH_MRID_COL,
            meas_mrid_col=sparql.SWITCH_MEAS_MEAS_MRID_COL)

        # Update inverters and switches with measurements.
        with open(_df.INVERTER_MEAS_MSG_9500, 'r') as f:
            inverter_meas_msg = json.load(f)

        inverter_mgr.update_state(msg=inverter_meas_msg, sim_dt=None)

        with open(_df.SWITCH_MEAS_MSG_9500, 'r') as f:
            switch_meas_msg = json.load(f)

        switch_mgr.update_state(msg=switch_meas_msg, sim_dt=None)

        # Initialize GLM Manager.
        cls.glm_mgr = glm.GLMManager(IEEE_9500)

        # Prep and update the manager.
        app._prep_glm(cls.glm_mgr)
        app._update_glm_inverters_switches(
            glm_mgr=cls.glm_mgr, inverters=inverters, switches=switches)

        cls.starttime = datetime(2013, 4, 1, 12, 0)
        cls.stoptime = datetime(2013, 4, 1, 12, 1)
        cls.out_file = 'tmp.glm'

        # Prep the manager again via the GA's method.
        ga.prep_glm_mgr(cls.glm_mgr, cls.starttime, cls.stoptime)

        # Write file.
        cls.glm_mgr.write_model(cls.out_file)

    @classmethod
    def tearDownClass(cls):
        # Get a connection to the database.
        db_conn = db.connect_loop()

        # Truncate the tables.
        db.truncate_table(db_conn=db_conn, table=ga.TRIPLEX_TABLE)
        db.truncate_table(db_conn=db_conn, table=ga.SUBSTATION_TABLE)

        try:
            # noinspection PyUnresolvedReferences
            os.remove(cls.out_file)
        except FileNotFoundError:
            pass

        try:
            os.remove('gridlabd.xml')
        except FileNotFoundError:
            pass

    def test_model_runs(self):
        result = utils.run_gld(self.out_file)
        self.assertEqual(0, result.returncode)


if __name__ == '__main__':
    unittest.main()
