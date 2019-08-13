import unittest
from unittest.mock import patch, Mock
from pyvvo import zip
from pyvvo import glm
from pyvvo import utils
import pandas as pd
import numpy as np
import os
import math
from scipy.optimize import OptimizeResult

# BAD PRACTICE: file dependencies across tests.
from tests.test_utils import TEST_ZIP1
# Get model directory.
from tests.models import MODEL_DIR

# Determine if GridLAB-D is at our disposal
GLD_PRESENT = utils.gld_installed()

TEST_FILE = os.path.join(MODEL_DIR, 'test_zip.glm')

# Define tolerances for using numpy's isclose function. Use different
# tolerances for P and Q. These values are the maximum relative
# differences found for the TestZipFitSLSQPAlt test case.
# R_TOL_P = 0.047
R_TOL_P = 0.012
# R_TOL_Q = 0.03
R_TOL_Q = 0.011
A_TOL = 0

# More tolerances, but same tolerance for P and Q. This will be used for
# comparing GridLAB-D and zip_model for the same parameters.
R_TOL = 0.001
# When checking if a value is close to 0, just use the absolute tolerance
A_TOL_0 = 1e-4

# NOTE: fmin_powell tests have been commented out, as it doesn't work
# well and is slow.
# TODO: if worth it, pursue fmin_powell. Though SLSQP seems to be doing
# the trick.

# Define zip fractions and power factors from PNNL CVR report:
# https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-19596.pdf
# Initial ordering: (Z%, Z_pf, I%, I_pf, P%, P_pf).
# NOTE: The power factors will be converted to power angles, so that
# the final ordering is Z%, Z_theta, I%, I_theta, P%, P_theta.

# Grab some references to shorten things up
AM = zip.ANGLE_MASK
# noinspection PyProtectedMember
pf2a = zip._angles_from_power_factors

# Incandescent light bulb (70W)
ZIP_INCANDESCENT = np.array([0.5711, 1, 0.4257, -1, 0.0032, 1])
ZIP_INCANDESCENT[AM] = pf2a(ZIP_INCANDESCENT[AM])

# Magnavox Television (Cathode Ray Tube)
ZIP_CRT_TV = np.array([0.0015, -0.99, 0.8266, 1, 0.1719, -0.92])
ZIP_CRT_TV[AM] = pf2a(ZIP_CRT_TV[AM])

# Oscillating Fan
ZIP_FAN = np.array([0.7332, 0.97, 0.2534, 0.95, 0.0135, -1])
ZIP_FAN[AM] = pf2a(ZIP_FAN[AM])

# Liquid Crystal Display (LCD) Dell
ZIP_LCD = np.array([-0.407, -0.97, 0.4629, -0.98, 0.9441, -0.97])
ZIP_LCD[AM] = pf2a(ZIP_LCD[AM])

# Plasma TV - Sony
ZIP_PLASMA = np.array([-0.3207, 0.85, 0.4836, 0.91, 0.8371, -0.99])
ZIP_PLASMA[AM] = pf2a(ZIP_PLASMA[AM])

# Liquid Crystal Display (LCD) - Clarity TV
ZIP_LCD_2 = np.array([-0.0383, 0.61, 0.0396, -0.54, 0.9987, -1])
ZIP_LCD_2[AM] = pf2a(ZIP_LCD_2[AM])

# Compact Fluorescent Light (CFL) 13 W
ZIP_CFL_13W = np.array([0.4085, -0.88, 0.0067, 0.42, 0.5849, -0.78])
ZIP_CFL_13W[AM] = pf2a(ZIP_CFL_13W[AM])

# Compact Fluorescent Light (CFL) 20 W
ZIP_CFL_20W = np.array([-0.0105, 0, 1, -0.81, 0.0105, 0.9])
ZIP_CFL_20W[AM] = pf2a(ZIP_CFL_20W[AM])

# Compact Fluorescent Light (CFL) 42W
ZIP_CFL_42W = np.array([0.4867, -0.97, -0.3752, -0.70, 0.8884, -0.79])
ZIP_CFL_42W[AM] = pf2a(ZIP_CFL_42W[AM])

# Map these into a dictionary.
ZIP_DICT = {'ZIP_INCANDESCENT': ZIP_INCANDESCENT, 'ZIP_CRT_TV': ZIP_CRT_TV,
            'ZIP_FAN': ZIP_FAN, 'ZIP_LCD': ZIP_LCD, 'ZIP_PLASMA': ZIP_PLASMA,
            'ZIP_LCD_2': ZIP_LCD_2, 'ZIP_CFL_13W': ZIP_CFL_13W,
            'ZIP_CFL_20W': ZIP_CFL_20W, 'ZIP_CFL_42W': ZIP_CFL_42W}

# Create some more constants for testing.
V_N = 120
S_N = 1000
V_SWEEP = np.arange(0.9 * V_N, 1.1 * V_N + 1)


class TestZipModelHelpers(unittest.TestCase):
    """Test _estimate_nominal_power, _get_vpq_bar, _zip_obj_and_jac,
    _zip_model, _power_factors_from_zip_terms, and _angles_from_power_factors.
    """

    def test_estimate_nominal_power(self):
        # Create simple DataFrame.
        vpq = pd.DataFrame({'v': (1, 2, 3), 'p': (2, 1, 3), 'q': (2, 1, 3)})

        # Median will be the first p-q pair.
        median_expected = abs(vpq.iloc[0]['p'] + 1j * vpq.iloc[0]['q'])

        # Call function.
        median_actual = zip._estimate_nominal_power(vpq)

        self.assertEqual(median_expected, median_actual)

    def test_get_vpq_bar(self):
        """Simple hard-coded test of _get_vpq_bar."""
        vpq = pd.DataFrame({'v': (1, 2, 3), 'p': (4, 5, 6), 'q': (7, 8, 9)})
        vpq_bar = zip._get_vpq_bar(vpq=vpq, v_n=V_N, s_n=S_N)

        expected = pd.DataFrame({'v_bar': (1/V_N, 2/V_N, 3/V_N),
                                 'p_bar': (4/S_N, 5/S_N, 6/S_N),
                                 'q_bar': (7/S_N, 8/S_N, 9/S_N)})

        pd.testing.assert_frame_equal(expected, vpq_bar)

    def test_zip_obj_and_jac_zero_error(self):
        """Given the correct zip_terms, our objective and Jacobian
        should be zero (to within reasonable rounding error)."""
        p, q = zip._zip_model(v=V_SWEEP, v_n=V_N, s_n=S_N,
                              zip_terms=zip.PAR_0)

        vpq = pd.DataFrame({'v': V_SWEEP, 'p': p, 'q': q})
        vpq_bar = zip._get_vpq_bar(vpq=vpq, v_n=V_N, s_n=S_N)

        obj, jac = zip._zip_obj_and_jac(zip_terms=zip.PAR_0,
                                        v_s=vpq_bar['v_bar'].values**2,
                                        v_bar=vpq_bar['v_bar'].values,
                                        p_bar=vpq_bar['p_bar'].values,
                                        q_bar=vpq_bar['q_bar'].values)

        # Our p and q are on the order of several hundred, so matching
        # to within 1 decimal place is acceptable.
        self.assertAlmostEqual(0, obj)
        np.testing.assert_allclose(jac, 0, rtol=0, atol=1e-10)

    def test_zip_model(self):
        """Simple test of _zip_model to ensure accuracy.
        """
        v = np.array([10])
        v_n = np.array([11])
        s_n = 100
        zip_terms = np.array([1/3, np.pi/4, 1/3, np.pi/4, 1/3, np.pi/4])

        # Get our p and q
        p_actual, q_actual = zip._zip_model(v=v, v_n=v_n, s_n=s_n,
                                            zip_terms=zip_terms)

        # sin and cos of pi/4 evaluate to sqrt(2)/2
        r22 = 2 ** 0.5 / 2
        # Hard-coding to make the test less maintainable but prevent
        # myself from copying from the function itself.
        p_expected = 100 * np.array([
            (10 ** 2) / (11 ** 2) * (1/3) * r22 + 10/11 * (1/3) * r22
            + (1/3) * r22
        ])
        # Since we used an angle of pi/4, p and q should be equal.
        q_expected = p_expected

        # Test.
        np.testing.assert_array_almost_equal(p_expected, p_actual)
        np.testing.assert_array_almost_equal(q_expected, q_actual)

    def test_get_power_factors(self):
        """Simple test of _power_factors_from_zip_terms."""
        # Use angles that'll span different parts of the unit circle.
        zip_terms = np.array([np.nan, np.pi/4, np.nan, -np.pi/6, np.nan,
                              np.pi/2])

        # Do a hard-coded check to ensure our ANGLE_MASK is correct.
        np.testing.assert_array_equal(zip_terms[zip.ANGLE_MASK],
                                      np.array([np.pi/4, -np.pi/6,
                                                np.pi/2]))

        # Use an alternate way of computing the power factors to double
        # check. Set p = 1.
        p = np.ones(3)
        q = np.tan(zip_terms[zip.ANGLE_MASK]) / p
        pf_expected = p / np.abs(p + 1j * q)
        # Hard-code the negative pf.
        pf_expected[1] *= -1

        # Compute the power factors.
        pf_actual = zip._power_factors_from_zip_terms(zip_terms)

        # Compare.
        np.testing.assert_array_equal(pf_actual, pf_expected)

    def test_power_factor_to_angle(self):
        """Simple test of _angles_from_power_factors."""
        zip_terms = np.array([np.nan, np.pi/4, np.nan, -np.pi/6, np.nan,
                              np.pi/2])
        pf = zip._power_factors_from_zip_terms(zip_terms=zip_terms)
        angles_expected = zip_terms[zip.ANGLE_MASK]
        angles_actual = zip._angles_from_power_factors(pf)

        np.testing.assert_array_almost_equal(angles_actual, angles_expected)


class TestZipFitSLSQP(unittest.TestCase):
    """Test _zip_fit_slsqp with all the PNNL models in ZIP_DICT."""

    @classmethod
    def setUpClass(cls):
        """Initialize all our expected results."""

        # Use 120V nominal.
        cls.v_n = V_N

        # We'll use a 1000VA base.
        cls.s_n = S_N

        # Sweep voltage from 90% to 110% of nominal.
        cls.v = V_SWEEP

        # Loop and assign.
        for key, value in ZIP_DICT.items():
            # Compute P and Q for the given model.
            p, q = zip._zip_model(v=cls.v, v_n=cls.v_n, s_n=cls.s_n,
                                  zip_terms=value)

            # Normalize.
            vpq_bar = zip._get_vpq_bar(
                vpq=pd.DataFrame({'v': cls.v, 'p': p, 'q': q}), v_n=cls.v_n,
                s_n=cls.s_n)

            setattr(cls, key, {'vpq_bar': vpq_bar,
                               'p': p, 'q': q})

    def run_fit(self, key):
        """Helper to perform the fit and tests."""
        # Grab attributes.
        vpq_bar = getattr(self, key)['vpq_bar']
        p_expected = getattr(self, key)['p']
        q_expected = getattr(self, key)['q']
        # zip_terms = getattr(self, key)['zip_terms']

        #
        result = zip._zip_fit_slsqp(vpq_bar=vpq_bar)

        with self.subTest('{}, success'.format(key)):
            self.assertTrue(result.success)

        p_actual, q_actual = zip._zip_model(v=self.v, v_n=self.v_n,
                                            s_n=self.s_n, zip_terms=result.x)

        with self.subTest('{}, p'.format(key)):
            np.testing.assert_allclose(p_actual, p_expected, rtol=R_TOL_P,
                                       atol=A_TOL)

        # If all the Q values are essentially 0 (like for the
        # incandescent bulb), we need to take a different approach.
        if not np.allclose(q_expected, np.zeros_like(q_expected), atol=A_TOL_0,
                           rtol=0):
            rtol = R_TOL_Q
            atol = A_TOL
        else:
            rtol = 0
            atol = 0.05

        with self.subTest('{}, q'.format(key)):
            np.testing.assert_allclose(q_actual, q_expected, rtol=rtol,
                                       atol=atol)

    def test_all(self):
        for key in ZIP_DICT.keys():
            self.run_fit(key)


class TestZipFitTestCase(unittest.TestCase):
    """Simple test of zip_fit."""
    @classmethod
    def setUpClass(cls):
        cls.v_n = V_N
        cls.s_n = S_N
        cls.v = V_SWEEP
        p, q = zip._zip_model(v=cls.v, v_n=cls.v_n, s_n=cls.s_n,
                              zip_terms=zip.PAR_0)
        cls.vpq = pd.DataFrame({'v': cls.v, 'p': p, 'q': q})

    def test_runs(self):
        """Ensure zip_fit runs and our return has the expected fields.
        """
        # Note we're giving it the correct answer here on purpose.
        # The solver itself has already been tested.
        out = zip.zip_fit(vpq=self.vpq, v_n=self.v_n, s_n=self.s_n,
                          par_0=zip.PAR_0, fit_data=True)

        # Check sol
        self.assertIn('sol', out)
        self.assertIsInstance(out['sol'], OptimizeResult)
        self.assertTrue(out['sol'].success)

        # Check zip_gld
        self.assertIn('zip_gld', out)
        self.assertIsInstance(out['zip_gld'], dict)
        self.assertIn('base_power', out['zip_gld'])
        self.assertIn('impedance_fraction', out['zip_gld'])
        self.assertIn('impedance_pf', out['zip_gld'])
        self.assertIn('current_fraction', out['zip_gld'])
        self.assertIn('current_pf', out['zip_gld'])
        self.assertIn('power_fraction', out['zip_gld'])
        self.assertIn('power_pf', out['zip_gld'])

        # Check our fit data
        self.assertIn('p_pred', out)
        self.assertIn('q_pred', out)
        self.assertIn('mse_p', out)
        self.assertIn('mse_q', out)

    @patch('pyvvo.zip.mean_squared_error')
    @patch('pyvvo.zip._zip_model', return_value=(1, 2))
    @patch('pyvvo.zip._zip_to_gld')
    @patch('pyvvo.zip._zip_fit_slsqp')
    @patch('pyvvo.zip._get_vpq_bar')
    @patch('pyvvo.zip._estimate_nominal_power')
    def test_patched(self, p_enp, p_vpqb, p_zf, p_ztg, p_zm, p_mse):
        """We'll patch our helpers and ensure they're called."""
        out = zip.zip_fit(vpq=self.vpq, v_n=self.v_n, s_n=None,
                          par_0=zip.PAR_0, fit_data=True)

        p_enp.assert_called_once()
        p_vpqb.assert_called_once()
        p_zf.assert_called_once()
        p_ztg.assert_called_once()
        p_zm.assert_called_once()
        self.assertEqual(2, p_mse.call_count)

        self.assertEqual(1, out['p_pred'])
        self.assertEqual(2, out['q_pred'])

    def test_warns(self):
        """Force a failed solve via patching, ensure we get a warning.
        """
        # Create return value for _zip_fit_slsqp.
        rv = Mock()
        rv.success = False

        with patch('pyvvo.zip._zip_fit_slsqp', return_value=rv) as p:
            with self.assertLogs(logger=zip.LOG, level='WARNING'):
                out = zip.zip_fit(vpq=self.vpq, v_n=self.v_n, s_n=None,
                                  par_0=zip.PAR_0, fit_data=True)

        self.assertIn('sol', out)
        self.assertIs(out['sol'], rv)

        p.assert_called_once()


class TestClusterAndFit(unittest.TestCase):
    """Tests for the cluster_and_fit function."""

    @classmethod
    def setUpClass(cls):
        """Initialize some ZIP outputs with different base powers."""

        # Use 120V nominal.
        cls.v_n = V_N

        # Use logarithmically varying S_n
        cls.s_n = [1, 10, 100, 1000]

        # Use four sets of ZIP coefficients that can use the default
        # initial parameters.
        cls.zip = [ZIP_CFL_42W, ZIP_LCD, ZIP_CFL_13W, ZIP_FAN]

        # Initialize results for the 4 coefficients.
        cls.results = [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]

        # Sweep voltage from 90% to 110% of nominal.
        cls.v = V_SWEEP

        # Loop and create output
        for i in range(len(cls.zip)):
            p, q = zip._zip_model(v=cls.v, v_n=cls.v_n, s_n=cls.s_n[i],
                                  zip_terms=cls.zip[i])

            # Get into format for calling zip_fit.
            vpq = pd.DataFrame({'v': cls.v, 'p': p, 'q': q})

            cls.results[i] = vpq

        # Done.

    def check_pq(self, expected, predicted):
        """Helper to test P/Q"""

        # Test p.
        self.assertTrue(np.allclose(expected['p'], predicted['p_pred'],
                                    rtol=R_TOL_P))

        # Test q.
        self.assertTrue(np.allclose(expected['q'], predicted['q_pred'],
                                    rtol=R_TOL_Q))

    def test_cluster_and_fit_no_cluster(self):
        # Ensure that with no clustering, we get the expected answer.

        # Grab some data.
        data = self.results[3]
        # Define inputs for actual zip fitting.
        zip_fit_inputs = {'s_n': self.s_n[3], 'v_n': self.v_n}

        # Use the 10th element for cluster selection.
        selection_data = None

        # Call cluster_and_fit.
        fit_data = zip.cluster_and_fit(data=data,
                                       zip_fit_inputs=zip_fit_inputs,
                                       selection_data=selection_data,
                                       n_clusters=None,
                                       random_state=None)

        self.check_pq(expected=data, predicted=fit_data)

    def test_cluster_and_fit_1_cluster(self):
        # Ensure that with 1 cluster, we get the expected answer.

        # Grab some data.
        data = self.results[0]
        # Define inputs for actual zip fitting.
        zip_fit_inputs = {'s_n': self.s_n[0], 'v_n': self.v_n}

        # Use the 10th element for cluster selection.
        selection_data = data.iloc[10][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip.cluster_and_fit(data=data,
                                       zip_fit_inputs=zip_fit_inputs,
                                       selection_data=selection_data,
                                       n_clusters=1, random_state=2)

        self.check_pq(expected=data, predicted=fit_data)

    def test_cluster_and_fit_2_cluster(self):
        # Ensure that with 2 clusters, we get the expected answer.

        # Grab data for fitting.
        data = pd.concat([self.results[0], self.results[1]])

        # Define inputs for zip fitting. We'll try to match results[1].
        zip_fit_inputs = {'s_n': self.s_n[1], 'v_n': self.v_n}

        # Use last element for cluster selection.
        selection_data = data.iloc[-1][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip.cluster_and_fit(data=data,
                                       zip_fit_inputs=zip_fit_inputs,
                                       selection_data=selection_data,
                                       n_clusters=2, random_state=2)

        self.check_pq(expected=self.results[1], predicted=fit_data)

    def test_cluster_and_fit_3_cluster(self):
        # Ensure that with 3 clusters, we get the expected answer.

        # Grab data for fitting.
        data = pd.concat([self.results[0], self.results[1], self.results[2]])

        # Define inputs for zip fitting. We'll try to match results[1].
        zip_fit_inputs = {'s_n': self.s_n[1], 'v_n': self.v_n}

        # Use first element for cluster selection.
        selection_data = self.results[1].iloc[0][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip.cluster_and_fit(data=data,
                                       zip_fit_inputs=zip_fit_inputs,
                                       selection_data=selection_data,
                                       n_clusters=3, random_state=2)

        self.check_pq(expected=self.results[1], predicted=fit_data)

    def test_cluster_and_fit_4_cluster(self):
        # Ensure that with 4 clusters, we get the expected answer.

        # Grab data for fitting.
        data = pd.concat([self.results[0], self.results[1], self.results[2],
                          self.results[3]])

        # Define inputs for zip fitting. We'll try to match results[2].
        zip_fit_inputs = {'s_n': self.s_n[2], 'v_n': self.v_n}

        # Use 15th element for cluster selection.
        selection_data = self.results[2].iloc[15][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip.cluster_and_fit(data=data,
                                       zip_fit_inputs=zip_fit_inputs,
                                       selection_data=selection_data,
                                       n_clusters=4, random_state=2)

        self.check_pq(expected=self.results[2], predicted=fit_data)

    def test_cluster_and_fit_cluster_too_small(self):
        # If a cluster is too small, None should be returned.

        # Grab data for fitting.
        data = self.results[0]

        # Define inputs for zip fitting. We'll try to match results[2].
        zip_fit_inputs = {'s_n': self.s_n[0], 'v_n': self.v_n}

        # Use 15th element for cluster selection.
        selection_data = data.iloc[15][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip.cluster_and_fit(data=data,
                                       zip_fit_inputs=zip_fit_inputs,
                                       selection_data=selection_data,
                                       n_clusters=data.shape[0],
                                       random_state=2,
                                       min_cluster_size=10)

        self.assertEqual(None, fit_data)

    def test_get_best_fit_from_clustering(self):
        # NOTE: This is the only test for get_best_fit_from_clustering,
        # as it can take a while to run and we don't want to bog our
        # tests down.

        # Put four different ZIP fits in, ensure it finds the best.

        # Grab data for fitting.
        data = pd.concat([self.results[0], self.results[1], self.results[2],
                          self.results[3]])

        # Define inputs for zip fitting. We'll try to match results[2].
        zfi = {'s_n': self.s_n[2], 'v_n': self.v_n}

        # Use 15th element for cluster selection.
        sd = self.results[2].iloc[15][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = \
            zip.get_best_fit_from_clustering(data=data,
                                             zip_fit_inputs=zfi,
                                             selection_data=sd,
                                             random_state=2)

        self.check_pq(expected=self.results[2], predicted=fit_data)


@unittest.skipIf(condition=(not GLD_PRESENT),
                 reason='gridlabd could not be found.')
class TestGLDZIP(unittest.TestCase):
    """Compare outputs between GridLAB-D ZIP load outputs and pyvvo."""

    @classmethod
    def setUpClass(cls):
        """Read and run GridLAB-D model, store ZIP parameters."""
        # Read GridLAB-D model.
        glm_manager = glm.GLMManager(TEST_FILE, True)

        # Run GridLAB-D model.
        result = utils.run_gld(model_path=TEST_FILE)

        # Raise an error if the model didn't successfully run.
        if result.returncode != 0:
            raise UserWarning('Failed to run model, {}'.format(TEST_FILE))

        # Get a listing of the recorders.
        recorders = glm_manager.get_objects_by_type('recorder')

        # Grab triplex_loads, keyed by name.
        load_data = glm_manager.get_items_by_type(item_type='object',
                                                  object_type='triplex_load')

        # Extract file and load names from the recorder dictionaries.
        load_dict = {}

        # Get lists of ZIP terms to use. zip.py doesn't have the
        # '_12' at the end.
        zip_terms = ['base_power_12', 'impedance_fraction_12',
                     'current_fraction_12', 'power_fraction_12',
                     'impedance_pf_12', 'current_pf_12', 'power_pf_12']
        zip_keys = [s.replace('_12', '') for s in zip_terms]

        cls.out_files = []

        for r in recorders:
            # Grab the meter the recorder is associated with.
            meter_name = r['parent']

            # Loop over the nodes and figure out which one is the child
            # of this meter.
            # TODO: This is terrible. While it'll be very fast for this
            # toy example, it's an expensive lookup when we have a
            # large number of nodes. We may want to add graph
            # functionality to the GLMManager.
            for name, data in load_data.items():
                if data['parent'] == meter_name:
                    # Track this load name.
                    load_name = name

            # Initialize entry.
            load_dict[load_name] = {}

            # Read the file into a DataFrame.
            this_file = os.path.join(MODEL_DIR, r['file'])
            gld_out = utils.read_gld_csv(this_file)
            cls.out_files.append(this_file)

            # Rename columns.
            gld_out.rename(columns={'measured_real_power': 'p',
                                    'measured_reactive_power': 'q'},
                           inplace=True)

            # Combine the two voltage measurements, get the magnitude.
            v = (gld_out['measured_voltage_1']
                 + gld_out['measured_voltage_2']).abs()

            # Add v to the DataFrame.
            gld_out['v'] = v

            # Drop the "measured_voltage" columns.
            gld_out.drop(['measured_voltage_1', 'measured_voltage_2'], axis=1,
                         inplace=True)

            # Add the data to the dictionary.
            load_dict[load_name]['gld_out'] = gld_out

            # Add the ZIP terms.
            zip_gld = {}
            for idx in range(len(zip_terms)):
                # Map the zip_term in the model into a zip_key in the
                # load_dict.
                zip_gld[zip_keys[idx]] = \
                    float(load_data[load_name][zip_terms[idx]])

            load_dict[load_name]['zip_gld'] = zip_gld

            # Set the nominal voltage. Note that GridLAB-D nominal
            # voltage is phase to neutral, but we want line to line.
            v_n = float(load_data[load_name]['nominal_voltage']) * 2

            load_dict[load_name]['v_n'] = v_n

            # Use zip.py to compute P and Q.
            p, q = zip._zip_model_gld(v=gld_out['v'], v_n=v_n,
                                      s_n=zip_gld['base_power'],
                                      gld_terms=zip_gld)
            load_dict[load_name]['zip_model_out'] = \
                pd.DataFrame({'p_predicted': p, 'q_predicted': q})

        # Assign the final load dictionary.
        cls.load_dict = load_dict

    @classmethod
    def tearDownClass(cls):
        """Remove the files that were created."""
        for f in cls.out_files:
            # BAD PRACTICE: Skip the file that's used in test_utils.
            if os.path.basename(TEST_ZIP1) not in f:
                os.remove(f)

    def compare_results(self, f):
        """Helper for comparing results"""
        # Loop over p, q
        for col in ['p', 'q']:
            # Grab data
            d1 = self.load_dict[f]['zip_model_out'][col + '_predicted']
            d2 = self.load_dict[f]['gld_out'][col]

            # If both arrays are near to zero, we need to handle this
            # differently (np.allclose doesn't behave well).
            if (np.allclose(d1, 0, rtol=0, atol=A_TOL_0)
                    and np.allclose(d2, 0, rtol=0, atol=A_TOL_0)):
                # Both arrays are quite close to 0, call it a day.
                bool_val = True
                # For reporting, the arrays have a 0% pct diff.
                diff_str = '0%'
            else:
                # Ensure both arrays are close.
                bool_val = np.allclose(d1.values, d2.values, atol=A_TOL,
                                       rtol=R_TOL)

                # Get string of max pct diff for reporting.
                pct_diff = ((d2 - d1) / d2) * 100
                try:
                    max_pct_diff = pct_diff[np.argmax(np.abs(pct_diff.values))]
                except AttributeError:
                    max_pct_diff = pct_diff[np.argmax(np.abs(pct_diff))]

                diff_str = '{:.2f}%'.format(max_pct_diff)

            # Actually run the test.
            with self.subTest(load=f, type=col, max_pct_diff=diff_str):
                self.assertTrue(bool_val)

    # TODO: Might be better to automatically create these functions
    # rather than manual hard-coding.

    def test_load_1(self):
        f = 'load_1'
        self.compare_results(f)

    def test_load_2(self):
        f = 'load_2'
        self.compare_results(f)

    def test_load_3(self):
        f = 'load_3'
        self.compare_results(f)

    def test_load_4(self):
        f = 'load_4'
        self.compare_results(f)

    def test_load_5(self):
        f = 'load_5'
        self.compare_results(f)

    def test_load_6(self):
        f = 'load_6'
        self.compare_results(f)


if __name__ == '__main__':
    unittest.main()
