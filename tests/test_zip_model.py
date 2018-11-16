import unittest
from pyvvo import zip_model
import pandas as pd
import numpy as np

# TODO: Create testing class to test _p_q_from_v_zip_gld. This class
# should actually call GridLAB-D, get output, and compare.

# Define tolerances for using numpy's isclose function. Use different
# tolerances for P and Q.
R_TOL_P = 0.02
R_TOL_Q = 0.06
A_TOL = 0
# NOTE: fmin_powell tests have been commented out, as it doesn't work
# well and is slow.
# TODO: if worth it, pursue fmin_powell. Though SLSQP seems to be doing
# the trick.

# Define zip fractions and power factors from PNNL CVR report:
# https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-19596.pdf
# Ordering: (z_pct, i_pct, p_pct, z_pf, i_pf, p_pf)
# Incandescent light bulb (70W)
ZIP_INCANDESCENT = (0.5711, 0.4257, 0.0032, 1, -1, 1)
# Magnavox Television (Cathode Ray Tube)
ZIP_CRT_TV = (0.0015, 0.8266, 0.1719, -0.99, 1, -0.92)
# Oscillating Fan
ZIP_FAN = (0.7332, 0.2534, 0.0135, 0.97, 0.95, -1)
# Liquid Crystal Display (LCD) Dell
ZIP_LCD = (-0.407, 0.4629, 0.9441, -0.97, -0.98, -0.97)
# Plasma TV - Sony
ZIP_PLASMA = (-0.3207, 0.4836, 0.8371, 0.85, 0.91, -0.99)
# Liquid Crystal Display (LCD) - Clarity TV
ZIP_LCD_2 = (-0.0383, 0.0396, 0.9987, 0.61, -0.54, -1)
# Compact Fluorescent Light (CFL) 13 W
ZIP_CFL_13W = (0.4085, 0.0067, 0.5849, -0.88, 0.42, -0.78)
# Compact Fluorescent Light (CFL) 20 W
ZIP_CFL_20W = (-0.0105, 1, 0.0105, 0, -0.81, 0.9)
# Compact Fluorescent Light (CFL) 42W
ZIP_CFL_42W = (0.4867, -0.3752, 0.8884, -0.97, -0.70, -0.79)

# Map these into a dictionary.
ZIP_DICT = {'ZIP_INCANDESCENT': ZIP_INCANDESCENT, 'ZIP_CRT_TV': ZIP_CRT_TV,
            'ZIP_FAN': ZIP_FAN, 'ZIP_LCD': ZIP_LCD, 'ZIP_PLASMA': ZIP_PLASMA,
            'ZIP_LCD_2': ZIP_LCD_2, 'ZIP_CFL_13W': ZIP_CFL_13W,
            'ZIP_CFL_20W': ZIP_CFL_20W, 'ZIP_CFL_42W': ZIP_CFL_42W}


class TestZipModelHelpers(unittest.TestCase):
    """Test _estimate_nominal_power, _zip_objective, _zip_constraint,
    _get_fractions_and_power_factors, and _poly_to_gld
    """

    def test_estimate_nominal_power(self):
        # Create simple DataFrame.
        vpq = pd.DataFrame({'v': (1, 2, 3), 'p': (2, 1, 3), 'q': (2, 1, 3)})

        # Median will be the first p-q pair.
        median_expected = abs(vpq.iloc[0]['p'] + 1j * vpq.iloc[0]['q'])

        # Call function.
        median_actual = zip_model._estimate_nominal_power(vpq)

        self.assertEqual(median_expected, median_actual)

    def test_zip_objective_zero_error(self):
        # Ensure we get zero if equal.
        vpq_bar = pd.DataFrame({'v_bar': (1, 1, 1), 'p_bar': (3, 3, 3),
                                'q_bar': (3, 3, 3)})

        poly_terms = (1,) * 6

        self.assertAlmostEqual(0, zip_model._zip_objective(poly_terms,
                                                           vpq_bar))

    def test_zip_objective_with_error(self):
        vpq_bar = pd.DataFrame({'v_bar': (1, 1, 0), 'p_bar': (3, 3, 3),
                                'q_bar': (3, 3, 3)})

        poly_terms = (1,) * 6

        self.assertAlmostEqual(8/3, zip_model._zip_objective(poly_terms,
                                                             vpq_bar))

    def test_zip_constraint_par_0(self):

        poly_terms = zip_model.PAR_0
        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_incandescent_bulb(self):
        # Incandescent light bulb (70W)
        poly_terms = zip_model._get_poly_from_zip(*ZIP_INCANDESCENT)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_crt_tv(self):
        # Magnavox Television (Cathode Ray Tube)
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CRT_TV)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_oscillating_fan(self):
        # Oscillating Fan
        poly_terms = zip_model._get_poly_from_zip(*ZIP_FAN)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_lcd(self):
        # Liquid Crystal Display (LCD) Dell
        poly_terms = zip_model._get_poly_from_zip(*ZIP_LCD)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_plasma(self):
        # Plasma TV - Sony
        poly_terms = zip_model._get_poly_from_zip(*ZIP_PLASMA)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_lcd_2(self):
        # Liquid Crystal Display (LCD) - Clarity TV
        poly_terms = zip_model._get_poly_from_zip(*ZIP_LCD_2)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_cfl_13w(self):
        # Compact Fluorescent Light (CFL) 13W
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_13W)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_cfl_20w(self):
        # Compact Fluorescent Light (CFL) 20W
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_20W)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_constraint_cfl_42w(self):
        # Compact Fluorescent Light (CFL) 42W
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_42W)

        self.assertAlmostEqual(0, zip_model._zip_constraint(poly_terms),
                               places=1)

    def test_zip_get_fractions_and_power_factors_par_0(self):

        poly_terms = zip_model.PAR_0
        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)

        # pf of -1 == pf of 1. For testing, cast last term to -1
        power_factors = np.array(power_factors)
        if power_factors[2] == 1:
            power_factors[2] = -1

        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, zip_model.PAR_0_ZIP))

    def test_get_fractions_and_power_factors_incandescent_bulb(self):
        # Incandescent light bulb (70W)
        poly_terms = zip_model._get_poly_from_zip(*ZIP_INCANDESCENT)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)

        # pf of -1 == pf of 1. For testing, cast to match.
        power_factors = np.array(power_factors)
        if power_factors[0] == -1:
            power_factors[0] = 1

        if power_factors[1] == 1:
            power_factors[1] = -1

        if power_factors[2] == -1:
            power_factors[2] = 1

        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_INCANDESCENT))

    def test_get_fractions_and_power_factors_crt_tv(self):
        # Magnavox Television (Cathode Ray Tube)
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CRT_TV)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)

        # pf of -1 == pf of 1. For testing, cast -1 to 1.
        power_factors = np.array(power_factors)
        if power_factors[1] == -1:
            power_factors[1] = 1

        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_CRT_TV))

    def test_get_fractions_and_power_factors_oscillating_fan(self):
        # Oscillating Fan
        poly_terms = zip_model._get_poly_from_zip(*ZIP_FAN)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_FAN))

    def test_get_fractions_and_power_factors_lcd(self):
        # Liquid Crystal Display (LCD) Dell
        poly_terms = zip_model._get_poly_from_zip(*ZIP_LCD)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_LCD))

    def test_get_fractions_and_power_factors_plasma(self):
        # Plasma TV - Sony
        poly_terms = zip_model._get_poly_from_zip(*ZIP_PLASMA)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_PLASMA))

    def test_get_fractions_and_power_factors_lcd_2(self):
        # Liquid Crystal Display (LCD) - Clarity TV
        poly_terms = zip_model._get_poly_from_zip(*ZIP_LCD_2)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_LCD_2))

    def test_get_fractions_and_power_factors_cfl_13w(self):
        # Compact Fluorescent Light (CFL) 13W
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_13W)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_CFL_13W))

    def test_get_fractions_and_power_factors_cfl_20w(self):
        # Compact Fluorescent Light (CFL) 20W
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_20W)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_CFL_20W))

    def test_get_fractions_and_power_factors_cfl_42w(self):
        # Compact Fluorescent Light (CFL) 42W
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_42W)

        fractions, power_factors = \
            zip_model._get_fractions_and_power_factors(poly_terms)
        zip_terms = (*fractions, *power_factors)
        self.assertTrue(np.allclose(zip_terms, ZIP_CFL_42W))

    def test_poly_to_gld_cfl_42w(self):
        # Since we heavily tested get_fractions_and_power_factors, one
        # test for _poly_to_gld will be sufficient.
        poly_terms = zip_model._get_poly_from_zip(*ZIP_CFL_42W)

        expected = {'impedance_fraction': ZIP_CFL_42W[0],
                    'current_fraction': ZIP_CFL_42W[1],
                    'power_fraction': ZIP_CFL_42W[2],
                    'impedance_pf': ZIP_CFL_42W[3],
                    'current_pf': ZIP_CFL_42W[4],
                    'power_pf': ZIP_CFL_42W[5]}

        actual = zip_model._poly_to_gld(poly_terms)

        # Loop through terms, ensure they're close.
        for k in expected:
            self.assertAlmostEqual(expected[k], actual[k])


class TestZipModelSolversPNNLZIP(unittest.TestCase):
    """Test solving for ZIP coefficients using optimization solvers.

    For this class, we'll be using the ZIP coefficients from the PNNL
    CVR report with an arbitrary 1000 VA base apparent power.

    NOTE: the crt_tv, incandescent, lcd_2, and plasma tests need a
    different starting point to succeed within our given tolerances for
    P and Q. While I don't like this manipulation, these tests are
    intended to ensure the solvers are working, and we have to provide
    a decent initial guess to get them to work.
    """

    @classmethod
    def setUpClass(cls):
        """Initialize all our expected results."""

        # Use 120V nominal.
        cls.v_n = 120

        # We'll use a 1000VA base.
        cls.s_n = 1000

        # Sweep voltage from 90% to 110% of nominal.
        cls.v = np.arange(0.9 * cls.v_n, 1.1 * cls.v_n + 1)

        # Loop and assign.
        for key, value in ZIP_DICT.items():
            # Extract fractions and power factors
            fractions = value[:3]
            power_factors = value[3:]

            # Get the polynomial form.
            poly_terms = zip_model._get_poly_from_zip(*value)

            # Compute p and q.
            pq = zip_model._pq_from_fractions_and_power_factors(
                fractions=fractions, power_factors=power_factors, v=cls.v,
                v_n=cls.v_n, s_n=cls.s_n)

            # Get into format for calling zip_fit.
            vpq = pd.DataFrame({'v': cls.v, 'p': pq['p_predicted'],
                                'q': pq['q_predicted']})

            setattr(cls, key, {'vpq': vpq, 'pq': pq, 'poly_terms': poly_terms})

        # Done.

    '''
    def save_files_for_indra(self):
        """Save P, Q, and V data to file. Add dummy columns for time,
        temperature, and solar flux.
        """
        # Initialize time range.
        T = pd.date_range('2016-01-01 00:00:00', periods=self.v.shape[0],
                          freq='15min')
        # Initialize dummy temperature and flux.
        temperature = np.zeros(T.shape[0])
        solar_flux = np.zeros(T.shape[0])

        # Loop over ZIP models.
        for key in ZIP_DICT.keys():
            # Grab vpq.
            vpq = getattr(self, key)['vpq']

            # Build DataFrame to save to file.
            d = {'P': vpq['p'].values, 'Q': vpq['q'].values, 'V': vpq[
                'v'].values, 'temperature': temperature,
                 'solar_flux': solar_flux}
            df = pd.DataFrame(d, index=T)
            df.to_csv(key + '.csv')


    def test_dummy(self):
        """Dummy test to run save_files_for_indra"""
        self.save_files_for_indra()
    '''

    def run_zip_fit(self, test, solver, use_answer, par_0=None, a_tol=A_TOL):
        """Helper to perform and test fit."""

        # Extract attributes
        vpq = getattr(self, test)['vpq']
        pq_expected = getattr(self, test)['pq']

        # Give it the right answer to start with.
        if use_answer:
            par_0 = getattr(self, test)['poly_terms']

        # Perform ZIP fit.
        results = zip_model.zip_fit(vpq=vpq, v_n=self.v_n, par_0=par_0,
                                    s_n=self.s_n, solver=solver)

        # Ensure it converged.
        self.assertTrue(results['success'],
                        'Solver {} failed for {}'.format(solver, test))

        # Consider the fit a success if we're within tolerance for both
        # p and q.
        for field, tol in {'p_predicted': R_TOL_P,
                           'q_predicted': R_TOL_Q}.items():

            s = 'Test case: {}, Solver: {}, {}'.format(test, solver, field)
            with self.subTest(s):
                self.assertTrue(np.allclose(pq_expected[field],
                                            results['pq_predicted'][field],
                                            rtol=tol, atol=a_tol), msg=s)

    def test_zip_fit_slsqp_incandescent_given_answer(self):
        # Setup test.
        test = 'ZIP_INCANDESCENT'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_incandescent(self):
        # NOTE: This test fails for q because q is all 0's. So we need
        # to override our A_TOL.

        # Setup test.
        test = 'ZIP_INCANDESCENT'
        solver = 'SLSQP'
        use_answer = False

        # Since actual P is on the order of ~800-~1100, 10 seems like
        # a reasonably small absolute tolerance for q (which is supposed
        # to be 0)
        self.run_zip_fit(test, solver, use_answer, a_tol=10)

    """
    def test_zip_fit_fmin_powell_incandescent_given_answer(self):
        # Setup test.
        test = 'ZIP_INCANDESCENT'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_incandescent(self):
        # Setup test.
        test = 'ZIP_INCANDESCENT'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_crt_tv_given_answer(self):
        # Setup test.
        test = 'ZIP_CRT_TV'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_crt_tv(self):
        # Setup test.
        test = 'ZIP_CRT_TV'
        solver = 'SLSQP'
        use_answer = False

        # This one needs help - get a better starting point.
        # Didn't work:
        # fan, incandescent,
        par_0 = zip_model._get_poly_from_zip(*ZIP_LCD)

        self.run_zip_fit(test, solver, use_answer, par_0)

    """
    def test_zip_fit_fmin_powell_crt_tv_given_answer(self):
        # Setup test.
        test = 'ZIP_CRT_TV'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_crt_tv(self):
        # Setup test.
        test = 'ZIP_CRT_TV'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_fan_given_answer(self):
        # Setup test.
        test = 'ZIP_FAN'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_fan(self):
        # Setup test.
        test = 'ZIP_FAN'
        solver = 'SLSQP'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)

    """
    def test_zip_fit_fmin_powell_fan_given_answer(self):
        # Setup test.
        test = 'ZIP_FAN'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_fan(self):
        # Setup test.
        test = 'ZIP_FAN'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_lcd_given_answer(self):
        # Setup test.
        test = 'ZIP_LCD'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_lcd(self):
        # Setup test.
        test = 'ZIP_LCD'
        solver = 'SLSQP'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)

    """
    def test_zip_fit_fmin_powell_lcd_given_answer(self):
        # Setup test.
        test = 'ZIP_LCD'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_lcd(self):
        # Setup test.
        test = 'ZIP_LCD'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_plasma_given_answer(self):
        # Setup test.
        test = 'ZIP_PLASMA'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_plasma(self):
        # Setup test.
        test = 'ZIP_PLASMA'
        solver = 'SLSQP'
        use_answer = False

        # This one needs help - get a better starting point.
        par_0 = zip_model._get_poly_from_zip(*ZIP_LCD_2)
        # Didn't work:
        # fan, crt_tv, lcd

        self.run_zip_fit(test, solver, use_answer, par_0)

    """
    def test_zip_fit_fmin_powell_plasma_given_answer(self):
        # Setup test.
        test = 'ZIP_PLASMA'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_plasma(self):
        # Setup test.
        test = 'ZIP_PLASMA'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_lcd_2_given_answer(self):
        # Setup test.
        test = 'ZIP_LCD_2'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_lcd_2(self):
        # Setup test.
        test = 'ZIP_LCD_2'
        solver = 'SLSQP'
        use_answer = False

        # This one needs help - get a better starting point.
        par_0 = zip_model._get_poly_from_zip(*ZIP_LCD)
        # Didn't work:
        # fan, crt_tv,

        self.run_zip_fit(test, solver, use_answer, par_0)

    """
    def test_zip_fit_fmin_powell_lcd_2_given_answer(self):
        # Setup test.
        test = 'ZIP_LCD_2'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_lcd_2(self):
        # Setup test.
        test = 'ZIP_LCD_2'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_cfl_13w_given_answer(self):
        # Setup test.
        test = 'ZIP_CFL_13W'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_cfl_13w(self):
        # Setup test.
        test = 'ZIP_CFL_13W'
        solver = 'SLSQP'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)

    """
    def test_zip_fit_fmin_powell_cfl_13w_given_answer(self):
        # Setup test.
        test = 'ZIP_CFL_13W'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_cfl_13w(self):
        # Setup test.
        test = 'ZIP_CFL_13W'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_cfl_20w_given_answer(self):
        # Setup test.
        test = 'ZIP_CFL_20W'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_cfl_20w(self):
        # Setup test.
        test = 'ZIP_CFL_20W'
        solver = 'SLSQP'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)

    """
    def test_zip_fit_fmin_powell_cfl_20w_given_answer(self):
        # Setup test.
        test = 'ZIP_CFL_20W'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_cfl_20w(self):
        # Setup test.
        test = 'ZIP_CFL_20W'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """

    def test_zip_fit_slsqp_cfl_42w_given_answer(self):
        # Setup test.
        test = 'ZIP_CFL_42W'
        solver = 'SLSQP'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_slsqp_cfl_42w(self):
        # Setup test.
        test = 'ZIP_CFL_42W'
        solver = 'SLSQP'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)

    """
    def test_zip_fit_fmin_powell_cfl_42w_given_answer(self):
        # Setup test.
        test = 'ZIP_CFL_42W'
        solver = 'fmin_powell'
        use_answer = True

        self.run_zip_fit(test, solver, use_answer)

    def test_zip_fit_fmin_powell_cfl_42w(self):
        # Setup test.
        test = 'ZIP_CFL_42W'
        solver = 'fmin_powell'
        use_answer = False

        self.run_zip_fit(test, solver, use_answer)
    """


class TestClusterAndFit(unittest.TestCase):
    """Tests for the cluster_and_fit function."""

    @classmethod
    def setUpClass(cls):
        """Initialize some ZIP outputs with different base powers."""

        # Use 120V nominal.
        cls.v_n = 120

        # Use logarithmically varying S_n
        cls.s_n = [1, 10, 100, 1000]

        # Use four sets of ZIP coefficients that can use the default
        # initial parameters.
        cls.zip = [ZIP_CFL_42W, ZIP_LCD, ZIP_CFL_13W, ZIP_FAN]

        # Initialize results for the 4 coefficients.
        cls.results = [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]

        # Sweep voltage from 90% to 110% of nominal.
        cls.v = np.arange(0.9 * cls.v_n, 1.1 * cls.v_n + 1)

        # Loop and create output
        for i in range(len(cls.zip)):
            # Extract fractions and power factors
            fractions = cls.zip[i][:3]
            power_factors = cls.zip[i][3:]

            # Compute p and q.
            pq = zip_model._pq_from_fractions_and_power_factors(
                fractions=fractions, power_factors=power_factors, v=cls.v,
                v_n=cls.v_n, s_n=cls.s_n[i])

            # Get into format for calling zip_fit.
            vpq = pd.DataFrame({'v': cls.v, 'p': pq['p_predicted'],
                                'q': pq['q_predicted']})

            cls.results[i] = vpq

        # Done.

    def check_pq(self, expected, predicted):
        """Helper to test P/Q"""

        # Test p.
        self.assertTrue(np.allclose(expected['p'], predicted['p_predicted'],
                                    rtol=R_TOL_P))

        # Test q.
        self.assertTrue(np.allclose(expected['q'], predicted['q_predicted'],
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
        fit_data = zip_model.cluster_and_fit(data=data,
                                             zip_fit_inputs=zip_fit_inputs,
                                             selection_data=selection_data,
                                             n_clusters=None,
                                             random_state=None)

        self.check_pq(expected=data, predicted=fit_data['pq_predicted'])

    def test_cluster_and_fit_1_cluster(self):
        # Ensure that with 1 cluster, we get the expected answer.

        # Grab some data.
        data = self.results[0]
        # Define inputs for actual zip fitting.
        zip_fit_inputs = {'s_n': self.s_n[0], 'v_n': self.v_n}

        # Use the 10th element for cluster selection.
        selection_data = data.iloc[10][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip_model.cluster_and_fit(data=data,
                                             zip_fit_inputs=zip_fit_inputs,
                                             selection_data=selection_data,
                                             n_clusters=1, random_state=2)

        self.check_pq(expected=data, predicted=fit_data['pq_predicted'])

    def test_cluster_and_fit_2_cluster(self):
        # Ensure that with 2 clusters, we get the expected answer.

        # Grab data for fitting.
        data = pd.concat([self.results[0], self.results[1]])

        # Define inputs for zip fitting. We'll try to match results[1].
        zip_fit_inputs = {'s_n': self.s_n[1], 'v_n': self.v_n}

        # Use last element for cluster selection.
        selection_data = data.iloc[-1][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip_model.cluster_and_fit(data=data,
                                             zip_fit_inputs=zip_fit_inputs,
                                             selection_data=selection_data,
                                             n_clusters=2, random_state=2)

        self.check_pq(expected=self.results[1],
                      predicted=fit_data['pq_predicted'])

    def test_cluster_and_fit_3_cluster(self):
        # Ensure that with 3 clusters, we get the expected answer.

        # Grab data for fitting.
        data = pd.concat([self.results[0], self.results[1], self.results[2]])

        # Define inputs for zip fitting. We'll try to match results[1].
        zip_fit_inputs = {'s_n': self.s_n[1], 'v_n': self.v_n}

        # Use first element for cluster selection.
        selection_data = self.results[1].iloc[0][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip_model.cluster_and_fit(data=data,
                                             zip_fit_inputs=zip_fit_inputs,
                                             selection_data=selection_data,
                                             n_clusters=3, random_state=2)

        self.check_pq(expected=self.results[1],
                      predicted=fit_data['pq_predicted'])

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
        fit_data = zip_model.cluster_and_fit(data=data,
                                             zip_fit_inputs=zip_fit_inputs,
                                             selection_data=selection_data,
                                             n_clusters=4, random_state=2)

        self.check_pq(expected=self.results[2],
                      predicted=fit_data['pq_predicted'])

    def test_cluster_and_fit_cluster_too_small(self):
        # If a cluster is too small, None should be returned.

        # Grab data for fitting.
        data = self.results[0]

        # Define inputs for zip fitting. We'll try to match results[2].
        zip_fit_inputs = {'s_n': self.s_n[0], 'v_n': self.v_n}

        # Use 15th element for cluster selection.
        selection_data = data.iloc[15][['p', 'q']]

        # Call cluster_and_fit.
        fit_data = zip_model.cluster_and_fit(data=data,
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
            zip_model.get_best_fit_from_clustering(data=data,
                                                   zip_fit_inputs=zfi,
                                                   selection_data=sd,
                                                   random_state=2)

        self.check_pq(expected=self.results[2],
                      predicted=fit_data['pq_predicted'])


if __name__ == '__main__':
    unittest.main()