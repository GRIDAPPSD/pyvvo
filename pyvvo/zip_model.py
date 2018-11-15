"""
Code for creating ZIP load models.

ZIP load models represent a load as part constant impedance (Z), part
constant current (I) and part constant power (P).

Since pyvvo uses GridLAB-D, we'll be formulating the problem the same
way GridLAB-D does.

P_i = S_n * [(V_i/V_n)^2 * Z% * cos(Z_theta)
      + V_i/V_n * I% * cos(I_theta) + P% * cos(P_theta)]

Q_i = S_n * [(V_i/V_n)^2 * Z% * sin(Z_theta)
      + V_i/V_n * I% * sin(I_theta) + P% * sin(P_theta)]

Contrained by:
Z% + I% + P% = 1

Where:
    P_i: Predicted real power for time i
    Q_i: Predicted reactive power for time i
    S_n: Magnitude of nominal power
    V_i: Magnitude of input voltage for time i
    V_n: Nominal voltage
    Z%: Impedance fraction
    Z_theta: Impedance angle
    I%: Current fraction
    I_theta: Current angle
    P%: Power fraction
    P_theta: Power angle

To simplify for solving, we'll reformulate the problem as follows:

P_bar = P_i/S_n
Q_bar = Q_i/S_n
V_bar = V_i/V_n

The following polynomial terms are what we solve for (and then back out
the individual terms):

a1 = Z% * cos(Z_theta)
a2 = I% * cos(I_theta)
a3 = P% * cos(P_theta)

b1 = Z% * sin(Z_theta)
b2 = I% * sin(I_theta)
b3 = P% * sin(P_theta)


Original code in R written by Dave Engel, and originally transcribed to
Python by Bin Zheng.

All further modifications by Brandon Thayer.
"""

# Standard library
import math
import multiprocessing as mp
from queue import Empty, Queue
from time import process_time

# Installed packages
import numpy as np
import pandas as pd
from mystic.constraints import with_penalty
import mystic.penalty
from scipy.optimize import minimize
# import mystic.solvers as my

# pyvvo
from pyvvo import cluster

# Make numpy error out for floating point errors.
np.seterr(all='raise')

# Constant for ZIP coefficients. ORDER MATTERS!
ZIP_TERMS = ['impedance', 'current', 'power']

# List of solvers which are set up.
# SOLVERS = ['fmin_powell', 'SLSQP']
# NOTE: fmin_powell just doesn't seem to work well, so has been removed.
# TODO: Put more effort into get fmin_powell working well.
SOLVERS = ['SLSQP']

# FTOL is for convergence tolerance. From scipy docs, once we no longer
# get FTOL improvement between iterations, we consider it converged
F_TOL = 5e-5

# Cap iterations.
MAX_ITER = 500

# Bound the polynomial terms (a1-a3 and b1-b3). Note sin/cos are always
# on interval [-1, 1], and ZIP fractions shouldn't normally exceed 1.
POLY_BOUNDS = [(-1, 1) for x in range(6)]

# NOTE: THERE'S A CONSTANT DEFINITION AFTER THE FOLLOWING FUNCTION.


def _get_poly_from_zip(z_pct, i_pct, p_pct, z_pf, i_pf, p_pf):
    """Get polynomial terms from ZIP coefficients.

    :param z_pct: Impedance fraction.
    :param i_pct: Current fraction.
    :param p_pct: Power fraction.
    :param z_pf: Impedance power factor (cos(Z_theta)). Negative if the
                 power factor is leading for the impedance component.
    :param i_pf: Current "..." for the current component.
    :param p_pf: Power "..." for the power component.

    :returns poly_terms: tuple of (a1, a2, a3, b1, b2, b3). See module
             docstring for more details (or just read the code).
    """

    # GridLAB-D uses absolute value on the pf's when computing real
    # power.
    poly_terms = [z_pct * abs(z_pf), i_pct * abs(i_pf), p_pct * abs(p_pf),
                  z_pct * abs(math.sin(math.acos(z_pf))),
                  i_pct * abs(math.sin(math.acos(i_pf))),
                  p_pct * abs(math.sin(math.acos(p_pf)))]

    # Keep it simple and hard code. If power factor is negative, flip
    # the b terms (reactive power). This is how GridLAB-D does it.
    if z_pf < 0:
        poly_terms[3] *= -1

    if i_pf < 0:
        poly_terms[4] *= -1

    if p_pf < 0:
        poly_terms[5] *= -1

    return tuple(poly_terms)


# Define default initial guess for ZIP models.
# We'll take the Oscillating Fan from the CVR report:
# https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-19596.pdf
# Z%: 73.32, I%: 25.34, P%: 1.35
# Zpf: 0.97, Ipf: 0.95, Ppf: -1.0
PAR_0_ZIP = (0.7332, 0.2534, 0.0135, 0.97, 0.95, -1)
PAR_0 = _get_poly_from_zip(*PAR_0_ZIP)


def zip_fit(vpq, v_n=240, s_n=None, solver='SLSQP', par_0=PAR_0,
            f_tol=F_TOL, max_iter=MAX_ITER, fit_data=True):
    """Given V, P, and Q data, perform ZIP fit and get coefficients.

    :param vpq: pandas DataFrame with columns 'v' for voltage
           magnitude, 'p' for real power, and 'q' for reactive power.
    :param v_n: nominal voltage magnitude.
    :param s_n: nominal apparent power magnitude. If None, it will be
           estimated/inferred from the vpq data.
    :param solver: Solver to use. Supported solvers listed in SOLVERS
    :param par_0: Initial guess/starting point for optimization. Should
           be array like so: (a1, a2, a3, b1, b2, b3). If given None,
           the default PAR_0 will be used.
    :param f_tol: Precision goal for optimization. Terminates after
           change between iterations is < f_tol
    :param max_iter: Maximum number of iterations for optimization.
    :param fit_data: Boolean flag. If true, include fitted p and q along
           with root mean square deviation data.

    :return: dict of ZIP coefficients for GridLAB-D.

    Fields for GridLAB-D modeling:
        base_power: S_n
        impedance_fraction: Z%
        impedance_pf: Impedance "power factor," cos(Z_theta). Should be
            negative if the power factor is leading for GridLAB-D
            conventions
        current_fraction: I%
        current_pf: Current "power factor," cos(I_theta). Negative if
            power factor is leading.
        power_fraction: P%
        power_pf: Power "power factor," cos(P_theta). Negative if
            power factor is leading.

    Other fields:

    """

    # Estimate nominal power if not provided.
    if s_n is None:
        s_n = _estimate_nominal_power(vpq)

    # If given None for par_0, grab default.
    if par_0 is None:
        par_0 = PAR_0

    # Variable substitution to get to polynomial format.
    vpq_bar = pd.DataFrame(data={'v_bar': vpq['v'] / v_n,
                                 'p_bar': vpq['p'] / s_n,
                                 'q_bar': vpq['q'] / s_n})

    # Solve.
    if solver == 'SLSQP':
        sol = _zip_fit_slsqp(vpq_bar, par_0, f_tol, max_iter)
        """
        elif solver == 'fmin_powell':
            sol = _zip_fit_fmin_powell(vpq_bar, par_0, f_tol, max_iter)
        """
    else:
        raise UserWarning('Unknown solver: {}'.format(solver))

    # Get fractions and power factors from the polynomial terms.
    fractions, power_factors = \
        _get_fractions_and_power_factors(sol['poly_terms'])

    # Get fractions and power_factors in GridLAB-D format.
    zip_gld = _fractions_power_factors_to_gld_terms(fractions, power_factors)

    # Add base power.
    zip_gld['base_power'] = s_n

    # Add success and message
    out = {'zip_gld': zip_gld, 'success': sol['success'],
           'message': sol['message']}

    # Fit the data.
    if fit_data:
        pq_predicted = _pq_from_fractions_and_power_factors(fractions,
                                                            power_factors,
                                                            v=vpq['v'],
                                                            v_n=v_n, s_n=s_n)
        out['pq_predicted'] = pq_predicted

        # Compute root mean square deviation.
        out['rmsd_p'] = compute_rmsd(actual=vpq['p'],
                                     predicted=pq_predicted['p_predicted'])
        out['rmsd_q'] = compute_rmsd(actual=vpq['q'],
                                     predicted=pq_predicted['q_predicted'])

    # Done.
    return out


def _estimate_nominal_power(vpq):
    """Estimate nominal power from p and q.

    :param vpq: pandas dataframe with columns 'v' for voltage
           magnitude, 'p' for real power, and 'q' for reactive power.
    :return: s_n: our estimate for nominal power.
    """
    # |S| = sqrt(P^2 + Q^2)
    s_n = np.median(np.sqrt(np.square(vpq['p']) + np.square(vpq['q'])))
    return s_n


def _zip_fit_slsqp(vpq_bar, par_0=PAR_0, f_tol=F_TOL, max_iter=MAX_ITER):
    """Use scipy minimize with 'SLSQP' solver to find ZIP parameters.

    :param vpq_bar: pandas dataframe with columns v_bar, p_bar, q_bar
    :param par_0: array of (a1, a2, a3, b1, b2, b3) for optimimization
           starting point.
    :param f_tol: from scipy: "Precision goal for the value of f in the
           stopping criterion."
    :param max_iter: maximum number of optimization iterations.
    :return: dictionary with fields:
             poly_terms: (a1, a2, a3, b1, b2, b3)
             success: boolean, whether optimization succeeded
             message: Message from solver.
    """

    # Solve.
    sol = minimize(fun=_zip_objective, x0=par_0, args=(vpq_bar,),
                   method='SLSQP', constraints={'type': 'eq',
                                                'fun': _zip_constraint,
                                                'bounds': POLY_BOUNDS},
                   options={'ftol': f_tol, 'maxiter': max_iter})

    # Return.
    return {'poly_terms': sol.x, 'success': sol.success,
            'message': sol.message}


'''
def _zip_fit_fmin_powell(vpq_bar, par_0=PAR_0, f_tol=F_TOL, max_iter=MAX_ITER):
    """Use mystic fmin_powell solver to find ZIP parameters.

    :param vpq_bar: pandas dataframe with columns v_bar, p_bar, q_bar
    :param par_0: array of (a1, a2, a3, b1, b2, b3) for optimimization
           starting point.
    :param f_tol: from scipy: "Precision goal for the value of f in the
           stopping criterion."
    :param max_iter: maximum number of optimization iterations.
    :return: dictionary with fields:
             poly_terms: (a1, a2, a3, b1, b2, b3)
             success: boolean, whether optimization succeeded
             message: Message from solver.
    """
    sol = my.fmin_powell(cost=_zip_objective, x0=par_0, args=(vpq_bar,),
                         bounds=POLY_BOUNDS,
                         penalty=_zip_constraint_fmin_powell,
                         disp=False, ftol=f_tol, full_output=True,
                         maxiter=max_iter)

    # Get message.
    if sol[4] == 1:
        msg = 'Maximum number of function evaluations'
        success = False
    elif sol[4] == 2:
        msg = 'Maximum number of iterations'
        success = False
    elif sol[4] == 0:
        msg = 'Success'
        success = True
    else:
        s = 'Unexpected warnflag for fmin_powell : {}'.format(sol[4])
        raise UserWarning(s)

    # Return.
    return {'poly_terms': sol[0], 'success': success, 'message': msg}
'''


def _zip_objective(poly_terms, vpq_bar):
    """Objective function to minimize. Normalized sum squared error.

    :param poly_terms: array of (a1, a2, a3, b1, b2, b3)
    :param vpq_bar: pandas DataFrame with columns v_par, p_bar, q_bar
    :return: scalar sum squared error divided by length of elements.
    """
    # Get fractions and power factors from the polynomial terms.
    fractions, power_factors = _get_fractions_and_power_factors(poly_terms)

    # Compute predicted p and q values. Note that s_n and v_n are
    # already baked into the 'bar' terms. It's important to use 'values'
    # from vpq_bar['v_bar'] to ensure a Pandas Series doesn't get passed
    # in.
    pq_predicted = \
        _pq_from_fractions_and_power_factors(fractions, power_factors,
                                             v=vpq_bar['v_bar'].values,
                                             v_n=1, s_n=1)

    # Compute sum of squared error.
    err = np.sum(np.square(vpq_bar['p_bar'] - pq_predicted['p_predicted'])
                 + np.square(vpq_bar['q_bar'] - pq_predicted['q_predicted']))

    # Normalize error by length of elements.
    return err / vpq_bar.shape[0]


def _zip_constraint(poly_terms):
    """Evaluate ZIP constraint of Z% + I% + P% = 1.

    :param poly_terms: array of (a1, a2, a3, b1, b2, b3)
    :return: sum of ZIP fractions minus one.
    """

    # Get ZIP fractions.
    fractions, _ = _get_fractions_and_power_factors(poly_terms)

    # Return difference with 1.
    return np.sum(fractions) - 1


@with_penalty(mystic.penalty.quadratic_equality)
def _zip_constraint_fmin_powell(poly_terms):
    """Penalty construction for fmin_powell solver."""
    return _zip_constraint(poly_terms)


def _get_fractions_and_power_factors(poly_terms):
    """Extract ZIP fractions and power factors from polynomial terms.

    :param poly_terms: array of (a1, a2, a3, b1, b2, b3)
    :return: tuple of (fractions, power_factors) IN ORDER (Z-I-P)
    """

    # Grab a and b tuples from poly_terms for readability.
    a = np.array(poly_terms[:3], dtype=float)
    b = np.array(poly_terms[3:], dtype=float)

    # Initialize fractions. The reduces correctly, but lacks sign
    # information.
    fractions = np.sqrt(np.square(a) + np.square(b))

    # Initialize power factors. Using np.divide's 'out' and 'where'
    # arguments, we ensure that division by zero results in a 1 for the
    # power factors.
    power_factors = np.absolute(np.divide(a, fractions,
                                          out=np.ones_like(a),
                                          where=(fractions != 0)))

    ####################################################################
    # GridLAB-D computes the real component using the absolute value of
    # the power factor. So we have to be really careful in getting the
    # signs right. The ASCII art below shows the four-quadrants of the
    # real/reactive plane, and the requisite signs of the power factors
    # and fractions. "Illustration 2" found at following link was
    # helpful: https://www.landisgyr.com/webfoo/wp-content/uploads/2012/
    # 09/Power_Flow_2.pdf
    #
    #
    #                                Q
    #              Quadrant 2        #  Quadrant 1
    #              Leading current   #  Lagging current
    #              P < 0, Q > 0      #  P > 0, Q > 0
    #              power factor < 0  #  power factor > 0
    #              fraction < 0      #  fraction > 0
    #                                #
    #                ################################# P
    #                                #
    #              Quadrant 3        #  Quadrant 4
    #              Lagging Current   #  Leading Current
    #              P < 0, Q < 0      #  P > 0, Q < 0
    #              power factor > 0  #  power factor < 0
    #              fraction < 0      #  fraction > 0
    #                                #
    #
    ####################################################################

    # Get boolean arrays for positive polynomial terms.
    pos_p = a > 0
    neg_p = ~pos_p
    pos_q = b > 0
    neg_q = ~pos_q

    # Quadrant 1:
    # No sign flipping needed, terms initialize to positive.
    
    # Quadrant 2:
    # Make both power factor and fraction negative.
    q2_bool = neg_p & pos_q
    power_factors[q2_bool] = power_factors[q2_bool] * -1
    fractions[q2_bool] = fractions[q2_bool] * -1
    
    # Quadrant 3:
    # Make fraction negative.
    q3_bool = neg_p & neg_q
    fractions[q3_bool] = fractions[q3_bool] * -1
    
    # Quadrant 4:
    # Make power factor negative.
    q4_bool = pos_p & neg_q
    power_factors[q4_bool] = power_factors[q4_bool] * -1

    return fractions, power_factors


def _poly_to_gld(poly_terms):
    """Take polynomial terms and convert to GridLAB-D format.

    :param poly_terms array of (a1, a2, a3, b1, b2, b3)
    :return dict with the following fields: impedance_fraction,
            impedance_pf, current_fraction, current_pf, power_fraction,
            power_pf
    """

    # Get fractions and power factors.
    fractions, power_factors = _get_fractions_and_power_factors(poly_terms)

    return _fractions_power_factors_to_gld_terms(fractions, power_factors)


def _fractions_power_factors_to_gld_terms(fractions, power_factors):
    """Given GridLAB-D ZIP terms from fractions and power factors.

    :param fractions: tuple of ZIP fractions in order (Z%, I%, P%)
    :param power_factors: tuple of ZIP power fractions in order
    :return: dictionary with GridLAB-D ZIP parameter names. These are:
             impedance_fraction, impedance_pf, current_fraction,
             current_pf, power_fraction, power_pf
    """
    out = {}

    # Name in GridLAB-D convention. THIS IS ORDER DEPENDENT.
    for i, k in enumerate(ZIP_TERMS):
        # Assign.
        out[k + '_fraction'] = fractions[i]
        out[k + '_pf'] = power_factors[i]

    # That's it.
    return out


def _pq_from_fractions_and_power_factors(fractions, power_factors, v, v_n,
                                         s_n):
    """Get P/Q given fractions, power_factors, voltage, and base power.

    This is written to emulate how GridLAB-D performs this calculation.

    WARNING: Ensure the v input is a numpy array, not a Pandas Series.
    To keep the optimization snappy, we won't be performing checks.

    :param fractions: tuple of ZIP fractions in order (Z%, I%, P%)
    :param power_factors: tuple of ZIP power_factors in order.
    :param v: numpy array of input voltage magnitudes.
    :param v_n: nominal voltage magnitude.
    :param s_n: nominal apparent power magnitude.
    :return: Pandas DataFrame with columns 'v,' 'p_predicted,' and
            'q_predicted.'
    """
    # Get fractions and power_factors as numpy arrays.
    f = np.array(fractions)
    pf = np.array(power_factors)

    # Initialize arrays for holding p and q terms.
    p_terms = np.zeros_like(f)
    q_terms = np.zeros_like(f)

    # Get boolean array of where the power factor is 0.
    zero_pf = pf == 0
    # In the case of a zero power factor, the p term will be 0. q term
    # simplifies to base power times fraction. (multiplication by
    # base power comes later)
    q_terms[zero_pf] = f[zero_pf]

    # If the power factor is not zero, compute p and q.
    nz_pf = ~zero_pf
    p_terms[nz_pf] = f[nz_pf] * abs(pf[nz_pf])
    q_terms[nz_pf] = p_terms[nz_pf] * np.sqrt(1 / np.square(pf[nz_pf]) - 1)

    # Flip sign if the power factor is less than 0 (leading).
    neg_pf = pf < 0
    q_terms[neg_pf] = q_terms[neg_pf] * -1

    # Pre-compute voltages.
    v_v_n = (v / v_n)
    v_squared = np.square(v_v_n)

    # Compute p and q.
    p = s_n * (v_squared * p_terms[0] + v_v_n * p_terms[1] + p_terms[2])
    q = s_n * (v_squared * q_terms[0] + v_v_n * q_terms[1] + q_terms[2])

    # Put in DataFrame and return.
    return pd.DataFrame({'v': v, 'p_predicted': p, 'q_predicted': q})


def _pq_from_v_zip_gld(v, v_n, zip_gld):
    """Compute P and Q given ZIP coefficients and voltage.

    This is just a convenience wrapper for calling
    _pq_from_fractions_and_power_factors.

    :param v: Array of voltage values. Could be numpy array or Pandas
           Series
    :param v_n: Nominal voltage.
    :param zip_gld: Dictionary of ZIP coefficients in GridLAB-D form.
    :return: See _pq_from_fractions_and_power_factors
    """
    # Map dictionary into arrays in Z-I-P order.
    fractions = (zip_gld['impedance_fraction'], zip_gld['current_fraction'],
                 zip_gld['power_fraction'])
    power_factors = (zip_gld['impedance_pf'], zip_gld['current_pf'],
                     zip_gld['power_pf'])

    # Call _pq_from_fractions_and_power_factors and return.
    return _pq_from_fractions_and_power_factors(fractions, power_factors, v=v,
                                                v_n=v_n,
                                                s_n=zip_gld['base_power'])


def compute_rmsd(actual, predicted):
    """Compute root mean square deviation for actual and predicted data.

    :param actual: nx1 Pandas Series or numpy array with the 'actual'
                   data
    :param predicted: "..." with the 'predicted' data
    :return: root mean square deviation.
    """
    out = math.sqrt(np.sum(np.square(actual - predicted)) / actual.shape[0])
    return out


def cluster_and_fit(data, zip_fit_inputs, selection_data=None, n_clusters=1,
                    min_cluster_size=4, random_state=None):
    """Cluster data and perform ZIP fit.

    Note that voltage will not be included in clustering.

    :param data: pandas DataFrame containing all data needed to cluster
                 (optional) and fit data. At the minimum, columns must
                 include v (voltage magnitude), p (active power), and q
                 (reactive power).
    :param zip_fit_inputs: dictionary of key word arguments to be passed
                           to the function zip_fit.
    :param selection_data: pandas Series with data to be used for
                           cluster selection. Index can only contain
                           labels that exist in data. NOTE:
                           selection_data should not have 'v' in it.
                           Optional. If None, no clustering is
                           performed.
    :param n_clusters: Integer of clusters to create for clustering.
                       Optional. Only required if selection_data is not
                       None.
    :param min_cluster_size: Minimum allowed number of data points in
                             the selected cluster. Optional. Only
                             required if selection_data is not None.
    :param random_state: Integer, numpy.random RandomState object
                         (preferred) or None. Used for random seeding of
                         K-Means clustering.

    :return: fit_outputs: outputs from ZIP fit, or None. None will be
                          returned if we're clustering and the
                          min_cluster_size requirement is not met.
    """

    # If we're clustering, do so.
    if selection_data is not None:
        # Note that 'v' is dropped from the cluster_data.
        fit_data, best_bool, _ = \
            cluster.find_best_cluster(cluster_data=data.drop('v', axis=1),
                                      selection_data=selection_data,
                                      n_clusters=n_clusters,
                                      random_state=random_state)
        # Re-associate voltage data.
        fit_data['v'] = data[best_bool]['v']
    else:
        # No clustering.
        fit_data = data

    # If we aren't clustering, or if we are and have enough data,
    # perform the fit.
    if (selection_data is None) or (fit_data.shape[0] >= min_cluster_size):
        fit_outputs = zip_fit(fit_data[['v', 'p', 'q']], **zip_fit_inputs)
    else:
        # Otherwise,
        fit_outputs = None

    return fit_outputs


def get_best_fit_from_clustering(data, zip_fit_inputs, selection_data=None,
                                 min_cluster_size=4, random_state=None):
    """Loop over different numbers of clusters to find the best ZIP fit.

    This calls cluster_and_fit function for each loop iteration.

    For input descriptions, see cluster_and_fit.

    NOTE: data and selection_data are assumed to already be normalized.

    NOTE: the 'fit_data' field of zip_fit_inputs will be overridden to
    be true, as this function won't work otherwise.

    :returns: best_fit. 'Best' output (smallest rmsd_p + rmsd_q) from
              calling cluster_and_fit
    """

    # Override zip_fit_inputs
    zip_fit_inputs['fit_data'] = True

    # Track best coefficients and minimum root mean square deviation.
    best_fit = None
    min_rmsd = np.inf

    # Compute maximum possible number of clusters.
    n = np.floor(data.shape[0] / min_cluster_size).astype(int)

    # Loop over different cluster sizes from maximum to minimum (1).
    for k in range(n, 0, -1):
        # Call cluster_and_fit
        fit_outputs = \
            cluster_and_fit(data=data, zip_fit_inputs=zip_fit_inputs,
                            selection_data=selection_data, n_clusters=k,
                            min_cluster_size=min_cluster_size,
                            random_state=random_state)

        # If None was returned, the cluster was too small. Move along.
        if fit_outputs is None:
            continue

        # Check to rmsd.
        rmsd = fit_outputs['rmsd_p'] + fit_outputs['rmsd_q']
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_fit = fit_outputs

    # That's it. Return the best fit.
    return best_fit


class ZIPManager():
    """Class for managing ZIP fits for a collection of meters."""

    def __init__(self, meters, num_workers):
        """

        :param meters: List of meter names for this manager to track.
        :param num_workers: Number of workers (processors) to use for
               performing ZIP fits.
        """
        # Track meters.
        self.meters = meters

        # Initialize queues for performing ZIP fitting. The input queue
        # will two slots per worker.
        self.input_queue = mp.JoinableQueue(maxsize=num_workers * 2)
        self.output_queue = mp.Queue()

        # TODO
        # Start processes.
