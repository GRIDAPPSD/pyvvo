"""
Code for creating ZIP load models. Note that this module follows Python
conventions: functions which start with an underscore are "private." So,
a user of this module should only use methods which do not start with an
underscore. At the time of writing (2020-06-17), the public methods are:
    - cluster_and_fit
    - get_best_fit_from_clustering
    - zip_fit

A discussion of load modeling in PyVVO can be found in `this paper
<http://hdl.handle.net/10125/64115>`__.

Discussion of the ZIP modeling follows:

ZIP load models represent a load as part constant impedance (Z), part
constant current (I) and part constant power (P).

Since PyVVO uses GridLAB-D, we'll be formulating the problem the same
way GridLAB-D does.

.. math::

    P_k \\! = \\! S_n \\! \\bigg[\\! \\frac{V_k^2}{V_n^2} Z_\\% \\cos(Z_\\theta) + \\frac{V_k}{V_n} I_\\% \\cos(I_\\theta) + P_\\%  \\cos(P_\\theta) \\bigg]

    Q_k \\! = \\! S_n \\! \\bigg[\\! \\frac{V_k^2}{V_n^2} Z_\\%  \\sin(Z_\\theta) + \\frac{V_k}{V_n}  I_\\% \\sin(I_\\theta) + P_\\% \\sin(P_\\theta) \\bigg]

    1 = Z_\\% + I_\\% + P_\\%

Where:

    :math:`P_k`: Predicted real power for time/interval :math:`k`

    :math:`Q_k`: Predicted reactive power for time/interval :math:`k`

    :math:`S_n`: Magnitude of nominal power

    :math:`V_k`: Magnitude of input voltage for time/interval :math:`k`

    :math:`V_n`: Nominal voltage

    :math:`Z\\%`: Impedance fraction

    :math:`Z_\\theta`: Impedance angle

    :math:`I\\%`: Current fraction

    :math:`I_\\theta`: Current angle

    :math:`P\\%`: Power fraction

    :math:`P_\\theta`: Power angle

To reduce computations during optimization, we'll make the following
variable substitutions:

.. math::

    \\bar{P}:=\\frac{P_k}{S_n}

    \\bar{Q}:=\\frac{Q_k}{S_n}

    \\bar{V}:=\\frac{V_a}{V_n}

In this module, a "zip_terms" parameter will be used frequently. This
parameter is a numpy array with six entries in the following order:

:math:`Z\\%`, :math:`Z_\\theta`, :math:`I\\%`, :math:`I_\\theta`,
:math:`P\\%`, :math:`P_\\theta`.
"""

# Standard library
import math
import logging

# Installed packages
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# pyvvo
from pyvvo import cluster

# Set up a log.
LOG = logging.getLogger(__name__)

# Make numpy error out for floating point errors.
np.seterr(all='raise')

# Constant for ZIP coefficients. ORDER MATTERS!
ZIP_TERMS = ['impedance', 'current', 'power']

# FTOL is for convergence tolerance. From scipy docs, once we no longer
# get FTOL improvement between iterations, we consider it converged.
# The value here is the default as documented here:
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
F_TOL = 1e-6

# Cap iterations.
MAX_ITER = 500

# Define bounds for our terms.
# Order: Z%, Z_theta, I%, I_theta, P%, P_theta
# Since we're dealing with power angles of loads, we'll bound them to
# the right-half-plane.
BOUNDS = Bounds([-np.inf, -np.pi/2, -np.inf, -np.pi/2, -np.inf, -np.pi/2],
                [np.inf, np.pi/2, np.inf, np.pi/2, np.inf, np.pi/2])

# Use a simple starting point where each fraction is 1/3, and each angle
# is pi / 6 (resulting in a PF of ~0.9). This was set through some
# light-weight manual trial and error.
PAR_0 = (1/3, np.pi/6, 1/3, np.pi/6, 1/3, np.pi/6)

# Jacobian for our equality constraint is constant and not a function of
# the parameters.
EQ_JAC = np.array([1, 0, 1, 0, 1, 0])

# We'll be passing zip_terms around in the order Z%, Z_theta, I%,
# I_theta, P%, P_theta. Create masks to pull the fractions and angles.
FRACTION_MASK = np.array([True, False, True, False, True, False])
ANGLE_MASK = ~FRACTION_MASK


def zip_fit(vpq, v_n=240, s_n=None, par_0=PAR_0,
            f_tol=F_TOL, max_iter=MAX_ITER, fit_data=True):
    """Given V, P, and Q data, perform ZIP fit and get coefficients.

    :param vpq: pandas DataFrame with columns 'v' for voltage
           magnitude, 'p' for real power, and 'q' for reactive power.
    :param v_n: nominal voltage magnitude.
    :param s_n: nominal apparent power magnitude. If None, it will be
           estimated/inferred from the vpq data.
    :param par_0: Initial guess/starting point for optimization. Should
           be array in the order Z%, Z_theta, I%, I_theta, P%, P_theta.
    :param f_tol: Precision goal for optimization. Terminates after
           change between iterations is < f_tol
    :param max_iter: Maximum number of iterations for optimization.
    :param fit_data: Boolean flag. If true, include fitted p and q along
           with the corresponding mean square error.

    :return: dictionary with several fields:

        -   zip_gld:  Dictionary with all the terms needed for GridLAB-D
            modeling. These include:

            -   base_power: S_n
            -   impedance_fraction: Z%
            -   impedance_pf: Impedance "power factor," cos(Z_theta).
                Will be negative if the power factor is leading for
                GridLAB-D conventions
            -   current_fraction: I%
            -   current_pf: Current "power factor," cos(I_theta).
                Negative if leading pf.
            -   power_fraction: P%
            -   power_pf: Power "power factor," cos(P_theta). Negative
                if leading pf.

        -   sol: scipy.optimize.OptimizeResult object from performing
            the ZIP fit (`docs
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult>`__).
            Fields of note include 'x' (contains the zip_terms),
            'success' (indicating if the optimizer exited successfully),
            and 'message' (description of the cause of optimizer
            termination).

        If fit_data is true, the following fields will be included:

        -   p_pred: numpy array of predicted real power for the
            resultant ZIP model.
        -   q_pred: numpy array of predicted reactive power for the
            resultant ZIP model.
        -   mse_p: mean-squared error for real power.
        -   mse_q: mean-squared error for reactive power.
        **IMPORTANT NOTE**: If the optimization fails, the only field
        in the return will be 'sol.' So it's up to the caller to either
        explicitly check sol.success or handle missing fields.
    """

    # Estimate nominal power if not provided.
    if s_n is None:
        s_n = _estimate_nominal_power(vpq)

    # Variable substitution to reduce multiplication/division during
    # optimization.
    vpq_bar = _get_vpq_bar(vpq=vpq, v_n=v_n, s_n=s_n)

    # Solve.
    sol = _zip_fit_slsqp(vpq_bar=vpq_bar, par_0=par_0, f_tol=f_tol,
                         max_iter=max_iter)

    # Initialize return.
    out = {'sol': sol}

    # If this failed, log and return.
    if not sol.success:
        LOG.warning('Unable to solve. Message: {}'.format(sol.message))
        return out

    # Get ZIP terms in GridLAB-D format.
    zip_gld = _zip_to_gld(sol.x)

    # Add base power.
    zip_gld['base_power'] = s_n

    # Add zip_gld to our return.
    out['zip_gld'] = zip_gld

    # Compute the predicted values if asked to.
    if fit_data:
        p_pred, q_pred = _zip_model(v=vpq['v'].values, v_n=v_n, s_n=s_n,
                                    zip_terms=sol.x)
        out['p_pred'] = p_pred
        out['q_pred'] = q_pred

        # Compute mean squared error.
        out['mse_p'] = \
            mean_squared_error(y_true=vpq['p'].values, y_pred=p_pred)

        out['mse_q'] = \
            mean_squared_error(y_true=vpq['q'].values, y_pred=q_pred)

    # Done.
    return out


def _get_vpq_bar(vpq, v_n, s_n):
    """Helper to scale our v, p, and q. This helps reduce the amount
    of floating point operations used during optimization.

    :param vpq: Pandas DataFrame with columns v, p, and q.
    :param v_n: scalar, nominal voltage.
    :param s_n: scalar, nominal apparent power magnitude.

    :returns: DataFrame with v_bar, p_bar, and q_bar, which are
        scaled parameters to be used in optimization.
    """
    return pd.DataFrame(data={'v_bar': vpq['v'] / v_n,
                              'p_bar': vpq['p'] / s_n,
                              'q_bar': vpq['q'] / s_n})


def _estimate_nominal_power(vpq):
    """Estimate nominal power from p and q.

    :param vpq: pandas DataFrame with columns 'v' for voltage
           magnitude, 'p' for real power, and 'q' for reactive power.
    :return: s_n: our estimate for nominal power.
    """
    # |S| = sqrt(P^2 + Q^2)
    s_n = np.median(np.sqrt(np.square(vpq['p']) + np.square(vpq['q'])))
    return s_n


def _zip_fit_slsqp(vpq_bar, par_0=PAR_0, f_tol=F_TOL,
                   max_iter=MAX_ITER):
    """Wrapper to call scipy.optimize.minimize.

    :param vpq_bar: Pandas DataFrame with columns v_bar, p_bar, and
        q_bar.
    :param par_0: Initial guess of zip parameters. Should be in order
        Z%, Z_theta, I%, I_theta, P%, P_theta.
    :param f_tol: Precision goal for the value of f in the stopping
        criterion of SLSQP.
    :param max_iter: Maximum number of iterations to solve.

    :return: scipy OptimizeResult object for this problem.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    """
    # Simply call minimize and return the result.
    return \
        minimize(
            # Call _zip_obj_and_jac for our objective function.
            fun=_zip_obj_and_jac, x0=par_0,
            # By setting jac=True, we indicate the objective function
            # will also return the Jacobian. This is for efficiency.
            jac=True,
            # _zip_obj_and_jac takes additional args v_s, v_bar, p_bar,
            # and q_bar.
            args=(np.square(vpq_bar['v_bar'].values),
                  vpq_bar['v_bar'].values,
                  vpq_bar['p_bar'].values,
                  vpq_bar['q_bar'].values),
            # Use sequential least squares programming, which is great if
            # you're fitting to a known model.
            method='SLSQP',
            # We only have one constraint: Z% + I% + P% = 1.
            # Note the Jacobian for that constraint is always the same.
            # TODO: is initializing lambda functions each time adding
            #   overhead? Should there be be regular functions defined
            #   for 'fun' and 'jac'?
            constraints={
                'type': 'eq',
                'fun': lambda x: np.array([np.sum(x[FRACTION_MASK]) - 1]),
                'jac': lambda x: EQ_JAC},
            # We don't have any bounds on our fractions, but we'll be
            # keeping the angles within the right-half-plane.
            bounds=BOUNDS,
            # Pass in additional options. We don't want to print convergence
            # messages, but will rather rely on the caller to check the
            # solution.
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
            options={'ftol': f_tol, 'maxiter': max_iter, 'disp': False}
        )


def _zip_obj_and_jac(zip_terms, v_s, v_bar, p_bar, q_bar):
    """ZIP objective function and Jacobian calculations. The objective
    function is the normalized sum of squared error, handling error for
    P and Q separately.

    :param zip_terms: numpy array with length 6. Should have terms Z%,
        Z_theta, I%, I_theta, P%, and P_theta in that order.
    :param v_s: numpy array, (v / v_nominal)^2
    :param v_bar: numpy array, v / v_nominal
    :param p_bar: numpy array, p / |s_n|
    :param q_bar: numpy array, q / |s_n|

    :returns: obj, jac. obj is the scalar value of the objective
        function, and jac is a numpy array the same shape as zip_terms
        with the partial derivative of the objective function with
        respect to each term in zip_terms.

    TODO: Should we normalize? Does it really matter? What helps the
        numerical stability of SLSQP?
    """
    # Pre-compute some terms we'll be using frequently.
    cos_z_t = math.cos(zip_terms[1])
    sin_z_t = math.sin(zip_terms[1])
    cos_i_t = math.cos(zip_terms[3])
    sin_i_t = math.sin(zip_terms[3])
    cos_p_t = math.cos(zip_terms[5])
    sin_p_t = math.sin(zip_terms[5])

    # The objective function and every partial derivative will involve
    # P - .... and Q - ... Pre-compute them here.
    p_delta = (p_bar
               - v_s * zip_terms[0] * cos_z_t
               - v_bar * zip_terms[2] * cos_i_t
               - zip_terms[4] * cos_p_t)

    q_delta = (q_bar
               - v_s * zip_terms[0] * sin_z_t
               - v_bar * zip_terms[2] * sin_i_t
               - zip_terms[4] * sin_p_t)

    # Compute some other terms which will be used more than once. Using
    # 'dot' as it's presumably faster than elementwise multiplication
    # followed by a sum.
    p_v_s_dot = np.dot(p_delta, v_s)
    q_v_s_dot = np.dot(q_delta, v_s)
    p_v_dot = np.dot(p_delta, v_bar)
    q_v_dot = np.dot(q_delta, v_bar)
    p_sum = np.sum(p_delta)
    q_sum = np.sum(q_delta)

    # Compute the value of the objective function.
    obj = np.sum(np.square(p_delta) + np.square(q_delta))

    # Initialize our Jacobian return.
    jac = np.zeros_like(zip_terms)

    # Compute the partial derivative w.r.t. Z% (0th term)
    jac[0] = (2 *
              (-p_v_s_dot * cos_z_t
               +
               -q_v_s_dot * sin_z_t)
              )

    # Compute the partial derivative w.r.t Z_theta (1st term)
    jac[1] = (2 *
              (-p_v_s_dot * zip_terms[0] * -sin_z_t
               +
               -q_v_s_dot * zip_terms[0] * cos_z_t)
              )

    # Compute the partial derivative w.r.t I% (2nd term)
    jac[2] = (2 *
              (-p_v_dot * cos_i_t
               +
               -q_v_dot * sin_i_t)
              )

    # Compute the partial derivative w.r.t. I_theta (3rd term)
    jac[3] = (2 *
              (-p_v_dot * zip_terms[2] * -sin_i_t
               +
               -q_v_dot * zip_terms[2] * cos_i_t)
              )

    # Compute the partial derivative w.r.t. P% (4th term)
    jac[4] = (2 * (p_sum * -cos_p_t - q_sum * sin_p_t))

    # Compute the partial derivative w.r.t. P_theta (5th term)
    jac[5] = (2 *
              (p_sum * -zip_terms[4] * -sin_p_t
               +
               q_sum * -zip_terms[4] * cos_p_t)
              )

    return obj, jac


def _zip_model(v, v_n, s_n, zip_terms):
    """Compute P and Q for a given ZIP model. This is generally used
    for testing, not for the fitting/optimization itself.

    :param v: numpy array of voltages.
    :param v_n: scalar, nominal voltage.
    :param s_n: scalar, nominal apparent power magnitude.
    :param zip_terms: numpy array with the terms Z%, Z_theta, I%,
        I_theta, P%, and P_theta, in that order.
    """
    v_s = np.square(v / v_n)
    p = s_n * (
            v_s * zip_terms[0] * math.cos(zip_terms[1])
            + v / v_n * zip_terms[2] * math.cos(zip_terms[3])
            + zip_terms[4] * math.cos(zip_terms[5])
    )

    q = s_n * (
            v_s * zip_terms[0] * math.sin(zip_terms[1])
            + v / v_n * zip_terms[2] * math.sin(zip_terms[3])
            + zip_terms[4] * math.sin(zip_terms[5])
    )

    return p, q


def _zip_model_gld(v, v_n, s_n, gld_terms):
    """Wrapper to call _zip_model given a dictionary of GridLAB-D terms.

    :param v: numpy array of voltages.
    :param v_n: scalar, nominal voltage.
    :param gld_terms: Dictionary of GridLAB-D terms for the ZIP model,
        as would be output from _zip_to_gld.
    """
    zip_terms = np.zeros(6)

    # Dump the fractions into the right slots.
    zip_terms[FRACTION_MASK] = np.array([
        gld_terms['impedance_fraction'], gld_terms['current_fraction'],
        gld_terms['power_fraction']
    ])

    # Compute angles. Start by extracting the power factors.
    pf = np.array([
        gld_terms['impedance_pf'], gld_terms['current_pf'],
        gld_terms['power_pf']])
    angles = _angles_from_power_factors(pf=pf)

    # Dump the angles into the correct slots.
    zip_terms[ANGLE_MASK] = angles

    return _zip_model(v=v, v_n=v_n, s_n=s_n, zip_terms=zip_terms)


def _power_factors_from_zip_terms(zip_terms):
    """Given the ZIP terms, compute the power factors related to them.

    Note that the percentages are not relevant to this function, but
    we'll take all the terms so the caller can be agnostic about which
    terms are used.

    :param zip_terms: numpy array, Z%, Z_theta, I%, I_theta, P%, and
        P_theta, in that order.
    :returns: numpy array, Z_pf, I_pf, and P_pf in that order.
    """
    # Extract the angles.
    angles = zip_terms[ANGLE_MASK]

    # Initialize the power factors to be the cosine of the power angles.
    pf = np.cos(angles)

    # Negative angles result in negative (leading) power factors.
    pf[angles < 0] *= -1

    # All done.
    return pf


def _angles_from_power_factors(pf):
    """Helper to convert a given power factor to a power angle.

    :param pf: Numpy array of power factors. Negative means leading.
        Note that all angles will be assumed to be in the right-half
        plane.

    :returns: angles. Numpy array with angles (radians) corresponding
        to each pf.
    """
    # The power angle is simply the inverse cosine of the power factor.
    angles = np.arccos(np.abs(pf))
    # Get the sign right.
    angles[pf < 0] *= -1

    # Done.
    return angles


def _zip_to_gld(zip_terms):
    """Given zip_terms, return a dictionary of GridLAB-D terms.

    :param zip_terms: numpy array of our ZIP terms in the order Z%,
        Z_theta, I%, I_theta, P%, P_theta.

    :returns: dictionary with the following fields:
        - impedance_fraction
        - impedance_pf
        - current_fraction
        - current_pf
        - power_fraction
        - power_pf
    """
    # Simply assign the fractions.
    out = {'impedance_fraction': zip_terms[0],
           'current_fraction': zip_terms[2],
           'power_fraction': zip_terms[4]}

    # Convert the angle to power factors.
    pf = _power_factors_from_zip_terms(zip_terms)

    # Assign.
    out['impedance_pf'] = pf[0]
    out['current_pf'] = pf[1]
    out['power_pf'] = pf[2]

    # All done.
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

    :return: fit_outputs: outputs from ZIP fit plus a 'data_len' field,
        or None. None will be returned if we're clustering and the
        min_cluster_size requirement is not met. The 'data_len' field
        indicates the length/height of the data which went into fitting.
    """

    # If we're clustering, do so.
    if selection_data is not None:
        # For K-Means, it's best to first standardize the data so that
        # it looks Gaussian.
        #
        # Initialize a StandardScaler, and fit it to our data.
        scaler = StandardScaler()
        scaler.fit(data.values)
        # TODO: MOVE THIS OUTSIDE OF THIS FUNCTION. THE DATA CAN BE
        #   SCALED ONCE IN THE CALLING FUNCTION, e.g.,
        #   get_best_fit_from_clustering. Likewise, the selection_data
        #   can be scaled outside this function.
        # Create a DataFrame for holding scaled data.
        # TODO: We're adding extra over-head to use a DataFrame, but
        #   this is a quick fix without messing with
        #   cluster.find_best_cluster.
        scaled_data = pd.DataFrame(scaler.transform(data.values),
                                   index=data.index, columns=data.columns)

        # We also need to scale the selection data.
        # Initialize a Series which has all the "columns" of our data.
        tmp_series = pd.Series(0, index=data.columns)
        # Fill the Series with our selection data values.
        tmp_series[selection_data.index] = selection_data
        # Now scale the temporary Series. Note the reshaping is for a
        # single sample (1 row by X columns), and ravel puts the data
        # back into a 1D array for Series creation.
        scaled_selection = pd.Series(
            scaler.transform(tmp_series.values.reshape(1, -1)).ravel(),
            index=tmp_series.index)

        # Note that 'v' is dropped from the cluster_data, and we're
        # plucking the appropriate selection data.
        data_out, best_bool, _ = cluster.find_best_cluster(
            cluster_data=scaled_data.drop('v', axis=1),
            selection_data=scaled_selection[selection_data.index],
            n_clusters=n_clusters,
            random_state=random_state)

        # Re-associate voltage data.
        data_out['v'] = scaled_data[best_bool]['v']

        # "Un-scale" the fit_data.
        # TODO: Again, we've got extra overhead by using DataFrames.
        fit_data = pd.DataFrame(
            scaler.inverse_transform(data_out[data.columns]),
            index=data_out.index, columns=data.columns)
    else:
        # No clustering.
        fit_data = data

    # If we aren't clustering, or if we are and have enough data,
    # perform the fit.
    if (selection_data is None) or (fit_data.shape[0] >= min_cluster_size):
        fit_outputs = zip_fit(fit_data[['v', 'p', 'q']], **zip_fit_inputs)
        fit_outputs['data_len'] = fit_data.shape[0]
    else:
        # Otherwise,
        fit_outputs = None

    return fit_outputs


def get_best_fit_from_clustering(data, zip_fit_inputs, selection_data=None,
                                 min_cluster_size=4, random_state=None):
    """Loop over different numbers of clusters to find the best ZIP fit.

    For input descriptions, see ``cluster_and_fit`` function.

    This calls cluster_and_fit function for each loop iteration.

    NOTE: the 'fit_data' field of zip_fit_inputs will be overridden to
    be true, as this function won't work otherwise.

    :returns: best_fit. 'Best' output (smallest normalized mse_p
        + mse_q) from calling cluster_and_fit. It will also have a 'k'
        field added, indicating the number of clusters used.
    """
    # The length of our data must be larger than our minimum cluster
    # size.
    if len(data) < min_cluster_size:
        raise ValueError('The given data has length {}, but the given '
                         'minimum cluster size is {}.'
                         .format(len(data), min_cluster_size))

    # Override zip_fit_inputs
    zip_fit_inputs['fit_data'] = True

    # Track best coefficients and minimum normalized mean squared error.
    # Note we normalize the MSE by dividing by the length of data used
    # to come up with the ZIP coefficients. This normalization is
    # important, because you're likely to have a large MSE with more
    # data points.
    best_fit = None
    min_norm_mse = np.inf

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

        # Check normalized mse.
        norm_mse = ((fit_outputs['mse_p'] + fit_outputs['mse_q'])
                    / fit_outputs['data_len'])

        if norm_mse < min_norm_mse:
            min_norm_mse = norm_mse
            best_fit = fit_outputs
            best_fit['k'] = k

    # That's it. Return the best fit.
    return best_fit
