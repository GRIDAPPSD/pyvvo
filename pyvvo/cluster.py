"""Module for performing K-means clustering on data.

In the context of PyVVO, functions here are used before performing fits
to the ZIP load model (see zip.py).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def find_best_cluster(cluster_data, selection_data,
                      n_clusters, random_state=None):
    """

    :param cluster_data: pandas DataFrame containing data which will be
                         clustered via K-means. NOTE: data should be
                         normalized.
    :param selection_data: pandas Series containing data which will
                           be used to select the best cluster. NOTE:
                           data should be normalized in same fashion as
                           cluster_data.
    :param n_clusters: int, number of clusters to create.
    :param random_state: int, numpy random.RandomState object, or None.
                         Used to seed the KMeans object.
    :return:
    """

    # Initialize K-means object.
    km = KMeans(n_clusters=n_clusters, random_state=random_state)

    # Perform clustering.
    km.fit(cluster_data)

    # Get cluster centers as a pandas DataFrame.
    centers = pd.DataFrame(km.cluster_centers_, columns=cluster_data.columns)

    # Find the sum squared Euclidean distance. Note we're only looking
    # at columns from the selection data.
    squared_distance = \
        euclidean_distance_squared(selection_data,
                                   centers[selection_data.index])

    # Get the minimum squared distance, which represents the best
    # "label" from the KMeans object.
    try:
        # noinspection PyUnresolvedReferences
        best_label = np.argmin(squared_distance.values)
    except AttributeError:
        best_label = np.argmin(squared_distance)

    # Get a boolean of where data in cluster_data belongs to the best
    # cluster.
    label_match = km.labels_ == best_label

    # Grab the best data.
    best_data = cluster_data[label_match]

    # Done. Return the KMeans object and best data.
    return best_data, label_match, km


def euclidean_distance_squared(v1, v2):
    """Find squared Euclidean distance between two data sets.

    :param v1: 1st data set. Can be numpy array, pandas Series, or
               pandas DataFrame. DataFrame is assumed to be simple 2-D
               DataFrame without a MultiIndex.
    :param v2: 2nd data set. "..."
    :return: scalar value representing squared Euclidean distance
             distance between the two data sets.
    """
    # Determine what axis to sum across. NOTE: this means that for
    # DataFrames, we're finding the square difference for each row
    # (across columns).
    axis = max((len(v1.shape), len(v2.shape))) - 1

    # Compute and return distance.
    return np.sum(np.square(v1 - v2), axis=axis)


def feature_scale(x, x_ref=None):
    """Perform feature scaling: all data mapped to set [0, 1].

    NOTE: DataFrames are feature scaled by column.

    NOTE: If x_ref creates a column of NaNs or infs (x is DataFrame) or
    the entire Series NaN or infs (x is Series), NaNs are zeroed out.

    :param x: Data to feature scale. pandas DataFrame or Series
    :param x_ref: Reference data to scale x to. None or pandas DataFrame

    :return: x_prime: feature scaled version of x
    """
    # Check if x is a DataFrame.
    x_is_df = isinstance(x, pd.DataFrame)

    # If x isn't a DataFrame, and it isn't a Series, we have troubles.
    if (not x_is_df) and (not isinstance(x, pd.Series)):
        raise TypeError('x must be a DataFrame or Series!')

    # If x_ref is None, set it to x.
    if x_ref is None:
        x_ref = x
    elif not isinstance(x, pd.Series):
        raise TypeError('x_ref must be None or a pandas Series.')

    # Scale.
    x_prime = (x - x_ref.min()) / (x_ref.max() - x_ref.min())

    # If an entire column is NaN, it should be zeroed out. This will
    # happen if x_ref has a column of zeros.
    if x_is_df:
        # For DataFrame, zero out columns which are entirely NaN.
        x_prime[x_prime.columns[x_prime.isnull().all()]] = 0
        # Zero out columns which are entirely infinite.
        x_prime[x_prime.columns[np.isinf(x_prime).all()]] = 0
    else:
        # Series is simpler - just 0 NaN's.
        x_prime[x_prime.isnull()] = 0
        # Zero infinite values.
        x_prime[np.isinf(x_prime)] = 0

    # If we still have any NaN's, raise an error.
    if x_prime.isnull().any().any():
        raise UserWarning('NaN values found in x_prime!')

    # Done.
    return x_prime
