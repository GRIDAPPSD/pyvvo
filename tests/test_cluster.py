import unittest
import numpy as np
import pandas as pd
from pyvvo import cluster

class TestCluster(unittest.TestCase):

    def test_euclidean_distance_sum_squared_array(self):
        # All numbers are one away, so squares will be 1. Sum of squares
        #
        a1 = np.array((1, 2, 3))
        a2 = np.array((2, 1, 4))

        self.assertEqual(3, cluster.euclidean_distance_squared(a1, a2))

    def test_euclidean_distance_sum_squared_series(self):
        s1 = pd.Series((1, 10, 7, 5), index=['w', 'x', 'y', 'z'])
        s2 = pd.Series((3, 5, -1, 10), index=['w', 'x', 'y', 'z'])
        # 4+25+64+25 = 118
        self.assertEqual(118, cluster.euclidean_distance_squared(s1, s2))

    def test_euclidean_distance_sum_squared_dataframe(self):
        d1 = pd.DataFrame({'c1': [1, 2, 3], 'c2': [1, 4, 9]})
        d2 = pd.DataFrame({'c1': [2, 3, 4], 'c2': [0, -2, 7]})
        # row one: 1 + 1
        # row two: 1 + 36
        # row three: 1 + 4

        expected = pd.Series([2, 37, 5])
        actual = cluster.euclidean_distance_squared(d1, d2)
        self.assertTrue(expected.equals(actual))

    def test_euclidean_distance_sum_squared_df_series(self):
        v1 = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
        v2 = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]})

        # row one: 0+1+4
        # row two: 1 + 0 + 1
        # row three: 4 + 1 + 0
        expected = pd.Series([5, 2, 5])
        actual = cluster.euclidean_distance_squared(v1, v2)
        actual2 = cluster.euclidean_distance_squared(v2, v1)
        for s in [actual, actual2]:
            with self.subTest():
                self.assertTrue(expected.equals(s))

    def test_find_best_cluster_1(self):
        # Simple test to exactly match one row in cluster data.

        cluster_data = pd.DataFrame({'x': [0.6, 0.5, 0.4, 0.3],
                                     'y': [0.6, 0.5, 0.4, 0.3],
                                     'z': [1.0, 1.0, 1.0, 1.0]})

        # Selection data should match the second row.
        selection_data = pd.Series([0.5, 0.5], index=['x', 'y'])

        # Run, use four clusters (so each row becomes a cluster).
        best_data, _, _ = cluster.find_best_cluster(cluster_data,
                                                    selection_data, 4, 42)

        # since we're using 4 clusters, best_data should return a single
        # row.
        self.assertTrue(best_data.iloc[0].equals(cluster_data.iloc[1]))

    def test_find_best_cluster_2(self):
        # Simple test to match a pair of rows.

        cluster_data = pd.DataFrame({'x': [0.51, 0.50, 0.33, 0.30],
                                     'y': [0.49, 0.50, 0.28, 0.30],
                                     'z': [1.00, 1.00, 1.00, 1.00]})

        # Selection data should put us closest to last two rows.
        selection_data = pd.Series([0.1, 0.1], index=['x', 'y'])

        # Run, use two clusters.
        best_data, _, _ = cluster.find_best_cluster(cluster_data,
                                                    selection_data, 2, 42)

        # Using two clusters, best_data should have two rows.
        self.assertTrue(best_data.equals(cluster_data.iloc[-2:]))

    def test_find_best_cluster_3(self):
        # Clusters influenced by last column (z).

        cluster_data = pd.DataFrame({'x': [0.10, 0.11, 0.10, 0.11],
                                     'y': [0.10, 0.11, 0.10, 0.11],
                                     'z': [1.00, 2.00, 8.00, 9.00]})

        # Just use a z to select.
        selection_data = pd.Series([3], index=['z'])

        # Run, use two clusters.
        best_data, _, _ = cluster.find_best_cluster(cluster_data,
                                                    selection_data, 2, 42)

        # Using two clusters, best_data should have two rows.
        self.assertTrue(best_data.equals(cluster_data.iloc[0:2]))

    def test_feature_scale_1(self):
        # Simple Series
        x = pd.Series([1, 2, 3, 4])
        x_ref = None

        expected = pd.Series([0, 1/3, 2/3, 1])
        actual = cluster.feature_scale(x, x_ref)
        self.assertTrue(expected.equals(actual))

    def test_feature_scale_2(self):
        # Simple DataFrame
        x = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [-1, -2, -3, -4]})
        x_ref = None

        expected = pd.DataFrame({'one': [0, 1/3, 2/3, 1],
                                 'two': [1, 2/3, 1/3, 0]})
        actual = cluster.feature_scale(x, x_ref)
        self.assertTrue(expected.equals(actual))

    def test_feature_scale_3(self):
        # Scale Series given reference series.
        x = pd.Series([2, 4, 6, 8])
        x_ref = pd.Series([1, 3, 5, 10])

        expected = pd.Series([1/9, 3/9, 5/9, 7/9])
        actual = cluster.feature_scale(x, x_ref)
        self.assertTrue(expected.equals(actual))

    def test_feature_scale_4(self):
        # DataFrame with all 0 column.
        x = pd.DataFrame({'a': [0, 0, 0, 0], 'b': [10, 0, 5, 3]})

        expected = pd.DataFrame({'a': [0.0, 0.0, 0.0, 0.0],
                                 'b': [1.0, 0.0, 0.5, 0.3]})

        actual = cluster.feature_scale(x, None)
        self.assertTrue(expected.equals(actual))

    def test_feature_scale_5(self):
        # All zero reference.
        x = pd.Series([1, 2, 3, 4])
        x_ref = pd.Series([0.0, 0.0, 0.0, 0.0])

        actual = cluster.feature_scale(x, x_ref)
        self.assertTrue(actual.equals(x_ref))

    def test_feature_scale_6(self):
        # Pass in numpy array, get TypeError
        a = np.array([1, 2, 3, 4])
        self.assertRaises(TypeError, cluster.feature_scale, a)


if __name__ == '__main__':
    unittest.main()
