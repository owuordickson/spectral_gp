# -*- coding: utf-8 -*-
"""
@author: Dickson Owuor

@credits: Thomas Runkler, Lesley Bonyo and Anne Laurent

@license: MIT

@version: 0.0.1

@email: owuordickson@gmail.com

@created: 01 March 2022

@modified: 01 March 2022

Clustering Gradual Items
------------------------

A gradual pattern (GP) is a set of gradual items (GI) and its quality is measured by its computed support value. A GI is
a pair (i,v) where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2
GIs; for example, column age yields GI age+ or age-. For example given a data set with 3 columns (age, salary, cars) and
10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have
the values of column age 'increasing' and column 'salary' decreasing.

We borrow the net-wins concept used in the work 'Clustering Using Pairwise Comparisons' proposed by R. Srikant to the
problem of extracting gradual patterns (GPs). In order to mine for GPs, each feature yields 2 gradual items which we use
to construct a bitmap matrix comparing each row to each other (i.e., (r1,r2), (r1,r3), (r1,r4), (r2,r3), (r2,r4),
(r3,r4)).

In this approach, we convert the bitmap matrices into 'net wins matrices'. Finally, we apply spectral clustering to
determine which gradual items belong to the same group based on the similarity of gradual dependency. Gradual items in
the same cluster should have almost similar score vector.

"""

import numpy as np
import so4gp as sgp


# Configuration Parameters
from sklearn.cluster import KMeans

MIN_SUPPORT = 0.5


def construct_net_wins(d_gp):
    # TO BE REMOVED
    # Function for constructing GP pairs for Mx2 matrix
    net_wins = []
    attr_data = d_gp.data.T
    n = d_gp.row_count
    for col in d_gp.attr_cols:
        col_data = np.array(attr_data[col], dtype=float)
        incr = np.array((col, '+'), dtype='i, S1')
        decr = np.array((col, '-'), dtype='i, S1')

        bitmap_pos = np.where(col_data > col_data[:, np.newaxis], 1, np.where(col_data < col_data[:, np.newaxis], -1,
                                                                              0))
        # Remove invalid candidates
        supp = float(np.sum(bitmap_pos[bitmap_pos == 1])) / float(n * (n - 1.0) / 2.0)
        if supp >= d_gp.thd_supp:
            # print(bitmap_pos)
            row_sum = np.sum(bitmap_pos, axis=1)
            row_sum[row_sum > 0] = 1
            row_sum[row_sum < 0] = -1

            net_wins.append(np.array([incr.tolist(), row_sum, supp], dtype=object))
            net_wins.append(np.array([decr.tolist(), -row_sum, supp], dtype=object))
    return np.array(net_wins)


def clugps(f_path=None, min_sup=MIN_SUPPORT):

    # Create a DataGP object
    d_gp = sgp.DataGP(f_path, min_sup)
    """:type d_gp: DataGP"""

    # Generate net-win matrices
    n_wins = construct_net_wins(d_gp)

    # Perform single value distribution to determine the independent rows
    u, s, vt = np.linalg.svd(n_wins)

    # Compute rank of net-wins matrix
    r = np.linalg.matrix_rank(n_wins)

    # Rank approximation
    n_wins_approx = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

    # 1a. Clustering using KMeans
    kmeans = KMeans(n_clusters=r, random_state=0)
    predicted_clusters = kmeans.fit_predict(n_wins_approx)
    # 1b. Infer GPs
    gps = infer_gps(predicted_clusters)

    return gps


def infer_gps(clusters):

    patterns = []
    idx_grp = [np.where(clusters == element)[0] for element in np.unique(clusters)]

    for grp in idx_grp:
        if grp.size > 1:
            # Estimate support of clusters
            supports = s[grp]
            cluster_m = N[grp]
            m = cluster_m.shape[0]
            xor = np.ones(cluster_m.shape[1], dtype=bool)
            for i in range(m):
                if (i + 1) < m:
                    temp = np.equal(cluster_m[i], cluster_m[i + 1])
                    xor = np.logical_and(xor, temp)
            prob = np.sum(xor) / cluster_m.shape[1]
            est_sup = prob * np.min(supports)

            # Infer GPs from the clusters
            gis = f[grp]
            gp = sgp.GP()
            for obj in gis:
                gi = sgp.GI(obj[0], obj[1].decode())
                gp.add_gradual_item(gi)
            gp.set_support(est_sup)
            # print(gp.print(ds.titles))
            patterns.append(gp)
    return patterns


def compare_gps(clustered_gps, f_path, min_sup):
    real_gps = sgp.graank(f_path, min_sup)
    pass
