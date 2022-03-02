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
import json

import numpy as np
import so4gp as sgp


# Configuration Parameters
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering

MIN_SUPPORT = 0.5
CLUSTER_ALGORITHM = 'kmeans'


def clugps(f_path=None, min_sup=MIN_SUPPORT, algorithm=CLUSTER_ALGORITHM, return_gps=False):

    # Create a DataGP object
    d_gp = sgp.DataGP(f_path, min_sup)
    """:type d_gp: DataGP"""

    # Generate net-win matrices
    d_gp.construct_net_wins()
    n_wins = d_gp.net_wins.matrix
    # print(n_wins)

    # Spectral Clustering: perform SVD to determine the independent rows
    u, s, vt = np.linalg.svd(n_wins)

    # Spectral Clustering: compute rank of net-wins matrix
    r = np.linalg.matrix_rank(n_wins)

    # Spectral Clustering: rank approximation
    n_wins_approx = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

    # 1a. Clustering using KMeans or MiniBatchKMeans or SpectralClustering or AgglomerativeClustering
    y_pred = predict_clusters(n_wins_approx, r, algorithm=algorithm)
    # 1b. Infer GPs
    str_gps, gps = infer_gps(y_pred, d_gp)

    # Compare inferred GPs with real GPs
    compare_gps(gps, f_path, min_sup)

    # Output
    out = json.dumps({"Algorithm": "Clu-GRAD", "Patterns": str_gps})
    """:type out: object"""
    if return_gps:
        return out, gps
    else:
        return out


def predict_clusters(nw_matrix, r, algorithm):

    if algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=r, random_state=0)
        y_pred = kmeans.fit_predict(nw_matrix)
    elif algorithm == 'mbkmeans':
        kmeans = MiniBatchKMeans(n_clusters=r)
        y_pred = kmeans.fit_predict(nw_matrix)
    elif algorithm == 'sc':
        spectral_model = SpectralClustering(n_clusters=r)
        y_pred = spectral_model.fit_predict(nw_matrix)
    elif algorithm == 'ac':
        model = AgglomerativeClustering(n_clusters=r)
        y_pred = model.fit_predict(nw_matrix)
    else:
        raise Exception("Error: unknown clustering algorithm selected!")
    return y_pred


def infer_gps(clusters, d_gp):

    patterns = []
    str_patterns = []

    n_wins = d_gp.net_wins
    sups = n_wins.supports
    n_matrix = n_wins.matrix
    all_gis = n_wins.gradual_items

    lst_indices = [np.where(clusters == element)[0] for element in np.unique(clusters)]
    for grp_idxs in lst_indices:
        if grp_idxs.size > 1:
            cluster = n_matrix[grp_idxs]
            cluster_sups = sups[grp_idxs]
            cluster_gis = all_gis[grp_idxs]

            # Estimate support
            m = cluster.shape[0]
            xor = np.ones(cluster.shape[1], dtype=bool)
            for i in range(m):
                if (i + 1) < m:
                    temp = np.equal(cluster[i], cluster[i + 1])
                    xor = np.logical_and(xor, temp)
            prob = np.sum(xor) / cluster.shape[1]
            est_sup = prob * np.min(cluster_sups)

            # Infer GPs from the clusters
            gp = sgp.GP()
            for gi in cluster_gis:
                gp.add_gradual_item(gi)
            gp.set_support(est_sup)
            patterns.append(gp)
            str_patterns.append(gp.print(d_gp.titles))

    return str_patterns, patterns


def compare_gps(clustered_gps, f_path, min_sup):
    same_gps = []
    miss_gps = []
    str_gps, real_gps = sgp.graank(f_path, min_sup, return_gps=True)
    for est_gp in clustered_gps:
        check, real_sup = sgp.contains_gp(est_gp, real_gps)
        if check:
            same_gps.append([est_gp, real_sup])
        else:
            miss_gps.append(est_gp)
    print(same_gps)
    return same_gps, miss_gps

# print(clugps('../data/DATASET.csv', min_sup=0.5))
print(clugps('../data/breast_cancer.csv', min_sup=0.5))