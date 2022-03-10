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
import random
import numpy as np
from ypstruct import structure

import so4gp as sgp

# Configuration Parameters
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering

MIN_SUPPORT = 0.5
ERASURE_PROBABILITY = 0
CLUSTER_ALGORITHM = 'kmeans'

FILE = '../data/DATASET.csv'


def clugps(f_path=None, min_sup=MIN_SUPPORT, algorithm=CLUSTER_ALGORITHM, return_gps=False):
    # Create a DataGP object
    d_gp = sgp.DataGP(f_path, min_sup)
    """:type d_gp: DataGP"""

    # Generate net-win matrices
    # d_gp.construct_net_wins()
    # n_wins = d_gp.net_wins.matrix
    r_matrix = construct_pairs(d_gp, e=ERASURE_PROBABILITY)
    n_wins = r_matrix.net_wins
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
    str_gps, gps = infer_gps(y_pred, d_gp, r_matrix)
    # print(str_gps)

    # Compare inferred GPs with real GPs
    compare_gps(gps, f_path, min_sup)

    # Output
    out = json.dumps({"Algorithm": "Clu-GRAD", "Patterns": str_gps})
    """:type out: object"""
    if return_gps:
        return out, gps
    else:
        return out


def construct_pairs(d_gp, e):

    # Sample pairs
    n = d_gp.row_count
    pair_count = int(n * (n - 1) * 0.5)
    p = 1 - e
    sampled_pairs = []
    # sample_idx = random.sample(range(pair_count), int(p*pair_count))  # normal distribution
    sample_idx = [0, 9, 6, 7, 3]  # For testing
    # print(sample_idx)
    # for i in range(pair_count):
    for i in sample_idx:
        # Retrieve group
        g, i_g = get_group(n, i)
        # print(str(i) + ' grp: ' + str(g))
        pair = [(g-1), (g + i_g)]
        sampled_pairs.append(pair)
    # sampled_pairs = np.array(sampled_pairs)
    # print(sampled_pairs)

    # Compute gradual relation
    attr_data = d_gp.data.T
    pairs = []
    n_mat = []
    lst_gis = []
    for col in d_gp.attr_cols:
        col_data = np.array(attr_data[col], dtype=float)
        pr_pos = []
        pr_neg = []
        for pr in sampled_pairs:
            if col_data[pr[0]] < col_data[pr[1]]:
                pr_pos.append(pr)
                pr_neg.append([pr[1], pr[0]])
            elif col_data[pr[0]] > col_data[pr[1]]:
                pr_neg.append(pr)
                pr_pos.append([pr[1], pr[0]])
        if len(pr_pos) > 0:
            n_vec = construct_net_win(n, np.array(pr_pos))
            n_mat.append(n_vec)
            pairs.append(pr_pos)
            lst_gis.append(sgp.GI(col, '+'))

            n_mat.append(-n_vec)
            pairs.append(pr_neg)
            lst_gis.append(sgp.GI(col, '-'))
    r_matrix = structure()
    r_matrix.gradual_items = np.array(lst_gis)
    r_matrix.pairs = np.array(pairs, dtype=object)
    r_matrix.net_wins = np.array(n_mat)
    return r_matrix


def construct_net_win(n, arr_pairs):
    s_vector = np.zeros(shape=(n,), dtype=int)
    for i in range(n):
        x_i = np.count_nonzero(arr_pairs[:, 0] == i)
        x_j = np.count_nonzero(arr_pairs[:, 1] == i)
        s_vector[i] = (x_i-x_j)
    s_vector[s_vector > 0] = 1
    s_vector[s_vector < 0] = -1
    return s_vector


def estimate_score_vector(w_mat):
    n, m = w_mat.shape
    score_vector = np.ones(shape=(n,))
    temp = score_vector.copy()
    for i in range(n):
        nume = np.sum(w_mat[i])
        deno = 0
        for j in range(m):
            if i != j:
                deno += (w_mat[i][j] + w_mat[j][i]) / (score_vector[i] + score_vector[j])
        temp[i] = nume / deno
    score_vector = temp / np.sum(temp)
    return score_vector


def get_group(n, i):
    # Retrieve group
    lb = 0
    k = 1
    x = n - k
    while k < n:
        if i < x:
            return k, (i-lb)
        else:
            lb = x
            k += 1
            x += (n - k)
    return -1, -1


def construct_win_matrix(n, cluster_pairs):
    w_vector = np.zeros(shape=(n,n), dtype=int)
    for lst_pairs in cluster_pairs:
        for pair in lst_pairs:
            w_vector[pair[0]][pair[1]] += 1
    return w_vector


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


def infer_gps(clusters, d_gp, r_mat):
    patterns = []
    str_patterns = []

    # n_wins = r_mat.net_wins
    # sups = n_wins.supports
    n = d_gp.row_count
    n_matrix = r_mat.net_wins
    r_pairs = r_mat.pairs
    all_gis = r_mat.gradual_items

    # print(win_matrix)

    lst_indices = [np.where(clusters == element)[0] for element in np.unique(clusters)]
    for grp_idxs in lst_indices:
        if grp_idxs.size > 1:
            cluster = n_matrix[grp_idxs]
            cluster_pairs = r_pairs[grp_idxs]
            # cluster_sups = sups[grp_idxs]
            cluster_gis = all_gis[grp_idxs]
            cluster_wins = construct_win_matrix(d_gp.row_count, cluster_pairs)

            # Estimate support
            score_vector = estimate_score_vector(cluster_wins)
            count = 0
            for i in range(n):
                temp = (n-(i+1)) * score_vector[i]
                count += temp
            print(count)
            # m = cluster.shape[0]
            # xor = np.ones(cluster.shape[1], dtype=bool)
            # for i in range(m):
            #    if (i + 1) < m:
            #        temp = np.equal(cluster[i], cluster[i + 1])
            #        xor = np.logical_and(xor, temp)
            # prob = np.sum(xor) / cluster.shape[1]
            est_sup = 0  # prob  * np.min(cluster_sups)

            print(score_vector)
            print(cluster_pairs)
            print(cluster_wins)
            print("\n")

            # Infer GPs from the clusters
            gp = sgp.GP()
            for gi in cluster_gis:
                gp.add_gradual_item(gi)
            gp.set_support(est_sup)
            patterns.append(gp)
            str_patterns.append(gp.print(d_gp.titles))
            # print(gp.print(d_gp.titles))
    return str_patterns, patterns


def compare_gps(clustered_gps, f_path, min_sup):
    same_gps = []
    miss_gps = []
    str_gps, real_gps = sgp.graank(f_path, min_sup, return_gps=True)
    for est_gp in clustered_gps:
        check, real_sup = sgp.contains_gp(est_gp, real_gps)
        # print([est_gp, est_gp.support, real_sup])
        if check:
            same_gps.append([est_gp, est_gp.support, real_sup])
        else:
            miss_gps.append(est_gp)
    # print(same_gps)
    print(str_gps)
    return same_gps, miss_gps


print(clugps('../data/DATASET.csv', min_sup=0.5))
# print(clugps('../data/breast_cancer.csv', min_sup=0.6))

#dset = sgp.DataGP(FILE, MIN_SUPPORT)
#r_mat = construct_pairs(dset)
#print(r_mat.net_wins)
#print(r_mat.wins)
#print(r_mat.pairs)
#print(r_mat.gradual_items)
