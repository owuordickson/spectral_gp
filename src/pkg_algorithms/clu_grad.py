# -*- coding: utf-8 -*-
"""
@author: Dickson Owuor

@credits: Thomas Runkler, Lesley Bonyo and Anne Laurent

@license: MIT

@version: 0.1.2

@email: owuordickson@gmail.com

@created: 01 March 2022

@modified: 17 March 2022

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
import math
import random
import numpy as np
from ypstruct import structure
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering

import so4gp as sgp

# Clustering Configurations
MIN_SUPPORT = 0.5
CLUSTER_ALGORITHM = 'kmeans'  # selects algorithm to be used for clustering the net-win matrices
ERASURE_PROBABILITY = 0.5  # determines the number of pairs to be ignored
SCORE_VECTOR_ITERATIONS = 10  # maximum iteration for score vector estimation


def clugps(f_path, min_sup=MIN_SUPPORT, e_probability=ERASURE_PROBABILITY,
           sv_max_iter=SCORE_VECTOR_ITERATIONS, algorithm=CLUSTER_ALGORITHM, return_gps=False, testing=False):
    # Create a DataGP object
    d_gp = sgp.DataGP(f_path, min_sup)
    """:type d_gp: DataGP"""

    # Generate net-win matrices
    mat_obj = construct_matrices(d_gp, e=e_probability)
    s_matrix = mat_obj.nwin_matrix
    # print(s_matrix)

    # Spectral Clustering: perform SVD to determine the independent rows
    u, s, vt = np.linalg.svd(s_matrix)

    # Spectral Clustering: compute rank of net-wins matrix
    r = np.linalg.matrix_rank(s_matrix)

    # Spectral Clustering: rank approximation
    s_matrix_approx = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

    # 1a. Clustering using KMeans or MiniBatchKMeans or SpectralClustering or AgglomerativeClustering
    y_pred = predict_clusters(s_matrix_approx, r, algorithm=algorithm)
    # 1b. Infer GPs
    max_iter = sv_max_iter
    str_gps, gps = infer_gps(y_pred, d_gp, mat_obj, max_iter)
    # print(str_gps)

    # Output - DO NOT ADD TO PyPi Package
    out = structure()
    out.estimated_gps = gps
    # out.iteration_count = it_count
    out.max_iteration = sv_max_iter
    out.titles = d_gp.titles
    out.col_count = d_gp.col_count
    out.row_count = d_gp.row_count
    out.e_prob = e_probability
    if testing:
        return out

    # Output
    out = json.dumps({"Algorithm": "Clu-GRAD", "Patterns": str_gps})
    """:type out: object"""
    if return_gps:
        return out, gps
    else:
        return out


def construct_matrices(d_gp, e):

    # Sample pairs
    n = d_gp.row_count
    pair_count = int(n * (n - 1) * 0.5)
    p = 1 - e
    sample_idx = random.sample(range(pair_count), int(p*pair_count))  # normal distribution
    # sample_idx = [0, 9, 6, 7, 3]  # For testing
    # print(sample_idx)

    # mat = np.arange(10)
    # choice = np.random.choice(range(mat.shape[0]), size=(int(mat.shape[0] / 2),), replace=False)
    # ind = np.zeros(mat.shape[0], dtype=bool)
    # ind[choice] = True
    # rest = mat[~ind]

    # Compute gradual relation
    attr_data = d_gp.data.T
    lst_gis = []
    r_mat_idx = []
    s_mat = []
    a_mat = np.zeros(shape=(n, pair_count))

    # Construct A matrix
    for idx in range(pair_count):
        ei = np.zeros(shape=(n,)).T
        ej = np.zeros(shape=(n,)).T
        g, i_g = get_pair_partition(n, idx)
        i = (g - 1)
        j = (g + i_g)
        ei[i] = 1
        ej[j] = 1
        a_mat[:, idx] = ei - ej

    # Construct R matrix from data set
    for col in d_gp.attr_cols:
        col_data = np.array(attr_data[col], dtype=float)
        r_vec = np.zeros(shape=(pair_count,))
        r_idx_pos = []
        r_idx_neg = []
        for idx in sample_idx:
            g, i_g = get_pair_partition(n, idx)
            i = (g - 1)
            j = (g + i_g)

            # Construct R vector
            if col_data[i] < col_data[j]:
                r_vec[idx] = 1
                r_idx_pos.append([idx, 1])  # For estimating score-vector
                r_idx_neg.append([idx, -1])  # For estimating score-vector
            elif col_data[i] > col_data[j]:
                r_vec[idx] = -1
                r_idx_pos.append([idx, -1])  # For estimating score-vector
                r_idx_neg.append([idx, 1])  # For estimating score-vector

        if np.count_nonzero(r_vec) > 0:
            # Compute net-win vector
            s_vec = np.dot(r_vec, a_mat.T)
            s_vec[s_vec > 0] = 1
            s_vec[s_vec < 0] = -1

            r_mat_idx.append(r_idx_pos)
            s_mat.append(s_vec)
            lst_gis.append(sgp.GI(col, '+'))

            r_mat_idx.append(r_idx_neg)
            s_mat.append(-s_vec)
            lst_gis.append(sgp.GI(col, '-'))

    res = structure()
    res.gradual_items = np.array(lst_gis)
    res.r_idx = np.array(r_mat_idx, dtype=object)
    res.nwin_matrix = np.array(s_mat)
    res.a_matrix = a_mat
    return res


def get_pair_partition(n, i):
    # Retrieve group from: (n-1), (n-2), (n-3) ..., (n-(n-1)) using index i
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


def infer_gps(clusters, d_gp, mat_obj, max_iter):

    patterns = []
    str_patterns = []

    n = d_gp.row_count
    all_gis = mat_obj.gradual_items
    r_idx = mat_obj.r_idx
    a_mat = mat_obj.a_matrix

    lst_indices = [np.where(clusters == element)[0] for element in np.unique(clusters)]
    # lst_indices = list([np.array([0, 5, 7])])  # Hard coded - for testing
    # print(lst_indices)
    for grp_idxs in lst_indices:
        if grp_idxs.size > 1:
            # 1. Retrieve all cluster-pairs and the corresponding GIs
            cluster_gis = all_gis[grp_idxs]
            cluster_ridxs = r_idx[grp_idxs]

            # 2. Compute score vector from R matrix
            score_vectors = []  # Approach 2
            for idxs in cluster_ridxs:  # Approach 2
                temp = estimate_score_vector(n, idxs, max_iter)
                score_vectors.append(temp)

            # 3. Estimate support
            est_sup = estimate_support(n, score_vectors)
            # est_sup = estimate_support_v2(score_vectors, a_mat)  # Has a bug

            # 4. Infer GPs from the clusters
            gp = sgp.GP()
            for gi in cluster_gis:
                gp.add_gradual_item(gi)
            gp.set_support(est_sup)
            patterns.append(gp)
            str_patterns.append(gp.print(d_gp.titles))
            # print(gp.print(d_gp.titles))
    return str_patterns, patterns


def estimate_score_vector(n, r_idxs, max_iter):
    # Estimate score vector from pairs
    score_vector = np.ones(shape=(n,))

    # Construct a win-matrix
    temp_vec = np.zeros(shape=(n,))

    # Compute score vector
    for k in range(max_iter):
        if np.count_nonzero(score_vector == 0) > 1:
            break
        else:
            # for idx in range(r_mat.shape[0]):
            for i_obj in r_idxs:
                idx = i_obj[0]
                pr_val = i_obj[1]
                g, i_g = get_pair_partition(n, idx)
                i = (g - 1)
                j = (g + i_g)
                if pr_val >= 1:
                    log = math.log(math.exp(score_vector[i]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                                   10)
                    temp_vec[i] += pr_val * log
                elif pr_val >= -1:
                    log = math.log(math.exp(score_vector[j]) / (math.exp(score_vector[i]) + math.exp(score_vector[j])),
                                   10)
                    temp_vec[j] += -pr_val * log
            score_vector = temp_vec / np.sum(temp_vec)
    return score_vector


def estimate_support(n, score_vectors):
    # Estimate support - use different score-vectors to construct pairs
    sim_pairs = 0
    is_common = False
    for i in range(n):
        for j in range(n):
            if is_common:
                sim_pairs += 1
            is_common = True
            for s_vec in score_vectors:
                prob = math.exp(s_vec[i]) / (math.exp(s_vec[i]) + math.exp(s_vec[j]))
                if prob <= 0.5:
                    is_common = False
    est_sup = sim_pairs / (n * (n - 1) / 2)
    # print(sim_pairs)
    return est_sup


def estimate_support_v2(score_vectors, a_mat):
    n = a_mat.shape[1]
    r_vec = np.ones(shape=(n,))
    for s_vec in score_vectors:
        temp_vec = np.dot(s_vec, a_mat)
        temp_vec[temp_vec > 0] = 1
        temp_vec[temp_vec < 0] = 0
        r_vec = np.multiply(r_vec, temp_vec)
    est_sup = np.sum(r_vec) / (n * (n - 1) / 2)
    return est_sup


# DO NOT ADD TO PyPi Package
def execute(f_path, min_supp, e_prob, max_iter, cores):
    Profile = sgp.Profile
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        out = clugps(f_path, min_supp, e_prob, max_iter, testing=True)
        list_gp = out.estimated_gps

        wr_line = "Algorithm: Clu-GRAD (v1.2)\n"
        wr_line += "No. of (dataset) attributes: " + str(out.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(out.row_count) + '\n'
        wr_line += "Erasure probability: " + str(out.e_prob) + '\n'

        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(list_gp)) + '\n'
        wr_line += "Number of iterations: " + str(out.iteration_count) + '\n'

        for txt in out.titles:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in list_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(round(gp.support, 3)) + '\n')

        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line