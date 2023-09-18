import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import style
# from sklearn import preprocessing
# from functools import reduce
# import sys
# from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
# from random import randint
style.use('ggplot')


class MyCanopy(BaseEstimator, TransformerMixin):
    def __init__(self, remove_outliers=True):
        self.remove_outliers = remove_outliers

    def calc_meanDist(self, data, dists=None):
        n = data.shape[0]
        if dists is None:
            dists = pairwise_distances(data, metric='euclidean')
        # if dists.shape[0] == dists.shape[1]:
        triang_dists = dists[np.arange(dists.shape[0])[:, None] > np.arange(dists.shape[1])].sum()
        # else:
        #  ####### here!
        #  triang_dists = dists.sum()
        meanDist = 2 * triang_dists / (n * (n - 1))
        return dists, meanDist

    def p_density(self, dp, dists_i, meanDist):
        f = np.where((dists_i - meanDist) < 0, 1, 0)
        p_i = f.sum()
        return p_i

    def a_density(self, meanDist, dists_i, p_i):
        cluster_dists = dists_i[dists_i < meanDist]
        d = cluster_dists.sum()
        if p_i - 1 == 0:
            return 0
        else:
            return 2 * d / (p_i * (p_i - 1))

    def s_distance(self, p, p_i, dists_i):
        dist_i_less_j = dists_i[p - p_i > 0]
        if dist_i_less_j.size > 0:
            return dist_i_less_j.min()
        else:
            return dists_i.max()

    def w_weight(self, p_i, a_i, s_i):
        if a_i == 0:
            return 0
        else:
            return p_i * s_i / a_i

    def removeData(self, meanDist, dists, data, ind, centroids_dists=np.array([])):
        # in a row of dists var we have all the distances of a data point to the rest
        dists_i = dists[ind, :]
        dist_filter = dists_i >= meanDist
        new_dists = dists[dist_filter, :]
        # if dists.shape[0] == dists.shape[1]:
        new_dists = new_dists[:, dist_filter]
        new_data = data[dist_filter, :]

        # dists from centroids
        new_centroids_dists = []
        for ind in range(centroids_dists.shape[0]):
            centroid_dists = centroids_dists[ind]
            new_centroids_dists.append(centroid_dists[dist_filter])
        new_centroids_dists = np.array(new_centroids_dists)
        return new_dists, new_data, new_centroids_dists, dist_filter

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        elif isinstance(dt, list):
            data = np.array(dt)
        else:
            raise Exception('dt should be a DataFrame or a numpy array')
        self.centroids = {}
        centroids_dists = np.array([])
        p_centroid = np.array([])

        # c1
        p = np.array([[]])
        ############
        dists, meanDist = self.calc_meanDist(data)
        for ind in range(data.shape[0]):
            p = np.append(p, self.p_density(ind, dists[ind, :], meanDist))

        max_p_sample_ind = p.argmax()
        centroid = data[max_p_sample_ind, :]

        # centroid
        centroid_index = 0
        self.centroids[centroid_index] = centroid
        centroids_dists = np.concatenate([centroids_dists, dists[max_p_sample_ind, :]],
                                         axis=0).reshape(1, -1)
        p_centroid = np.append(p_centroid, p[max_p_sample_ind])
        '''                         
        print('MD', meanDist)
        print('1 data',data.shape)
        print('1 dists', dists.shape)
        print('1 p', len(p))
        '''
        dists, data, centroids_dists, dist_filter = self.removeData(meanDist, dists, data, max_p_sample_ind,
                                                                    centroids_dists=centroids_dists)
        p = p[dist_filter]
        '''
        print('2 data', data.shape)
        print('2 dists', dists.shape)
        print('cd', centroids_dists)
        print('2 p',len(p))
        '''
        ############

        # c2
        p = np.array([])
        a = np.array([])
        s = np.array([])
        w = np.array([])

        ############

        _, meanDist = self.calc_meanDist(data, dists=dists)
        for ind in range(data.shape[0]):
            p_i = self.p_density(ind, dists[ind, :], meanDist)
            p = np.append(p, p_i)
            # p_i = p[ind]
            a = np.append(a, self.a_density(meanDist, dists[ind, :], p_i))

        for ind in range(data.shape[0]):
            s_i = self.s_distance(p, p[ind], dists[ind, :])
            s = np.append(s, s_i)
            w = np.append(w, self.w_weight(p[ind], a[ind], s_i))

        max_w_sample_ind = w.argmax()
        centroid = data[max_w_sample_ind, :]

        # centroid
        centroid_index += 1
        self.centroids[centroid_index] = centroid
        centroids_dists = np.concatenate([centroids_dists, [dists[max_w_sample_ind, :]]],
                                         axis=0)
        p_centroid = np.append(p_centroid, p[max_w_sample_ind])
        dists, data, centroids_dists, dist_filter = self.removeData(meanDist, dists, data, max_w_sample_ind,
                                                                    centroids_dists=centroids_dists)
        p_prev = p[dist_filter]
        s_prev = s[dist_filter]
        p = p[dist_filter]
        a = a[dist_filter]
        '''
        print('3 data', data.shape)
        print('3 dist', dists.shape)
        print('3 p',len(p))
        print('3 cdists', centroids_dists.shape)
        '''
        ############

        # p_prev = np.zeros((len(data))) #p
        # s_prev = np.ones((len(data))) * 999 #s
        # print(self.centroids)
        # print(centroids_dists)

        c_remove = 0
        while data.shape[0] > 1:
            print(data.shape, dists.shape)
            w = np.array([])
            # p_new = np.array([])
            # a = np.array([])
            # s = np.array([])

            ind = 0
            # for ind in range(data.shape[0]):
            _, meanDist = self.calc_meanDist(data, dists=dists)
            if meanDist == 0:
                break
            while ind < data.shape[0]:
                p_i = self.p_density(ind, dists[ind, :], meanDist)
                # p_new = np.append(p, p_i)
                a_i = self.a_density(meanDist, dists[ind, :], p_i)
                # a = np.append(a, self.a_density(meanDist, dists[ind, :], p_i))

                # ind = 0
                # toRemove = False
                # print(p_prev.shape, p.shape, data.shape)

                # s_i = self.s_distance(p, p[ind], dists[ind,:])
                s_centroid = []
                w_i = 1

                for centroid_dists in centroids_dists:
                    s_i = centroid_dists[ind]
                    # s_i = self.s_distance(p, p[ind], centroid_dists)
                    # s_centroid.append(s_i)
                    s_centroid.append(s_i)

                    w_i *= self.w_weight(p_i, a_i, s_i)
                # print('e',ind, data)
                # if w.shape[0] == data.shape[0]:
                #  w[ind] *= w_i
                # else:

                # remove outliers
                if p_prev[ind] > p_i and s_prev[ind] < min(s_centroid) and self.remove_outliers:
                    c_remove += 1
                    # toRemove = True
                    data = np.delete(data, ind, axis=0)
                    dists = np.delete(dists, ind, axis=0)
                    dists = np.delete(dists, ind, axis=1)
                    p = np.delete(p, ind)
                    a = np.delete(a, ind)
                    # s = np.delete(s, ind)
                    p_prev = np.delete(p_prev, ind)
                    s_prev = np.delete(s_prev, ind)
                    # w = np.delete(w, ind)
                    centroids_dists = np.delete(centroids_dists, ind, axis=1)
                    _, meanDist = self.calc_meanDist(data, dists=dists)
                    # break
                else:
                    p_prev[ind] = p_i
                    s_prev[ind] = min(s_centroid)
                    w = np.append(w, w_i)
                    ind += 1
            # if p_i == 0:
            #  print('end', dists.shape)
            #  break
            if w.size > 0:
                max_w_sample_ind = w.argmax()
                centroid = data[max_w_sample_ind, :]
                centroids_dists = np.concatenate([centroids_dists, [dists[max_w_sample_ind, :]]], axis=0)
                p_centroid = np.append(p_centroid, p[max_w_sample_ind])

                centroid_index += 1
                self.centroids[centroid_index] = centroid

                dists, data, centroids_dists, dist_filter = self.removeData(meanDist, dists, data, max_w_sample_ind,
                                                                            centroids_dists=centroids_dists)
                p_prev = p_prev[dist_filter]
                s_prev = s_prev[dist_filter]
                p = p[dist_filter]
                a = a[dist_filter]

            # print(f'{4+ind} data', data.shape)
            # print(f'{4+ind} dist', dists.shape)
            # print(f'{4+ind} p',len(p), len(p_prev))
            # print(f'{4+ind} cdists', centroids_dists.shape)

        print('Canopy found %d centers' % (len(self.centroids)))
        print('Removed %d data points' % (c_remove))


class DC_KMeans:
    def __init__(self, k=2, tol=0.01, max_rep=100,
                 init_type='random', init_centers=None):
        self.k = k
        self.tol = tol
        self.max_rep = max_rep
        self.centroids = {}
        self.clusters = {}
        self.init_type = init_type
        if init_type == 'random':
            self.name = 'KMeans'
        elif init_type == 'kmeans++':
            self.name = 'KMeans++'
        elif init_type == 'canopy':
            self.centroids = init_centers
            self.k = len(init_centers)
            self.name = 'CanopyKmeans'
        else:
            self.name = 'NotSpecified'

    def init_centroids(self, data, init_type):
        if self.k < len(data):
            if init_type == 'random':
                # np.random.seed(randint(1,42))
                # seeds = np.random.randint(0, len(df), self.k)
                seeds = np.random.choice(len(data), self.k, replace=False)
                for index in range(self.k):
                    self.centroids[index] = data[seeds[index], :]

            elif init_type == 'kmeans++':
                len_data = data.shape[0]
                seed1 = np.random.choice(len_data, 1, replace=False)[0]
                centroid_index = 0
                self.centroids[centroid_index] = data[seed1, :]
                seeds = [seed1]

                # Here starts the for-loop for the other seeds:
                for cluster_index in range(self.k - 1):
                    dist2centroids = np.array([self.find_mindist(data, seed)
                                               for seed in self.centroids]) ** 2
                    # dist_df = dist2centroids.argmin(axis=0)
                    for seed in seeds:
                        dist2centroids[:, seed] = 0
                    dist_sum = dist2centroids.sum()
                    D2 = (dist2centroids / dist_sum).sum(axis=0)

                    # cumprobs = D2.cumsum()
                    # r = np.random.uniform(0, 1)
                    new_seed = np.random.choice(len_data, 1, replace=False, p=D2)[0]
                    seeds.append(new_seed)
                    centroid_index += 1
                    self.centroids[centroid_index] = data[new_seed, :]
            '''
            elif init_type == 'canopy':
                canopy = MyCanopy()
                canopy.fit(data)
                self.centroids = canopy.centroids
                self.k = len(self.centroids)
            '''

        else:
            raise Exception('# of desired clusters should be < total data points')

    def find_mindist(self, data, seed):
        # print(self.centroids[seed])
        # seed_df = pd.DataFrame([self.centroids[seed]]*len(df.index))
        return self.distance_metric(data, self.centroids[seed])

    def distance_metric(self, a, b, dist='Euclidean'):
        """
        Define the distance metric used
        This can be: 'Euclidean' (default)
        """
        # a numpy matrix, b numpy vector of the centroid
        if a.shape[1] == b.shape[0]:
            """
            We assume that:
            - the numerical values of a and are normalized
            - a and b have the same columns from now on
            """
            # a_num = a.select_dtypes(exclude='object')
            # a_cat = a.select_dtypes(include='object')
            ## make the same size as a
            # b_num = b.select_dtypes(exclude='object')
            # b_cat = b.select_dtypes(include='object')
            # print(a)
            # print(a-b)
            distance = ((a - b) ** 2).sum(axis=1)

            # dist_cat = pd.DataFrame(np.where(a_cat==b_cat, 0, 1)).sum(axis=1)
            # return (distance + dist_cat)**0.5
            return distance ** 0.5

    def handle_empty_cluster(self, dist2centroids, data, seed, emptySeeds):
        # choose non empty seeds from distance matrix
        nonEmpty_dist2centroids = np.delete(dist2centroids, emptySeeds, axis=0)
        dat_point_maxDist = nonEmpty_dist2centroids.sum(axis=0).argmax()
        self.centroids[seed] = data[dat_point_maxDist, :]
        return np.array(self.find_mindist(data, seed))

    def fit(self, dt):
        if isinstance(dt, pd.DataFrame):
            data = dt.values
        elif isinstance(dt, np.ndarray):
            data = dt
        else:
            raise Exception('dt should be a DataFrame or a numpy array')

        if not len(self.centroids) == self.k:
            print('No init centers', self.name)
            # get random indexes from data
            self.init_centroids(data, self.init_type)

        converge = False
        while self.max_rep > 0 and converge == False:
            emptyCluster = False
            emptySeeds = []
            dist2centroids = np.array([self.find_mindist(data, seed)
                                       for seed in self.centroids])
            # dist2centroids has k rows which correspond to the dist from each centroid
            # dist_df = pd.concat(dist2centroids, axis=1).idxmin(axis=1)
            self.labels_ = dist2centroids.argmin(axis=0)

            for seed in self.centroids:
                self.clusters[seed] = np.where(self.labels_ == seed)
                if self.clusters[seed][0].size == 0:
                    print("Cluster %s with centroid %s is empty!"
                          % (seed, self.centroids[seed]))
                    emptySeeds.append(seed)
                    emptyCluster = True

            # check for empty clusters
            if emptyCluster:
                for seed in emptySeeds:
                    dist2centroids[seed] = self.handle_empty_cluster(dist2centroids, data, seed, emptySeeds)
                    emptySeeds.pop(emptySeeds.index(seed))

                # find new clusters after fixing empty ones
                self.labels_ = dist2centroids.argmin(axis=0)
                for seed in self.centroids:
                    self.clusters[seed] = np.where(self.labels_ == seed)

            prev_centroids = self.centroids.copy()
            for seed in self.clusters:
                self.centroids[seed] = data[self.clusters[seed]].mean(axis=0)

            converge = True
            for seed in self.clusters:
                # if euclidean(prev_centroids[seed], self.centroids[seed]) <= self.tol:
                # if np.array_equal(prev_centroids[seed], self.centroids[seed]):
                dist_diff = np.linalg.norm(prev_centroids[seed] - self.centroids[seed],
                                           ord=2)
                if dist_diff < self.tol:
                    converge = converge and True
                else:
                    converge = converge and False

            self.max_rep -= 1
        print('Remaining repetitions: %s' % (self.max_rep))

        self.inertia_ = 0
        for seed in self.centroids:
            self.inertia_ += np.array([self.find_mindist(data[self.clusters[seed]], seed) ** 2]).sum()


if __name__ == "__main__":
    k = 3
    tol = 0.001
    max_rep = 100
    fuzzy_m = 2

    # Data
    data = np.array([[2, 3],
                     [3, 5],
                     [1, 4],
                     [10, 12],
                     [11, 13],
                     [12, 10]])
    # plt.scatter(data[:, 0], data[:, 1], s=100)
    # plt.show()
    df = pd.DataFrame(data)

    # MyCanopy
    remove_outliers = True
    mycanopy = MyCanopy(remove_outliers=remove_outliers)
    mycanopy.fit(df)
    # centers = list(mycanopy.centroids.values())
    # print('Canopy centers', centers)

    dckm = DC_KMeans(k, tol, max_rep, 'canopy', mycanopy.centroids)
    # DC_KMeans(n_clusters=len(centers), tol=tol, max_iter=max_rep, init=np.array(centers)),
    dckm.fit(df)
    color = ['g', 'c', 'y']
    print(dckm.clusters)
    for centroid in dckm.centroids:
        plt.scatter(dckm.centroids[centroid][0], dckm.centroids[centroid][1], marker='o', color=color[centroid])
        plt.scatter(data[dckm.clusters[centroid][0], 0], data[dckm.clusters[centroid][0], 1], marker='+',
                    color=color[centroid])
    plt.show()
    # dckm.predict(pd.DataFrame([[1, 5]]))
