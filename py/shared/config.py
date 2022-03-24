# -*- coding: utf-8 -*-

# Configurations for Gradual Patterns:
MIN_SUPPORT = 0.5
CPU_CORES = 4  # Depends on your computer

# DATASET = "../../data/DATASET.csv"
# DATASET = "../../data/hcv_data.csv"

# Uncomment for Main:
DATASET = "../data/DATASET.csv"
# DATASET = '../data/breast_cancer.csv'


# Uncomment for Terminal:
# DATASET = "data/DATASET.csv"

# Clustering Configurations
CLUSTER_ALGORITHM = 'kmeans'  # selects algorithm to be used for clustering the net-win matrices
ERASURE_PROBABILITY = 0.5  # determines the number of pairs to be ignored
SCORE_VECTOR_ITERATIONS = 10  # maximum iteration for score vector estimation
