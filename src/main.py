# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@created: "16 Mar 2022"
Usage:
    $python main.src -f ../data/DATASET.csv -s 0.5
Description:
    f -> file path (CSV)
    s -> minimum support
    a -> clustering algorithm
    e -> erasure probability
    i -> maximum iteration
"""

import sys
import so4gp
from optparse import OptionParser
import config as cfg

if __name__ == "__main__":
    if not sys.argv:
        filePath = sys.argv[0]
        minSup = sys.argv[1]
        algChoice = sys.argv[2]
        eProb = sys.argv[3]
        itMax = sys.argv[4]
        numCores = sys.argv[5]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             default=cfg.DATASET,
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=cfg.MIN_SUPPORT,
                             type='float')
        optparser.add_option('-a', '--algorithmChoice',
                             dest='algChoice',
                             help='select algorithm for clustering',
                             default=cfg.ALGORITHM,
                             type='string')
        optparser.add_option('-e', '--eProb',
                             dest='eProb',
                             help='erasure probability',
                             default=cfg.ERASURE_PROBABILITY,
                             type='float')
        optparser.add_option('-i', '--maxIteration',
                             dest='itMax',
                             help='maximum iteration for score vector estimation',
                             default=cfg.SCORE_VECTOR_ITERATIONS,
                             type='float')
        optparser.add_option('-c', '--cores',
                             dest='numCores',
                             help='number of cores',
                             default=cfg.CPU_CORES,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python3 main.src -f filename.csv -a 'clugrad'")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        algChoice = options.algChoice
        eProb = options.eProb
        itMax = options.itMax
        numCores = options.numCores

    import time
    import tracemalloc
    from pkg_algorithms import clu_grad, aco_grad, graank
    Profile = so4gp.Profile

    if algChoice == 'clugrad':
        # CLU-GRAD
        start = time.time()
        tracemalloc.start()
        res_text = clu_grad.execute(filePath, minSup, eProb, itMax, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_cgp_v1' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name, wr=False)
        print(wr_text)
    elif algChoice == 'acograd':
        # ACO-GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = aco_grad.execute(filePath, minSup, numCores, cfg.EVAPORATION_FACTOR, cfg.MAX_ITERATIONS)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_aco_v1' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name, wr=False)
        print(wr_text)
    elif algChoice == 'graank':
        # GRAANK
        start = time.time()
        tracemalloc.start()
        res_text = graank.execute(filePath, minSup, numCores)
        snapshot = tracemalloc.take_snapshot()
        end = time.time()

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += (Profile.get_quick_mem_use(snapshot) + "\n")
        wr_text += str(res_text)
        f_name = str('res_graank' + str(end).replace('.', '', 1) + '.txt')
        Profile.write_file(wr_text, f_name, wr=False)
        print(wr_text)
    else:
        print("Invalid Algorithm Choice!")