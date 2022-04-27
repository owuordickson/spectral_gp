from src.pkg_algorithms import clu_grad_v2 as cgp
from src import config as cfg
import so4gp as sgp


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


def run_comparison():
    output, est_gps = cgp.clugps(f_path=cfg.DATASET, min_sup=cfg.MIN_SUPPORT, return_gps=True)
    print(output)

    # Compare inferred GPs with real GPs
    hit_gps, miss_gps = compare_gps(est_gps, cfg.DATASET, cfg.MIN_SUPPORT)
    d_gp = sgp.DataGP(cfg.DATASET, cfg.MIN_SUPPORT)
    for gp in miss_gps:
        print(gp.print(d_gp.titles))


# run_comparison()
