import cluster_gps as cgp
import so4gp as sgp
from profile import Profile


def execute(f_path, min_supp,  algorithm, e_prob, max_iter, cores):
    try:
        if cores > 1:
            num_cores = cores
        else:
            num_cores = Profile.get_num_cores()

        out = cgp.clugps(f_path, min_supp, algorithm, e_prob, max_iter)
        list_gp = out.estimated_gps

        wr_line = "Algorithm: Clu-GRAD (v1.0)\n"
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
    except Exception as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


def run_comparison():
    output, est_gps = cgp.clugps(f_path=cgp.FILE, min_sup=cgp.MIN_SUPPORT, return_gps=True)
    print(output)

    # Compare inferred GPs with real GPs
    hit_gps, miss_gps = cgp.compare_gps(est_gps, cgp.FILE, cgp.MIN_SUPPORT)
    d_gp = sgp.DataGP(cgp.FILE, min_sup=cgp.MIN_SUPPORT)
    for gp in miss_gps:
        print(gp.print(d_gp.titles))
