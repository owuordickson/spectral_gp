# Spectral clustering for GPs

We borrow the net-win concept used in the work 'Clustering Using Pairwise Comparisons' proposed by R. Srikant and apply it to the
problem of extracting gradual patterns (GPs). In order to mine for GPs, each feature yields 2 gradual items which we use
to construct a ranking matrix that compares rows/objects to one another (i.e., (r1,r2), (r1,r3), (r1,r4), (r2,r3), (r2,r4),
(r3,r4)).

In this algorithm, we transform the object pairs to *net-win matrices*. Finally, we apply spectral clustering to
determine which gradual items belong to the same group based on the similarity of their object pairs. Gradual items in
the same cluster should have almost similar score vector. Every gradual pattern is inferred from a cluster of gradual items. <!-- The research paper is available via this link: -->

<!-- * Owuor, D.O., Runkler, T., Laurent, A. et al. Ant colony optimization for mining gradual patterns. Int. J. Mach. Learn. & Cyber. (2021). https://doi.org/10.1007/s13042-021-01390-w -->


### Requirements:

You will be required to install the following python dependencies before using <em><strong>Clu-GRAD</strong></em> algorithm:<br>
```
                   install python (version => 3.6)

```
<!-- python-dateutil scikit-fuzzy cython h5py mpi4py -->
```
                    $ pip3 install so4gp~=0.1.9 numpy~=1.21.5 ypstruct~=0.0.2 scikit-learn~=1.0.2

```

### Usage:
Use it a command line program with the local package to mine gradual patterns:

For example we executed the <em><strong>Clu-GRAD</strong></em> algorithm on a sample data-set<br>
```
$python3 src/main.py -a 'clugrad' -f data/DATASET.csv
```

where you specify the input parameters as follows:<br>
* <strong>filename.csv</strong> - [required] a file in csv format <br>
* <strong>minSup</strong> - [optional] minimum support ```default = 0.5``` <br>


<strong>Output</strong><br>
```
1. Age
2. Salary
3. Cars
4. Expenses

File: ../data/DATASET.csv

Pattern : Support
[1-, 4+] : 1.0
[1+, 2+, 4-] : 0.6

0.08473014831542969 seconds
```

### License:

* MIT

### Experimental Study

This algorithm was tested on several real-life data sets and its performance analyzed. The analysis of its performance is available through this [link](https://github.com/owuordickson/meso-hpc-lr/tree/master/results/clugp).

### References

* Owuor, D.O., Runkler, T., Laurent, A. et al. Ant colony optimization for mining gradual patterns. Int. J. Mach. Learn. & Cyber. (2021). https://doi.org/10.1007/s13042-021-01390-w
* Anne Laurent, Marie-Jeanne Lesot, and Maria Rifqi. 2009. GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. In Proceedings of the 8th International Conference on Flexible Query Answering Systems (FQAS '09). Springer-Verlag, Berlin, Heidelberg, 382-393.
* Wu, R., Xu, J., Srikant, R., Massoulié, L., Lelarge, M., & Hajek, B. (2015, June). Clustering and inference from pairwise comparisons. In Proceedings of the 2015 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems (pp. 449-450).