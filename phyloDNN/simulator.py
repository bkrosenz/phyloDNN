# from functools import partial, lru_cache
# from multiprocess import pool
import argparse
import collections
from pathlib import Path
from sys import argv
from typing import List
from joblib import Parallel, delayed
import numba
import subprocess
from collections import namedtuple
import tempfile
import os
from itertools import *
import toytree
# import ipcoal
import msprime
import numpy as np
# from datasets import alignment_to_datum
# from sklearn.utils import parallel_backend
__package__ = 'phyloDNN'

NSTATES = 4  # dna
MAX_NE = 1e14
MIN_NE = 1e3
MIN_LOCI = 30
MAX_RECOMB = 1e-3
MAX_BLOCKS = 10
MIN_SITES = 1e1
MAX_SITES = 1e9

SpeciesTreeParameters = namedtuple(
    "SpeciesTreeParameters",
    "speciation_rate extinction_rate outgroup_ratio substition_rate generation_time",
)
LocusTreeParameters = namedtuple(
    "LocusTreeParameters", "duplication_rate loss_rate hgt_rate gc_rate generation_time"
)
SubstitutionRateParameters = namedtuple(
    "SubstitutionRateParameters",
    "species_branch_rate gene_family_rate locus_lineage_rate gen_lineage_rate",
)
INDELIbleParams = namedtuple(
    "INDELIbleParams", "submodel statefreq rates indelmodel indelrate"
)


@numba.njit(parallel=True)
def vec_scaler(x, a=0, b=1):
    xmin = xmax = x[0]
    for i in x[1:]:
        if i > xmax:
            xmax = i
        elif i < xmin:
            xmin = i
    return (x - xmin) * (b - a) / (xmax - xmin) + a


@numba.njit
def minmax_scaler(X, a=0, b=1):
    """"scales a [0,1] rv to arbirtrary interval"""
    return X * (b - a) + a


@numba.njit(parallel=True)
def param_scaler(X):
    """scales hyperparams to """
    z = np.empty(*X.shape)
    for i, x in enumerate(z):
        z[i] = minmax_scaler(x, *bounds[i])
    return z


class TreeSimulator(object):
    tree_height = 1

    def __init__(self, njobs):
        self.njobs = njobs

        # genome-specific priors
        self.birth_death_prior = np.random.gamma
        self.inv_birth_death_prior = np.random.gamma  # std prior, alpha,beta >0

        self.height_prior = np.random.uniform  # low=0, high = tree_height in coal units
        # per-block recomb
        self.ntaxa_prior = np.random.poisson  # mean_taxa > 0

    def run(self, arglist):
        subprocess.run([self.executable] + arglist)

    def make_argument_list(self):
        return


class SeqSimulator(object):
    tree_height = 1

    def __init__(self, njobs):
        self.njobs = njobs
        pass

    def run(arglist):
        subprocess.run([self.executable] + arglist)

    def make_argument_list(self):
        return


class IpCoal(TreeSimulator):
    def __init__(self, genomes_per_run=10, ntaxa=None, type="dna", seed=1234, njobs=4):
        super().__init__(njobs)
        """set random params"""
        # run-specific hyperpriors
        self.N = genomes_per_run
        self.ntaxa = np.repeat(ntaxa, genomes_per_run)
        self.nloci_prior = np.random.lognormal  # mean_loci > 0
        self.nblocks_prior = np.random.triangular  # mean_blocks >0
        # blocks of genes will share common recomb, length, ne, and mutation model
        self.blocks_prior = np.random.multinomial  # no params
        self.block_size_prior = (
            np.random.dirichlet
        )  # symmetric alpha in (0,1) of length nblocks
        self.recomb_prior = np.random.lognormal  # negative mu, sigma >0
        # per-block Ne
        self.ne_prior = np.random.lognormal  # mu >0,sigma >0
        self.nsites_prior = np.random.gamma  # p in (0,1)
        self.state_freq_prior = np.random.dirichlet  # symmetric alpha > 0 * 4
        self.kappa_prior = np.random.exponential  # scale > 0

    def sample(self, p):
        """sample new parameters from the hyperparams given in a vector in (0,1]^21."""
        from joblib import Parallel, delayed

        N = self.N
        # transform param ranges
        # TODO: define appropriate ranges for param vals

        mu = minmax_scaler(p[0], 2, 7)
        s = minmax_scaler(p[1], 1e-2, 1)
        nloci = self.nloci_prior(mu, s, size=N).astype(int)

        lam = minmax_scaler(p[2], 2, 500)

        ntaxa = (
            self.ntaxa if self.ntaxa is not None else self.ntaxa_prior(
                lam, size=N) + 3
        )

        # to avoid combinatorial explosion, each run will have same block sizes and subst model params
        mx = nloci.min()
        nblocks = int(self.nblocks_prior(1, minmax_scaler(p[3], mx / 10), mx))

        a = minmax_scaler(p[5], 2, 20)
        b = minmax_scaler(p[6], 10, 200)
        self.nsites = self.nsites_prior(a, b, size=(N, nblocks)).astype(int)

        alpha = np.repeat(minmax_scaler(p[7], 0, 10), nblocks)

        if nblocks > 1:
            self.block_sizes = np.array(
                [
                    1 + self.blocks_prior(n - nblocks,
                                          self.block_size_prior(alpha))
                    for n in nloci
                ]
            )
        else:
            self.block_sizes = nloci.reshape(-1, 1)

        mu_ne = minmax_scaler(p[8], 10, 16)
        sig_ne = minmax_scaler(p[9], 0, 0.6)

        l, u = sorted(vec_scaler(p[10:12], 50, 500))
        height = self.height_prior(l, u, N)

        s = minmax_scaler(p[12], 1e-5, 0.75)
        mu = minmax_scaler(p[13], -55, -2)
        recomb = self.recomb_prior(mu, s, size=nblocks)
        a = minmax_scaler(p[14], 0.1, 10)
        alpha = np.repeat(a, NSTATES)
        state_freq = self.state_freq_prior(alpha, size=nblocks)
        # TODO: Kappa
        lam = minmax_scaler(p[15], 0, 50)
        kappa = self.kappa_prior(lam, size=nblocks)
        a, b_birth = vec_scaler(p[16:18], 2, 5)
        b_death = minmax_scaler(p[18], 0, 0.05)
        a_sd_bd, b_sd_bd = minmax_scaler(p[19:21], 2, 20)
        birth = self.birth_death_prior(a, b_birth, N)
        death = self.birth_death_prior(a, b_death, N)
        birth_sd = 1.0 / self.inv_birth_death_prior(a_sd_bd, b_sd_bd, N)
        death_sd = 1.0 / self.inv_birth_death_prior(a_sd_bd, b_sd_bd, N)
        print(
            "ntax {}, loci {}, blocksizes {}, birth {}, death {}, birth_sd {}, death_sd {}".format(
                ntaxa, nloci, self.block_sizes, birth, death, birth_sd, death_sd
            )
        )

        def ne_prior(size): return self.ne_prior(
            mean=mu_ne, sigma=sig_ne, size=size
        ).astype(np.int64)

        # number of loci in each block
        self.models = Parallel(n_jobs=self.njobs)(
            delayed(self.model_factory(
                ne_prior, state_freq, kappa, recomb))(*params)
            for params in zip(birth, death, birth_sd, death_sd, ntaxa, height)
        )

    @staticmethod
    def model_factory(ne_prior, state_freq, kappa, recomb):
        """TODO: this method uses the same tree for each locus"""
        import ipcoal

        def make_model(b, d, bs, ds, nx, h):
            from dendropy.simulate import treesim
            tree = toytree.tree(
                treesim.birth_death_tree(
                    b,
                    d,
                    birth_rate_sd=bs,
                    death_rate_sd=ds,
                    num_extant_tips=nx,
                    repeat_until_success=False,
                ).as_string(schema="newick")
            )
            nblocks = state_freq.shape[0]
            n_nodes = len(tree.get_node_values())
            # TODO: add a tree-specific prior and a parent-child prior
            size = (n_nodes, nblocks)
            nes = ne_prior(size)
            tree.mod.node_scale_root_height(nes.max() * 2 * h)

            models = []
            for b in range(nblocks):
                # print(
                #     "b: {} sfreqs: {}".format(b, state_freq[b]),
                #     sum(state_freq[b].tolist()),
                # )
                for node, ne in zip(tree.treenode.traverse(), nes):
                    node.add_feature("Ne", ne[b])
        # NOTE: must modify ~/.conda/envs/py38/lib/python3.8/site-packages/ipcoal/Model.py as follows
        #                 if not np.allclose(fsum, 1, rtol=1e-10):
        # NOTE: must also modify ~/.conda/envs/py38/lib/python3.8/site-packages/ipcoal/Writer.py to
        #         self.ancestral_seq = ancestral_seq.copy() if ancestral_seq is not None else None

                model = ipcoal.Model(
                    tree,
                    recomb=recomb[b],
                    substitution_model={
                        "state_frequencies": state_freq[b].tolist(),
                        "kappa": kappa[b],
                    },
                )
                models.append(model)
            return models

        return make_model

    def simulate(self, trees=False):
        """simulate with the cp"""
        for (i, j), block_size in np.ndenumerate(self.block_sizes):
            print(
                "simulating i={} j={} nblocks={} nsites={}".format(
                    i, j, block_size, self.nsites[i, j]
                )
            )
            self.models[i][j].sim_loci(nloci=block_size,
                                       nsites=self.nsites[i, j])
            if trees:
                self.models[i][j].infer_gene_trees(inference_method="raxml",
                                                   inference_args={"T": self.njobs})


class IpCoalSimple(TreeSimulator):
    def __init__(self, independent_loci=10,
                 ntaxa=None, type="dna", seed=1234, njobs=4):
        super().__init__(njobs)
        """set random params"""
        # run-specific hyperpriors
        self.N = independent_loci
        self.ntaxa = ntaxa
        # genome-specific priors
        self.birth_death_prior = np.random.gamma
        self.inv_birth_death_prior = np.random.gamma  # std prior, alpha,beta >0

        self.height_prior = np.random.uniform  # low=0, high = tree_height in coal units
        # per-block recomb
        self.ntaxa_prior = np.random.poisson  # mean_taxa > 0
        self.recomb_prior = np.random.lognormal  # negative mu, sigma >0
        # per-block Ne
        self.ne_prior = np.random.lognormal  # mu >0,sigma >0
        self.nsites_prior = np.random.gamma  # p in (0,1)
        self.state_freq_prior = np.random.dirichlet  # symmetric alpha > 0 * 4
        self.kappa_prior = np.random.exponential  # scale > 0

    def sample(self, p):
        """sample new parameters from the hyperparams given in a vector in (0,1]^13."""
        N = self.N

        mu_ne = minmax_scaler(p[0], 2, 12)
        sig_ne = minmax_scaler(p[1], 1e-5, 1)
        l, u = sorted(vec_scaler(p[2:4], 1e3, 1e7))
        a_birth_death, b_birth = vec_scaler(p[4:6], 2, 6)
        b_death = minmax_scaler(p[6], 0, 0.05)
        s_recomb = minmax_scaler(p[7], 1e-5, 2)
        a = minmax_scaler(p[8], 0.05, 11)
        a_sites = minmax_scaler(p[9], 4, 30)
        b_sites = minmax_scaler(p[10], 5, 70)
        mu_recomb = minmax_scaler(p[11], -40, -1.5)
        lam = minmax_scaler(p[12], 0, 50)

        height = self.height_prior(l, u)  # generations
        recomb = self.recomb_prior(mu_recomb, s_recomb, size=N)
        state_freq_alpha = np.repeat(a, NSTATES)
        state_freq = self.state_freq_prior(state_freq_alpha, size=N)
        ntaxa = self.ntaxa if self.ntaxa is not None else self.ntaxa_prior(
            lam) + 3
        kappa = self.kappa_prior(lam, size=N)
        birth = self.birth_death_prior(a_birth_death, b_birth)
        death = self.birth_death_prior(a_birth_death, b_death)

        nsites = self.nsites_prior(a_sites, b_sites,
                                   size=N).astype(int)
        nloci = np.ones(N)  # self.nloci_prior(mu, s, size=N).astype(int)

        # make tree
        tree = toytree.rtree.bdtree(ntips=ntaxa, b=birth, d=death, stop='taxa')
        # tree = toytree.tree(
        #     treesim.birth_death_tree(
        #         birth,
        #         death,
        #         birth_rate_sd=birth_sd,
        #         death_rate_sd=death_sd,
        #         num_extant_tips=ntaxa,
        #         repeat_until_success=False,
        #     ).as_string(schema="newick")
        # )
        tree = tree.mod.node_scale_root_height(height)

        self.tree = tree

        nes = self.ne_prior(
            mean=mu_ne, sigma=sig_ne, size=(N, tree.nnodes)
        ).astype(np.int64)
        self.nes = nes
        self.recomb = recomb
        self.state_freq, self.kappa, self.nloci, self.nsites = state_freq, kappa, nloci, nsites
#
#        print('''built model
#        height {}
#        nes {} recomb {}, state_freq {},
#        kappa {}, nloci {}, nsites {}'''.format(height, nes, recomb, state_freq,
#                                                kappa, nloci, nsites))
        print('total length: {} nes {} '.format(
            nsites.sum(),
            nes)
        )
        self.models = [self.make_model(
            *params) for params in zip(repeat(tree), nes, recomb, state_freq, kappa, nloci, nsites)]
        # simulate seqs
        # with parallel_backend("loky", inner_max_num_threads=4):
        #     self.models = Parallel(n_jobs=self.njobs)(
        #         delayed(self.make_model)(*params)
        #         for params in zip(repeat(tree), nes, recomb, state_freq, kappa, nloci, nsites)
        #     )
        print('job', os.getpid(), 'finished models')
        return self

    @ staticmethod
    def make_model(tree, ne, recomb, state_freq, kappa, nloci, nsites):
        ndict = dict(enumerate(ne))
        tree = tree.set_node_values("Ne", values=ndict)

        model = ipcoal.Model(
            tree,
            Ne=None,
            recomb=recomb,
            substitution_model={
                "state_frequencies": state_freq.tolist(),
                "kappa": kappa,
            },
        )
        model.sim_loci(nloci=nloci, nsites=nsites)
        return model

    def get_alignment(self):
        from dendropy import Tree
        from torch import tensor, from_numpy, long
        from torch_geometric.data import Data

        seqs = np.concatenate(
            [m.seqs.squeeze() for m in self.models],
            axis=-1
        ).squeeze() + 1  # reserve 0 for padding
        fully_connected = tensor(
            list(permutations(range(self.ntaxa), 2)), dtype=long).T
        dist = (Tree.get(data=self.tree.newick,
                         schema="newick")
                .phylogenetic_distance_matrix()
                .distances())
        target = tensor(dist)
        return Data(x=from_numpy(seqs), edge_index=fully_connected, y=target)


class FastIpCoal(IpCoalSimple):
    """simulate with IpCoal's built-in simulator using a fixed recombination rate"""

    def __init__(self,
                 nloci=10,
                 nsites=1000,
                 ne=1e5,
                 recomb=1e-9,
                 seed=1234,
                 max_taxa=50,
                 mut=1e-08,
                 replicates=5,
                 save_models=False,
                 njobs=4):
        super().__init__(njobs)
        if save_models:
            self.models = []
        else:
            self.models = None
        self.max_taxa = max_taxa
        self.nsites = nsites
        self.ne = int(ne)
        self.recomb = recomb
        self.mut = mut
        self.replicates = replicates
        self.nloci = nloci

    def sample_trees(self):
        """sample new parameters from the hyperparams given in a vector in (0,1]^13, generates n trees"""
        p = np.random.rand(6)
        b_death = minmax_scaler(p[0], 0, 1)
        lam = minmax_scaler(p[1], 0, 20)
        l, u = sorted(minmax_scaler(x, .01, 40) for x in p[2:4])
        a, b = sorted(minmax_scaler(x, 2, 6) for x in p[4:6])

        height = self.height_prior(l, u, size=self.replicates)  # generations
        birth = self.birth_death_prior(a, b, size=self.replicates)
        death = self.birth_death_prior(a/3, b/3, size=self.replicates)
        taxa = self.ntaxa_prior(lam, size=self.replicates).clip(
            min=3, max=self.max_taxa)

        self.trees = []

        for b, d, h, t in zip(birth, death, height, taxa):
            tree = (toytree.rtree
                    .bdtree(ntips=t, b=b, d=d, stop='taxa')
                    .mod.node_scale_root_height(h*self.ne)
                    )
            self.trees.append(tree)
        return self

    def __len__(self):
        return self.replicates * self.nloci

    # @lru_cache  # NOTE: caching means sims are not random
    def __getitem__(self, i):
        tree = self.trees[i//self.nloci]
        model = ipcoal.Model(
            tree,
            # Ne=self.ne, # use default 10000
            mut=self.mut,
            recomb=self.recomb)
        model.sim_loci(nloci=self.nloci, nsites=self.nsites)
        # return alignment_to_datum(seqs, tree)
        return model.seqs, tree

    def __iter__(self):
        """get nloci seqs for tree i"""
        for tree in self.trees:
            model = ipcoal.Model(
                tree,
                Ne=self.ne,
                mut=self.mut,
                recomb=self.recomb)
            model.sim_loci(nloci=self.nloci, nsites=self.nsites)
            if self.models is not None:
                self.models.append(model)
            for seqs in model.seqs:
                # yield alignment_to_datum(seqs, tree)
                yield model.seqs, tree


def sim_tree(taxa, b, d, h, ne):
    import toytree
    tree = (toytree.rtree
            .bdtree(ntips=taxa, b=b, d=d, stop='taxa')
            .mod.node_scale_root_height(h*ne*2)
            .mod.make_ultrametric()
            )
    tree = tree.set_node_values(
        feature="name",
        values={f'r{i}': f'r{i+1}' for i in range(len(tree))},
    )
    return tree


def simulate_ancestry(nwtree, recomb, nsites, ne):
    import collections
    import msprime
    import numpy
    initial_size = collections.defaultdict(lambda: ne)
    demography = msprime.Demography.from_species_tree(
        nwtree,
        initial_size)
    # assumes rooted binary tree -> N-1 internal nodes
    samps = {f'r{n}': 1 for n in range(1, demography.num_events+2)}
    # print(demography.keys())
    # print(samps)
    # can add recomb map
    ts = msprime.sim_ancestry(
        samples=samps,
        model="smc_prime",
        ploidy=1,
        recombination_rate=recomb,
        gene_conversion_rate=recomb/2,
        gene_conversion_tract_length=5,
        sequence_length=nsites,
        demography=demography)
    ntrees = ts.num_trees
    tract_lengths = numpy.diff(tuple(ts.breakpoints())).astype(int)
    trees = (t.newick() for t in ts.trees())
    s = [f'[{tract}]{tree}\n' for tract,
         tree in zip(tract_lengths, trees)]
    return s


def count_lines(fn):
    with open(fn) as f:
        for i, _ in enumerate(fn):
            pass
    return i


def sim_alignment(cmd: List, seqfile=None):
    # WARNING mutates cmd list
    import subprocess
    with open(cmd[-1]) as f:
        for ntrees, _ in enumerate(f, 1):
            pass

    cmd.insert(2, str(ntrees))
    process = subprocess.run(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    exit_code = process.returncode
    if exit_code != 0:
        print(process.stdout, process.stderr, ' '.join(cmd))
        raise (subprocess.CalledProcessError(
            exit_code, ))
    res = process.stdout.decode()
    if seqfile:
        with open(seqfile, 'w') as f:
            f.write(res)


class MsPrime(TreeSimulator):

    def __init__(self,
                 nloci=1,
                 nsites=10_000,
                 ne=10_000,
                 seed=1234,
                 max_taxa=50,
                 replicates=5,
                 njobs=4):
        super().__init__(njobs)
        self.max_taxa = max_taxa
        self.nsites = nsites
        self.ne = int(ne)
        self.replicates = replicates
        self.nloci = nloci

    def __repr__(self) -> str:
        return super().__repr__()+f'Ne {self.ne}, nsites: {self.nsites}, reps {self.replicates}'

    def load_species_trees(self, filename, l: int = None, u: int = None):
        import dendropy
        with open(filename, 'r') as f:
            self.trees = []
            for t in f:
                # tree = dendropy.Tree.get(
                #     data=t.replace('taxon', 'r'),
                #     schema="newick",
                #     rooting='default-rooted').as_string(schema="newick")
                tree = toytree.tree(
                    t.replace('taxon', 'r')) # toytree defaults to rooted
                if 'r0' in tree.get_tip_labels():
                    tree = (tree.set_node_values(
                        feature="name",
                        values={f'r{i}': f'r{i+1}' for i in range(len(tree))})
                        .mod.make_ultrametric())

                self.trees.append(tree)
        if l and u:
            heights = self.height_prior(
                l, u,
                size=len(self.trees))  # x*Ne generations
            self.trees = [t.mod.node_scale_root_height(
                h*self.ne*2) for h, t in zip(heights, self.trees)]

    def sample_species_trees(self, l=20, u=120,
                             theta=11,
                             k=25,
                             ratio=None, taxa=None,
                             adjust_terminal=None):
        """sample new parameters from the hyperparams given in a vector in (0,1]^13, generates n trees"""
        p = np.random.rand(6)
        # b_death = minmax_scaler(p[0], 0, 1)
        lam = minmax_scaler(p[1], 0, 50)
        # l, u = 20, 120  # sorted(minmax_scaler(x, 1, 30) for x in p[2:4])
        # theta, k = sorted(minmax_scaler(x, 6, 30) for x in p[4:6])

        height = self.height_prior(
            l, u,
            size=self.replicates)  # x*Ne generations
        birth = self.birth_death_prior(k, theta, size=self.replicates)
        if ratio is not None:
            death = ratio*birth
        else:
            death = self.birth_death_prior(
                3*k/4, 3*theta/4,
                size=self.replicates)
            death = np.minimum(birth, death)
        print(
            f'hyperparams: {self.birth_death_prior} {k}, {theta}. {self.height_prior} {l}, {u}, {ratio}')
        if taxa is None:
            taxa = self.ntaxa_prior(lam, size=1).clip(
                min=3, max=self.max_taxa)  # for now, each samp has same # of tips
        # taxa = self.ntaxa_prior(lam, size=self.replicates).clip(
        #     min=3, max=self.max_taxa)
    # TODO: iter over taxa

        self.trees = Parallel(self.njobs)(delayed(sim_tree)(taxa, *args, self.ne)
                                          for args in zip(birth, death, height))
        # if adjust_terminal is not None:
        #     for t in self.trees:
        #         for n in t.no
        return self

    def sample_gene_trees(self,
                          recomb=1e-9,
                          ):
        self.recomb = recomb
        from joblib.externals.loky import set_loky_pickler
        set_loky_pickler('dill')
        # simulate_ancestry(
        #     self.trees[0].write(tree_format=5),
        #     # {n: 1 for n in tree.get_tip_labels()},
        #     self.recomb,
        #     self.nsites,
        #     self.ne)
        gene_trees = Parallel(self.njobs, )(
            delayed(simulate_ancestry)(
                tree.write(tree_format=5),
                # {n: 1 for n in tree.get_tip_labels()},
                self.recomb,
                self.nsites,
                self.ne)
            for tree in self.trees)
        self.max_trees = max(map(len, gene_trees))
        # with open(filename, 'w') as f:
        #     for gt in gene_trees:
        #         f.writelines(gt)
        self.gene_trees = gene_trees

    def write_species_trees(self, filename):
        with open(filename, 'w') as f:
            for s in self.trees:
                f.write(s.write(tree_format=5)+'\n')

    def write_gene_trees(self, filename, separate=False):
        # unused
        for i, t in enumerate(self.gene_trees):
            with open(filename/f'{i}.nw', 'w') as f:
                f.writelines(t)
        return
        self.max_trees = max(ts.num_trees for ts in self.gene_trees)
        if separate:
            for i, t in enumerate(self.gene_trees):
                with open(filename/f'{i}.nw', 'w') as f:
                    for ds in self.gene_trees:
                        tract_lengths = np.diff(
                            tuple(ds.breakpoints())).astype(int)
                        trees = (t.newick() for t in ds.trees())
                        s = ''.join(f'[{tract}]{tree}\n' for tract,
                                    tree in zip(tract_lengths, trees))
                        f.write(s)
        else:
            with open(filename, 'w') as f:
                for ds in self.gene_trees:
                    tract_lengths = np.diff(
                        tuple(ds.breakpoints())).astype(int)
                    trees = (t.newick() for t in ds.trees())
                    s = ''.join(f'[{tract}]{tree}\n' for tract,
                                tree in zip(tract_lengths, trees))
                    f.write(s+'\n')
        print(self.max_trees)

    def sample_alignments_parallel(self,
                                   mut=1e-08,
                                   alpha=2,
                                   treedir=None,
                                   invariant=0,
                                   seqdir=None):
        self.mut = mut
        self.alpha = alpha
        self.invariant = invariant

        cmd = ['seq-gen',
               '-p',  # self.max_trees,
               '-s', mut,
               '-op',
               '-q',
               '-l', self.nsites,
               '-mPAM',
               '-i', self.invariant,
               ]
        if alpha:
            cmd.extend(['-a', self.alpha])
        cmd = list(map(str, cmd))
        # for t in treedir.rglob('*.nw'):
        #     print(t)
        #     sim_alignment(cmd+[str(t)], seqdir/f'{t.stem}.phy')
        #     break
        res = Parallel(self.njobs)(
            delayed(sim_alignment)(cmd+[str(t)], seqdir/f'{t.stem}.phy') for t in treedir.rglob('*.nw')
        )
        return res

    def sample_alignments(self,
                          mut=1e-08,
                          alpha=2,
                          treefile=None,
                          invariant=0,
                          seqfile=None):
        self.mut = mut
        self.alpha = alpha
        self.invariant = invariant
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            if treefile is None:
                treefile = d/'trees'
                self.write_gene_trees(treefile)
            cmd = ['seq-gen',
                   '-s', mut,
                   '-op',
                   '-l', self.nsites,
                   '-mPAM',
                   '-p', self.max_trees,
                   '-i', self.invariant,
                   treefile]
            if alpha:
                cmd.extend(['-a', self.alpha])
            cmd = map(str, cmd)
            process = subprocess.run(
                cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            exit_code = process.returncode
            if exit_code != 0:
                print(process.stdout, process.stderr, ' '.join(cmd))
                raise (subprocess.CalledProcessError(
                    exit_code, ))
        res = process.stdout.decode()
        if seqfile:
            with open(seqfile, 'w') as f:
                f.write(res)
        return res

        # model = msprime.PAM()
        # self.loci = []
        # for ts in self.gene_trees:
        #     mts = msprime.sim_mutations(ts,
        #                                 model=model,
        #                                 rate=self.mut,
        #                                 random_seed=5678)
        #     self.loci.append(mts)


class INDELIble(SeqSimulator):
    def __init__(self):
        pass

    def initialize(self):
        self.npartitions
        self.models


class SimPhy(TreeSimulator):
    individuals_per_species = 1
    distance_dep_hgt = True
    verbosity = 0
    distributions = {
        "F": "fixed",
        "U": "uniform",
        "N": "normal",
        "E": "exponential",
        "G": "gamma",
        "LN": "lognormal",
        "LU": "loguniform",
        "SL": "lognormal*const",
        "D": "dirichlet",
    }
    dist_params = {"F": ()}

    def initialize(self, simphy, seed=1234):
        self.simphy = simphy
        self.indelible = indelible

    def get_indelible_params(self):
        """extends the INDELible_wrapper.pl options from mallo&al"""
        return

    def simulate(self, indelible=None):
        """can pipe directly to an indelible simulator, or just store trees"""
        with tempfile.TemporaryDirectory() as output_dir, tempfile.NamedTemporaryFile() as config_file:
            config_file.writelines(indelible_params)
            subprocess.run(self.get_argument_list(self.simphy))
            if indelible is not None:
                indelible.run(config_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and test.')

    parser.add_argument('--dir',
                        type=Path,
                        help='''root directory''')
    parser.add_argument('--treefile',
                        type=Path,
                        help='''path to species tree file if they've already been generated''')
    parser.add_argument('--replicates',
                        type=int,
                        help='''num species trees''')
    parser.add_argument('--nsites',
                        type=int,
                        help='''sites''')
    parser.add_argument('--ntaxa',
                        type=int,
                        help='''number of taxa. ''')
    parser.add_argument('--ne',
                        type=int,
                        default=10_000,
                        help='''pop size. ''')
    parser.add_argument('--lower',
                        type=int,
                        help='''lower limit of height prior''')
    parser.add_argument('--upper',
                        type=int,
                        help='''upper limit of height prior''')
    parser.add_argument('--njobs',
                        type=int,
                        default=6,
                        help='''number of jobs''')
    parser.add_argument('--mutation_rate',
                        type=float,
                        default=5e-6,
                        help='''mutation rate per site per generation''')
    parser.add_argument('--alpha',
                        type=float,
                        default=2,
                        help='''alpha param for site-specific rate heterogeneity''')
    parser.add_argument('--recombination_rate',
                        type=float,
                        default=1e-8,
                        help='''number of jobs''')
    parser.add_argument('--terminal',
                        type=float,
                        default=None,
                        help='''add x*Ne to all terminal branches''')
    args = parser.parse_args()
    # print('usage: simulator.py <path> <nreps> <nsites> <ntaxa> <min height> <max height> <njobs> <mut> <recomb>')
    print(argv)
    s = MsPrime(njobs=args.njobs,
                nsites=args.nsites,
                ne=args.ne,
                replicates=args.replicates)
    print(s.njobs)
    root = args.dir
    root.mkdir(parents=True, exist_ok=True)

    genes_dir = root/'genes'
    seqs_dir = root/'seqs'
    genes_dir.mkdir(exist_ok=True)
    seqs_dir.mkdir(exist_ok=True)

    with open(root/'cmd.txt', 'w') as f:
        f.write(' '.join(argv))
    if args.treefile:
        s.load_species_trees(args.treefile,
                             l=args.lower,
                             u=args.upper,
                             )
    else:
        s.sample_species_trees(taxa=args.ntaxa,
                               l=args.lower,
                               u=args.upper,
                               ratio=None,
                               adjust_terminal=args.terminal)
    s.write_species_trees(root/'trees.nw')
    print('wrote species trees')

    s.sample_gene_trees(recomb=args.recombination_rate)  # 1e-8
    s.write_gene_trees(genes_dir)

    print('wrote gene trees')

    s.sample_alignments_parallel(
        mut=args.mutation_rate,
        alpha=args.alpha,
        treedir=genes_dir,
        seqdir=seqs_dir)  # 5e-6
    print('wrote alignments')

    # p = np.random.uniform(size=100)
    # ip = IpCoal(5, ntaxa=5)
    # ip.sample(p)
    # ip.simulate()
    # print(ip.models)
