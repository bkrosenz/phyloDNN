from collections import namedtuple
from itertools import *
from scipy.spatial.distance import squareform
from typing import Iterable, List, Tuple
import re

from joblib import Parallel, delayed
import numpy.typing as npt
from functools import partial

import dendropy
import numpy as np
from Bio import AlignIO
from dendropy import PhylogeneticDistanceMatrix, TaxonNamespace
from torch import Tensor
import torch
from pathlib import Path
from ete3 import Tree as ETree
from torch.nn.utils.rnn import pad_sequence

from phyloDNN.bionj import BioNJ

MSA = namedtuple("AlignmentData", ["x", "ids", "filename", "idx", "dists"])

LETTERS = (
    "-",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)
# there are 17 degenerate IUPAC DNA codes, 21 AA, + 1 for zero padding (value=0) to match array dim
# in ungapped version, we ignore gaps, but still keep this embedding dim

N_STATES = len(LETTERS) + 1
let2int = dict(zip(LETTERS, range(1, 1 + N_STATES)))
letter_to_int = np.vectorize(lambda let: let2int[let], otypes=[np.uint8])


class PhyloFormerDS(object):
    ALPHABET = "ARNDCQEGHILKMFPSTWYVX-"
    LOOKUP = {char: index for index, char in enumerate(ALPHABET)}
    lookup_func = np.vectorize(
        lambda let: PhyloFormerDS.LOOKUP[let], otypes=[np.uint8])

    def __call__(self, s):
        """convert letter to int"""
        return self.lookup_func(s)


class AlignmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        directory: Path,
        max_length: int | None = None,
        preprocessor: torch.nn.Module | None = None,
        alignments_per_file: int | None = None,
        matrix_type="distance",
        device: str = "cpu",
        transform=None,
        keep_invariant=True,
        true_distances: bool = False,
        seqdir: str = "seqs",
        treefile: str = "trees.tsv",
        phyloformer_style: bool = False,
    ):
        """generate alignment tensors with or without species trees

        :param directory: location of dataset
        :type directory: Path
        :param max_length: if not None, will take a random contiguous block of this length, defaults to None
        :type max_length: int, optional
        :param preprocessor: process aligns, defaults to None
        :type preprocessor: torch.nn.Module, optional
        :param alignments_per_file: number of aligns in each file, defaults to None
        :type alignments_per_file: int, optional
        :param matrix_type: distance or covariance, defaults to 'distance'
        :type matrix_type: str, optional
        :param device: cpu or cuda, defaults to 'cpu'
        :type device: str, optional
        :param transform: transform, defaults to None
        :type transform: _type_, optional
        :param keep_invariant: whether to keep invariant sites.  If False, alignments must be padded, defaults to True
        :type keep_invariant: bool, optional
        :param true_distances: include true_distances (patristic distance matrix of trees from trees.nw file in directory), defaults to False
        :param seqdir: subdirectory holding sequences, defaults to 'seqs'
        :type seqdir: str, optional
        :param treefile: path from root directory to file containing trees, defaults to 'trees.tsv'
        """
        super().__init__()
        self.phyloformer_style = phyloformer_style
        self.keep_invariant = keep_invariant
        self.root = directory
        self.transform = transform
        self.device = device
        if preprocessor is not None:
            self.preprocessor = preprocessor.to(device)
        self.matrix_type = matrix_type
        seqdir = self.root / seqdir
        if not seqdir.exists():
            raise FileNotFoundError(f"directory {seqdir} not found")
        # assume that bash lex sort from concatenating trees is the same as lex sort here
        self.files = tuple(
            fn for fn in seqdir.rglob("*.phy") if "uniqueseq" not in fn.name
        )  # don't need to sort
        if alignments_per_file is not None:
            self.alignments_per_file = alignments_per_file
        else:
            try:
                # assume all files have same number of alignments
                self.alignments_per_file = sum(
                    1 for _ in AlignIO.parse(self.files[0], format="phylip")
                )
            except:
                print(self.files[0])
                raise
        self.max_length = max_length
        self.true_distances = true_distances
        if true_distances:
            self.treefile = self.root / treefile
            self.rx = re.compile("^(\d+)")
            with open(self.treefile) as f:
                self.trees = dict(s.strip().split() for s in f.readlines())
            if any(x.stem not in self.trees for x in self.files):
                print(self.files[0].stem)
                raise ValueError(
                    f"Tree file {self.treefile} and alignment files in {seqdir} do not match. Please check the treefile and alignment files."
                )

    # @ lru_cache
    def __len__(self):
        return self.nfiles * self.alignments_per_file

    # @ lru_cache
    @property
    def nfiles(self):
        return len(self.files)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get_item(idx)
        elif isinstance(idx, Iterable):
            # aligns,files, indices,true_distances
            if self.true_distances:
                aligns, * \
                    fields, true_distances = zip(
                        *(self.get_item(i) for i in idx))
                item = torch.stack(aligns), * \
                    fields, torch.stack(true_distances)
            else:
                aligns, *fields = zip(*(self.get_item(i) for i in idx))
                item = torch.stack(aligns), *fields

        return item

    def get_item(self, idx: int) -> MSA:
        """get alignment tensor, tree, and metadata for a given index"""

        file_no = idx // self.alignments_per_file
        seq_no = idx % self.alignments_per_file
        filename = self.files[file_no]
        # TODO: rewrite this to return Data objects OR rewrite axial so it doesn't need ground truth y.

        x, ids = alignment_to_torch(
            filename,
            return_keys=True,
            letter_to_int=PhyloFormerDS() if self.phyloformer_style else letter_to_int)

        ids = ids[seq_no]
        L = x.shape[-1]
        if self.max_length is not None and L > self.max_length:
            start = np.random.randint(L - self.max_length)
            end = start + self.max_length
            x = x[seq_no][..., start:end]
        else:  # take the whole alignment
            x = x[seq_no]
        if hasattr(self, "preprocessor"):
            device = x.device
            x = self.preprocessor(x.to(self.device).unsqueeze(0)).to(device)
        x = x.squeeze()
        if not self.keep_invariant:
            mask = (x.max(0).values - x.min(0).values).bool()
            x = x[:, mask]

        if self.true_distances:
            if self.nfiles == 1:
                tree_index = seq_no
            elif self.alignments_per_file == 1:
                # int(self.rx.findall(filename.stem)[0])
                tree_index = filename.stem
            else:  # WARNING: aligns in sorted file list must have same order as trees.nw
                try:
                    tree_index = (
                        int(self.rx.findall(filename.stem)[0])
                        * self.alignments_per_file
                        + seq_no
                    )
                except:
                    tree_index = file_no * self.alignments_per_file + seq_no
            dists = self.get_distances(tree_index)
            return MSA(x, ids, filename, idx, dists)
        else:
            return MSA(x, ids, filename, idx, None)

    def get_distances(self, idx: int | str) -> Tensor:
        """get distance matrix for a given tree index, optionally applying self.transform"""
        # print(treefile, idx)
        tree = self.trees[idx]
        tree = ETree(tree)
        if self.transform:
            tree = self.transform(tree)
        if self.matrix_type == "covariance":
            mat = tree_cov(tree)
        else:
            mat = tree_dist(tree)
        return mat


def collate_alignment_dataset(data: List, true_distances: bool = True):
    """collate function for alignment dataset.
    If true_distances is True, will return a tuple of (alignments, labels, ids, filename, indices, distances).
    Otherwise, will return (alignments, labels, ids, filename, indices)"""
    if len(data) > 3 and isinstance(data[2], Path):
        return data
    elif len(data) == 1:  # single item
        aligns, *fields = data[0]
        return aligns[None, ...], *fields
    dists = None
    if data[0].dists is not None:
        aligns, *fields, dists = zip(*data)
        dists = torch.stack(dists)
    else:
        aligns, *fields = zip(*data)
    try:
        aligns = torch.stack(aligns, dim=0)
    except (TypeError, RuntimeError):  # if alignments are not the same size
        # print([len(a) for a in aligns])
        aligns = pad_sequence(
            [a.transpose(-1, -2) for a in aligns], padding_value=0.0, batch_first=True
        ).transpose(-1, -2)
    if dists is None:
        return aligns, *fields
    else:
        return aligns, *fields, dists


def tree_cov(t: ETree, normalization=False, rooted=False):
    distances = {}
    if not rooted:
        t.set_outgroup(t.get_midpoint_outgroup())
    root = t.get_tree_root()
    leaves = sorted(t.get_leaves(), key=lambda l: l.name)
    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i <= j:
                mrca = leaf1.get_common_ancestor(leaf2)
                distances[(i, j)] = root.get_distance(mrca)
    if normalization:
        diam = max(distances.values())
        for dist in distances:
            distances[dist] /= diam
    y = torch.tensor(list(distances.values()))

    return y


def tree_dist_dpy(t: dendropy.Tree, normalization=False):
    distances = {}
    leaves = sorted(t.leaf_nodes(), key=lambda n: n.taxon.label)
    pdm = t.phylogenetic_distance_matrix()
    for j, leaf2 in enumerate(leaves):
        for i in range(j):
            leaf1 = leaves[i]
            distances[(i, j)] = pdm(leaf1, leaf2)
    if normalization:
        diam = max(distances.values())
        for dist in distances:
            distances[dist] /= diam
    y = torch.Tensor(list(distances.values()))

    return y


def tree_dist(t: ETree, normalization: bool = False, with_labels=False):
    """calculate pairwise patristic distances between leaves of a tree.
    leaf names must be unique"""
    distances = {}
    leaves = sorted(t.get_leaves(), key=lambda l: l.name)
    n = len(leaves)
    for i, leaf1 in enumerate(leaves):
        for j in range(i + 1, n):
            leaf2 = leaves[j]
            distances[(leaf1.name, leaf2.name)] = leaf1.get_distance(leaf2)
    if normalization:
        diam = max(distances.values())
        for dist in distances:
            distances[dist] /= diam
    y = torch.tensor(list(distances.values()))
    if with_labels:
        return y, tuple(distances.keys())
        # order is important
    else:
        return y


def get_sub_alignment(align, l, max_start, min_len, max_len):
    start = np.random.randint(max_start)
    if min_len == max_len:
        end = start + min_len
    else:
        end = np.random.randint(
            min(l - 1, start + min_len), min(l, start + max_len))
    a = align[:, start:end]
    return a


def array_to_mat(ix: torch.LongTensor, arr: torch.Tensor, **kwargs):
    return sparse_coo_tensor(ix, arr, **kwargs).to_dense()


def tree2dist(tree: dendropy.Tree, square=True) -> Tensor:
    """convert a dendropy tree to a distance matrix tensor
    Args:
        tree (dendropy.Tree): dendropy tree object
        square (bool, optional): return square distance matrix. Defaults to True.
    Returns:
        Tensor: distance matrix
    """

    dmat = tree.phylogenetic_distance_matrix()

    ns = sorted(dmat.taxon_namespace.get_taxa(map(str, range(len(tree)))))
    dist = [dmat.distance(*pair) for pair in combinations(ns, 2)]
    if square:
        dist = squareform(dist)
    return torch.tensor(dist)


def alignment_to_torch(
    align_filename: Path | str,
    ntaxa: int = None,
    no_gaps=False,
    return_keys=False,
    taxon_mapper: dict = None,
    letter_to_int=letter_to_int,
) -> List:
    """Sorts sequences by name alphabetically

    Args:
        align_filename (Path): phylip file
        ntaxa (int, optional): number of taxa. Defaults to None.
        no_gaps (bool, optional): ungap sequences. Defaults to False.
        return_keys (bool, optional): return names. Defaults to False.
        taxon_mapper (dict, optional): relabel sequences. Defaults to None.

    Returns:
        List: _description_
    """
    aligns = AlignIO.parse(align_filename, format="phylip")
    alignments = []
    taxon_labels = []
    for i, rep in enumerate(aligns):
        if ntaxa is None:
            ntaxa = len(rep)
        # TODO: need to handle different size genes since we'll be concatenating them...
        alignment_dict = dict()
        keys = []
        seqs = []
        rep.sort()
        for seq_record in rep:
            # TODO: handle gene duplication
            if taxon_mapper is not None:
                tid = taxon_mapper[seq_record.id.split("_")[0]]
            else:
                tid = seq_record.id
            if no_gaps:
                seq = seq_record.seq.ungap()
            else:
                seq = seq_record.seq
            keys.append(tid)
            seqs.append(letter_to_int(seq))
        # keys = sorted(alignment_dict)
        # seq_array = [alignment_dict[k] for k in keys]
        # NOTE: do not sort keys! this will mess up the alignment-tree mapping
        # keys, seq_array = map(list, zip(*alignment_dict.items()))
        alignments.append(seqs)
        taxon_labels.append(keys)
    alignments = torch.tensor(np.array(alignments), dtype=torch.uint8)
    if return_keys:
        return alignments, taxon_labels
    return alignments


def d_RF(y: torch.Tensor,
         ypred: torch.Tensor,
         workers: int = 4,
         weighted: bool = False,
         matrix_type: str = "distance",
         algorithm: str = 'bionj'
         ):
    """build (Bio)NJ tree from distance matrices, calculate  Robinson-Foulds distance between true and predicted trees
    Args:
        y (): true trees
        ypred (torch.Tensor): predicted pairwise distances
        workers (int, optional): number of workers. Defaults to 4.
        weighted (bool, optional): use weighted RF distance. Defaults to False.
    Returns:
        Tuple: mean and std of distances
    """
    # TODO: this is wasteful, should load tree str from val dataset so don't have to perform NJ
    calculate_dRF = wrf_distance if weighted else partial(
        rf_distance, normalize=True)
    if y.dim() == 1:
        y = y.unsqueeze(0)
        ypred = ypred.unsqueeze(0)
    n = d = 0.0
    if matrix_type == "covariance":
        ypred = squareform(ypred)
        ypred = (
            covariance_to_distance(ypred_batch) for ypred_batch in ypred
        )  # requires squareform matrix
    # ns=sq.TaxonNamespace(range()) # find out how many taxa
    # drf=Parallel(self.workers,backend='threading')(delayed(rf_distance)(*ts) for ts in zip(true_trees,pred_trees))

    with Parallel(workers, require="sharedmem") as parallel:
        true_trees = parallel(delayed(njtree)(
            t, algorithm=algorithm) for t in y)
        pred_trees = parallel(
            delayed(njtree)(pred, true_tree.taxon_namespace, algorithm=algorithm)
            for pred, true_tree in zip(ypred, true_trees)
        )
        distances = parallel(
            delayed(calculate_dRF)(pred_tree, true_tree)
            for pred_tree, true_tree in zip(pred_trees, true_trees)
        )
    distances = np.array(distances)
    return distances.mean(), distances.std()


def yh_prob(t, d_rf):
    """probability that a random BINARY UNROOTED YH tree has distance < d_rf from t
    (i.e. shares less than n-3-d_rf/2 nontrivial partitions).
    Uses Poisson approximation"""
    from scipy.stats import poisson

    n = len(t)
    if n == 4:
        return 0.5, (1 + (d_rf != 0)) / 2
    elif n < 4:
        return 1, 1
    s = n - 3 - d_rf / 2
    lam = cherries(t) / (2 * n)
    pois = poisson(mu=lam)
    return 1 - pois.pmf(s), 1 - pois.cdf(s)


class ExtendedPDM(PhylogeneticDistanceMatrix):

    def bionj_tree(self,
                   is_weighted_edge_distances=True,
                   tree_factory=None,
                   ):
        """
        Returns an Neighbor-Joining (NJ) tree based on the distances in the matrix.

        Calculates and returns a tree under the Neighbor-Joining algorithm of
        Saitou and Nei (1987) for the data in the matrix.

        Parameters
        ----------
        is_weighted_edge_distances: bool
            If ``True`` then edge lengths will be considered for distances.
            Otherwise, just the number of edges.

        Returns
        -------
        t : |Tree|
            A |Tree| instance corresponding to the Neighbor-Joining (NJ) tree
            for this data.

        Examples
        --------

        ::

            import dendropy

            # Read data from a CSV file into a PhylogeneticDistanceMatrix
            # object
            with open("distance_matrix.csv") as src:
                pdm = dendropy.PhylogeneticDistanceMatrix.from_csv(
                        src,
                        is_first_row_column_names=True,
                        is_first_column_row_names=True,
                        is_allow_new_taxa=True,
                        delimiter=",",
                        )

            # Calculate the tree
            nj_tree = pdm.nj_tree()

            # Print it
            print(nj_tree.as_string("nexus"))


        References
        ----------
        Gascuel

        """

        if is_weighted_edge_distances:
            original_dmatrix = self._taxon_phylogenetic_distances
        else:
            original_dmatrix = self._taxon_phylogenetic_path_steps
        if tree_factory is None:
            tree_factory = dendropy.Tree
        tree = tree_factory(taxon_namespace=self.taxon_namespace)
        tree.is_rooted = False

        # initialize node pool
        node_pool = []
        for t1 in self._mapped_taxa:
            nd = tree.node_factory()
            nd.taxon = t1
            nd._nj_distances = {}
            node_pool.append(nd)

        # initialize factor
        n = len(self._mapped_taxa)

        # cache calculations
        for nd1 in node_pool:
            nd1._nj_xsub = 0.0
            for nd2 in node_pool:
                if nd1 is nd2:
                    continue
                d = original_dmatrix[nd1.taxon][nd2.taxon]
                nd1._nj_distances[nd2] = d
                nd1._nj_xsub += d

        while n > 1:

            # calculate the Q-matrix
            min_q = None
            nodes_to_join = None
            for idx1, nd1 in enumerate(node_pool[:-1]):
                for idx2, nd2 in enumerate(node_pool[idx1+1:]):
                    v1 = (n - 2) * nd1._nj_distances[nd2]
                    qvalue = v1 - nd1._nj_xsub - nd2._nj_xsub
                    if min_q is None or qvalue < min_q:
                        min_q = qvalue
                        nodes_to_join = (nd1, nd2)

            # create the new node
            new_node = tree.node_factory()

            # attach it to the tree
            for node_to_join in nodes_to_join:
                new_node.add_child(node_to_join)
                node_pool.remove(node_to_join)

            # calculate the distances for the new node
            new_node._nj_distances = {}
            new_node._nj_xsub = 0.0
            for node in node_pool:
                # actual node-to-node distances
                v1 = 0.0
                for node_to_join in nodes_to_join:
                    v1 += node._nj_distances[node_to_join]
                v3 = nodes_to_join[0]._nj_distances[nodes_to_join[1]]
                dist = 0.5 * (v1 - v3)
                new_node._nj_distances[node] = dist
                node._nj_distances[new_node] = dist

                # Adjust/recalculate the values needed for the Q-matrix
                # calculations
                new_node._nj_xsub += dist
                node._nj_xsub += dist
                for node_to_join in nodes_to_join:
                    node._nj_xsub -= node_to_join._nj_distances[node]

            # calculate the branch lengths
            if n > 2:
                v1 = 0.5 * nodes_to_join[0]._nj_distances[nodes_to_join[1]]
                v4 = 1.0/(2*(n-2)) * \
                    (nodes_to_join[0]._nj_xsub - nodes_to_join[1]._nj_xsub)
                delta_f = v1 + v4
                delta_g = nodes_to_join[0]._nj_distances[nodes_to_join[1]] - delta_f
                nodes_to_join[0].edge.length = delta_f
                nodes_to_join[1].edge.length = delta_g
            else:
                d = nodes_to_join[0]._nj_distances[nodes_to_join[1]]
                nodes_to_join[0].edge.length = d / 2
                nodes_to_join[1].edge.length = d / 2

            # clean up
            for node_to_join in nodes_to_join:
                del node_to_join._nj_distances
                del node_to_join._nj_xsub

            # add the new node to the pool of nodes
            node_pool.append(new_node)

            # adjust count
            n -= 1

        tree.seed_node = node_pool[0]
        del tree.seed_node._nj_distances
        del tree.seed_node._nj_xsub
        return tree


def njtree(
    mat: Tuple | Tensor | npt.NDArray,
    namespace: TaxonNamespace | List | None = None,
    midpoint: bool = False,
    algorithm: str = 'bionj',
) -> dendropy.Tree:
    """generate Neighbor-Joining tree from distance matrix

    Args:
        mat (Union[tuple, Tensor, np.array]): full distance matrix
        namespace (TaxonNamespace, optional): dendropy namespace for tree. Defaults to None.
        midpoint (bool, optional): Midpoint-root the tree. Defaults to False.

    Returns:
        Tree: dendropy Tree object
    """
    # TODO: does this REQUIRE dmat in squareform?
    import io
    if isinstance(mat, tuple):
        if isinstance(mat[0], Tensor):
            mat = array_to_mat(
                mat[0].detach().long().cpu(),
                mat[1].detach().float().cpu()
            ).numpy()
        else:
            mat = np.array(mat)
    elif isinstance(mat, Tensor):
        mat = mat.detach().float().cpu().numpy()
        if len(mat.shape) == 1:  # numpy array
            mat = squareform(mat)
        elif mat.shape[0] == 1:
            mat = squareform(mat[0])

        # elif > 1 and mat.size(0) == mat.size(1):
        #     mat = mat.cpu().detach().numpy()
    # mat = (mat+mat.T)/2  # force symmetric
    if algorithm == 'bionj':
        if namespace is None:
            names = list(map(str, range(len(mat))))
        elif isinstance(namespace, TaxonNamespace):
            names = namespace.labels()
        else:
            names = namespace
        t = dendropy.Tree.get(
            data=BioNJ().reconstruct_tree(mat, names),
            taxon_namespace=namespace,
            schema="newick")

    elif algorithm == "nj":
        if namespace is None:
            # names = list(string.ascii_lowercase[:len(mat)])
            names = list(map(str, range(len(mat))))
            namespace = TaxonNamespace(names)
            namespace.sort()  # NOTE: sort func must be consistent with dataset.parse_alignments; ensure labels are strings
        elif isinstance(namespace, list):
            namespace = TaxonNamespace(namespace)
        names = namespace.labels()
        # save to string buffer
        fh = io.StringIO(",".join(names) + "\n")
        fh.seek(0, 2)
        np.savetxt(fh, mat, delimiter=",", fmt="%10.8f")
        fh.seek(0)
        pdm = PhylogeneticDistanceMatrix.from_csv(
            fh,
            taxon_namespace=namespace,
            is_first_column_row_names=False,
            delimiter=",",
        )
        t = pdm.nj_tree()
    else:
        raise NotImplementedError(f"unknown alg {algorithm}")
    if midpoint:
        t.reroot_at_midpoint()

    return t


def get_sorted_leaves(t: str):
    """get lexicographically sorted leaf names of a tree"""
    t = ETree(t)
    leaves = sorted(t.get_leaves(), key=lambda l: l.name)
    return leaves


def cherries(t):
    """count cherries in dendropy tree"""
    n = 0
    l = len(t) - 2
    if t.bipartition_encoding is None:
        t.encode_bipartitions()
    for b in t.bipartition_encoding:
        ones = b.split_as_bitstring().count("1")
        if ones == 2 or ones == l:
            n += 1
    return n


def partition_sizes(bipartition_encoding):
    """count all non-trivial partitions in dendropy bipartition set"""
    from collections import Counter

    n = len(bipartition_encoding)
    counts = Counter(b.split_as_bitstring().count("1")
                     for b in bipartition_encoding)
    for i in range(1, n // 2):
        counts[i] += counts[n - i]
        counts[n - i] = counts[i]
    del counts[1], counts[n], counts[n - 1]
    return counts


def make_tree(
    tree, namespace: TaxonNamespace | None = None, with_distances=False
) -> dendropy.Tree | Tuple:
    if isinstance(tree, dendropy.Tree):
        return tree
    if not isinstance(tree, str):
        tree = tree.newick
    if not tree.endswith(";"):
        tree += ";"
    tree = dendropy.Tree.get(data=tree, taxon_namespace=namespace, schema="newick")
    if with_distances:
        return tree, tree2dist(tree, square=True)
    else:
        return tree


def rf_distance(
    t1: dendropy.Tree | str, t2: dendropy.Tree | str, normalize: bool = True
):
    """unroot, then calculate unweighted Robinson-Foulds dist"""
    from dendropy.calculate import treecompare

    if isinstance(t1, str):
        t1 = make_tree(t1)
    if isinstance(t2, str):
        t2 = make_tree(t2, namespace=t1.taxon_namespace)
    if t1.is_rooted:
        t1 = t1.clone()
        t1.deroot()
    if t2.is_rooted:
        t2 = t2.clone()
        t2.deroot()
    d = treecompare.symmetric_difference(t1, t2)
    if normalize:
        d /= len(t2.internal_edges()) + len(t1.internal_edges())
    return d


def covariance_to_distance(c):
    return c.diag() + c.diag().unsqueeze(1) - 2 * c


def wrf_distance(t1: dendropy.Tree | str, t2: dendropy.Tree | str):
    """unroot, then calculate weighted Robinson-Foulds dist"""
    from dendropy.calculate import treecompare
    if isinstance(t1, str):
        t1 = make_tree(t1)
    if isinstance(t2, str):
        t2 = make_tree(t2, namespace=t1.taxon_namespace)

    if t1.is_rooted:
        t1 = t1.clone()
        t1.deroot()
    if t2.is_rooted:
        t2 = t2.clone()
        t2.deroot()
    return treecompare.weighted_robinson_foulds_distance(t1, t2)


class ModelBasedDist:
    def __init__(self) -> None:
        from Bio.Phylo.TreeConstruction import (
            DistanceCalculator,
            DistanceTreeConstructor,
        )

        self.tc = DistanceTreeConstructor()
        self.dc = DistanceCalculator("pam70")

    def __call__(
        self, msa: AlignIO.MultipleSeqAlignment | List, as_string: bool = False
    ):
        if isinstance(msa, AlignIO.MultipleSeqAlignment):
            msa = [msa]
        output = []
        for a in msa:
            dmat = self.dc.get_distance(a)
            n = len(a)

            tree = self.tc.nj(dmat)
            tree = dendropy.Tree.get_from_string(
                tree.format("newick"), schema="newick")
            if as_string:
                tree = tree.as_string(
                    "newick", suppress_internal_node_labels=True)
            output.append(tree)
        return tree if len(tree) > 1 else tree[0]
