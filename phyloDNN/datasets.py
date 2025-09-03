import os
import re
import tempfile
from ctypes import alignment
from fileinput import filename
from functools import cached_property, lru_cache
from glob import glob
from hashlib import new
from itertools import *
from operator import *
from pathlib import Path
from typing import Callable, Iterable, List, Union

import numpy as np
import torch
from Bio import AlignIO
from dendropy import TaxonNamespace, Tree
from joblib import Parallel, delayed, dump, load
from scipy.spatial.distance import squareform
# from sklearn.utils import shuffle
from torch import Tensor
from torch_geometric import data, is_debug_enabled
# from torch_sparse import SparseTensor, cat

from .seq_utils import alignment_to_torch, tree2dist, get_sub_alignment, make_tree, squareform, tree_cov, tree_dist
# from .utils import

__package__ = 'phyloDNN'

# import torch_geometric


rx = re.compile('data_(\d+).pt')

LETTERS = ('-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
# there are 17 degenerate IUPAC DNA codes, 21 AA, + 1 for zero padding (value=0) to match array dim
# in ungapped version, we ignore gaps, but still keep this embedding dim

N_STATES = len(LETTERS)+1
let2int = dict(zip(LETTERS, range(1, 1+N_STATES)))

letter_to_int = np.vectorize(lambda l: let2int[l], otypes=[np.uint8])

# parse_phylip = partial(AlignIO.parse, format='phylip')

taxon_id = re.compile('^(\d+)_')


def remove_invariant(x: torch.Tensor):
    # TODO: fix
    _, u = x.unique(dim=-1, return_counts=True)
    return x[:, u>1]


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def get_id(s: str):
    return int(taxon_id.findall(s)[0])


def alignment_to_datum(a: AlignIO.MultipleSeqAlignment,
                       tree: Union[Tree, str],
                       form: str = "distance"):
    # TODO: add "covariance" option to compute covariance (shared distance to the root) instead of patristic distances
    tree = make_tree(tree)

    ntaxa = len(tree)
    # dists = tree2dist(tree, form)

    x = np.array(a)
    try:
        x = letter_to_int(x)
    except:
        pass

    fully_connected = torch.tensor(
        list(permutations(range(ntaxa), 2)),
        dtype=torch.long).T  # relabel taxa to consecutive integers

    d = data.Data(
        x=torch.from_numpy(x),
        edge_index=fully_connected,
        y=y
    )
    return d


class DataTransform:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, data: data.Data):
        if "Ne" in self.__dict__:
            s = self.__dict__["Ne"]
            if data.y is not None:
                data.y = data.y/s
        if "padding" in self.__dict__:
            p = self.__dict__["padding"]
            h, w = data.x.shape
            d = (p-w)
            l = r = d//2
            if d % 2 and l > 0:
                l += 1
            data.x = torch.pad(data.x, (l, r), "constant", 0)
        if "min_taxa" in self.__dict__ and "max_taxa" in self.__dict__:
            min_taxa = self.__dict__["min_taxa"]
            max_taxa = self.__dict__["max_taxa"]
            try:
                ntaxa = data.x.shape[0]
            except:
                print(data)
                raise
            if min_taxa <= ntaxa:
                n = np.random.randint(
                    min_taxa, min(max_taxa, ntaxa)+1)
                taxa = np.random.choice(ntaxa, size=n, replace=False)
                data.y = data.y[taxa, ...][..., taxa]
                data.x = data.x[taxa, ...]
                data.edge_index = torch.tensor(
                    list(combinations(range(n), 2)),
                    dtype=int).T
        return data

    @staticmethod
    def log(data: data.Data) -> data.Data:
        '''log transform distances.
        maintains zeros on diagonal: log(y+1)'''
        if data.y is not None:
            if data.y[0, 0] == 0:
                data.y += 1
            data.y = torch.log(data.y)

        return data

    @staticmethod
    def scale(data: data.Data, s=1e7) -> data.Data:
        if data.y is not None:
            data.y = data.y/s
        return data

    @staticmethod
    def pad(data: data.Data, p):
        h, w = data.x.shape
        d = (p-w)
        l = r = d//2
        if d % 2 and l > 0:
            l += 1
        data.x = torch.pad(data.x, (l, r), "constant", 0)
        return data

    @staticmethod
    def mkl(data: data.Data):
        data.x = data.x.float().to_mkldnn()
        return data

    @staticmethod
    def matrix(data: data.Data):
        '''turn edge features into dense dist mat'''
        if data.y is not None and data.y.dim() == 1:
            data.y = array_to_mat(data.edge_index, data.y)
        return data

    @staticmethod
    def subsample(data: data.Data, min_length=1e4, max_length=5e5):
        '''subsample x matrix from data object.
        Can't subsample taxa yet, because this would require recomputing the distance matrix.'''
        if data.x is not None:
            _, l = data.x.shape
            max_start = max(1, l-min_length)
            data.x = get_sub_alignment(
                data.x, l, max_start, min_length, max_length)
        return data

    def subsampler(data: data.Data,
                   min_taxa, max_taxa,
                   max_length=None, num_genes=None, gene_frac=None) -> data.Data:
        '''subsample taxa and genes from data object of dims:
            ntaxa x ngenes x seq_length.
            min_taxa is strict.
            TODO: add outgroup option for covariance format'''
        if data.x is not None:
            ntaxa = data.x.shape[0]
            if min_taxa > ntaxa:
                return
            n = np.random.randint(
                min_taxa, min(max_taxa, ntaxa)+1)
            taxa = np.random.choice(ntaxa, size=n, replace=False)
            if max_length is not None:
                data.x = data.x[taxa, ...][..., :max_length]
            data.y = data.y[taxa, :][:, taxa]
            data.edge_index = torch.tensor(
                list(combinations(range(n), 2)),
                dtype=int).T
            # print('subsample shape', data.x.shape)
            if len(data.x.shape) <= 3:
                ngenes = data.x.shape[1]
                # TODO: add a check to ignore
                if ngenes > 1:
                    if num_genes is not None:
                        genes = np.random.choice(
                            ngenes, size=num_genes, replace=False)
                    else:
                        genes = np.random.binomial(1, gene_frac, ngenes)
                    data.x = data.x[:, genes, :]
            else:
                r, c = data.x.shape
                data.x = data.x.reshape(r, 1, c)
        # print('subsample shape (after)', data.x.shape)
        return data

    @staticmethod
    def make_subsampler(
            gene_frac: float = None,
            num_genes: int = None,
            min_taxa: int = 4,
            max_taxa: int = 4,
            max_length: int = None) -> Callable[[data.Data], data.Data]:
        """could also use Beta (a,b) prior for per-species Bernoulli drop param p
        Args:
            gene_frac (_type_, optional): _description_. Defaults to None.
            num_genes (_type_, optional): _description_. Defaults to None.
            min_taxa (int, optional): inclusive. Defaults to 4.
            max_taxa (int, optional): inclusive. Defaults to 4.
            max_length (int, optional): deterministic: sequences will be truncated to this length. Defaults to 2000.

        Returns:
            function: a function that transforms a Data object
        """        ''''''
        def subsampler(data: data.Data) -> data.Data:
            '''subsample taxa and genes from data object of dims:
                ntaxa x ngenes x seq_length.
                min_taxa is strict.
                TODO: add outgroup option for covariance format'''
            if data.x is not None:
                ntaxa = data.x.shape[0]
                if min_taxa > ntaxa:
                    return
                n = np.random.randint(
                    min_taxa, min(max_taxa, ntaxa)+1)
                taxa = np.random.choice(ntaxa, size=n, replace=False)
                if max_length is not None:
                    data.x = data.x[taxa, ...][..., :max_length]
                data.y = data.y[taxa, :][:, taxa]
                data.edge_index = torch.tensor(
                    list(combinations(range(n), 2)),
                    dtype=int).T
                # print('subsample shape', data.x.shape)
                if len(data.x.shape) <= 3:
                    ngenes = data.x.shape[1]
                    # TODO: add a check to ignore
                    if ngenes > 1:
                        if num_genes is not None:
                            genes = np.random.choice(
                                ngenes, size=num_genes, replace=False)
                        else:
                            genes = np.random.binomial(1, gene_frac, ngenes)
                        data.x = data.x[:, genes, :]
                else:
                    r, c = data.x.shape
                    data.x = data.x.reshape(r, 1, c)
            # print('subsample shape (after)', data.x.shape)
            return data
        return subsampler


class Batch(data.Batch):
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

    @staticmethod
    def from_data_list(data_points: 'list[data.Data]',
                       follow_batch=[],
                       concat: bool = False) -> Union[data.Batch, None]:
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.

        Args:
            data_points (list[data.Data]): Data objects to batch.
            follow_batch (list, optional): Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`. Defaults to [].
            concat (bool, optional): if True, collapses batch of n Data objects to single graph x with n disconnected subgraphs, and a block diagonal target y.
                Defaults to False.

        Returns:
            Union[data.Batch, None]: The assignment vector :obj:`batch` is created on the fly.
        .
        """
        data_list = list(filter(None, data_points))
        keys = [set(data.keys) for data in data_list]
        # if 0 == len(keys):
        #     raise ValueError(f"empty batch: {data_points}")
        keys = list(set.union(*keys))
        assert 'batch' not in keys
        batch = Batch()

        data_list = [d for d in data_list if d.keys]  # drop empty data objects
        if data_list == []:
            return None
        batch_size = len(data_list)

        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                # elif isinstance(item, SparseTensor):
                #     value = item.storage.value()
                #     if value is not None and value.dtype != torch.bool:
                #         if not isinstance(cum, int) or cum != 0:
                #             value = value + cum
                #         item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Treat 0-dimensional tensors as 1-dimensional.
                if isinstance(item, Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)

                batch[key].append(item)

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                cat_dims[key] = cat_dim
                if isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device
                # elif isinstance(item, SparseTensor):
                #     size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                #     device = item.device()

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size, ), i, dtype=torch.long,
                                       device=device))

            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            if isinstance(item, Tensor):
                if key == 'x':  # hardcode for now
                    if not all_equal(it.shape[-1] for it in items):
                        pad_dim = max(it.shape[-1] for it in items)
                        for i in range(len(items)):
                            *item_dims, last_dim = items[i].shape
                            new_item = torch.zeros((*item_dims, pad_dim))
                            new_item[:, :, :last_dim] = items[i]
                            items[i] = new_item

                    # NOTE: this code is for the old [taxa,sequence] format, not [taxa,gene,sequence]
                    # items = [it for item in items for it in item]
                    # items = (torch.nn.utils.rnn
                    #          .pad_sequence(items, batch_first=True)
                    #          .transpose(0, ref_data.__cat_dim__(key, item)))
                    if concat:
                        batch[key] = torch.cat(items)
                    else:
                        batch[key] = items
                elif key == 'y':
                    if concat:  # breaking change!
                        batch[key] = torch.block_diag(*items)
                    else:
                        batch[key] = items
                else:
                    batch[key] = torch.cat(
                        items, cat_dim)
            # elif isinstance(item, SparseTensor):
            #     batch[key] = cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if is_debug_enabled():
            batch.debug()
        batch.batch_size = batch_size
        return batch.contiguous()


# torch_geometric.data.Batch = Batch


class Collater(object):
    def __init__(self, follow_batch, concat):
        self.follow_batch = follow_batch
        self.concat = concat

    def collate(self, batch):
        return Batch.from_data_list(batch, self.follow_batch, self.concat)

    def __call__(self, batch):
        return self.collate(batch)

    def __repr__(self) -> str:
        return (f'Collater(follow_batch={self.follow_batch}, concat={self.concat})')



class Dataset(data.Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def processed_file_names(self):
        return glob(os.path.join(self.processed_dir, 'data_*.pt'))

    @lru_cache
    def len(self):
        r"""This is called by __len__ in parent"""
        return self.n_raw*self.n_genes*self.n_alignments

    @cached_property
    def n_raw(self):
        return len(self.raw_file_names)

    def process(self):
        from functools import partial

        raw_dir, dirs, _ = next(os.walk(self.raw_dir))
        raw_dir = Path(raw_dir)
        n = len(dirs)
        # self.parse_alignments(raw_dir, dirs[0])
        params = {
            k: v for k, v in self.__dict__.items() if type(v) in (int,
                                                                  str, float, bool)
        }
        torch.save(params, Path(self.root) / 'config.torch')
        with Parallel(n_jobs=self.njobs) as parallel:
            parallel(
                delayed(self.parse_alignments)(raw_dir, d) for i, d in enumerate(dirs)
            )

    def parse_alignments(*args):
        return

    @cached_property
    def raw_file_names(self):
        _, dirs, _ = next(os.walk(self.raw_dir))
        return dirs

    def make_processed_file_path(self, dirname: Union[int, str]):
        num_processed_files = len(self.processed_file_names)
        if 0 < num_processed_files < 1000:
            if isinstance(dirname, int):
                dirname = f'{dirname:03d}'
        elif 1000 <= num_processed_files < 10000:
            dirname = f'{dirname:04d}'

        return os.path.join(
            self.processed_dir, f'data_{dirname}.pt'
        )

    def get(self, idx):
        try:
            d = torch.load(os.path.join(
                self.processed_dir, 'data_{}.pt'.format(idx)))
        except FileNotFoundError:
            d = data.Data()
        return d

    @staticmethod
    def covariance_matrix(tree: Tree, ns=None):
        '''Calculate phylogenetic covariance matrix.
        Roots at midpoint of longest edge if tree is unrooted.'''
        if ns is None:
            ns = tree.taxon_namespace
        if tree.is_unrooted:
            tree.reroot_at_midpoint()
        tree.calc_node_root_distances()
        d = [
            tree.mrca(
                taxa=taxa).root_distance for taxa in combinations_with_replacement(ns, 2)
        ]
        n = len(ns)
        mat = np.zeros((n, n))
        mat[np.triu_indices(n)] = d
        mat += mat.T-np.diag(mat.diagonal())  # hack to make symmetric matrix
        return torch.tensor(mat)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 follow_batch: List = [],
                 concat: bool = False,
                 **kwargs):
        """Generate Data objects.

        Args:
            dataset (Dataset): type of dataset (LocusDataset for concatenated seqs + species tree, GeneDataset for indiv genes)
            batch_size (int, optional): number of distinct alignments (from different species trees) in a batch. Defaults to 1.
            shuffle (bool, optional): shuffle order of traversing alignments. Defaults to False.
            follow_batch (list, optional): _description_. Defaults to [].
            concat (bool, optional): merge each batch into block-diagonal x and y matrices.  If False, return list of x,y matrices. Defaults to False.
        """
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, concat) if batch_size else None,
            **kwargs)


class SequenceDataset(Dataset):
    """generates (align,species_tree) pairs for subsamples of concatenated alignments"""
    # could even set min_taxa to 2, since the loss fn includes the error in estimating (generation time) distances
    min_taxa = 4

    def __init__(self, root,
                 n_sub_samples,
                 sub_align=True,
                 min_length=1e4,
                 max_length=1e6,
                 max_taxa=9,
                 seed=1234,
                 njobs=4,
                 overwrite=False,
                 transform=None,
                 pre_transform=None):
        self.sub_align = sub_align
        self.n_sub_samples = n_sub_samples
        self.min_length = min_length
        self.max_length = max_length
        self.max_taxa = max_taxa+1
        self.seed = seed
        self.njobs = njobs
        self.overwrite = overwrite

        if self.overwrite:
            for filepath in glob(os.path.join(root, 'processed', '*')):
                os.remove(filepath)
        # must call after overwriting
        super().__init__(root, transform, pre_transform)

    @cached_property
    def raw_file_names(self):
        return glob(os.path.join(self.raw_dir, '*_concat.phy'))

    @property
    def processed_file_names(self):
        return glob(os.path.join(self.processed_dir, 'data_*.pt'))

    def process(self):
        import toytree
        '''process files, keeping old files if they exist'''
        print('creating trainset...')
        written = set()
        if os.listdir(self.processed_dir):
            written = set(int(rx.findall(fn)[0])
                          for fn in self.processed_file_names)

        with Parallel(n_jobs=self.njobs) as parallel, tempfile.TemporaryDirectory() as tmpdir:
            for i, raw_path in enumerate(self.raw_paths):

                file_nums = [
                    x for x in
                    (self.n_sub_samples*i + j for j in range(self.n_sub_samples)) if x not in written
                ]
                if file_nums == []:
                    continue
                else:
                    print(f'generating {len(file_nums)} subsamples')
                    # Read data from `raw_path`.
                align = AlignIO.read(raw_path, 'phylip')
                s_tree = toytree.tree(raw_path.replace(
                    '_concat.phy', '.species_tree'))
                l = align.get_alignment_length()
                n = len(align)
                max_start = max(1, l-self.min_length)
                min_drop = max(0, n-self.max_taxa)

                align_fn = os.path.join(tmpdir, 'align_memmap')
                dump(align, align_fn)
                align = load(align_fn, mmap_mode='r')

                parallel(
                    delayed(self.generate_datum)(
                        align, s_tree, l, n, max_start, min_drop, f_no) for f_no in file_nums
                )

    def generate_datum(self, a, s_tree, l, n, max_start, min_drop, i):
        # TODO: put most of the subsampling in a separate transform function, do it on the fly.
        if self.min_taxa == self.max_taxa:
            num_taxa = self.min_taxa
        else:
            num_taxa = np.random.randint(min_drop,
                                         n-self.min_taxa)
        drop_taxa = list(map(str,
                             np.random.choice(
                                 n,
                                 size=num_taxa,
                                 replace=False)
                             ))

        tree = s_tree.drop_tips(drop_taxa)
        if self.sub_align:
            a = get_sub_alignment(a, l,
                                  max_start,
                                  self.min_length,
                                  self.max_length)
        a = AlignIO.MultipleSeqAlignment(
            [seq for seq in a if seq.name not in drop_taxa])
        a.sort()

        d = alignment_to_datum(a, tree)

        if self.pre_transform is not None:
            d = self.pre_transform(d)

        outpath = self.make_processed_file_path(i)
        torch.save(d, outpath)

    def len(self):
        r"""This is called by __len__ in parent"""
        if self.overwrite:
            return len(self.raw_file_names) * self.n_sub_samples
        else:
            return len(self.processed_file_names)

    def get(self, idx):
        filepath = os.path.join(
            self.processed_dir, 'data_{}.pt'.format(idx))
        try:
            d = torch.load(filepath)
#            if self.transform is not None:
#                data = self.transform(data)
        except FileNotFoundError:
            print(f"couldn't find {filepath}")
            d = data.Data()
        return d


class LocusDataset(Dataset):
    """This class spits out a concatenated alignment of U([min_genes,max_genes]) along with their corresponding *species* trees"""

    def __init__(self, root: str,
                 n_genes: int = 500,
                 max_genes: int = None,
                 min_genes: int = None,
                 n_alignments: int = 1,
                 seed: int = 1234,
                 njobs: int = 4,
                 matrix_type='distance',
                 no_gaps: bool = True,
                 overwrite=False,
                 max_length: float = 4e3,
                 transform=None,
                 use_tree=False,
                 pre_transform=None):
        '''njobs is only used when regenerating a dataset
            n_genes is total # of genes per dataset,
            max_length is maximum length of each gene, '''
        if matrix_type not in ('distance', 'covariance'):
            raise ValueError("""matrix must be 'distance','covariance'""")
        if max_genes is None or max_genes > n_genes:
            max_genes = n_genes
        if min_genes is None:
            min_genes = max_genes
        self. matrix_type = matrix_type
        self.no_gaps = no_gaps
        self.max_genes = max_genes
        self.min_genes = min_genes
        self.n_genes = n_genes
        self.seed = seed
        self.max_length = int(max_length)
        self.njobs = njobs
        self.overwrite = overwrite
        self.use_tree = use_tree
        self.n_alignments = n_alignments

        if self.overwrite:
            for filepath in glob(os.path.join(root, 'processed/*')):
                os.remove(filepath)

        super().__init__(root, transform, pre_transform)
        dataset_config = torch.load(Path(root) / 'config.torch')
        if dataset_config['matrix_type'] != self.matrix_type:
            raise ValueError(
                f"matrix type {self.matrix_type} does not match generated dataset type {dataset_config['matrix_type']}.")
        if dataset_config['no_gaps'] != self.no_gaps:
            raise ValueError(
                f"gap type {self.no_gaps} does not match generated dataset type {dataset_config['no_gaps']}.")

    @property
    def processed_file_names(self):
        return glob(os.path.join(self.processed_dir, 'data_*.pt'))

    @lru_cache
    def len(self):
        r"""This is called by __len__ in parent"""
        return self.n_raw

    def process(self):
        from functools import partial

        raw_dir, dirs, _ = next(os.walk(self.raw_dir))
        raw_dir = Path(raw_dir)
        n = len(dirs)
        self.parse_alignments(raw_dir, dirs[0])
        params = {
            k: v for k, v in self.__dict__.items() if type(v) in (int,
                                                                  str, float, bool)
        }

        torch.save(params, Path(self.root) / 'config.torch')
        with Parallel(n_jobs=self.njobs) as parallel:
            parallel(
                delayed(self.parse_alignments)(raw_dir, d) for i, d in enumerate(dirs)
            )

    def parse_alignments(self, parentname: Path, dirname, index=0):
        from collections import defaultdict
        from glob import glob
        from os import path

        index *= self.n_genes
        # sort 0,1,2,...
        species_tree = (Tree
                        .get(path=path.join(parentname, dirname, 's_tree.trees'),
                             schema='newick')
                        )
        gene_length = self.max_length

        # rather than using the distances() method, we must ensure the taxon naming is consistent.
        ns = sorted(species_tree.taxon_namespace, key=lambda k: k.label)
        ntaxa = len(ns)
        taxon_mapper = {k.label.replace(
            ' ', '_'): v for v, k in enumerate(ns)}
        # TODO: handle multi-copy orthologs

        # def taxon_mapper(s): return int(s.split('_')[0])-1

        edge_index = torch.tensor(
            list(combinations(range(ntaxa), 2)),
            dtype=torch.long).T  # without self-loops
        if self.matrix_type == 'distance':
            pdm = species_tree.phylogenetic_distance_matrix()
            dmat = squareform([pdm.distance(*k)
                              for k in combinations(ns, 2)])
            dmat = torch.tensor(dmat)
        elif self.matrix_type == 'covariance':
            dmat = self.covariance_matrix(species_tree)

        fully_connected = torch.tensor(
            list(combinations(range(ntaxa), 2)),
            dtype=torch.long).T  # without self-loops

        alignment_filenames = glob(path.join(parentname, dirname, '*.phy'))
        if self.use_tree:
            edge_attr = torch.tensor(
                squareform(
                    [species_tree.distance(*k, is_weighted_edge_distances=False)
                     for k in combinations(ns, 2)]
                )
            )
        data_list = []
        for align_filename in alignment_filenames:
            # align_filename = parentname/dirname / (fn+'_TRUE.phy')
            alignments = alignment_to_torch(
                align_filename, ntaxa=ntaxa, no_gaps=self.no_gaps, taxon_mapper=taxon_mapper)
            data_list.append(alignments)

        d = data.Data(
            edge_index=edge_index,
            y=dmat
        )

        outpath = self.make_processed_file_path(dirname)

        torch.save((d, data_list), outpath)

    def get(self, idx: int) -> data.Data:
        """each dataset .pt file has a ngenes x nreplicate list of dicts ntaxa  x nsites
            data dictionaries.
        If self.n_alignments==1, randomly choose a gene from a particular dataset:
        idx 175 -> ds 18, replicate 5"""
        # TODO: need option to get ALL genes, possibly randomly select alignment
        file_no = idx+1
        filepath = self.make_processed_file_path(file_no)
        try:
            d, all_genes = torch.load(filepath)
            total_genes = len(all_genes)
            # TODO: handle all this with a Transform in the DataLoader constructor
            if total_genes >= self.min_genes:
                n_genes = np.random.randint(
                    self.min_genes, min(self.max_genes, total_genes)+1)
                gene_ix = np.random.choice(
                    total_genes, size=n_genes, replace=False)
                align_ix = np.random.choice(
                    self.n_alignments, size=n_genes, replace=True)
                all_genes = [all_genes[i][j]
                             for i, j in zip(gene_ix, align_ix)]
                d.x = torch.cat(all_genes, -1)
                if len(d.x.shape) == 2:  # hack
                    r, c = d.x.shape
                    d.x = d.x.reshape(r, 1, c)
            else:
                print(f"not enough genes: {total_genes}<{self.min_genes} ")
                # *dims, _ = all_genes[0][0].shape
                # d.x = torch.empty((*dims, 0))
                d = data.Data()
        except FileNotFoundError:
            print(filepath, 'not found')
        except IndexError as e:
            print(e, filepath, idx, d.shape)

        return d


class GeneDataset(Dataset):
    """This class spits out a collection of alignments (list of tensors),
    along with their gene trees (single block matrix).
    NOTE: if root has a dataset already, and overwrite is False,
    will not regenerate (even if some param values are different).
    NOTE: root directory raw subdir must contain ONLY directories that will be processed into datapoints.
    """

    def __init__(self,
                 root: Path,
                 n_genes: int = 10,
                 n_alignments: int = 1,
                 max_length: float = 2e3,
                 seed=1234,
                 njobs: int = 4,
                 overwrite=False,
                 transform=None,
                 no_gaps=True,
                 use_tree=False,
                 return_seq_paths=False,
                 with_likelihood=False,
                 matrix_type='distance',
                 pre_transform=None):
        """
        Args:
            root (Path): _description_
            n_genes (int, optional): how many genes are simulated per species tree.
                Can have any number of alignments per gene. Defaults to 10.
            n_alignments (int, optional): how many alignments are simulated per gene tree.
                Can have any number of alignments per gene. If set to 1, will randomly choose a different alignment at every query.
                Defaults to 1.
            max_length (_type_, optional): _description_. Defaults to 2e3.
            seed (int, optional): _description_. Defaults to 1234.
            njobs (int, optional): njobs is only used when regenerating a dataset
                Defaults to 4.
            overwrite (bool, optional): remove all files in the processed/ directory
                and regenerate in torch data. Defaults to False.
            transform (_type_, optional): _description_. Defaults to None.
            no_gaps (bool, optional): If no_gaps is true, will remove all gaps in the sequence - makes inference much harder!
                Defaults to True.  Seqs wil be padded with trailing 0's to match the ength of the longest seq in the alignment.
            use_tree (bool, optional): If use_tree is true, will return the target graph (unweighted).
                Use for branch length only inference.
                Defaults to False.
            matrix_type (str, optional): one of 'covariance' or 'distance'. Defaults to 'distance'.
            pre_transform (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if matrix_type not in ('distance', 'covariance'):
            raise ValueError("""matrix must be 'distance', 'covariance'""")
        self. matrix_type = matrix_type
        self.n_genes = n_genes
        self.n_alignments = n_alignments
        self.max_length = int(max_length)
        self.seed = seed
        self.njobs = njobs
        self.overwrite = overwrite
        self.no_gaps = no_gaps
        self.use_tree = use_tree
        self.return_seq_paths = return_seq_paths
        self.with_likelihood = with_likelihood

        if self.overwrite:
            for filepath in Path(root).glob('processed/*'):
                os.remove(filepath)

        super().__init__(str(root), transform, pre_transform)
        config_path = Path(self.root) / 'config.torch'
        if config_path.exists():
            dataset_config = torch.load(config_path)
            if dataset_config['matrix_type'] != self.matrix_type:
                raise ValueError(
                    f"matrix type {self.matrix_type} does not match generated dataset type {dataset_config['matrix_type']}.")
            if dataset_config['no_gaps'] != self.no_gaps:
                raise ValueError(
                    f"gap type {self.no_gaps} does not match generated dataset type {dataset_config['no_gaps']}.")

    def process(self):
        from functools import partial

        raw_dir = Path(self.raw_dir)
        params = {
            k: v for k, v in self.__dict__.items() if type(v) in (int,
                                                                  str, float, bool)
        }
        torch.save(params, Path(self.root) / 'config.torch')
        with Parallel(n_jobs=self.njobs) as parallel:
            parallel(
                delayed(self.parse_alignments)(d) for d in raw_dir.iterdir() if d.is_dir()
            )

    @ property
    def processed_file_names(self):
        return glob(os.path.join(self.processed_dir, 'data_*.pt'))

    @ lru_cache
    def len(self):
        r"""This is called by __len__ in parent"""
        return self.n_raw*self.n_genes*self.n_alignments

    def parse_alignments(
        self,
        # parentname: Union[Path, str],
        dirname: Union[Path, str],
    ):
        """parse all alignments for the genes in dirname.

        Args:
            parentname (Union[Path, str]): path to parent directory (e.g. raw file dir)
            dirname (Union[Path, str]): species tree dir (001, 002, etc)

        TODO: indexing, reading gene trees, etc.
        decide whether to add dummy dimension here or elsewhere in pipeline.
        TODO: implement distance/covariance methods w/outgroup"""
        # TODO: rewrite so this is more like the LocusDataset, with x=[replicate,gene,maxlength], y=block diagonal gene tree dmats, edge_index=separate graphs.  get method?
        # TODO: must fix indexing so that we match n_species_trees x n_genes x n_alignments.
        import pandas as pd
        dirname = Path(dirname)
        gene_tree_filename = dirname/'trees.txt'
        if gene_tree_filename.exists():
            try:
                gene_trees = pd.read_csv(gene_tree_filename,
                                         skiprows=2,
                                         sep='\t',
                                         usecols=['FILE', 'TREE STRING'],
                                         index_col='FILE')
            except ValueError:
                gene_trees = pd.read_csv(gene_tree_filename,
                                         skiprows=4,
                                         sep='\t',
                                         usecols=['FILE', 'TREE STRING'],
                                         index_col='FILE')
        else:
            gene_tree_filename = dirname/'gene_trees.nw'
            if gene_tree_filename.exists():
                with open(gene_tree_filename, 'r') as f:
                    gene_trees = [(f'dataset_{i:03}', tree)
                                  for i, tree in enumerate(f, 1)]
                    gene_trees = pd.DataFrame.from_records(
                        gene_trees,
                        columns=['FILE', 'TREE STRING'],
                        index='FILE')
            else:
                gene_tree_filename = dirname/'trees.nw'
                if gene_tree_filename.exists():
                    with gene_tree_filename.open('r') as f:
                        trees = f.readlines()
                    gene_trees = (pd
                                  .DataFrame(
                                      {'FILE': map(str, range(len(trees))), 'TREE STRING': trees})
                                  .set_index('FILE'))
                else:
                    gene_trees = pd.DataFrame.from_records(
                        [(fn.stem, open(fn, 'r').read())
                         for fn in dirname.glob('*.nwk')],
                        columns=['FILE', 'TREE STRING'],
                        index='FILE')
        # sort 0,1,2,...

        gene_length = self.max_length

        data_list = []

        if self.with_likelihood:
            likelihoods = pd.read_csv(
                dirname/'log_likelihoods.txt', header=None)

        for gene_no, (fn, gtree) in enumerate(gene_trees.itertuples()):
            gtree = make_tree(gtree)
            gtree.taxon_namespace.sort()
            ns = gtree.taxon_namespace
            ntaxa = len(ns)
            taxon_mapper = {k.label.replace(
                ' ', '_'): v for v, k in enumerate(ns)}
            # TODO: handle multi-copy orthologs

            # def taxon_mapper(s): return int(s.split('_')[0])-1

            edge_index = torch.tensor(
                list(combinations(range(ntaxa), 2)),
                dtype=torch.long).T  # without self-loops
            if self.matrix_type == 'distance':
                pdm = gtree.phylogenetic_distance_matrix()
                dmat = torch.tensor(
                    squareform([pdm.distance(*k)
                                for k in combinations(ns, 2)]))
            elif self.matrix_type == 'covariance':
                dmat = self.covariance_matrix(gtree)
            try:
                align_filename = dirname / (fn+'_TRUE.phy')
                alignments = alignment_to_torch(
                    align_filename, ntaxa=ntaxa, no_gaps=self.no_gaps, taxon_mapper=taxon_mapper)
            except:
                align_filename = dirname / (fn+'.phy')
                alignments = alignment_to_torch(
                    align_filename, ntaxa=ntaxa, no_gaps=self.no_gaps, taxon_mapper=taxon_mapper)

            # alignment_dict = {}
            # for i, rep in enumerate(aligns):
            #     taxa = []
            #     alignment_array = np.zeros(
            #         (ntaxa, gene_length), dtype=np.uint8)
            #     for seq_record in rep:
            #         tid = taxon_mapper[seq_record.id]
            #         if self.no_gaps:
            #             seq = seq_record.seq.ungap()
            #         else:
            #             seq = seq_record.seq
            #         alignment_array[tid, :min(gene_length, len(seq))] = letter_to_int(seq)[
            #             : gene_length]
            #         taxa.append(tid)
            #     alignment_dict[i] = alignment_array
            # seq_array = np.zeros(
            #     shape=(len(alignment_dict), ntaxa, 1, gene_length),
            #     dtype=np.uint8
            # )  # null dim for consistency

            # for k in sorted(alignment_dict):
            #     seq = alignment_dict[k]
            #     seq_array[k, :, 0, :] = seq

            d = data.Data(
                x=torch.stack(alignments, dim=0),
                edge_index=edge_index,
                y=dmat
            )
            if self.with_likelihood:
                d.likelihood = likelihoods[gene_no]

            if self.use_tree:
                d.edge_attr = torch.tensor(
                    squareform(
                        [gtree.distance(*k, is_weighted_edge_distances=False)
                         for k in combinations(ns, 2)]
                    )
                )
            data_list.append(d)

        outpath = self.make_processed_file_path(dirname.stem)
        torch.save(data_list, outpath)

    def get(self, idx: int):
        """each dataset .pt file has a ngenes-long list
            of Data objects with nreplicate x ntaxa x 1 x nsites
            data matrices.
        If self.n_alignments==1, randomly choose a gene from a particular dataset:
        idx 175 -> ds 18, replicate 5"""
        # TODO: need option to get ALL genes, possibly randomly select alignment
        file_no = idx//(self.n_genes*self.n_alignments)+1
        gene_no = idx % (self.n_genes*self.n_alignments) // self.n_alignments
        align_no = idx % (
            self.n_genes*self.n_alignments) % self.n_alignments
        filepath = self.make_processed_file_path(file_no)

        try:
            d = torch.load(filepath)
            d = d[gene_no]
            if self.n_alignments == 1:
                align_no = np.random.choice(d.x.shape[0])
            d.x = d.x[align_no, ...]
            if len(d.x.shape) == 2:  # hack
                r, c = d.x.shape
                d.x = d.x.reshape(r, 1, c)
        except FileNotFoundError:
            print(filepath, 'not found')
            d = data.Data()
        except IndexError as e:
            print(e, filepath, idx, gene_no, align_no, d.shape)
        # if return_seq_paths:
        #     d.seq_path=self.raw_dir/
        return d


def make_data_loader(dataset: Dataset,
                     seqdir: Union[Path, str],
                     batch_size: int = 12,
                     num_workers: int = 2,
                     overwrite=False,
                     multigpu=False,
                     timeout=40,
                     n_genes=10,
                     transforms=None,
                     shuffle=True,
                     concat: bool = False,
                     pin_memory=False,
                     **kwargs):
    """hack to get around dataloader issues at end of epoch.
    dataset and args must be in global scope.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load. Defaults to 12.
        seqdir (_type_): location of directory containing raw/ and processed/ subdirectories.
        num_workers (int, optional): _description_. Defaults to 2.
        overwrite (bool, optional): delete contents of processed/ directory and regenerate. Defaults to False.
        multigpu (bool, optional): _description_. Defaults to False.
        transforms (_type_, optional): mappers to pass each Data object through after loading. Can be nondeterministic.
            Defaults to None.
        concat (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    ds = dataset(
        root=seqdir,
        n_genes=n_genes,
        overwrite=overwrite,
        transform=transforms,
        **kwargs)

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        timeout=timeout,
        concat=concat,
        pin_memory=pin_memory,
        drop_last=False,)
    if multigpu:
        i1 = i2 = iter(data_loader)
        data_loader = zip(i1, i2)
    return data_loader

################## IPCOAL #################


class BirthDeathSimulator(torch.utils.data.IterableDataset):
    def __init__(self, td,
                 ntrees=100,
                 nloci=100,
                 ntaxa=7,
                 birth=1,
                 death=.5,
                 njobs=4,
                 transform=None,
                 max_height=40,
                 overwrite=False,
                 infer=False):
        '''initialize BD trees, run ipcoal simulations using <njobs> threads'''
        import shutil

        import ipcoal
        from dendropy.simulate import treesim

        super().__init__()
        self.transform = transform
        self.ntrees = ntrees
        self.nloci = nloci
        self.td = td
        taxa = TaxonNamespace(range(ntaxa))
        self.infer = infer

        def simulate(t):
            tree = treesim.birth_death_tree(birth_rate=birth,
                                            death_rate=death,
                                            num_extant_tips=ntaxa,
                                            taxon_namespace=taxa)
            Ne = 1e5
            l = np.random.uniform(10, max_height)
            for e in tree.edges():
                e.length = 2*l*Ne*e.length+1
            ts = tree.as_string('newick')
            model = ipcoal.Model(
                ts,
                Ne=Ne,
                recomb=0,
                mut=1e-8)
            model.sim_loci(nloci=nloci, nsites=1000)
            os.listdir(td)
            os.mkdir(f'{td}/tree_{t}')
            model.write_loci_to_phylip(
                outdir=td,
                name_prefix=f'{td}/tree_{t}/seqs_',
                quiet=True)

            with open(f'{td}/tree_{t}/gene_trees.nw', 'w') as f:
                f.write('\n'.join(model.df.genealogy))
            tree.write_to_path(f'{td}/tree_{t}/species_tree.nw', 'newick')

            if infer:
                model.infer_gene_trees(inference_method="raxml")
                model.df.inferred_tree.to_csv(
                    f'{td}/tree_{t}/inferred_trees.nw',
                    header=None,
                    index=None)

        dirs = glob(f'{td}/tree_*')

        if len(dirs) == ntrees and not overwrite:
            return

        done = set()

        if overwrite:
            for d in dirs:
                shutil.rmtree(d)
        else:
            rx = re.compile(r'^.*tree_(\d+)$')
            done = set(map(int, (rx.findall(fn)[0] for fn in dirs)))
            Parallel(njobs)(delayed(simulate)(t)
                            for t in range(ntrees) if t not in done)

    @ cached_property
    def __len__(self):
        return self.ntrees*self.nloci

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.ntrees
        else:  # in a worker process
            # split workload
            per_worker = int(
                np.ceil((self.ntrees) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.ntrees)
        return self.generator(iter_start, iter_end)

    def generator(self, iter_start, iter_end):
        for tree_idx in range(iter_start, iter_end):
            parentdir = f'{self.td}/tree_{tree_idx}'

            if self.infer:
                with open(f'{parentdir}/inferred_trees.nw', 'r') as f:
                    inferred_trees = f.readlines()

            with open(f'{parentdir}/gene_trees.nw', 'r') as f:
                for tree_no, tree in enumerate(f):
                    seqfile = f'{parentdir}/seqs_{tree_no}.phy'
                    try:
                        s = AlignIO.read(
                            seqfile,
                            'phylip')
                    except:
                        continue
                    d = alignment_to_datum(s, tree)
                    d.seqfile = seqfile
                    d.parentdir = parentdir
                    d.ytree = tree
                    if self.transform is not None:
                        d = self.transform(d)
                    if self.infer:
                        d.itree = inferred_trees[tree_no]

                    yield d
