import argparse
from math import inf
from pathlib import Path
import joblib
import pandas as pd
from joblib import Parallel, delayed
from phyloDNN.seq_utils import make_tree, rf_distance, wrf_distance
from dendropy.calculate import treecompare
import numpy as np
from scipy import spatial


def compute_distances(a:str, b:str) -> dict:
    """
    Compute distances between two trees.  
    Some trees may not have the same leaf set (iqtree removes identical sequences), 
    so we prune the trees to the same leaf set.
    """
    d = dict()
    a = make_tree(a)
    b = make_tree(b, namespace=a.taxon_namespace)
    ns = b.poll_taxa()
    a.retain_taxa(ns)

    d["wrf"] = wrf_distance(a, b)
    d["rf"] = rf_distance(a, b, normalize=True)
    d["bl_dist"] = treecompare.euclidean_distance(a, b)
    a_dist = np.array(a.phylogenetic_distance_matrix().distances())
    b_dist = np.array(b.phylogenetic_distance_matrix().distances())
    try:
        d["Frobenius"] = spatial.distance.euclidean(a_dist, b_dist)
    except ValueError as e:
        print(a,b,a_dist.shape, b_dist.shape,len(a),len(b))  
        raise e
    d["l_1"] = np.abs(a_dist - b_dist).sum()
    d["l_inf"] = np.abs(a_dist - b_dist).max()
    return d


def get_trees(dirname: Path, prefix: str) -> dict:
    """
    Get all trees in a directory with a given prefix
    """
    inferred_trees = dict(
        (fn.stem.replace(prefix,''), fn.read_text().strip()) for fn in dirname.glob(prefix + "*nj.bionj")
    )
    try:
        inferred_trees[prefix + "_iqtree"] = (
        (dirname / (prefix + ".iqtree.treefile")).read_text().strip()
    )
    except FileNotFoundError:
        pass
    return inferred_trees


def summarize(dir: Path, prefix: str, tree: str) -> dict:
    """
    Compare a reference tree with all inferred tree files in a directory
    """
    pred_trees = get_trees(dir, prefix)
    res = {}
    for tname in pred_trees:
        res[tname] = compute_distances(tree, pred_trees[tname])
    return prefix,res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, help="tree file")
    parser.add_argument("--dir", type=Path, help="input directory")
    parser.add_argument(
        "--out", type=str, default="tree_dists", help="output file prefix"
    )
    args = parser.parse_args()

    with Parallel(n_jobs=16) as parallel:
        with open(args.tree, "r") as f:
            # for line in f.readlines():
            #     summarize(args.dir, *line.split())
            dists = parallel(
                delayed(summarize)(args.dir, *line.split()) for line in f.readlines()
            )
    joblib.dump(dists, args.dir / "tree_dists.joblib")
    dists = pd.DataFrame(
        {
            (tree_name, alg): distances
            for tree_name, record in dists
            for alg, distances in record.items()
        },
    ).T
    if dists.empty:
        raise ValueError("no inferred trees in directory")
    dists.index = dists.index.map(
        # change _HKY.nj to HKY
        lambda s: (s[0], s[1][1:-3] if not 'iqt' in s[1] else 'iqtree'))
    dists.index.set_names(["tree_id", "alg"])

    # index with idx = pd.IndexSlice
    dists.to_pickle(args.dir / f"{args.out}.pd.gz")

# ds={s:pd.read_pickle(s+'_60/tree_dists.pd.gz') for s in ['hky','hky_f','jc','lg_gc','lg_indel']}
# pd.DataFrame( {s:ds[s].rf.groupby('alg').mean() for s in ds})
# r[['jc','hky','hky_f','lg_gc','lg_indel']].loc[['JC','HKY','HKY+G,'iqtree']]
# print((r).to_latex(escape=False))
