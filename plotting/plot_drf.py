from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from phyloDNN import seq_utils as sq
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Plot WRF and RF distances for phylogenetic trees."
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0,
    help="Max IQTREE drf.",
)
parser.add_argument(
    "--dir",
    type=Path,
    default="jc_60",
    help="Directory containing prediction files.",
)
parser.add_argument(
    "--treefile",
    type=Path,
    default="trees_60.tsv",
    help="Directory containing prediction files.",
)
args = parser.parse_args()

data_dir = Path("/N/project/phyloML/data/phyloDNN/test/")
true = pd.read_csv(
    data_dir / args.treefile, sep=r"\s+", header=None, names=["tree"]
).map(sq.make_tree)

include_iqtree = True
try:
    iqtree = pd.read_pickle(data_dir / args.dir / "tree_dists.pd.gz")
    iqtree.index = iqtree.index.map(
        lambda s: (s[0], s[1] if not "iqt" in s[1] else "iqtree")
    )
    iqtree = iqtree.xs("iqtree", level=1)[["wrf", "rf"]].rename(
        columns={"wrf": "wrf_m_iqtree", "rf": "rf_m_iqtree"}
    )
except FileNotFoundError:
    include_iqtree = False

for tree_col in ["trees_nj", "trees_bionj"]:
    predfiles = (data_dir / args.dir).glob("predictions_*.pd.gz")
    preds = []
    for fn in predfiles:
        try:
            preds.append(
                pd.read_pickle(fn)[["filenames", tree_col]]
                .rename(
                    columns={
                        tree_col: fn.stem.replace("predictions_", "m_").replace(
                            ".pd", ""
                        ),
                    }
                )
                .set_index("filenames")
            )
        except Exception as e:
            print(f"{fn} does not have the correct format")
            raise e
    preds = pd.concat(
        preds,
        axis=1,
    )
    models = [c for c in preds.columns if c.startswith("m_")]
    preds = preds.join(true)
    if include_iqtree:
        preds = preds.join(iqtree)

    drop = []
    for m in models:
        # try:
        preds["wrf_" + m] = [
            sq.wrf_distance(
                ttrue,
                tpred if isinstance(tpred, str) else tpred.as_string("newick"),
            )
            for ttrue, tpred in preds[["tree", m]].itertuples(False)
        ]
        preds["rf_" + m] = preds.apply(
            lambda x: sq.rf_distance(
                x["tree"], x[m] if isinstance(x[m], str) else x[m].as_string("newick")
            ),
            axis=1,
        )
        rf_mean, wrf_mean = preds["rf_" + m].mean(), preds["wrf_" + m].mean()
        print(
            "model:",
            m,
            "rf:",
            rf_mean,
            "wrf:",
            wrf_mean,
        )
        if rf_mean > 0.9 or wrf_mean > 100:
            preds.drop(columns=["rf_" + m], inplace=True)
            preds.drop(columns=["wrf_" + m], inplace=True)
            drop.append(m)
    models = [m for m in models if m not in drop]
    print(f"dropping {drop} because of high mean rf or wrf")

    if include_iqtree:
        models.append("m_iqtree")
    preds.to_csv(
        data_dir / args.dir / f"{tree_col}_all_preds.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    # TODO: calculate bl_dist  Frobenius          l_1     l_inf for DNN models
    if args.threshold > 0:
        if include_iqtree:
            preds = preds.query(f"rf_m_iqtree < {args.threshold}")
        else:
            print("no iqtree trees")

    for alg in ["wrf", "rf"]:
        p = preds.melt(
            # id_vars=['filenames', 'tree'],
            value_vars=["_".join([alg, m]) for m in models],
            var_name="model",
            value_name="statistic",
        )
        p.model = p.model.str.replace(f"{alg}_m_", "", regex=False).str.replace(
            "_", " ", regex=False
        )

        # means = p.groupby("model").statistic.mean()
        # drop = means[means > (0.9 if alg == "rf" else 100)].index.tolist()
        # print(f"Dropping {len(drop)} models with mean {alg} > 0.9: {drop}")
        # p = p[~p.model.isin(drop)]
        sns.violinplot(data=p, x="model", y="statistic", cut=0)
        plt.title(r"$d_{wRF}$" if alg == "wrf" else r"$d_{RF}$")
        # plt.legend(
        #     loc="upper right", title="Model", labels=[m.replace("m_", "") for m in models]
        # )
        plt.xticks(rotation=90)
        plt.xlabel("Model")
        plt.ylabel(r"$d_{WRF}$" if alg == "wrf" else r"$d_{RF}$")
        plt.tight_layout()
        plt.savefig(
            f"figs/plot_{str(args.dir)}_{tree_col}_{alg}_t{args.threshold}.png",
            bbox_inches="tight",
        )
        plt.close()
