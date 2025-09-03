import argparse
from phyloDNN.utils import (compose, log_transform, make_subsampler,
                            matrix_transform, scale_transform)

parser = argparse.ArgumentParser(description='Train and test.')
parser.add_argument('--config',
                    type=str,
                    help='''path to json-formatted config file.''',
                    default='')

args = parser.parse_args()
subsample_transform = make_subsampler(
    min_taxa=args.min_taxa,
    max_taxa=args.max_taxa,
    num_genes=1)

# TODO: use transforms, make_data_loader funcs
