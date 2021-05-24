"""
visualize how coverage varies with num train
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from common import pickle_from_file, process_params

METHOD_DICT = {
        "agg": "Aggregate",
        "individual": "Individual",
        "independent": "Aggregate"}

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--pi-alpha',
        type=float,
        help="(1-pi_alpha) prediction interval")
    parser.add_argument('--kfolds',
        type=str,
        default="1",
        help="comma separated")
    parser.add_argument('--coverage-files',
        type=str,
        help="comma separated")
    parser.add_argument('--plot-file',
        type=str,
        default="_output/coverage_vs_kfolds.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.coverage_files = process_params(args.coverage_files, str)
    args.kfolds = process_params(args.kfolds, int)
    return args

def plot_coverage_vs_kfolds(coverage_results, args):
    all_data = []
    for kfolds, coverage_dict in coverage_results:
        results_dict = coverage_dict[args.pi_alpha]
        for metric_key, res_dict in results_dict.items():
            if metric_key in ["is_covered", "ci_diams", "local_coverage_mse", "true_cov"]:
                for type_key, values in res_dict.items():
                    print(type_key, metric_key)
                    #if type_key == "individual":
                    #    # Do not plot individual
                    #    continue

                    for val in values:
                        data_row = {
                            "kfolds": kfolds,
                            "value": float(val),
                            "measure": metric_key,
                            "type": METHOD_DICT[type_key]}
                        all_data.append(data_row)

    coverage_data = pd.DataFrame(all_data)
    print(coverage_data)

    plt.clf()
    plt.figure(figsize=(2,4))
    sns.set(font_scale=1.25, style="white")
    sns.despine()
    g = sns.relplot(
            x="kfolds",
            y="value",
            hue="type",
            style="type",
            col="measure",
            kind="line",
            data=coverage_data,
            facet_kws={"sharey":False},
            ci=95)
            #ci="sd")
    g = g.set_titles("").set_xlabels("K folds")
    g.axes[0,0].set_ylabel("CI Coverage")
    g.axes[0,1].set_ylabel("CI Width")
    g.axes[0,2].set_ylabel("Marginal Coverage")
    g.axes[0,3].set_ylabel("PI Coverage MSE")
    plt.savefig(args.plot_file)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    coverage_results = []
    for coverage_file, kfolds in zip(args.coverage_files, args.kfolds):
        coverage_result = pickle_from_file(coverage_file)
        coverage_results.append((kfolds, coverage_result))

    plot_coverage_vs_kfolds(coverage_results, args)

if __name__ == "__main__":
    main(sys.argv[1:])
