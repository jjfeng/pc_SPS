"""
visualize how coverage varies with num train
"""
import os
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

    parser.add_argument('--file-seeds',
        type=str,
        default="0,1",
        help="min and max seeds")
    parser.add_argument('--pi-alpha',
        type=float,
        default=0.1,
        help="(1-pi_alpha) prediction interval")
    parser.add_argument('--num-trains',
        type=str,
        default="1",
        help="comma separated")
    parser.add_argument('--coverage-file-temp',
        type=str,
        default="_output/agg_coverages.pkl",
        help="comma separated")
    parser.add_argument('--plot-file',
        type=str,
        default="_output/coverage_vs_num_train.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.num_trains = process_params(args.num_trains, int)
    args.file_seeds = process_params(args.file_seeds, int)
    return args

def plot_coverage_vs_num_train(coverage_results, args):
    all_data = []
    for num_train, coverage_dict in coverage_results:
        results_dict = coverage_dict[args.pi_alpha]
        for metric_key, res_dict in results_dict.items():
            if metric_key in ["is_covered", "ci_diams", "local_coverage_mse", "true_cov"]:
                for type_key, values in res_dict.items():
                    for val in values:
                        data_row = {
                            "num_train": num_train,
                            "value": float(val),
                            "measure": metric_key,
                            "type": METHOD_DICT[type_key]}
                        all_data.append(data_row)

    coverage_data = pd.DataFrame(all_data)
    print(coverage_data)
    #print("agg coverage", coverage_data.value[coverage_data.measure == "is_covered" & coverage_data.num_train == 5760 & coverage_data.type == "Aggregate"].mean())
    print(coverage_data.groupby(["num_train", "measure", "type"]).mean())
    #is_covered_mask = coverage_data.measure == "is_covered"
    #is_train = coverage_data.num_train == 2880 * 2

    #method_mask = coverage_data.type == "Aggregate"
    #summ = np.sum(coverage_data[is_covered_mask & method_mask & is_train].value)
    #print("indpt", summ, "out of", np.sum(is_covered_mask & method_mask & is_train))

    #method_mask = coverage_data.type == "Individual"
    #summ = np.sum(coverage_data[is_covered_mask & method_mask & is_train].value)
    #print("indiv", summ, "out of", np.sum(is_covered_mask & method_mask & is_train))

    plt.clf()
    plt.figure(figsize=(2,4))
    sns.set(font_scale=1.25, style="white")
    sns.despine()
    g = sns.relplot(
            x="num_train",
            y="value",
            hue="type",
            style="type",
            col="measure",
            kind="line",
            data=coverage_data,
            facet_kws={"sharey":False},
            ci=95)
            #ci="sd")
    g = g.set_titles("").set_xlabels("Number of training obs")
    g.axes[0,0].set_ylabel("CI Coverage")
    g.axes[0,1].set_ylabel("CI Width")
    g.axes[0,2].set_ylabel("Marginal PI Coverage")
    g.axes[0,3].set_ylabel("PI Coverage MSE")
    plt.savefig(args.plot_file)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(0)

    coverage_results = []
    for num_train in args.num_trains:
        num_seeds_found = 0
        for seed in range(args.file_seeds[0], args.file_seeds[1]):
            coverage_file = args.coverage_file_temp % (num_train, seed)
            if os.path.exists(coverage_file):
                coverage_result = pickle_from_file(coverage_file)
                coverage_results.append((num_train, coverage_result))
                num_seeds_found += 1
        print("NUM SEEDS", num_train, num_seeds_found)

    plot_coverage_vs_num_train(coverage_results, args)

if __name__ == "__main__":
    main(sys.argv[1:])
