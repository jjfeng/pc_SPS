"""
Compare interval-fitted to density-fitted
"""
import sys
import argparse
from argparse import Namespace
import logging
from typing import List, Dict
import numpy as np
from numpy import ndarray
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from plot_fitted import eval_coverage
from data_generator import DataGenerator
from common import load_model, pickle_to_file, pickle_from_file, process_params, is_within_interval, get_normal_ci

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--orig-data-files',
        type=str,
        default="_output/data.pkl",
        help="comma separated")
    parser.add_argument('--num-test',
        type=int,
        default=1000,
        help="number of new observations to generate and test for")
    parser.add_argument('--num-trains',
        type=str,
        default="100",
        help="comma separated")
    parser.add_argument('--fitted-density-files',
        type=str,
        default="_output/fitted_density.pkl",
        help="comma separated")
    parser.add_argument('--fitted-interval-files',
        type=str,
        default="_output/fitted_interval.pkl",
        help="comma separated")
    parser.add_argument('--alpha',
        type=float,
        default=0.05)
    parser.add_argument('--plot-file',
        type=str,
        default="_output/interval_vs_density.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.orig_data_files = process_params(args.orig_data_files, str)
    args.fitted_interval_files = process_params(args.fitted_interval_files, str)
    args.fitted_density_files = process_params(args.fitted_density_files, str)
    args.num_trains = process_params(args.num_trains, int)
    return args

def get_distance_to_true_pi(fitted_models: List, xs_and_pis: List[Dict], args: Namespace, label: str):
    """
    @return avg value of |true PI boundary - estimated PI boundary|, weighted by accept probability
    """
    assert len(fitted_models) == len(xs_and_pis)
    dist_to_pis = []
    for model, xs_and_pi_dict in zip(fitted_models, xs_and_pis):
        xs = xs_and_pi_dict["xs"]
        true_pi = xs_and_pi_dict["true_pi"]
        est_pi = model.get_prediction_interval(xs, args.alpha)
        accept_prob = model.get_accept_prob(xs)
        print(label, ": avg accept prob=", np.mean(accept_prob))
        weighted_pi_distances = (true_pi - est_pi) * accept_prob
        avg_weighted_dist_to_pi = np.mean(np.abs(true_pi - est_pi))
        dist_to_pis.append(avg_weighted_dist_to_pi)

    data_rows = []
    for dist, num_train in zip(dist_to_pis, args.num_trains):
        data_rows.append({
            "dist": dist,
            "num_train": num_train,
            "type": label})
    return data_rows

def plot_distance_from_boundaries(
        fitted_density_models: List,
        fitted_interval_models: List,
        data_generators: List[DataGenerator],
        args: Namespace):
    xs_and_pis = []
    for data_generator in data_generators:
        xs = data_generator.generate_x(args.num_test)
        true_pi = data_generator.get_prediction_interval(xs, args.alpha)
        xs_and_pis.append({
            "xs": xs,
            "true_pi": true_pi})

    density_data = get_distance_to_true_pi(
            fitted_density_models,
            xs_and_pis,
            args,
            "density")
    interval_data = get_distance_to_true_pi(
            fitted_interval_models,
            xs_and_pis,
            args,
            "interval")

    df = pd.DataFrame(density_data + interval_data)
    print("data", df)

    plt.clf()
    sns.lineplot(
            x="num_train",
            y="dist",
            hue="type",
            data=df)
    plt.savefig(args.plot_file)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    data_generators = [
            pickle_from_file(orig_data_file)["data_gen"]
            for orig_data_file in args.orig_data_files]

    # Load models
    fitted_density_models = [load_model(fitted_file) for fitted_file in args.fitted_density_files]
    fitted_interval_models = [load_model(fitted_file) for fitted_file in args.fitted_interval_files]

    # Do all the plotting
    plot_distance_from_boundaries(
            fitted_density_models,
            fitted_interval_models,
            data_generators,
            args)

if __name__ == "__main__":
    main(sys.argv[1:])
