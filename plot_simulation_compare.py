"""
visualize where we accept (under model misspecification)
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

from decision_prediction_combo import DecisionPredictionModel, EntropyOutlierPredictionModel

from common import load_model, pickle_to_file, pickle_from_file, process_params, is_within_interval, get_normal_ci, get_normal_dist_entropy

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--accept-prob',
        type=float,
        default=0.5)
    parser.add_argument('--cost-decline',
        type=float,
        default=0.1)
    parser.add_argument('--eps',
        type=float,
        default=0.05)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--data-split-file',
        type=str,
        default="_output/data_split.pkl")
    parser.add_argument('--fitted-files',
        type=str,
        default="_output/fitted.pkl",
        help="comma separated")
    parser.add_argument('--add-thres',
        action="store_true",
        default=False,
        help="comma separated")
    parser.add_argument('--add-outlier',
        action="store_true",
        default=False,
        help="comma separated")
    parser.add_argument('--plot-accept-region-file',
        type=str,
        default="_output/accept_region.png")
    parser.set_defaults()
    args = parser.parse_args()
    args.fitted_files = process_params(args.fitted_files, str)
    return args

def plot_accepted_rejected_region(data_dict, fitted_models, args, mesh_size=0.01):
    """
    Plot acceptance region
    """
    num_models = len(fitted_models)
    # Look at the region we accepted
    support_sim = data_dict["support_sim_settings"]
    mesh_x = np.arange(support_sim.min_x, support_sim.max_x, mesh_size).reshape((-1,1))
    y_given_x_sigma = data_dict["data_gen"].sigma_func(mesh_x)
    true_mu = data_dict["data_gen"].mu_func(mesh_x)
    entropy = get_normal_dist_entropy(y_given_x_sigma)

    fitted_model  = fitted_models[0]
    x_accept_probs = fitted_model.get_accept_prob(mesh_x).flatten()
    pred_mus = fitted_model.get_prediction_interval(mesh_x, alpha=0.5).mean(axis=1)
    print("MAX ACCEPT PROB", x_accept_probs.max())
    accepted = x_accept_probs > args.accept_prob
    print("prob accepted", accepted.mean())

    plt.clf()
    mesh_x = mesh_x.flatten()
    plt.scatter(mesh_x, true_mu.flatten(), c="gray", s=1)
    plt.scatter(mesh_x[accepted], pred_mus[accepted], c="green")
    plt.scatter(data_dict["train"].x.flatten(), data_dict["train"].y.flatten(), c="red")
    plt.xlim(support_sim.min_x, support_sim.max_x)

    plt.savefig(args.plot_accept_region_file)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)

    sns.set_context("paper", font_scale=2.1)
    # Read all data
    orig_data_dict = pickle_from_file(args.data_file)
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    recalib_data = orig_data_dict["train"].subset(split_dict["recalibrate_idxs"])
    args.num_p = recalib_data.x.shape[1]

    # Load models
    fitted_models = [load_model(fitted_file) for fitted_file in args.fitted_files]
    if args.add_outlier or args.add_thres:
        thres = args.cost_decline if args.add_thres else np.inf
        eps = args.eps if args.add_outlier else 1
        fitted_models = [
                EntropyOutlierPredictionModel(
                    DecisionPredictionModel(m, thres),
                    thres,
                    eps=eps)
                for m in fitted_models]
        for m in fitted_models:
            m.fit_decision_model(orig_data_dict["train"].x)

    # Do all the plotting
    plot_accepted_rejected_region(
        orig_data_dict,
        fitted_models,
        args)

if __name__ == "__main__":
    main(sys.argv[1:])
