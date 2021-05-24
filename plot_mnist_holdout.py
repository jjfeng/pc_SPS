"""
visualize mnist
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from dataset import Dataset
from decision_prediction_combo import *
from ensemble_decision_density_nn import EnsembleSimultaneousDensityDecisionNNs
from common import *
from plot_common import *

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',
        type=int,
        default=0,
        help="random seed")
    parser.add_argument('--ensemble-eps',
        type=float,
        default=0.1,
        help="eps for outlier detector")
    parser.add_argument('--cost-decline',
        type=float,
        default=1)
    parser.add_argument('--ensemble-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--ensemble-borrow-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--ensemble-ulm-files',
        type=str,
        default="",
        help="comma separated")
    parser.add_argument('--train-data-file',
        type=str)
    parser.add_argument('--test-data-file',
        type=str)
    parser.add_argument('--weird-data-file',
        type=str)
    parser.add_argument('--log-file',
        type=str,
        default="_output/plot_mnist_log.txt")
    parser.add_argument('--out-results-file',
        type=str,
        default="_output/results.pkl")
    parser.add_argument('--out-examples-plot',
        type=str,
        default="_output/examples.pdf")
    parser.set_defaults()
    args = parser.parse_args()
    args.ensemble_borrow_files = process_params(args.ensemble_borrow_files, str)
    args.ensemble_ulm_files = process_params(args.ensemble_ulm_files, str)
    args.ensemble_files = process_params(args.ensemble_files, str)
    return args

def get_data(models, test_data, num_train_classes, weird_x, random_x, label):
    print(label)
    unseen_idxs = np.where(test_data.y > 0)[1] >= num_train_classes
    seen_idxs = np.logical_not(unseen_idxs)
    res_data = []
    # Collect results for the simultaneous decision density NNs
    for idx, model in enumerate(models):
        accept_probs = model.get_accept_prob(test_data.x).reshape((-1,1))
        accept_seen = np.mean(accept_probs[seen_idxs])
        accept_unseen = np.mean(accept_probs[unseen_idxs])

        seen_test_data = test_data.subset(seen_idxs)
        seen_test_data.y = seen_test_data.y[:,:num_train_classes]
        seen_test_data.num_classes = num_train_classes

        score_seen = model.score(test_data.x[seen_idxs], test_data.y[seen_idxs, :num_train_classes])

        accept_weird = model.get_accept_prob(weird_x).mean()

        accept_random = model.get_accept_prob(random_x).mean()

        res_data += extract_row(accept_seen, "accept_seen", label)
        res_data += extract_row(accept_unseen, "accept_unseen", label)
        res_data += extract_row(score_seen, "score_seen", label)
        res_data += extract_row(accept_weird, "accept_weird", label)
        res_data += extract_row(accept_random, "accept_random", label)
    return res_data

def plot_results(
        result_dict: Dict[str, List],
        test_data: Dataset,
        num_train_classes: int,
        weird_x: ndarray,
        random_x: ndarray):
    all_data = []
    for key, val in result_dict.items():
        new_data = get_data(val, test_data, num_train_classes, weird_x, random_x, key)
        all_data += new_data
    data_df = pd.DataFrame(all_data)
    logging.info(data_df.to_latex())
    print(data_df)
    return data_df

def plot_examples_unseen(
        model_dict: Dict,
        test_data_dict: Dict,
        num_train_classes: int,
        out_plot_file: str,
        alpha: float = 0.1,
        num_examples: int = 5):
    print("UNSEEN IMAGES")
    test_data = test_data_dict["data"]
    print(test_data.y)
    unseen_idxs = np.where(test_data.y > 0)[1] >= num_train_classes
    unseen_img_idxs = np.where(unseen_idxs)[0]
    unseen_x = test_data.x[unseen_idxs]

    accept_prob_dict = {}
    pred_set_dict = {}
    for model_label, model in model_dict.items():
        accept_prob_dict[model_label] = model.get_accept_prob(unseen_x)
        pred_set_dict[model_label] = model.get_prediction_interval(unseen_x, alpha)
    model_keys = list(model_dict.keys())
    print("MODEL KEYS", model_keys)
    accept_probs = np.hstack([accept_prob_dict[model_label] for model_label in model_keys])

    fig, ax = plt.subplots(num_examples)
    for i in range(num_examples):
        img_idx = unseen_img_idxs[i]
        orig_x = test_data_dict["x_orig"][img_idx]
        print("accept probs", accept_probs[i,:])
        for k in accept_prob_dict.keys():
            print("pred sets", k, pred_set_dict[k][i])
        im = ax[i].imshow(
                orig_x,
                cmap=plt.cm.binary,
                interpolation='nearest')

    plt.savefig(out_plot_file)

def plot_examples_seen(
        model_dict: Dict,
        test_data_dict: Dict,
        num_train_classes: int,
        out_plot_file: str,
        alpha: float = 0.1,
        num_examples: int = 5):
    print("SEEN IMAGES")
    test_data = test_data_dict["data"]
    unseen_idxs = np.where(test_data.y > 0)[1] >= num_train_classes
    seen_idxs = np.logical_not(unseen_idxs)
    seen_img_idxs = np.where(seen_idxs)[0]
    seen_x = test_data.x[seen_idxs]

    accept_prob_dict = {}
    pred_set_dict = {}
    for model_label, model in model_dict.items():
        accept_prob_dict[model_label] = model.get_accept_prob(seen_x)
        pred_set_dict[model_label] = model.get_prediction_interval(seen_x, alpha)
    model_keys = list(model_dict.keys())
    print("MODEL KEYS", model_keys)
    accept_probs = np.hstack([accept_prob_dict[model_label] for model_label in model_keys])

    fig, ax = plt.subplots(num_examples * 2)
    for i in range(num_examples):
        img_idx = seen_img_idxs[i]
        print("IMG IDX", img_idx)
        orig_x = test_data_dict["x_orig"][img_idx]
        print("accept probs", accept_probs[i,:])
        for k in accept_prob_dict.keys():
            print("pred sets", k, pred_set_dict[k][i])
        im = ax[i].imshow(
                orig_x,
                cmap=plt.cm.binary,
                interpolation='nearest')

    pred_set_sizes = np.sum(pred_set_dict["ensemble_ulm"], axis=1)
    keep_indices = np.where(pred_set_sizes > 1)[0]
    for i, keep_idx in enumerate(keep_indices[:num_examples]):
        img_idx = seen_img_idxs[keep_idx]
        print("IMG IDX", img_idx)
        orig_x = test_data_dict["x_orig"][img_idx]
        print("accept probs", accept_probs[keep_idx,:])
        for k in accept_prob_dict.keys():
            print("pred sets", k, pred_set_dict[k][keep_idx])
        im = ax[i + num_examples].imshow(
                orig_x,
                cmap=plt.cm.binary,
                interpolation='nearest')

    plt.savefig(out_plot_file)

def plot_examples_weird(
        model_dict: Dict,
        weird_x_dict: Dict[str, ndarray],
        out_plot_file: str,
        alpha: float = 0.1,
        num_examples: int = 5):
    print("WEIRD IMAGES")
    accept_prob_dict = {}
    pred_set_dict = {}
    for model_label, model in model_dict.items():
        accept_prob_dict[model_label] = model.get_accept_prob(weird_x_dict["data"].x)
        pred_set_dict[model_label] = model.get_prediction_interval(weird_x_dict["data"].x, alpha)
    model_keys = list(model_dict.keys())
    print("MODEL KEYS", model_keys)
    accept_probs = np.hstack([accept_prob_dict[model_label] for model_label in model_keys])
    keep_indices = np.where(accept_probs[:,0] != accept_probs[:,1])[0]
    if keep_indices.size == 0:
        keep_indices = range(num_examples)

    fig, ax = plt.subplots(num_examples)
    for i in range(num_examples):
        img_idx = keep_indices[i]
        orig_weird = weird_x_dict["orig"][img_idx]/255.
        print("accept probs", accept_probs[img_idx,:])
        for k in accept_prob_dict.keys():
            print("pred sets", k, pred_set_dict[k][img_idx])
        im = ax[i].imshow(
                orig_weird,
                cmap=plt.cm.binary,
                interpolation='nearest')

    plt.savefig(out_plot_file)

def generate_random_imgs(train_data_dict, n: int = 100):
    """
    @param train_data_dict: Dict with reference train dataset
    @param n: number of random imgs to create

    @return random imgs or embeddings
    """
    data_shape = train_data_dict["train"].x.shape
    if len(data_shape) == 4:
        # Generate random images
        return np.random.rand(n, data_shape[1], data_shape[2], data_shape[3])
    else:
        # Generate random embeddings
        return train_data_dict["support_sim_settings"].support_unif_rvs(n)

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    np.random.seed(args.seed)

    test_dataset_dict = pickle_from_file(args.test_data_file)
    test_dataset = test_dataset_dict["data"]
    train_data_dict = pickle_from_file(args.train_data_file)
    weird_x_dict = pickle_from_file(args.weird_data_file)
    weird_x = weird_x_dict["data"].x
    train_dataset = train_data_dict["train"]
    random_x = generate_random_imgs(train_data_dict)

    #embedded = train_dataset.dec_embedder.transform(random_x.reshape((random_x.shape[0], -1)))
    #check_ra = train_data_dict["support_sim_settings"].check_obs_x(embedded)
    #print("CHECK", check_ra)
    #print("MEAN", check_ra.mean())
    #assert check_ra.mean() >= 1.0

    nns_dict = {
        "ensemble": args.ensemble_files,
        "ensemble_borrow_ulm": args.ensemble_borrow_files,
        "ensemble_ulm": args.ensemble_ulm_files,
    }
    for key, val in nns_dict.items():
        print(key)
        nns_dict[key] = [load_model(file_name) for file_name in val]

    decision_density_dict = {}
    for key, val in nns_dict.items():
        if "ulm" in key:
            decision_density_dict[key] = val
        else:
            decision_density_dict["%s+cutoff" % key] = [
                DecisionPredictionModel(nns, args.cost_decline)
                for nns in val]

    model_types = {
            "ensemble": args.ensemble_eps,
    }
    for model_type, eps in model_types.items():
        # Do Outlier detection on raw variables
        combo_key = "%s+cutoff+OD_raw_%s" % (model_type, str(eps))
        decision_density_dict[combo_key] = [
                EntropyOutlierPredictionModel(DecisionPredictionModel(nns, args.cost_decline), args.cost_decline, eps=eps)
                for nns in nns_dict[model_type]]
        for m in decision_density_dict[combo_key]:
            m.fit_decision_model(train_dataset.x)

        ## Do Outlier detection on prediction NN embedding
        #combo_key = "%s+cutoff+OD_embed_%s" % (model_type, str(eps))
        #decision_density_dict[combo_key] = [
        #        EmbeddingEntropyOutlierPredictionModel(DecisionPredictionModel(nns, args.cost_decline), args.cost_decline, eps=eps)
        #        for nns in nns_dict[model_type]]
        #for m in decision_density_dict[combo_key]:
        #    m.fit_decision_model(train_dataset.x)

    data_df = plot_results(
            decision_density_dict,
            test_dataset,
            train_dataset.num_classes,
            weird_x,
            random_x)
    pickle_to_file(data_df, args.out_results_file)

    #model_plot_dict = {
    #            "ensemble_ulm": decision_density_dict["ensemble_ulm"][0],
    #            "ensemble+cutoff": decision_density_dict["ensemble+cutoff"][0]}
    #plot_examples_seen(
    #        model_plot_dict,
    #        test_dataset_dict,
    #        train_dataset.num_classes,
    #        out_plot_file=args.out_examples_plot.replace("images", "seen"))
    #plot_examples_unseen(
    #        model_plot_dict,
    #        test_dataset_dict,
    #        train_dataset.num_classes,
    #        out_plot_file=args.out_examples_plot.replace("images", "unseen"))
    #plot_examples_weird(
    #        model_plot_dict,
    #        weird_x_dict,
    #        out_plot_file=args.out_examples_plot.replace("images", "weird"))

if __name__ == "__main__":
    main(sys.argv[1:])

