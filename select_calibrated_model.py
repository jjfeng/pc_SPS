"""
Select the best model by gatekeeping
"""
import sys
import argparse
import logging
import tensorflow as tf
import numpy as np

from decision_interval_recalibrator import DecisionIntervalRecalibrator
from common import pickle_to_file, pickle_from_file, process_params, get_normal_ci, load_model


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--data-split-file',
        type=str,
        default="_output/data_split.pkl")
    parser.add_argument('--fitted-file',
        type=str,
        default="_output/fitted.pkl")
    parser.add_argument('--out-file',
        type=str,
        default="_output/calibrated_model.pkl")
    parser.add_argument('--log-file',
        type=str,
        default="_output/recalibrated_log.txt")
    parser.add_argument('--alphas',
        type=str,
        help="""
        Specifies which alpha values to calibrate for
        comma separated list
        """,
        default="0.05,0.1,0.2")
    parser.add_argument('--desired-alpha',
        type=float,
        default=0.05)
    parser.add_argument('--ci-alpha',
        type=float,
        default=0.05)
    parser.set_defaults()
    args = parser.parse_args()
    args.alphas = list(sorted(process_params(args.alphas, float)))
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    print(args)
    logging.info(args)

    # Read all data
    data_dict = pickle_from_file(args.data_file)
    # Get the appropriate datasplit
    split_dict = pickle_from_file(args.data_split_file)
    recalib_data = data_dict["train"].subset(split_dict["recalibrate_idxs"])

    # Load model
    fitted_model = load_model(args.fitted_file)

    coverage_dict = {}
    good_alpha = args.alphas[0]
    for alpha in args.alphas:
        recalibrator = DecisionIntervalRecalibrator(fitted_model, alpha)
        inference_dict = recalibrator.recalibrate(recalib_data)
        #print("RECALIB INF DICT", inference_dict["cov_given_accept"])
        est_cov_given_accept = inference_dict["cov_given_accept"]["mean"]
        logging.info("Alpha %f, ideal cov %f, est cov|accept %f", alpha, 1 - alpha, est_cov_given_accept)
        logging.info(get_normal_ci(inference_dict["cov_given_accept"], args.ci_alpha))
        coverage_dict[alpha] = inference_dict
        print("----------- ALPHA", alpha)
        print("CI cov given accpt", get_normal_ci(inference_dict["cov_given_accept"], args.ci_alpha))
        ci_lower, ci_upper = get_normal_ci(inference_dict["cov_given_accept"], args.ci_alpha)
        if ci_lower < 1 - args.desired_alpha:
            break
        else:
            good_alpha = alpha
    print("SELECTED ALPHA", good_alpha)
    logging.info("SELECTED ALPHA %f", good_alpha)
    pickle_to_file(good_alpha, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
