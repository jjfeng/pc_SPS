"""
Assess calibrated model
"""
import sys
import argparse
import logging
import numpy as np
import scipy.stats

from common import pickle_to_file, pickle_from_file, process_params, load_model, get_normal_ci
from decision_interval_recalibrator import DecisionIntervalRecalibrator


def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--data-file',
        type=str,
        default="_output/data.pkl")
    parser.add_argument('--num-test',
        type=int,
        default=6000)
    parser.add_argument('--fitted-file',
        type=str,
        default="_output/calibrated_model.pkl")
    parser.add_argument('--calibrated-alpha-file',
        type=str,
        default="_output/calibrated_alpha.pkl")
    parser.add_argument('--desired-alpha',
        type=float,
        default=0.05)
    parser.add_argument('--log-file',
        type=str,
        default="_output/log_eval.txt")
    parser.add_argument('--out-file',
        type=str,
        default="_output/eval_result.txt")
    parser.set_defaults()
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    logging.info(args)

    data_dict = pickle_from_file(args.data_file)
    test_data, _ = data_dict["data_gen"].create_data(args.num_test)
    fitted_model = load_model(args.fitted_file)
    calibrated_alpha = pickle_from_file(args.calibrated_alpha_file)
    print("calibrated alpha", calibrated_alpha)
    test_coverage_dict = DecisionIntervalRecalibrator(fitted_model, calibrated_alpha).recalibrate(test_data)
    test_coverage = test_coverage_dict["cov_given_accept"]["mean"]
    ci = get_normal_ci(test_coverage_dict["cov_given_accept"])
    logging.info("true coverage %f", test_coverage)
    print("true coverage %f" % test_coverage)
    print("CI %f -- %f" % ci)
    pickle_to_file({
        "coverage_is_good": test_coverage >= (1 - args.desired_alpha),
        "true_coverage": test_coverage,
        "calibrated_alpha": calibrated_alpha,
    }, args.out_file)

if __name__ == "__main__":
    main(sys.argv[1:])
