import sys
import argparse
import logging
import numpy as np
import pandas as pd

from common import pickle_from_file, process_params

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        default=0)
    parser.add_argument('--result-files',
        type=str,
        default="_output/result.pkl",
        help="comma separated")
    parser.add_argument('--result-binary-files',
        type=str,
        default="_output/result_binary.pkl",
        help="comma separated")
    parser.add_argument('--result-platt-files',
        type=str,
        default="_output/result_platt.pkl",
        help="comma separated")
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.result_files = process_params(args.result_files, str)
    args.result_binary_files = process_params(args.result_binary_files, str)
    args.result_platt_files = process_params(args.result_platt_files, str)
    return args

def print_evaluation_summary(label, file_names):
    print("=======", label)
    logging.info("EVAL SUMMARY %s", label)
    eval_dicts = [
            pickle_from_file(res_file) for res_file in file_names]
    average_good = np.mean([eval_dict["coverage_is_good"] for eval_dict in eval_dicts])
    logging.info("is good %f", average_good)
    true_coverages = [eval_dict["true_coverage"] for eval_dict in eval_dicts]
    logging.info("true_coverages %f (std dev %f)", np.mean(true_coverages), np.sqrt(np.var(true_coverages)))
    calibrated_alphas = [eval_dict["calibrated_alpha"] for eval_dict in eval_dicts]
    logging.info("calibrated_alpha %f (std dev %f)", np.mean(calibrated_alphas), np.sqrt(np.var(calibrated_alphas)))
    print("IS GOOD", average_good)
    print("true_coverages", np.mean(true_coverages))

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)

    print_evaluation_summary("GUARDS", args.result_files)
    print_evaluation_summary("BINARY SEARCH", args.result_binary_files)
    print_evaluation_summary("PLATT", args.result_platt_files)

if __name__ == "__main__":
    main(sys.argv[1:])
