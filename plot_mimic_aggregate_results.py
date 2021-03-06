"""
aggregate table result_files
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List

from common import *

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-files',
        type=str)
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.result_files = process_params(args.result_files, str)
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    args.result_files = [f for f in args.result_files if os.path.exists(f)]
    all_results = [pickle_from_file(f) for f in args.result_files]

    kept_results = []
    for res, res_f in zip(all_results, args.result_files):
        print(res_f)
        print(res)
        mdls = res["type"].unique()
        if "borrow_only_0" not in mdls or "ulm_0" not in mdls:
            continue

        ulm_accept = res[(res["key"] == "accept young") & (res["type"] == "ulm_0")]["value"].iloc[0] > 0.05
        borrow_only_accept = res[(res["key"] == "accept young") & (res["type"] == "borrow_only_0")]["value"].iloc[0] > 0.05
        if ulm_accept and borrow_only_accept:
            kept_results.append(res)
        else:
            print("IGNORE")
    print("kept results", len(kept_results))

    res = pd.concat(kept_results)
    agg_res = res.groupby(['key', 'type']).agg([
        np.mean,
        lambda x: np.std(x)/np.sqrt(len(args.result_files))])
    print(agg_res)
    logging.info(agg_res.reset_index().pivot(index="type", columns="key").to_latex(formatters={
        "value": lambda x: "%.3f" % x,
    }))


if __name__ == "__main__":
    main(sys.argv[1:])
