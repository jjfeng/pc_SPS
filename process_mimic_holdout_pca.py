"""
Run this after process_mimic_in_hospital_mortality.py
"""
import os
import argparse
import numpy as np
import sys

from sklearn.decomposition import PCA

from support_sim_settings import *
from common import pickle_to_file, pickle_from_file

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--holdout-age',
            action="store_true")
    parser.add_argument(
            '--holdout-min-age',
            type=int,
            default=40)
    parser.add_argument(
            '--holdout-max-age',
            type=int,
            default=100)
    parser.add_argument(
            '--num-pca',
            type=int,
            default=20)
    parser.add_argument(
            '--in-train-data',
            type=str,
            default='../data/mimic/in-hospital-mortality/mimic_in_hospital_train.pkl')
    parser.add_argument(
            '--out-train-data',
            type=str,
            help='file for storing output training dataset, removing holdout ages',
            default='../data/mimic/in-hospital-mortality/mimic_in_hospital_train_holdout_no_whiten.pkl')
    parser.add_argument(
            '--in-test-data',
            type=str,
            default='../data/mimic/in-hospital-mortality/mimic_in_hospital_test.pkl')
    parser.add_argument(
            '--out-test-data',
            type=str)
            #default='../data/mimic/in-hospital-mortality/mimic_in_hospital_test_holdout_no_whiten.pkl')
    parser.add_argument(
            '--out-age-labels',
            type=str)
            #default='../data/mimic/in-hospital-mortality/mimic_in_hospital_age.pkl')
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)

    # Save things
    test_data = pickle_from_file(args.in_test_data)
    test_data.dec_embedder = None
    print(test_data)
    print(test_data.dec_embedder)
    train_data_dict = pickle_from_file(args.in_train_data)
    train_data = train_data_dict["train"]
    train_data.dec_embedder = None
    num_meta_feats = 3
    assert train_data.x.shape[1] % 2 == 1
    num_non_meta_feats = int((train_data.x.shape[1] - num_meta_feats)/2)
    start_missing_idx = num_non_meta_feats + num_meta_feats
    print(start_missing_idx)
    is_missingness_acceptable = np.mean(train_data.x[:,start_missing_idx:], axis=0) < 0.1

    keep_cols = np.concatenate([
        np.array([True, True, True]),
        is_missingness_acceptable,
        np.zeros(num_non_meta_feats, dtype=bool)])
    print(keep_cols)
    print(keep_cols.shape)
    train_data.x = train_data.x[:,keep_cols]
    test_data.x = test_data.x[:,keep_cols]
    print("FINAL DATA BEFORE PCA", train_data.x.shape)

    orig_support_sim_settings = SupportSimSettingsComplex.create_from_dataset(
            train_data.x,
            inflation_factor=0)
    orig_cts_feature_idxs = orig_support_sim_settings.cts_feature_idxs
    orig_discrete_feature_idxs = orig_support_sim_settings.discrete_feature_idxs
    print("CTS", orig_cts_feature_idxs.size, orig_cts_feature_idxs[:10])
    print("DiSCRETE", orig_discrete_feature_idxs.size, orig_discrete_feature_idxs[:10])

    # Rescale the continuous features
    for feat_idx in orig_cts_feature_idxs[1:]:
        # process all features except for age (which is the first feature)
        feat_range = orig_support_sim_settings.feature_ranges[feat_idx]
        train_data.x[:, feat_idx] /= (feat_range.max_x - feat_range.min_x)
        test_data.x[:, feat_idx] /= (feat_range.max_x - feat_range.min_x)

    if args.holdout_age:
        age = train_data.x[:,0]
        age_mask = ((age < args.holdout_min_age) + (age > args.holdout_max_age)).astype(bool)
        heldin_train_data = train_data.subset(age_mask)
        # REMOVE AGE FROM CTS FEATURES
        orig_cts_feature_idxs = orig_cts_feature_idxs[1:]
    else:
        heldin_train_data = train_data
        offset_idx = 0
    print("max train age", np.max(heldin_train_data.x[:,0]))

    pca = PCA(n_components=args.num_pca, whiten=False)
    print("ORIG SHAPE", heldin_train_data.x.shape)
    heldin_train_data_x_cts = pca.fit_transform(heldin_train_data.x[:, orig_cts_feature_idxs])
    print("VAR EXPLAINED", pca.explained_variance_ratio_)
    print("VAR EXPLAINED", np.sum(pca.explained_variance_ratio_))
    print("NUM DIS", orig_discrete_feature_idxs.size)
    test_data_x_cts = pca.transform(test_data.x[:, orig_cts_feature_idxs])

    heldin_train_data.x = np.hstack([
        heldin_train_data.x[:, orig_discrete_feature_idxs],
        heldin_train_data_x_cts])

    pickle_to_file(test_data.x[:, 0:1], args.out_age_labels)
    if args.holdout_age:
        test_data.x = np.hstack([
            test_data.x[:, 0:1], # age feature
            test_data.x[:, orig_discrete_feature_idxs],
            test_data_x_cts])
    else:
        test_data.x = np.hstack([
            test_data.x[:, orig_discrete_feature_idxs],
            test_data_x_cts])
    print('NEW TEST SHAPE', test_data.x.shape)
    print('NEW TRAIN SHAPE', heldin_train_data.x.shape)

    support_sim_settings = SupportSimSettingsComplex.create_from_dataset(
            heldin_train_data.x,
            inflation_factor=0)
    support_sim_settings._process_feature_ranges()
    print("dataset check", support_sim_settings.check_dataset(heldin_train_data))
    if args.holdout_age:
        # Calculate OOO samples in support
        test_age = test_data.x[:,0]
        print("MAX AGE", np.max(test_age), np.max(age))
        test_age_mask = ((test_age >= args.holdout_min_age) & (test_age <= args.holdout_max_age)).astype(bool)
        test_check_supp = support_sim_settings.check_obs_x(
                test_data.x[test_age_mask,1:]
            )
        print("OOO: TEST CHECK", np.mean(test_check_supp))
        print("OOO: TEST AGE ", np.mean(test_age_mask))
        # filter for observations in support
        test_check_supp = support_sim_settings.check_obs_x(test_data.x[:,1:])
    else:
        # filter for observations in support
        test_check_supp = support_sim_settings.check_obs_x(test_data.x)
    print("ALL: TEST CHECK", np.mean(test_check_supp))
    test_data.x = test_data.x[test_check_supp]
    test_data.y = test_data.y[test_check_supp]
    test_data.group_id = test_data.group_id[test_check_supp]

    train_data_dict["train"] = heldin_train_data
    train_data_dict["support_sim_settings"] = support_sim_settings
    heldin_train_data.num_p = heldin_train_data.x.shape[1]
    pickle_to_file(train_data_dict, args.out_train_data)

    test_data.num_p = test_data.x.shape[1]
    pickle_to_file(test_data, args.out_test_data)

    print("num obs", heldin_train_data.num_obs)
    print("num obs", train_data.num_obs)
    print("FINAL NUM FEATS", heldin_train_data.num_p)
    print("FINAL NUM FEATS", test_data.num_p)

if __name__ == '__main__':
    main(sys.argv[1:])
