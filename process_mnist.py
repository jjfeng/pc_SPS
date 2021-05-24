import os
import argparse
import numpy as np
import sys
import tensorflow as tf
import cv2

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from dataset import Dataset
from support_sim_settings import SupportSimSettingsContinuousMulti, SupportSimSettingsInvPCA
from autoencoder import get_autoencoded_imgs, fit_autoencoder
from common import pickle_to_file

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data',
            type=str,
            help='Path to the data of mnist',
            default='../data/mnist')
    parser.add_argument(
            '--out-train-data',
            type=str,
            help='file for storing output training dataset',
            default=None)
    parser.add_argument(
            '--out-test-data',
            type=str,
            help='file for storing output test dataset',
            default=None)
    parser.add_argument(
            '--out-weird-data',
            type=str,
            help='file for storing output test dataset',
            default=None)
    parser.add_argument(
            '--do-aux-pca',
            action="store_true")
    parser.add_argument(
            '--do-pca',
            action="store_true")
    parser.add_argument(
            '--do-autoencoder',
            action="store_true")
    parser.add_argument(
            '--autoencoder-latent-dim',
            type=int,
            default=64)
    parser.add_argument(
            '--out-random-images-plot',
            type=str,
            default='_output/images/random_mnist_kinda_images.png')
    args = parser.parse_args()
    return args

def create_support_sim_settings(x_train, x_test, pca=None, orig_x_shape=None):
    num_p = x_train.shape[1]
    min_x = np.min(np.concatenate([x_train, x_test]), axis=0).reshape((1,-1))
    max_x = np.max(np.concatenate([x_train, x_test]), axis=0).reshape((1,-1))
    #print(max_x - min_x)
    if pca is not None:
        support_sim_settings = SupportSimSettingsInvPCA(
            pca,
            orig_x_shape,
            num_p,
            min_x=min_x,
            max_x=max_x)
    else:
        support_sim_settings = SupportSimSettingsContinuousMulti(
            num_p,
            min_x=min_x,
            max_x=max_x)
    #diffs = max_x - min_x
    #print(np.sum(diffs > 0))
    #print(np.sum(np.log(diffs[diffs > 0])))
    print(x_train.shape)
    print(x_test.shape)
    return support_sim_settings

def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)
    np.random.seed(0)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    x_test_orig = x_test

    orig_image_shape = x_train.shape[1:]

    num_classes = 10
    num_train_classes = 9

    data_mask = y_train < num_train_classes
    x_train = x_train[data_mask]
    y_train = y_train[data_mask]
    y_train_categorical = np.zeros((y_train.size, num_train_classes))
    y_train_categorical[np.arange(y_train.size),y_train] = 1

    y_test_categorical = np.zeros((y_test.size, num_classes))
    y_test_categorical[np.arange(y_test.size),y_test] = 1

    (_, _), (orig_weird_x,_) = tf.keras.datasets.fashion_mnist.load_data()
    weird_x = orig_weird_x.astype('float32') / 255.0
    print("TRAIN", x_train.shape)
    print("TEST", x_test.shape)
    print("WEIRD", weird_x.shape)

    # Consider various ways of compressing the data
    dec_embedder = None
    if args.do_pca:
        pca = PCA(n_components=300, whiten=False)
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))
        weird_x = weird_x.reshape((weird_x.shape[0], -1))
        x_train = pca.fit_transform(x_train)
        print(pca.explained_variance_ratio_)
        x_test = pca.transform(x_test)
        weird_x = pca.transform(weird_x)
        support_sim_settings = create_support_sim_settings(x_train, x_test)
        check_supp = support_sim_settings.check_obs_x(weird_x)
    elif args.do_autoencoder:
        # Note: we get pretty good results with loss 3.9
        # However, this doesn't actually capture all the variance of the image, and will fail to describe
        # unseen digits very accurately and struggle really bad with fashion images.
        # PCA turns out to be better, in fact
        x_train = x_train.reshape(list(x_train.shape)+ [1])
        x_test = x_test.reshape(list(x_test.shape) + [1])
        weird_x = weird_x.reshape(list(weird_x.shape) + [1])
        sess = tf.Session()
        with sess.as_default():
            autoencoder = fit_autoencoder(x_train, x_test, latent_dim=args.autoencoder_latent_dim, epochs=20)
            x_train = get_autoencoded_imgs(autoencoder, x_train)
            x_test = get_autoencoded_imgs(autoencoder, x_test)
            weird_x = get_autoencoded_imgs(autoencoder, weird_x)
        sess.close()
        support_sim_settings = create_support_sim_settings(x_train, x_test)
    elif args.do_aux_pca:
        dec_embedder = PCA(n_components=300, whiten=False)
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))
        x_train_transform = dec_embedder.fit_transform(x_train_flat)
        x_test_transform = dec_embedder.transform(x_test_flat)
        support_sim_settings = create_support_sim_settings(x_train_transform, x_test_transform)
        check_supp = support_sim_settings.check_obs_x(
                dec_embedder.transform(weird_x.reshape((weird_x.shape[0], -1))))
        x_train = x_train.reshape(list(x_train.shape)+ [1])
        x_test = x_test.reshape(list(x_test.shape) + [1])
        weird_x = weird_x.reshape(list(weird_x.shape) + [1])
    else:
        # reshape for image
        x_train = x_train.reshape(list(x_train.shape)+ [1])
        x_test = x_test.reshape(list(x_test.shape) + [1])
        weird_x = weird_x.reshape(list(weird_x.shape) + [1])

        # Fit PCA for making random image-looking things
        pca = PCA(n_components=300, whiten=False)
        x_train_flat = x_train.reshape((x_train.shape[0], -1))
        x_test_flat = x_test.reshape((x_test.shape[0], -1))
        x_train_transform = pca.fit_transform(x_train_flat)
        x_test_transform = pca.transform(x_test_flat)
        support_sim_settings = create_support_sim_settings(
                x_train_transform,
                x_test_transform,
                pca,
                list(x_train.shape[1:]))
        # Plot for checking
        randx = support_sim_settings.support_unif_rvs(4)
        plt.imshow(randx[3,:,:,0],cmap=plt.cm.binary,interpolation='nearest')
        plt.savefig("_output/rand.png")
        plt.clf()
        plt.imshow(x_train[0,:,:,0],cmap=plt.cm.binary,interpolation='nearest')
        plt.savefig("_output/xtrain.png")
        check_supp = support_sim_settings.check_obs_x(weird_x)

        print("doing nothing to the images")
        #raise NotImplementedError()

    train_data = Dataset(x=x_train, y=y_train_categorical, num_classes=num_train_classes, dec_embedder=dec_embedder) #lambda x: pca.transform(x.reshape((x.shape[0], -1))))
    train_data_dict = {
            "train": train_data,
            "support_sim_settings": support_sim_settings}
    pickle_to_file(train_data_dict, args.out_train_data)

    random_idx = np.random.choice(x_test.shape[0], size=4000, replace=False)
    test_data = Dataset(x=x_test[random_idx, :], y=y_test_categorical[random_idx, :], num_classes=num_classes, dec_embedder=dec_embedder)
    pickle_to_file({
        "data": test_data,
        "x_orig": x_test_orig[random_idx],
    }, args.out_test_data)

    print("NUM WEIRD", weird_x.shape)
    print("NUM WEiRD IN SUPPORT", np.sum(check_supp))
    weird_data = Dataset(x=weird_x[check_supp], y=None, dec_embedder=dec_embedder)
    pickle_to_file({
        "data": weird_data,
        "orig": orig_weird_x[check_supp],
    }, args.out_weird_data)

if __name__ == '__main__':
    main(sys.argv[1:])
