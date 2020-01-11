import pandas as pd
import numpy as np
import pickle
from utils import *
import argparse
from itertools import product, combinations
#from keras.preprocessing.image import load_img, img_to_array
#from scipy.misc import imresize
#from keras.layers import Input
#from keras.models import Model
#from googlenet_functional import *
#from sklearn.decomposition import PCA
from os.path import exists
from scipy.sparse import save_npz, load_npz

def create_partitions(rankings_all, n_fold, d_train):
    '''
    :param n_fold: number of cross validation folds
    :param rankings_all: [(i_1,i_2, ...), (i_1,i_2, ...), ...]
    :param d_train: number of training samples
    :return: rankings_train (n_fold x d), rankings_test (n_fold x len(rankings_all/n_fold))
    '''
    d_all = len(rankings_all)
    d_test = int(d_all / n_fold)
    if d_train > d_all - d_test:
        d_train = d_all - d_test
        print('Only', d_train, 'number of training observations available!')
    # partition observations into train and test data
    np.random.seed(1)
    # stock indices to a matrix of (n_fold, indices)
    shuffled_ind = np.reshape(np.random.permutation(d_test * n_fold), (n_fold, d_test))
    rankings_train = []
    rankings_test = []
    for test_fold in range(n_fold):
        train_ind = shuffled_ind[[fold for fold in range(n_fold) if (fold != test_fold)]]
        # put remaining samples back
        train_ind = np.concatenate((np.reshape(train_ind, (train_ind.shape[0] * train_ind.shape[1],)),
                                    np.arange(d_test * n_fold, d_all)))
        # subsample
        train_ind = train_ind[:d_train]
        # get training rankings
        rankings_train.append(rankings_all[train_ind])
        # get test rankings
        rankings_test.append(rankings_all[shuffled_ind[test_fold]])
    # dims: (n_fold, d_train, number of ranked items at a time (A_l))
    return rankings_train, rankings_test

def create_partitions_wrt_sample(rankings_all, n, n_fold, d_train):
    '''
    :param n: number of samples
    :param n_fold: number of cross validation folds
    :param rankings_all: [(i_1,i_2, ...), (i_1,i_2, ...), ...]
    :param d_train: number of training samples
    partition rankings by the samples participating in train or test. no rankings across
    :return: rankings_train (n_fold x d), rankings_test (n_fold x len(rankings_all/n_fold))
    '''
    # partition samples into train and test data
    np.random.seed(1)
    # stock indices to a matrix of (n_fold, indices)
    samp_test = int(n / n_fold)
    shuffled_samp = np.reshape(np.random.permutation(samp_test * n_fold), (n_fold, samp_test))
    rankings_train = []
    rankings_test = []
    for test_fold in range(n_fold):
        train_samp = shuffled_samp[[fold for fold in range(n_fold) if (fold != test_fold)]]
        test_samp = shuffled_samp[test_fold]
        # put remaining samples back
        train_samp = np.concatenate((np.reshape(train_samp, (train_samp.shape[0] * train_samp.shape[1],)),
                                    np.arange(samp_test * n_fold, n)))
        # get training rankings associated with only training samples
        rankings_train_fold = []
        for rank in rankings_all:
            if np.all(np.isin(rank, train_samp)):
                rankings_train_fold.append(rank)
        # subsample
        if d_train < len(rankings_train_fold):
            rankings_train_fold = rankings_train_fold[:d_train]
        rankings_train.append(rankings_train_fold)
        # get test rankings associated with only test samples
        rankings_test_fold = []
        for rank in rankings_all:
            if np.all(np.isin(rank, test_samp)):
                rankings_test_fold.append(rank)
        rankings_test.append(rankings_test_fold)
    # dims: (n_fold, d_train, number of ranked items at a time (A_l))
    return rankings_train, rankings_test

def save_synthetic_data(n_fold, n, p, k, d, rand_iter, dir='synthetic_', partition='per_obs', score_model='exp', samp_method='random'):
    '''
    generate Gaussian distribution features and parameters
    sigma2: sigma square of covariance matrix
    Make sure to keep test data size while subsampling training data
    :param n: number of items
    :param p: dimension
    :param k: size of partial ranking
    :param d: number of full rankings
    :param samp_method: hajek (dense case) or not
    :return: feature matrix, dataset {(winner, loser)}, true scores, true_beta
    '''
    np.random.seed(rand_iter)
    d_max = n * (n-1) / 2
    d_test = int(d_max / n_fold)
    d_total = d + d_test
    if d > int(d_max * (n_fold - 1) / n_fold):
        raise ValueError('Number of training samples cannot be greater than', d_max - d_test)
    else:
        print(d_total, 'rankings are being generated')
    if score_model == 'exp':
        sigma2Beta = 0.8
        sigma2Feature = 0.8
        Beta_center = np.zeros(p)
        Beta_cov = sigma2Beta * np.diag(np.ones(p))
        Feature_center = np.zeros(p)
        Feature_cov = sigma2Feature * np.diag(np.ones(p))
        Beta = np.random.multivariate_normal(Beta_center, Beta_cov, 1)[0, :]
        X = np.random.multivariate_normal(Feature_center, Feature_cov, n)
        b = 0
        scores = softmax(np.dot(X, Beta))
    else:
        sigma2Beta = 100
        sigma2Feature = 100
        Beta_center = np.zeros(p)
        Beta_cov = sigma2Beta * np.diag(np.ones(p))
        Feature_center = np.zeros(p)
        Feature_cov = sigma2Feature * np.diag(np.ones(p))
        Beta = np.random.multivariate_normal(Beta_center, Beta_cov, 1)[0, :]
        X = np.random.multivariate_normal(Feature_center, Feature_cov, n)
        b = np.max(-1.0 * np.dot(X, Beta)) + rtol
        scores = np.dot(X, Beta) + b*np.ones((n,), dtype=float)
        # Normalize scores
        Beta = Beta / sum(scores)
        b = b / sum(scores)
        scores = scores / sum(scores)
    # generate observations
    if samp_method != 'hajek':
        rankings_all = []
        while len(rankings_all) < d_total:
            # sample 2 items
            items = np.random.choice(n, k, replace=False)
            # get PL probabilities for pairwise comparison
            probabilities = scores[items] / sum(scores[items])
            comp = np.random.choice(items, k, p=probabilities, replace=False)
            rankings_all.append(comp)
            rankings_all = list(np.unique(rankings_all, axis=0))
        # cv partition
        rankings_all = np.array(rankings_all)
        if partition == 'per_samp':
            rankings_train, rankings_test = create_partitions_wrt_sample(rankings_all, X.shape[0], n_fold, d)
        else:
            rankings_train, rankings_test = create_partitions(rankings_all, n_fold, d)
    else:
        part_rankings = []
        M = int(n * d / k)  # number of training samples
        # get PL probabilities for full ranking
        probabilities = scores / sum(scores)
        for full_iter in range(d):
            # create full ranking
            full_ranking = np.random.choice(n, n, p=probabilities, replace=False)
            # length-k partition uniformly at random
            part_items = np.reshape(np.random.permutation(n), (int(n / k), k))
            for part_iter in range(int(n / k)):
                ranking_indices = []
                # sort the items in each partition with respect to the full ranking
                for item in part_items[part_iter, :]:
                    ranking_indices.append(np.where(full_ranking == item)[0])
                part_rankings.append([x for _, x in sorted(zip(ranking_indices, part_items[part_iter, :]))])
        part_indices = np.random.permutation(range(M))
        train_indices = part_indices[:M-100]
        test_indices = part_indices[M-100:]
        rankings_train = np.array(part_rankings[train_indices]).astype(int)
        rankings_test = np.array(part_rankings[test_indices]).astype(int)
    ################################################## save
    for test_fold in range(n_fold):
        print(test_fold, 'folds are generated')
        save_name = str(rand_iter) + '_' + str(test_fold) + '_n_' + str(n) + '_p_' + str(p) + '_k_' + str(k) + '_d_' + str(d)
        np.save('../data/' + dir + 'data/' + save_name + '_parameters', Beta)
        np.save('../data/' + dir + 'data/' + save_name + '_bias', b)
        np.save('../data/' + dir + 'data/' + save_name + '_scores', scores)
        np.save('../data/' + dir + 'data/' + save_name + '_features', X)
        np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
        np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
        n = X.shape[0]
        mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
        save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
    return X, rankings_train, rankings_test

def save_gifgif_happy_data(n_fold, d, dir='gifgif_happy_', partition='per_obs', n_gif = 1000):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param d: number of training samples
    :param dir: current directory to read features and labels
    :return: feature matrix, rankings_train, rankings_test
    '''
    input_shape = (3,224,224)
    p = 50
    pca = PCA(n_components=p, whiten=True)
    # First pass over the data to transform GIFGIF HAPPINESS IDs to consecutive integers.
    image_ids = set([])
    with open('../data/' + dir + 'data/'  + 'gifgif-dataset-20150121-v1.csv') as f:
        next(f)  # First line is header.
        for line in f:
            emotion, left, right, choice = line.strip().split(",")
            if len(left) > 0 and len(right) > 0 and (emotion == 'happiness' or emotion == 'sadness') and \
                exists('../data/' + dir + 'data/' + 'images/' + left + '.gif') and \
                                exists('../data/' + dir + 'data/' + 'images/' + right + '.gif'):
                    image_ids.add(left)
                    image_ids.add(right)
            # take n_gif images
            if len(image_ids) >= n_gif:
                image_ids = list(image_ids)[:n_gif]
                break
    int_to_idx = dict(enumerate(image_ids))
    idx_to_int = dict((v, k) for k, v in int_to_idx.items())
    # create googlenet base network
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    base_net.load_weights('../data/' + dir + 'data/' + 'googlenet_weights.h5', by_name=True)
    base_net.compile(loss='mean_squared_error', optimizer='sgd')
    # load images and googlenet features
    gg_feat = np.zeros((n_gif, 1024), dtype=float)
    for image_id, i in idx_to_int.items():
        # load
        image_mtx = img_to_array(load_img('../data/' + dir + 'data/' + 'images/' + image_id + '.gif')).astype(np.uint8)
        # resize
        image_mtx = np.reshape(imresize(image_mtx, input_shape[1:]), input_shape)[np.newaxis, :,:,:]
        # take googlenet features
        gg_feat[i, :] = base_net.predict(image_mtx)
    # take features with PCA
    X = pca.fit_transform(gg_feat)
    # take comparisons of images in image_ids
    rankings_all = []
    with open('../data/' + dir + 'data/' + 'gifgif-dataset-20150121-v1.csv') as f:
        next(f)  # First line is header.
        for line in f:
            emotion, left, right, choice = line.strip().split(",")
            if left in image_ids and right in image_ids:
                if emotion == 'happiness':  # left is happier
                    # Map ids to integers.
                    left = idx_to_int[left]
                    right = idx_to_int[right]
                    if choice == "left":
                        # Left image won the happiness comparison.
                        rankings_all.append((left, right))
                    elif choice == "right":
                        # Right image won the happiness comparison.
                        rankings_all.append((right, left))
                elif emotion == 'sadness':  # right is happier
                    # Map ids to integers.
                    left = idx_to_int[left]
                    right = idx_to_int[right]
                    if choice == "right":
                        # Left image won the sadness comparison.
                        rankings_all.append((left, right))
                    elif choice == "left":
                        # Right image won the sadness comparison.
                        rankings_all.append((right, left))
    rankings_all = np.array(rankings_all)
    if partition == 'per_samp':
        rankings_train, rankings_test = create_partitions_wrt_sample(rankings_all, X.shape[0], n_fold, d)
    else:
        rankings_train, rankings_test = create_partitions(rankings_all, n_fold, d)
    ##########################################################################save
    for test_fold in range(n_fold):
        print(test_fold, 'folds are generated')
        save_name = str(test_fold) + '_d_' + str(d)
        np.save('../data/' + dir + 'data/' + save_name + '_features', X)
        np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
        np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
        n = X.shape[0]
        mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
        save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
    return X, rankings_train, rankings_test

def save_fac_data(n_fold, d, dir='fac_', partition='per_obs', n_img = 1000):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param d: number of training samples
    :param dir: current directory to read features and labels
    :return: feature matrix, rankings_train, rankings_test
    '''
    input_shape = (3,224,224)
    p = 50
    pca = PCA(n_components=p, whiten=True)
    comp_label_file = "/pairwise_comparison.pkl"
    with open('../data/' + dir + 'data/' + comp_label_file, 'rb') as f:
        comp_label_matrix = pickle.load(f)
    image_ids = set([])
    # get all unique images in category
    for row in comp_label_matrix:
        # category, f1, f2, workerID, passDup, imgId, ans
        if row['category'] == 0:
            left = row['f1'] + '/' + row['imgId'] + '.jpg'
            right = row['f2'] + '/' + row['imgId'] + '.jpg'
            if exists('../data/' + dir + 'data/' + left) and exists('../data/' + dir + 'data/' + right):
                image_ids.add(left)
                image_ids.add(right)
        # take n_img images
        if len(image_ids) >= n_img:
            image_ids = list(image_ids)[:n_img]
            break
    int_to_idx = dict(enumerate(image_ids))
    idx_to_int = dict((v, k) for k, v in int_to_idx.items())
    # create googlenet base network
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    feature1, _ = create_googlenet(input1, input2)
    base_net = Model(input1, feature1)
    base_net.load_weights('../data/' + dir + 'data/' + 'googlenet_weights.h5', by_name=True)
    base_net.compile(loss='mean_squared_error', optimizer='sgd')
    # load images and googlenet features
    gg_feat = np.zeros((n_img, 1024), dtype=float)
    for image_id, i in idx_to_int.items():
        # load
        image_mtx = img_to_array(load_img('../data/' + dir + 'data/' + image_id)).astype(np.uint8)
        # resize
        image_mtx = np.reshape(imresize(image_mtx, input_shape[1:]), input_shape)[np.newaxis, :, :, :]
        # take googlenet features
        gg_feat[i, :] = base_net.predict(image_mtx)
    # take features with PCA
    X = pca.fit_transform(gg_feat)
    # take comparisons of images in image_ids
    rankings_all = []
    for row in comp_label_matrix:
        # category, f1, f2, workerID, passDup, imgId, ans
        if row['category'] == 0:
            left = row['f1'] + '/' + row['imgId'] + '.jpg'
            right = row['f2'] + '/' + row['imgId'] + '.jpg'
            choice = row['ans']
            if left in image_ids and right in image_ids:
                # Map ids to integers.
                left = idx_to_int[left]
                right = idx_to_int[right]
                if choice == "left":
                    # Left image won the comparison.
                    rankings_all.append((left, right))
                elif choice == "right":
                    # Right image won the comparison.
                    rankings_all.append((right, left))
    rankings_all = np.array(rankings_all)
    if partition == 'per_samp':
        rankings_train, rankings_test = create_partitions_wrt_sample(rankings_all, X.shape[0], n_fold, d)
    else:
        rankings_train, rankings_test = create_partitions(rankings_all, n_fold, d)
    ##########################################################################save
    for test_fold in range(n_fold):
        print(test_fold, 'folds are generated')
        save_name = str(test_fold) + '_d_' + str(d)
        np.save('../data/' + dir + 'data/' + save_name + '_features', X)
        np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
        np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
        n = X.shape[0]
        mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
        save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
    return X, rankings_train, rankings_test

def save_rop_data(n_fold, d, dir='rop_', partition='per_obs', n_img = 100):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param d: number of training samples
    :param dir: current directory to read features and labels
    :return: feature matrix, rankings_train, rankings_test
    '''
    # read features
    df = pd.read_excel('../data/' + dir + 'data/' + '100Features.xlsx')
    image_names = df.as_matrix()[:n_img, 0]
    image_names = np.array([name[:-4] for name in image_names])  # delete extension
    X = df.as_matrix()[:n_img, 1:144]
    # load all comparisons
    with open('../data/' + dir + 'data/' + 'Partitions.p', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        label_cmp = u.load()['cmpData']  # (expert,pair_index,label)
    # Number of comparisons per expert
    M_per_expert = len(label_cmp[0])
    rankings_all = []
    for expert in range(5):
        for pair_ind in range(M_per_expert):
            item1 = np.where(image_names == label_cmp[expert][pair_ind][0])[0]
            item2 = np.where(image_names == label_cmp[expert][pair_ind][1])[0]
            if item1 != np.empty((1,)) and item2 != np.empty((1,)):
                item1 = np.asscalar(item1)
                item2 = np.asscalar(item2)
                if label_cmp[expert][pair_ind][2] == 1:
                    rankings_all.append((item1, item2))
                else:
                    rankings_all.append((item2, item1))
    rankings_all = np.array(rankings_all)
    if partition == 'per_samp':
        rankings_train, rankings_test = create_partitions_wrt_sample(rankings_all, X.shape[0], n_fold, d)
    else:
        rankings_train, rankings_test = create_partitions(rankings_all, n_fold, d)
    ##########################################################################save
    for test_fold in range(n_fold):
        print(test_fold, 'folds are generated')
        save_name = str(test_fold) + '_d_' + str(d)
        np.save('../data/' + dir + 'data/' + save_name + '_features', X)
        np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
        np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
        n = X.shape[0]
        mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
        save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
    return X, rankings_train, rankings_test

def save_sushi_data(n_fold, d, A_l=2, dir='sushi_', partition='per_obs'):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param d: number of training samples
    :param dir: current directory to read features and labels
    :return: feature matrix, rankings_train, rankings_test
    '''
    n_sushi = 100
    n_labeler = 10
    # load the feature matrix
    sushi_i = np.genfromtxt('../data/' + dir + 'data/' + 'sushi3.idata')
    num_sushi = sushi_i.shape[0]
    feature_matrix = np.zeros([num_sushi, 6])
    feature_matrix[:, 0:2] = sushi_i[:, 2:4]
    feature_matrix[:, 2:6] = sushi_i[:, 5:9]
    # convert the categorical feature to binary
    minor_group_matrix = np.zeros([num_sushi, 12])
    for row in range(num_sushi):
        minor_group = int(sushi_i[row, 4])
        minor_group_matrix[row, minor_group] = 1
    X = np.concatenate((feature_matrix, minor_group_matrix), axis=1)
    # load the total rankings
    sushi_order_file = np.genfromtxt('../data/' + dir + 'data/' + 'sushi3b.5000.10.order')
    all_unique_rankings = []
    for ind_labeler in range(n_labeler):
        rankings = np.array(sushi_order_file)[ind_labeler, 2:].astype(int)
        all_unique_rankings.extend(list(combinations(rankings, A_l)))  # get all rankings of size A_l
    all_unique_rankings = np.array(all_unique_rankings)
    if partition == 'per_samp':
        rankings_train, rankings_test = create_partitions_wrt_sample(all_unique_rankings, X.shape[0], n_fold, d)
    else:
        rankings_train, rankings_test = create_partitions(all_unique_rankings, n_fold, d)
    ##########################################################################save
    for test_fold in range(n_fold):
        print(test_fold, 'folds are generated')
        save_name = str(test_fold) + '_d_' + str(d)
        np.save('../data/' + dir + 'data/' + save_name + '_features', X)
        np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
        np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
        n = X.shape[0]
        mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
        save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
    return X, rankings_train, rankings_test

def save_netflix_data(n_fold, d, dir='netflix_', partition='per_obs'):
    '''
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param d: number of training samples
    :param dir: current directory to read features and labels
    :param per_obs or per_samp
    :return: feature matrix, rankings_train, rankings_test
    '''
    n_movies = 1000
    n_labeler = 10
    # load the feature matrix
    with open('../data/' + dir + 'data/' + 'NetflixFeature.p', "rb") as pickle_in:
        data_file = pickle.load(pickle_in)
    X = data_file.pop('Feature')[:n_movies]
    labelers = list(data_file.keys())
    all_unique_rankings = []
    for ind_labeler in range(n_labeler):
        labeler = labelers[ind_labeler]
        # (movie, score) pairs
        movies_labeler = np.array(data_file[labeler]).astype(int)
        # Take the ranking for movies in the subset
        movies_labeler = movies_labeler[np.where(np.isin(movies_labeler[:, 0], range(n_movies)))[0], :]
        # Group movies by scores
        movies_from_5_to_1 = []
        movies_from_5_to_1.append(list(movies_labeler[np.where(movies_labeler[:, 1] == 5)[0], 0]))
        movies_from_5_to_1.append(list(movies_labeler[np.where(movies_labeler[:, 1] == 4)[0], 0]))
        movies_from_5_to_1.append(list(movies_labeler[np.where(movies_labeler[:, 1] == 3)[0], 0]))
        movies_from_5_to_1.append(list(movies_labeler[np.where(movies_labeler[:, 1] == 2)[0], 0]))
        movies_from_5_to_1.append(list(movies_labeler[np.where(movies_labeler[:, 1] == 1)[0], 0]))
        # for each user, load all unique triplet rankings
        for first in np.arange(0, 3):
            for second in np.arange(first+1, 4):
                for third in np.arange(second+1, 5):
                    ranked_movies = []
                    ranked_movies.append(movies_from_5_to_1[first])
                    ranked_movies.append(movies_from_5_to_1[second])
                    ranked_movies.append(movies_from_5_to_1[third])
                    all_unique_rankings.extend(list(product(*ranked_movies)))
    all_unique_rankings = np.array(all_unique_rankings)
    if partition == 'per_samp':
        rankings_train, rankings_test = create_partitions_wrt_sample(all_unique_rankings, X.shape[0], n_fold, d)
    else:
        rankings_train, rankings_test = create_partitions(all_unique_rankings, n_fold, d)
    ##########################################################################save
    for test_fold in range(n_fold):
        print(test_fold, 'folds are generated')
        save_name = str(test_fold) + '_d_' + str(d)
        np.save('../data/' + dir + 'data/' + save_name + '_features', X)
        np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
        np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
        n = X.shape[0]
        mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
        save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
    return X, rankings_train, rankings_test

def save_mslr_data(n_fold, d=None, dir='mslr_'):
    '''
    TRAIN PER QUERY, FEATURES DEPEND ON QUERY
    each row in data is a query-url pair.
    The first column: relevance label of the pair,
    the second column: query id
    the following columns: features for pairs
    n: number of items
    p: feature dimension
    X: n*p, feature matrix
    :param n_fold: number of cross validation folds
    :param d: number of training samples
    :param dir: current directory to read features and labels
    :return: feature matrix, rankings_train, rankings_test
    '''
    n_url = 400
    n_labeler = 10
    ind_labeler = 0
    total_label = 0
    with open('../data/' + dir + 'data/' + 'Fold1/test.txt') as myfile:
        data1 = myfile.readlines()
    # separate each entry
    data1 = np.array([line.split(' ') for line in data1])[:,:-1]
    with open('../data/' + dir + 'data/' + 'Fold1/vali.txt') as myfile:
        data2 = myfile.readlines()
    # separate each entry
    data2 = np.array([line.split(' ') for line in data2])[:,:-1]
    with open('../data/' + dir + 'data/' + 'Fold1/train.txt') as myfile:
        data3 = myfile.readlines()
    # separate each entry
    data3 = np.array([line.split(' ') for line in data3])[:,:-1]
    data_file = np.concatenate((data1, data2, data3))
    # get all queries
    labelers = np.unique(data_file[:, 1])
    for labeler in labelers:
        all_unique_rankings = []
        # get score for each query-url pair
        scores_labeler = data_file[data_file[:, 1] == labeler, 0].astype(float)
        if len(scores_labeler) >= n_url:
            # get features for each query-url pair
            X = np.array([float(elm[elm.index(':') + 1:]) for line in data_file[data_file[:, 1] == labeler] for elm in line[2:]])
            # reshape and boost diagonals for inversion
            X = X.reshape((len(scores_labeler), 136)) + np.eye(len(scores_labeler), 136) * rtol
            # Group urls by scores
            url_from_5_to_1 = []
            url_from_5_to_1.append(list(np.where(scores_labeler == 4)[0]))
            url_from_5_to_1.append(list(np.where(scores_labeler == 3)[0]))
            url_from_5_to_1.append(list(np.where(scores_labeler == 2)[0]))
            url_from_5_to_1.append(list(np.where(scores_labeler == 1)[0]))
            url_from_5_to_1.append(list(np.where(scores_labeler == 0)[0]))
            # for each user, load all unique triplet rankings
            for first in np.arange(0, 3):
                for second in np.arange(first + 1, 4):
                    for third in np.arange(second + 1, 5):
                        ranked_urls = []
                        ranked_urls.append(url_from_5_to_1[first])
                        ranked_urls.append(url_from_5_to_1[second])
                        ranked_urls.append(url_from_5_to_1[third])
                        all_unique_rankings.extend(list(product(*ranked_urls)))
            all_unique_rankings = np.array(all_unique_rankings)
            # partition into cv folds
            rankings_train, rankings_test = create_partitions(all_unique_rankings, n_fold, len(all_unique_rankings))
            ##########################################################################save
            for test_fold in range(n_fold):
                save_name = str(test_fold) + '_d_' + str(ind_labeler)  # no need to downsample in this case
                np.save('../data/' + dir + 'data/' + save_name + '_features', X)
                np.save('../data/' + dir + 'data/' + save_name + '_train', np.array(rankings_train[test_fold]))
                np.save('../data/' + dir + 'data/' + save_name + '_test', np.array(rankings_test[test_fold]))
                n = X.shape[0]
                mat_Pij = est_Pij(n, np.array(rankings_train[test_fold]))
                save_npz('../data/' + dir + 'data/' + save_name + '_mat_Pij', mat_Pij)
            # number of labels collected so far
            ind_labeler += 1
            total_label += len(all_unique_rankings)
        if ind_labeler >= n_labeler:
            break
    print('MSLR data total number of comparisons', total_label)
    return X, rankings_train, rankings_test

if __name__ == "__main__":
    n_fold = 10  ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser = argparse.ArgumentParser(description='prep', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('d', type=int)
    parser.add_argument('dir', type=str)
    # Dataset
    synthetic = True
    real_data = False
    if synthetic:
        parser.add_argument('n', type=int)
        parser.add_argument('p', type=int)
        parser.add_argument('k', type=int)
        parser.add_argument('rand_iter', type=int)
        args = parser.parse_args()
        rand_iter = args.rand_iter
        n = args.n
        p = args.p
        k = args.k
        d = args.d
        dir = args.dir
        if dir == 'synthetic_':
            save_synthetic_data(n_fold, n, p, k, d, rand_iter, dir=dir)
        elif dir == 'synthetic_par_':
            save_synthetic_data(n_fold, n, p, k, d, rand_iter, dir=dir, partition='per_samp')
    elif real_data:
        args = parser.parse_args()
        d = args.d
        dir = args.dir
        if dir == 'rop_':
            save_rop_data(n_fold, d, dir=dir)
        elif dir == 'fac_':
            save_fac_data(n_fold, d, dir=dir)
        elif dir == 'gifgif_happy_':
            save_gifgif_happy_data(n_fold, d, dir=dir)
        elif dir == 'sushi_pair_':
            save_sushi_data(n_fold, d, A_l=2, dir=dir)
        elif dir == 'sushi_triplet_':
            save_sushi_data(n_fold, d, A_l=3, dir=dir)
        elif dir == 'sushi_pair_3fold_':
            n_fold = 3
            save_sushi_data(n_fold, d, A_l=2, dir=dir)
        elif dir == 'rop_par_':
            save_rop_data(n_fold, d, dir=dir, partition='per_samp')
        elif dir == 'sushi_pair_par_3fold_':
            n_fold = 3
            save_sushi_data(n_fold, d, A_l=2, dir=dir, partition='per_samp')
        elif dir == 'sushi_triplet_par_':
            n_fold = 3
            save_sushi_data(n_fold, d, A_l=3, dir=dir, partition='per_samp')
        elif dir == 'gifgif_happy_par_':
            save_gifgif_happy_data(n_fold, d, dir=dir, partition='per_samp')
        elif dir == 'fac_par_':
            save_fac_data(n_fold, d, dir=dir, partition='per_samp')
        if dir == 'rop_n_':
            save_rop_data(n_fold, 50, dir=dir, n_img=d)
        elif dir == 'fac_n_':
            save_fac_data(n_fold, 250, dir=dir, n_img=d)
        elif dir == 'gifgif_happy_n_':
            save_gifgif_happy_data(n_fold, 250, dir=dir, n_gif=d)


