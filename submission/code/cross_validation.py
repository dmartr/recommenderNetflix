import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from sklearn import cross_validation as cv
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import math
from plots import plot_raw_data

from helpers import *
from preprocess_data import *
from SGD import *
from extra_helpers import *

"""Generates a K length array for training and another K length array for testing. 
Every training item contains k-1 random indexes generated, an the correspondient testing item the last one
With this indixes, we can build the matrix needed for the cross validation"""
def build_k_indices(ratings, k_fold=4):
    """build k indices for k-fold."""
    np.random.seed(988)
    row, col = ratings.shape
    indices = np.random.permutation(np.arange(len(valid_ratings.nonzero()[0])))
    division_point = int(len(indices)/k_fold)
    cross_ratings = []
    for k in range(k_fold):
        new_cross_rating = indices[k*division_point:(k+1)*division_point]
        cross_ratings.append(new_cross_rating)

    training = []
    testing = []
    for k in range(k_fold):
        testing.append(cross_ratings[k])
        training_list = np.array([])
        for k_train in range(k_fold):
            if k_train != k:
                training_list = np.append(training_list, cross_ratings[k_train])
        training.append(training_list)
    return training, testing

"""Get the indices of build_k_indices and build the matrices with the values of the non-zero ratings"""
def cross_validation(ratings, training, testing):
    data = ratings.nonzero()
    data_values = ratings.todense()
    cross_training = []
    cross_testing = []
    for tr in training:
        training_matrix = np.zeros(ratings.shape)
        for i in tr:
            row = data[0][i]
            col = data[1][i]
            training_matrix[row, col] = data_values[row, col]
        cross_training.append(training_matrix)
    for te in testing:
        testing_matrix = np.zeros(ratings.shape)
        for i in te:
            row = data[0][i]
            col = data[1][i]
            testing_matrix[row, col] = data_values[row, col]
        cross_testing.append(testing_matrix)    
    return np.asarray(cross_training), np.asarray(cross_testing)

"""Normalize all the training sets (in order to avoid doing it in every iteration) """
def normalize_cross_validation(training):
    normalized = []
    means = []
    for k in range(len(training)):
        print("Normalizing", k+1, 'of', len(training))
        norm_train, mean = normalize(sp.lil_matrix(training[k]))
        normalized.append(norm_train)
        means.append(mean)
    return normalized, means

"""Once you have all th edifferent training and test sets, perform the SGD with each one of them to get the best gamma value"""
def cross_validation_SGD(training, testing, item_bias, user_bias, normalized, means):
    """matrix factorization by SGD."""
    # define parameters
    gammas = [0.0001, 0.0005, 0.001, 0.005]
    num_features = 10   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.1
    max_it = 10
    rmse_list_train = []
    rmse_list_test = []
    # set seed
    np.random.seed(988)
    for gamma in gammas:
        rmse_list_train_local = []
        rmse_list_test_local = []
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("Gamma:", gamma)
        for k in range(len(training)):
            it = 0
            train = sp.lil_matrix(training[k])
            test = sp.lil_matrix(testing[k])
            norm_train = normalized[k]
            mean = means[k]
            train_lil = sp.lil_matrix(norm_train)

            test_dense = sp.lil_matrix.todense(test)

            # init matrix
            user_features, item_features = init_SVD(train_dense, num_features) #user Z N na K item W d na K

            # find the non-zero ratings indices 
            nz_row, nz_col = train.nonzero()
            nz_train = list(zip(nz_row, nz_col))
            nz_row, nz_col = test.nonzero()
            nz_test = list(zip(nz_row, nz_col))

            nonzero_rows_train = np.where(norm_train!=0)[0]
            nonzero_cols_train = np.where(norm_train!=0)[1]
            nonzero_rows_test = np.where(test_dense!=0)[0]
            nonzero_cols_test = np.where(test_dense!=0)[1]

            while (it<max_it): 

                it += 1
                np.random.shuffle(nz_train)
                predict = np.dot(user_features, item_features)

                gamma /= 1.2
                nb = int(len(nz_train)/2)
                for d, n in nz_train[:nb]: 
                    predict_error=norm_train[d,n]-predict[d,n]

                    item_features[:,n]+=gamma*(predict_error*user_features[d,:].T-lambda_item*item_features[:,n])
                    user_features[d,:]+=gamma*(predict_error*item_features[:,n].T-lambda_user*user_features[d,:])

                predict = np.dot(user_features, item_features)
                rmse_train = compute_rmse(norm_train, predict, nonzero_rows_train, nonzero_cols_train)
                #print("It", it, "training error:", rmse_train)
            rmse_test = compute_rmse(test, denormalize(predict, mean), nonzero_rows_test, nonzero_cols_test)
            print("-- CV", k+1, '/', len(training), '--- Training error:', rmse_train)
            print("             Testing error:", rmse_test)
            rmse_list_train_local.append(rmse_train)
            rmse_list_test_local.append(rmse_test)
            #predictions_local.append(np.dot(user_features, item_features))
        rmse_train_final = np.mean(rmse_list_train_local)
        rmse_test_final = np.mean(rmse_list_test_local)
        print("--- Gamma", gamma, "-------------------------")
        print("--- Training error:", rmse_train_final)
        print("--- Testing error:", rmse_test_final)
        print("---------------------------------------")
        rmse_list_train.append(rmse_train_final)
        rmse_list_test.append(rmse_test_final)
    return rmse_list_train, rmse_list_test