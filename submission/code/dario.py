
# Useful starting lines
from helpers import *
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


# Loading the training data...
path_dataset = "../data/data_train.csv"
ratings = load_data(path_dataset)
def matrix_factorization_SGD(training, testing, item_bias, user_bias, normalized, means):
    """matrix factorization by SGD."""
    # define parameters
    gammas = 0.0005
    num_features = [5, 7, 10, 12]   # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.1
    max_it = 10
    rmse_list_train = []
    rmse_list_test = []
    # set seed
    np.random.seed(988)
    for num_feature in num_features:
        rmse_list_train_local = []
        rmse_list_test_local = []
        print("++++++++++++++++++++++++++++++++++++++++++++")
        print("K:", num_feature)
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
        print("--- K", num_feature, "-------------------------")
        print("--- Training error:", rmse_train_final)
        print("--- Testing error:", rmse_test_final)
        print("---------------------------------------")
        rmse_list_train.append(rmse_train_final)
            rmse_list_test.append(rmse_test_final)
        return rmse_list_train, rmse_list_test

training, testing = build_k_indices(ratings, 5)
training, testing = cross_validation(ratings, training, testing)
normalized, means = normalize_cross_validation(training)
rmse_list_train, rmse_list_test = matrix_factorization_SGD(training, testing, item_bias, user_bias, normalized, means)