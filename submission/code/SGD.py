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
from MF_helpers import *

predictions = []

"""Perform the matrix factorization problem using Stochastic Gradient Descent. Starting with a training set, test set, 
the means obtained by the normalization and the properties we want to use.
Returns the predictions for each iteration performed"""
def SGD(train, test, means, gamma=0.0005, num_features=10, lambda_user=0.1, lambda_item=0.1, max_it=10):
    """matrix factorization by SGD."""
    it = 0
    # set seed
    np.random.seed(988)

    train_lil = sp.lil_matrix(train)
    train_dense = sp.lil_matrix.todense(train_lil)

    # init matrix
    user_features, item_features = init_SVD(train_dense, num_features) #user Z N na K item W d na K

    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    nonzero_rows_train = np.where(train_dense!=0)[0]
    nonzero_cols_train = np.where(train_dense!=0)[1]
    nonzero_rows_test = np.where(test.todense()!=0)[0]
    nonzero_cols_test = np.where(test.todense()!=0)[1]

    while (it<max_it): 
    
        it += 1
        np.random.shuffle(nz_train)
        # decrease step size
        gamma /= 1.2
        nb = int(len(nz_train)/2)
        
        predict = np.dot(user_features, item_features)
        for d, n in nz_train[:nb]: 
            # Calculate error on the prediction
            predict_error=train[d,n]-predict[d,n]
            # Update user and item features
            item_features[:,n]+=gamma*(predict_error*user_features[d,:].T-lambda_item*item_features[:,n])
            user_features[d,:]+=gamma*(predict_error*item_features[:,n].T-lambda_user*user_features[d,:])

        # Calculate again the predictino
        predict = np.dot(user_features, item_features)

        # Get the RMSE errors for this prediction and go to the next iteration
        rmse_train = compute_rmse(train, predict, nonzero_rows_train, nonzero_cols_train)
        denormalized = denormalize(predict, means)
        rmse_test = compute_rmse(test, denormalized , nonzero_rows_test, nonzero_cols_test)
        predictions.append(predict)
        print("It", it, "training error:", rmse_train, "testing error:", rmse_test)

    rmse_test = compute_rmse(test, denormalize(predict, means), nonzero_rows_test, nonzero_cols_test)
    print("Testing error:", rmse_test)

    return predictions