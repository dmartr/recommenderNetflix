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
from extra_helpers import *


"""Normalize the matrix by item mean. For every non-zero element, substract the mean of the corresponding item"""
def normalize(data_set):
    normalized_set = np.zeros(data_set.shape)
    means = np.zeros(data_set.shape[1])
    for i in range(data_set.shape[1]):
        movie_ratings = np.array(data_set[:,i].toarray())
        n_ratings = len(np.where(movie_ratings>0)[0])
        ratings_mean = np.sum(movie_ratings)/n_ratings
        means[i] = ratings_mean
        movie_ratings[np.where(movie_ratings!=0)] -= (ratings_mean)
        normalized_set[:,i] = movie_ratings[:,0]
    return normalized_set, means

"""Inverse process of the normalize function. For every non-zero element, add the mean of the corresponding item"""
def denormalize(data_set, means): 
    train = np.zeros(data_set.shape)
    
    for i in range(data_set.shape[1]):
        train[:,i] = (data_set[:,i] + means[i]).ravel()
    return train


"""Split the data into training and testing given a probability."""
def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
   Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(988)
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0].tolist()
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0].tolist()
    valid_ratings = ratings[valid_users, :][: , valid_items]  
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data and return train and test data. TODO
    # NOTE: we only consider users and movies that have more
    # than 10 ratings
    # ***************************************************
    train = valid_ratings.copy()
    test = sp.lil_matrix((valid_ratings.shape[0], valid_ratings.shape[1])) 
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            v = train[i, j]
            if v != 0:
                p = np.random.random()
                if(p < p_test):
                    train[i, j] = 0
                    test[i, j] = v
    print("Total number of nonzero elements in original data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in valid data:{v}".format(v=valid_ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test, valid_users, valid_items