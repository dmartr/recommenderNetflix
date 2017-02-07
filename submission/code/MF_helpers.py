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
from extra_helpers import *

"""Get the movie with more ratings"""
def get_most_rated_movie(dataset):
    best_movie = -1
    ratings = 0
    for i in range(dataset.shape[1]):
        a = np.array(dataset[:,i].toarray())
        n = len(np.where(a>0)[0])
        if n > ratings:
            ratings = n
            best_movie = i
    return best_movie
"""Get the user with more ratings"""
def get_most_rated_user(dataset):
    best_movie = -1
    ratings = 0
    for i in range(dataset.shape[0]):
        a = np.array(dataset[i,:].toarray())
        n = len(np.where(a>0)[0])
        if n > ratings:
            ratings = n
            best_movie = i
    return best_movie

"""Generation of the initial item features matrix taking into account the pearson correlation with the most rated item """
def generate_item_features(normalized_train, num_features, best_movie):
    item_features = np.zeros([normalized_train.shape[1], num_features])
    for i in range(normalized_train.shape[1]):
        p = pearson_correlation(normalized_train[:,best_movie], normalized_train[:,i])
        if math.isnan(p):
            p=0
        item_features[i,:] = 1 + p
    return item_features.T

"""Generation of the initial user features matrix taking into account the pearson correlation with the user with more ratings """
def generate_user_features(normalized_train, num_features, best_user):
    user_features = np.zeros([normalized_train.shape[0],num_features])
    for i in range(normalized_train.shape[0]):
        p = 1 + pearson_correlation(normalized_train[best_user,:], normalized_train[i,:])
        if math.isnan(p):
            p=0
        user_features[i,:] = p
    return user_features

"""Generation of the initial item and user features matrix with the pearson correlations"""
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    best_user = get_most_rated_user(train)
    best_movie = get_most_rated_movie(train)
    item_features = generate_item_features(train, num_features, best_movie)
    user_features = generate_user_features(train, num_features, best_user)
    return user_features, item_features

"""Generation of the initial item features matrix by random values"""
def init_random(train, num_features):
    """init the parameter for matrix factorization."""
    (num_user, num_item) = train.shape
    user_features = np.random.rand(num_user, num_features)
    item_features = np.random.rand(num_item, num_features)
    return user_features, item_features.T

"""Generation of the initial item features matrix with SVD """
def init_SVD(train, k):
    U, s, V  = np.linalg.svd(train, full_matrices=False)
    S = np.diag(s)
    item_features = U[:, 0:k]
    features_S = S[0:k, 0:k]
    features_V = V[0:k, :]
    user_features = np.dot(features_S, features_V)
    return item_features, user_features

