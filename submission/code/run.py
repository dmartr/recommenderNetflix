# Useful starting lines
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

# Loading the training data...
path_dataset = "../data/data_train.csv"
print("Loading ratings...")
ratings = load_data(path_dataset)

# Get the number of items per user and users per item
num_items_per_user = np.array((ratings != 0).sum(axis=1)).flatten()
num_users_per_item = np.array((ratings != 0).sum(axis=0).T).flatten()

print("Spliting data...")

# Split the data!
valid_ratings, train, test, valid_users, valid_items = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=0, p_test=0.1)

print("Normalizing training...")
# For every item, we normalize the values 
norm_train, means = normalize(train)

print("Finding MF SGD...")
# We get the prediction for the training data given and the properties
sgd = SGD(norm_train, test, means, gamma=0.001, num_features=10, lambda_user=0.1, lambda_item=0.1, max_it=20)

print("Denormalizing")
# Once we have a prediction we have to denormalize the value 
prediction = denormalize(predictions[-1], means)

# We create the final_submission.csv in ../data
submit(prediction, 'final_submission')