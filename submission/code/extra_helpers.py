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

"""Compute the RMSE error for a given dataset and the predictions we want to compare. Also, specify which indees are not zero in the array (only comparing those values"""
def compute_rmse(dataset, predictions, nonzero_rows, nonzero_cols):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    #nz_count = dataset.nnz
    nz_count = len(nonzero_rows)
    total = 0
    for i in range(len(nonzero_rows)):
        r = nonzero_rows[i]
        c = nonzero_cols[i]
        total += (predictions[r,c]-dataset[r,c])**2
    return np.sqrt(total/nz_count)

"""For a given prediction, create the csv file with the predictions required by kaggle"""
def submit(prediction, file_name, path_dataset="../data/sampleSubmission.csv"):
    testing = read_txt(path_dataset)[1:]
    rows = []
    cols = []
    preds = []
    for t in range(len(testing)):
        sp = testing[t].split('_c')
        cols.append(int(sp[1][:-2]))
        rows.append(int(sp[0][1:]))
    for i in range(len(cols)):
        preds.append(prediction[rows[i]-1, cols[i]-1])
    create_csv_submission(rows, cols, preds, file_name)