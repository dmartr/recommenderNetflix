# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=1)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=0).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_users_per_movie, color='blue')
    ax1.set_xlabel("items")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_movies_per_user)
    ax2.set_xlabel("users")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, 10000, 2000))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.5, markersize=0.5)
    ax1.set_xlabel("Items")
    ax1.set_ylabel("Users")
    ax1.set_title("Training data")
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 1000)
    ax1.set_aspect('equal')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.5, markersize=0.5)
    ax2.set_xlabel("Items")
    ax2.set_ylabel("Users")
    ax2.set_title("Test data")
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, 1000)
    #plt.tight_layout()
    plt.savefig("train_test")
    #plt.show()
