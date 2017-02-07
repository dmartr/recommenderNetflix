# Project Recommender System

For this choice of project task, you are supposed to predict good recommendations, e.g. of movies to users. We have acquired ratings of 10000 users for 1000 different items (think of movies). All ratings are integer values between 1 and 5 stars. No additional information is available on the movies or users.

### Evaluation Metric
Your collaborative filtering algorithm is evaluated according to the prediction error, measured by root-mean-squared error (RMSE).

## Instructions:

To get the submission, just run run.py directly. A new .csv file called finalSubmission.csv will be created at the same folder, containing the best Kaggle solution obtained.

## Data Structure

- data
	- **sampleSubmission.csv**: example of submission given by Kaggle. Used to get all the user-item tuples needed for the final submission
	- **data_train.csv**: training data. Matrix with all the user-item ratings used to predict the asked ones

- code
	- **run.py**: Trains the model to generate the best submission updloaded to Kaggle. It splits the data, normalized the training set, uses SGD to factorize the matrices (with the best parameters, obtained by cross validation) and creates the finalSubmission.csv file in the current directory.
	- **preprocess_data.py**: contains the functions used to normalized, denormalize and split the data.
	- **SGD.py**: contains the SGD Matrix Factorization used in the run.py file. 
	- **ALS.py**: contains the ALS Matrix Factorization functions used in the experiments and during the initial stages of the projects
	- **MF_helpers.py**: contains the initialization array features methods used in both ALS and SGD. 
	- **cross_validation.py**: contains the cross validation functions to generate the random indices, build the value matrices and perform the SGD cross validation for gamma and num_features (K) (although it only needs little changes for any other variable)
	- **extra_helpers.py**: compute_rmse own function and submit (generates csv with a prediction)
	- **helpers.py**: functions given by the course. 
	- **plots.py**: functions given by the course.

- report

## Getting Started

By executing run.py you'll be able to generate the best submission of the Kaggle competition. The other .py contain information

## Prerequisites

The whole project uses the functions given by the course, numpy, scipy and sklearn for the cosine similarity. The rest of the methods have been developed completely by the authors.

## Process followed:

In order to achieve the results uploaded to Kaggle, the team has followed the next process:

1. First, developing both the ALS and the SGD Matrix Factorization methods to fully underestand their differences and pros and cons respect each other.
2. Then, after some attempts with both of them, we could see that the best results were coming from the SGD method, so the team decided to focus there and try to improve.
3. As it has been seen in previous projects, cross validation was key to get the best possible parameters for the model. Number of features, learning rate and number of iterations were very important to get better results.
4. All the data was normalized to be able to deal with the sparsity. This way, a zero wasn't an outlier anymore.

## Authors

* Ignacio Aguado
* Anna Kubik
* Darío Martínez
