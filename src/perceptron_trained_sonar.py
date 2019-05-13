# Perceptron Algorithm on the Sonar Dataset
from random import randrange
from metrics_tools import accuracy_metric
from perceptron import perceptron, predict, train_weights
from cross_validation_split import cross_validation_split
import os
import pandas as pd


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataframe, algorithm, n_folds, *args):
	folds = cross_validation_split(dataframe, n_folds)
	scores = list()

	for foldIndex in range(len(folds)):
		# Building the train set
		train = list(folds)
		train.pop(foldIndex)
		train_set = pd.DataFrame()
		for i in range(len(train)):
			train_set = train_set.append(train[i], ignore_index=True)

		# Building the test set
		test_set = folds[foldIndex].copy(deep=True)
		test_set.drop(test_set.columns[test_set.shape[1] - 1], axis=1, inplace=True)

		train_set_labels = train_set.iloc[:, -1:]
		train_set.drop(train_set.columns[train_set.shape[1] - 1], axis=1, inplace=True)
		predicted = algorithm(train_set, train_set_labels, test_set, *args)
		labels = folds[foldIndex].iloc[:, -1]
		scores.append(accuracy_metric(labels, predicted))

	return scores
