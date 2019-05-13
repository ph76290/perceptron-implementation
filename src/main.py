#!/usr/bin/python3

# Main file which will be executed
# Test the Perceptron algorithm on the sonar dataset
from random import seed
from load_file import load_csv_into_dataframe
from convert_types import str_column_to_int
from perceptron_trained_sonar import evaluate_algorithm
from perceptron import perceptron

 
# Set the seed for the random numbers to set
# In order to keep consistency between results and be able to compare them
seed(1)


# Load and prepare data
filename = '../sonar_dataset/sonar_data.csv'
dataframe = load_csv_into_dataframe(filename, [])
# dataframe is a pandas.DataFrame
datasetLength = len(dataframe[0])

dataframe[60] = str_column_to_int(dataframe[60], datasetLength - 1)


# Evaluate algorithm
# We want to find the best hyper parameters to optimize our results 
best_n_folds = 0
best_n_epoch = 0
best_accuracy = 0.0
l_rate = 0.01

scores = evaluate_algorithm(dataframe, perceptron, 4, l_rate, 10)
print(scores)
# for n_folds in range(4, 5):
#         for n_epoch in range(400, 425, 25):
#                 scores = evaluate_algorithm(dataframe, perceptron, n_folds, l_rate, n_epoch)
#                 accuracy = (sum(scores)/float(len(scores)))
#                 if accuracy > best_accuracy:
#                         best_n_folds = n_folds
#                         best_n_epoch = n_epoch
#                         best_accuracy = accuracy


# Display the best hyper parameters with the respective accuracy
# print('Mean Accuracy: %.3f%%' % best_accuracy)
# print('This accuracy was obtained with %d folds and %d epochs' % (best_n_folds, best_n_epoch))
