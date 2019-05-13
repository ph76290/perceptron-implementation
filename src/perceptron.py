# Make a prediction with a given set of weights
# The format used for the weights row is: [bias, weights...]
import os
import pandas as pd


# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train_set, train_set_labels, test_set, l_rate, n_epoch):
	weights, bias = train_weights(train_set, train_set_labels, l_rate, n_epoch)
	predictions = [predict(row, weights, bias) for rowIndex, row in test_set.iterrows()]
	return predictions


# Make a prediction with weights
def predict(row, weights, activation):
	# Activation == bias
	activation += weights.dot(row)
	return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train_set, train_set_labels, l_rate, n_epoch):
	weights, bias = pd.Series(0.0, index=range(train_set.shape[1])), 1
	print("Training weights ...")
	for epoch in range(n_epoch):
		for rowIndex, row in train_set.iterrows():
			prediction = predict(row, weights, bias)
			error = train_set_labels.iloc[rowIndex, 0] - prediction
			bias += l_rate * error
			weights = weights.add(row * l_rate * error)
	print("done!")
	return (weights, bias)
