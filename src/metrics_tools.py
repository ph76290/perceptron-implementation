# This file gathers different metrics to visualize the efficiency
# of the machine learning algorithms


# Calculate accuracy percentage
def accuracy_metric(labels, predicted):
        correct = 0
        for i in range(len(labels)):
                if labels.iloc[i] == predicted[i]:
                        correct += 1
        return correct / float(len(labels)) * 100.0
