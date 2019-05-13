# File containing a function to split the dataframe into folds
# in order to apply cross validation


# Split a dataset into k folds
def cross_validation_split(dataframe, n_folds):
	dataframe_split = list()
	dataframe = dataframe.sample(n=dataframe.shape[0])
	dataframe.index = range(0, dataframe.shape[0])
	fold_size = int(dataframe.shape[0] / n_folds)
	for i in range(n_folds):
		new_fold = dataframe.loc[i * fold_size:(i + 1) * fold_size - 1, :]
		# Get all the rows split into fold indexes and all the columns
		dataframe_split.append(new_fold)
	return dataframe_split
