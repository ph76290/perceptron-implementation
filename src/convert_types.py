# Functions to convert column from the dataset into the right type
# These functions are used because we only want to work with numbers


# Convert string column to integer
def str_column_to_int(serie, column):
        unique = set(serie)
        lookup = dict()
        for i, value in enumerate(unique):
                lookup[value] = i
        serie = serie.apply(lambda x:lookup[x])
        return serie
