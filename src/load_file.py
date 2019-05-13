# Load a CSV file
import os
import pandas as pd


def load_csv_into_dataframe(csvFile, headers):
        header = None if len(headers) == 0 else headers
        dataframe = pd.read_csv(csvFile, header=header)
        return dataframe
