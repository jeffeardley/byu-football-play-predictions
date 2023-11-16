import csv
import pandas as pd


def readData():
    # Make an array of the DataFrames made from all the CSV files
    dataFrames = []
    # Make an array of the names of the CSV files
    for i in range(1, 10):
        dataFrames.append(pd.read_csv('play-data/2022_wk' + str(i) + '.csv'))

    return dataFrames
