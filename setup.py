import os
import pandas as pd
import numpy as np
import pip
import glob
from zipfile import ZipFile
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

with ZipFile('dataset.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()


def get_fnames(all_files):
    file_names = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        file_names.append(df)
    return file_names
