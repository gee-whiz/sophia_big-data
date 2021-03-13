import os
import pandas as pd
import numpy as np
import pip
import glob
from zipfile import ZipFile


with ZipFile('dataset.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

path = r'dataset' 
all_files = glob.glob(path + "/*.csv")


def get_fnames():
    file_names = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        file_names.append(df)
    return file_names

#we join our csv files
file_names = get_fnames()
df = pd.concat(file_names, axis=0, ignore_index=True, sort=True)

print(df.size)
print(df.head().to_string())