
!mkdir data
!cd data
!pip install kaggle-cli
!kg datasets download ['georgekapoya'/'sofia-air-quality-dataset']

!apt install unzip

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os


!unzip archive


data = pd.read_csv('archive/2017-07_bme280sof.csv')

data.head()