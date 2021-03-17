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

path = r'dataset' 
all_files = glob.glob(path + "/*_bme280sof.csv")


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
#Preprocess (drop nulls and sort)
df.dropna(subset=["pressure","humidity", 'temperature'])
df.sort_values(by=['timestamp'])
df = df[df.sensor_id == 3096]

X_train, x_test, Y_train, y_test = train_test_split(df[['temperature']], df[['humidity']], test_size=0.2)


#train
reg_model = LinearRegression()
reg_model.fit(X_train, Y_train)
score = reg_model.score(X_train, Y_train)

#Predict the PM2 concentration 
prediction = reg_model.predict(x_test)

#Print  score
print("Score: ", score)

# The coefficients
print('Coefficients: \n', reg_model.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, prediction))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, prediction))


plt.plot(x_test,y_test,".",label="temperature")
plt.plot(x_test,prediction,label="Score="+str(score))
plt.legend()
plt.show()


