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
import setup

path = r'dataset' 

all_files = glob.glob(path + "/*_bme280sof.csv")
#we join our csv files
file_names = setup.get_fnames(all_files)
df = pd.concat(file_names, axis=0, ignore_index=True, sort=True)

print(df.size)
print(df.head().to_string())

#Removing annoying values that are making us to get bad score.
df = df[(df['temperature']>0.01)]
df = df[(df['humidity']>0.01)]
df = df[(df['humidity']<99.99)]
#Preprocess (drop nulls and sort)
df.dropna(subset=["pressure","humidity", 'temperature'])
df.sort_values(by=['timestamp'])


#Removing the outliers of the humidity
df['z_score_hum']=stats.zscore(df['humidity'])
df = df.loc[df['z_score_hum'].abs()<=3]

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

fig, ax = plt.subplots(figsize=(16,10))
plt.plot(x_test,y_test,".",label="temperature")
plt.plot(x_test,prediction,label="Score="+str(score))
plt.legend()
plt.show()

