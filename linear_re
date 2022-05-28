import numpy as np
from sklearn.dummy import DummyRegressor
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.1, random_state=13)

#baseline
model = DummyRegressor(strategy='mean')
model.fit(X_train, y_train)
#dummy_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
#rmse
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Dummy RMSE:', rmse)

#LinearRegression
lin_reg = LinearRegression()
# fit on the training data
lin_reg.fit(X_train, y_train)
# make predictions on the test set
y_pred = lin_reg.predict(X_test)
# calculate root mean squared error
mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(mse)
print('Lin_reg RMSE:', lin_rmse)