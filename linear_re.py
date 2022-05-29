from tkinter.font import names
import numpy as np
from sklearn.dummy import DummyRegressor
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
names = diabetes.feature_names

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.1, random_state=13)

########
#BASELINE
########
model = DummyRegressor(strategy='mean')
model.fit(X_train, y_train)
#dummy_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)

#rmse
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Dummy RMSE:', rmse)
#squared
r_squared = r2_score(y_test, y_pred)
print('R-squared:', r_squared)

########
#LinearRegression
########
lin_reg = LinearRegression()
# fit on the training data
lin_reg.fit(X_train, y_train) #parameters
# make predictions on the test set
y_pred = lin_reg.predict(X_test)

# calculate root mean squared error
mse = mean_squared_error(y_test, y_pred)
lin_rmse = np.sqrt(mse)
print('Lin_reg RMSE:', lin_rmse)
#squared
r_squared_lin = r2_score(y_test, y_pred)
print('R-squared_lin:', r_squared_lin )

########
#LASSO
########
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
# Create lasso regression with alpha value
regr = Lasso(alpha=0.5)
# Fit the lasso regression
model_lasso = regr.fit(X_std, y)
y_pred = model_lasso.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
lasso_rmse = np.sqrt(mse)
print('Lasso RMSE:', lasso_rmse)
#squared
r_squared_lasso = r2_score(y_test, y_pred)
print('R-squared_lasso:', r_squared_lasso )

alphas = np.logspace(-4,0,50)


#Effect Of Alpha On Lasso Regression
def lasso(alphas):
     # Create an empty data frame
    df = pd.DataFrame()
    
    # Create a column of feature names
    df['Feature Name'] = names
    coefs = []
    alphas = np.logspace(-4,0,50)

    # For each alpha value in the list of alpha values,
    for alpha in alphas:
        # Create a lasso regression with that alpha value,
        lasso = Lasso(alpha=alpha)
        
        # Fit the lasso regression
        lasso.fit(X, y)
        
        # Create a column name for that alpha value
        column_name = 'Alpha = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = lasso.coef_
        coefs.append(lasso.coef_)
        
    # Return the dataframe    
    print(df)

#     plt.plot(alpha, cdf[column_name], label = "alpha")
#     #plt.plot(alpha, acc_train, label = "Train")
#     plt.title("Lasso - score vs alpha")
#     plt.legend()
#     plt.xlabel("alpha")
#     plt.ylabel("value")
#     plt.show()

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Lasso coefficients as a function of alpha')
    plt.show()


print('lasso', lasso([.0001, .5, 10]))




########
#Ridge
########
# Create ridge regression with an alpha value
regr = Ridge(alpha=0.5)
# Fit the linear regression
model_ridge = regr.fit(X_std, y)
y_pred = model_ridge.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
ridge_rmse = np.sqrt(mse)
print('Ridge RMSE:', ridge_rmse)
#squared
r_squared_ridge = r2_score(y_test, y_pred)
print('R-squared_ridge:', r_squared_ridge )

#Effect Of Alpha On Ridge Regression
def ridge(alphas):
     # Create an empty data frame
    df = pd.DataFrame()
    
    # Create a column of feature names
    df['Feature Name'] = names
    coefs = []
    alphas = np.logspace(-4,0,50)

    # For each alpha value in the list of alpha values,
    for alpha in np.logspace(-4, 0, 50):
        # Create a ridge regression with that alpha value,
        ridge = Ridge(alpha=alpha)
        
        # Fit the ridge regression
        ridge.fit(X, y)
        
        # Create a column name for that alpha value
        column_name = 'Alpha = %f' % alpha

        # Create a column of coefficient values
        df[column_name] = ridge.coef_
        coefs.append(ridge.coef_)
        
    # Return the datafram    
    print(df)
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('alpha')
    plt.ylabel('Standardized Coefficients')
    plt.title('Ridge coefficients as a function of alpha')
    plt.show()

print('ridge', ridge([.0001, .5, 10]))


