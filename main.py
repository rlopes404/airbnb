#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 06:28:50 2020

@author: ramon
"""
#TODO: latitude, longitude,

import pandas as pd
import numpy as np
np.random.seed(0)

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from xgboost import plot_importance
 
from sklearn.ensemble import StackingRegressor


#Previsão do preço da estadia (feature ‘price’)
#Classificação do room type (feature ‘room_type’)

#Faça uma análise exploratória para avaliar a consistência dos dados
#e identificar possíveis variáveis que impactam sua variável resposta

## load data
name = 'data.csv'
data = pd.read_csv(name, sep=',', header=0)
y_name = 'price'

##exploring the dataset
data.head(n=10)

data.shape
#(33715, 16)

data.info()
#name, host_name, neighbourhood, room_type, last_review are not numeric
#missing data: neighbourhood_group, last_review, reviews_per_month
#to keep: neibhourhood, room_type, last_review
#discard: id, name, host_id, host_name

#checking duplicate columns
np.sum(data.duplicated()) == 0

##evaluating for missing data
data.count(axis=0)/len(data)
#neighbourhood_group 0 -> discard
#reviews_per_month 0.56 -> imputation?


#analysis per neighborhourhood
data.groupby('neighbourhood')[y_name].agg('mean').sort_values(ascending=False)
data.groupby('neighbourhood')[y_name].agg('count').sort_values(ascending=False)

#ploting mean price per neighborhoud
y = data.groupby('neighbourhood')[y_name].agg('mean').sort_values(ascending=False).values
x = np.array(range(1, len(y)+1))
plt.plot(x, y, 'ro')

#lets create bins: [0, 250, 500, 1000, 3000, np.inf ] and names = ['b1', 'b2', 'b3', 'b4', 'b5']
bins = [0, 250, 500, 1000, 3000, np.inf]
names= ['B250', 'B500', 'B1k', 'B3k', 'BInf']
neigh_bins = pd.cut(data[y_name], bins, labels=names)

#one-hot encoding
data = pd.concat([data, pd.get_dummies(neigh_bins, prefix='is')], axis=1)


#analysis per room_type
data.groupby('room_type')[y_name].agg('mean').sort_values(ascending=False)
data.groupby('room_type')[y_name].agg('count').sort_values(ascending=False)

#one hot encoding
data = pd.concat([data, pd.get_dummies(data['room_type'], prefix='is')], axis=1)

#droping non-numeric columns
df = data.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 'last_review', 'reviews_per_month', 'neighbourhood', 'room_type', 'latitude', 'longitude'])
df.info()


##  exploring the target variable
df[y_name].describe()
#min price: 0 -> discard
(df[y_name] == 0).sum()

idxs = df[y_name] > 0
df = df[idxs]
#std is high

plt.figure(figsize=(9, 8))
sns.distplot(df[y_name], color='g', bins=100)
#high skewed

df.boxplot(column=[y_name])

#TODO: several outliers -> remove them
#idxs = (np.abs(stats.zscore(df['price'])) < 3)
#df = df[idxs]

df.hist(figsize=(12, 8), bins=50)

### assessing correlation
sns.heatmap(df.corr(),cmap='BrBG',annot=True)

df.corr()[y_name]
#some extracted features show higher correlation coeficient than the raw features

#pairplot
#pair_plot = sns.pairplot(df)
#no others patterns, correlations


X_train, X_test = train_test_split(df, test_size=0.2, random_state=0)
X_train, X_valid = train_test_split(X_train, test_size=0.1, random_state=0)

Y_train = X_train[y_name]
Y_test = X_test[y_name]
Y_valid = X_valid[y_name]

X_train.drop(columns=[y_name], inplace=True)
X_test.drop(columns=[y_name], inplace=True)
X_valid.drop(columns=[y_name], inplace=True)

### Weakest baseline: mean

MSE = metrics.mean_squared_error(Y_test, np.repeat(np.mean(Y_train), len(Y_test)))
print('Mean RMSE: %.2f'%(np.sqrt(MSE)))


### Linear Regression Raw Features
cols = ['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365']
lr_raw = LinearRegression().fit(X_train[cols], Y_train)
r2 = lr_raw.score(X_train[cols], Y_train)
print(r2)
Y_pred = lr_raw.predict(X_test[cols])
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('LR Raw RMSE: %.2f' %(np.sqrt(MSE)))

### KNN
grid_params = {
            'n_neighbors': [1, 5, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute']            
        }


knn = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=grid_params, verbose=1)
grid_results = knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
RMSE_knn = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('KNN RMSE: %.f' %(RMSE_knn))

### Polynomial Linear Regression
results = []
for d in range(2,5):
    poly = PolynomialFeatures(degree=d, interaction_only=True, include_bias = True)
    X_train_poly = poly.fit_transform(X_train)
    X_valid_poly = poly.fit_transform(X_valid)
    
    
    reg_poly =  LinearRegression().fit(X_train_poly, Y_train)
        
    Y_pred = reg_poly.predict(X_valid_poly)
    MSE = metrics.mean_squared_error(Y_valid, Y_pred)
    results.append((np.sqrt(MSE), d))

results.sort()
best_d = results[0][1]

poly = PolynomialFeatures(degree=best_d, interaction_only=True, include_bias = True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

lr_poly =  LinearRegression().fit(X_train_poly, Y_train)
r2 = lr_poly.score(X_train_poly, Y_train)
Y_pred = lr_poly.predict(X_test_poly)
print(r2)
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('Poly RMSE: %.2f' %(np.sqrt(MSE)))

### Improved Linear Regression
lr_improved = LinearRegression().fit(X_train, Y_train)
r2 = lr_improved.score(X_train, Y_train)
print(r2)
Y_pred = lr_improved.predict(X_test)
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('Improved LR RMSE: %.2f' %(np.sqrt(MSE)))


### XGBoost
grid_params = {
            'learning_rate': [0.001, 0.01],
            'n_estimators': [100, 500],
            'max_depth': [5, 10],
            'subsample': [0.5, 1.0],
            'colsample_bytree': [0.5, 1.0],
            'min_child_weight': [1, 10],
            'early_stopping_rounds': [10],
            'gamma': [0, 1]
            }
      
xgb = GridSearchCV(estimator=XGBRegressor(seed=42), cv=2, param_grid=grid_params, verbose=1)
xgb.fit(X_train, Y_train)


#model = XGBRegressor(learning_rate = 0.1, n_estimators=1000, max_depth=3, subsample=0.8, colsample_bytree=1, gamma= 1, seed=42)         
#model.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_train, Y_train), (X_valid, Y_valid)],  verbose=True, early_stopping_rounds = 10)

Y_pred = xgb.predict(X_test)
RMSE_xgb = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('XGB RMSE: %.f' %(RMSE_xgb))

### stacking manual


X_valid_poly = poly.fit_transform(X_valid)
lrp_y_pred = lr_poly.predict(X_valid_poly).reshape(-1,1)
knn_y_pred = knn.predict(X_valid).reshape(-1,1)
xgb_y_pred = xgb.predict(X_valid).reshape(-1,1)

_X_train = np.concatenate([xgb_y_pred, knn_y_pred, lrp_y_pred], axis=1)

_lr = LinearRegression().fit(_X_train, Y_valid)

X_test_poly = poly.fit_transform(X_test)
lrp_y_pred = lr_poly.predict(X_test_poly).reshape(-1,1)
knn_y_pred = knn.predict(X_test).reshape(-1,1)
xgb_y_pred = xgb.predict(X_test).reshape(-1,1)
_X_test = np.concatenate([xgb_y_pred, knn_y_pred, lrp_y_pred], axis=1)


Y_pred = _lr.predict(_X_test)
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('Improved LR RMSE: %.2f' %(np.sqrt(MSE)))

### Stacking
estimators = [('knn', knn), ('lr_poly', lr_poly), ('xgb', xgb)]
reg = StackingRegressor(estimators=estimators)
reg.fit(X_test, Y_test)
Y_pred = reg.predict(X_test)
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('Stacking RMSE: %.2f' %(np.sqrt(MSE)))

### XGBoost Poly
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias = True)
X_train_poly = poly.fit_transform(X_train)
X_valid_poly = poly.fit_transform(X_valid)
X_test_poly = poly.fit_transform(X_test)

grid_params = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'max_depth': [2, 6, 10],
            'subsample': [0.3, 0.7, 1.0],
            'colsample_bytree': [0.3, 0.7, 1.0],
            'min_child_weight': [1, 10, 100],
            'early_stopping_rounds': [10],
            'gamma': [0, 1, 5]
            }
      
grid = GridSearchCV(estimator=XGBRegressor(seed=42), param_grid=grid_params)
grid.fit(X_train_poly, Y_train)


#model = XGBRegressor(learning_rate = 0.1, n_estimators=1000, max_depth=3, subsample=0.8, colsample_bytree=1, gamma= 1, seed=42)         
#model.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_train, Y_train), (X_valid, Y_valid)],  verbose=True, early_stopping_rounds = 10)

Y_pred = grid.predict(X_test_poly)
RMSE_xgb = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE_xgb: %.f' %(RMSE_xgb))



#http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/
#https://stackoverflow.com/questions/51194303/how-to-run-a-python-script-in-a-py-file-from-a-google-colab-notebook
#https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

#https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
