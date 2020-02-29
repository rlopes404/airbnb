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
from sklearn.linear_model import Ridge

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
#(31757, 27)

data.info()
#name, host_name, neighbourhood, room_type, last_review are not numeric
#missing data: neighbourhood_group, last_review, reviews_per_month
#to keep: neibhourhood, room_type, last_review
#discard: id, name, host_id, host_name

#checking duplicate columns
np.sum(data.duplicated()) == 0

#checking price equals 0
data = data[data[y_name] > 0]

data = data[data['beds'] != 0]
data = data[data['bedrooms'] != 0]
data = data[data['accommodates'] != 0]


##evaluating for missing data
data.count(axis=0)/len(data)


X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)
X_train, X_valid = train_test_split(X_train, test_size=0.1, random_state=0)


plt.hist(X_train['review_scores_rating'])
plt.title('Review Scores Rating')
plt.show()

#### feature engineering

#analysis per neighborhourhood
X_train.groupby('neighbourhood')[y_name].agg('mean').sort_values(ascending=False)
X_train.groupby('neighbourhood')[y_name].agg('count').sort_values(ascending=False)

#ploting mean price per neighborhoud
y = X_train.groupby('neighbourhood')[y_name].agg('mean').sort_values(ascending=False).values
x = np.array(range(1, len(y)+1))
plt.plot(x, y, 'ro')


labels = data.groupby('neighbourhood')[y_name].agg('median').sort_values(ascending=True).index
_map = { v: (k+1) for k,v in enumerate(labels)}
replace_map = {'neighbourhood' : _map}

X_train.replace(replace_map, inplace=True)
X_valid.replace(replace_map, inplace=True)
X_test.replace(replace_map, inplace=True)



# #lets create bins: [0, 250, 500, 1000, 2000, 3000, np.inf ] 
# bins = [0, 250, 500, 1000, 2000, 3000, np.inf]
# names= ['B250', 'B500', 'B1k', 'B2k', 'B3k', 'BInf']


# #one-hot encoding
# neigh_bins = pd.cut(X_train[y_name], bins, labels=names)
# X_train = pd.concat([X_train, pd.get_dummies(neigh_bins, prefix='is')], axis=1)

# neigh_bins = pd.cut(X_valid[y_name], bins, labels=names)
# X_valid = pd.concat([X_valid, pd.get_dummies(neigh_bins, prefix='is')], axis=1)

# neigh_bins = pd.cut(X_test[y_name], bins, labels=names)
# X_test = pd.concat([X_test, pd.get_dummies(neigh_bins, prefix='is')], axis=1)


#analysis per bed_type
X_train.groupby('bed_type')[y_name].agg('mean').sort_values(ascending=False)
X_train.groupby('bed_type')[y_name].agg('count').sort_values(ascending=False)

replace_map = {'bed_type' : {'Futon': 1, 'Pull-out Sofa': 2, 'Airbed': 3, 'Couch': 4, 'Real Bed': 5}}
X_train.replace(replace_map, inplace=True)
X_valid.replace(replace_map, inplace=True)
X_test.replace(replace_map, inplace=True)

#X_train = pd.concat([X_train, pd.get_dummies(X_train['bed_type'], prefix='is')], axis=1)
#X_valid = pd.concat([X_valid, pd.get_dummies(X_valid['bed_type'], prefix='is')], axis=1)
#X_test = pd.concat([X_test, pd.get_dummies(X_test['bed_type'], prefix='is')], axis=1)

#analysis per room_type
X_train.groupby('room_type')[y_name].agg('mean').sort_values(ascending=False)
X_train.groupby('room_type')[y_name].agg('count').sort_values(ascending=False)

replace_map = {'room_type' : {'Shared room': 1, 'Private room': 2, 'Hotel room': 3, 'Entire home/apt': 4}}
X_train.replace(replace_map, inplace=True)
X_valid.replace(replace_map, inplace=True)
X_test.replace(replace_map, inplace=True)

#one hot encoding
#X_train = pd.concat([X_train, pd.get_dummies(X_train['room_type'], prefix='is')], axis=1)
#X_valid = pd.concat([X_valid, pd.get_dummies(X_valid['room_type'], prefix='is')], axis=1)
#X_test = pd.concat([X_test, pd.get_dummies(X_test['room_type'], prefix='is')], axis=1)

#droping some columns
#df = X_train.drop(columns=['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 'last_review', 'reviews_per_month', 'neighbourhood', 'room_type', 'latitude', 'longitude'])
#['neighbourhood', 'bed_type', 'room_type']
cols = ['availability_30', 'availability_60', 'availability_90', 'availability_365', 'maximum_nights', 'first_review']
X_train.drop(columns=cols, inplace=True)
X_valid.drop(columns=cols, inplace=True)
X_test.drop(columns=cols, inplace=True)
X_train.info()

assert X_train.isna().sum().sum() == 0
assert X_valid.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0


##  exploring the target variable
X_train[y_name].describe()
#std is high

plt.figure(figsize=(9, 8))
sns.distplot(X_train[y_name], color='g', bins=100)
#high skewed

X_train.boxplot(column=[y_name])

#removing outliers
idxs = (np.abs(stats.zscore(X_train['price'])) < 3)
X_train = X_train[idxs]

X_train.hist(figsize=(12, 8), bins=50)

### assessing correlation
#sns.heatmap(df.corr(),cmap='BrBG',annot=True)

X_train.corr()[y_name]
#some extracted features show higher correlation coeficient than the raw features
#TODO: remove features with correlation coeficicient values less than 0.01

#pairplot
#pair_plot = sns.pairplot(df)
#no others patterns, correlations

Y_train = X_train[y_name]
Y_test = X_test[y_name]
Y_valid = X_valid[y_name]

X_train.drop(columns=[y_name], inplace=True)
X_test.drop(columns=[y_name], inplace=True)
X_valid.drop(columns=[y_name], inplace=True)

### Weakest baseline: mean

MSE = metrics.mean_squared_error(Y_test, np.repeat(np.mean(Y_train), len(Y_test)))
print('Mean RMSE: %.2f'%(np.sqrt(MSE))) #1752.62


### Linear Regression
lr = LinearRegression().fit(X_train, Y_train)
r2 = lr.score(X_train, Y_train)
print('%.2f' %(r2)) # / 0.33
Y_pred = lr.predict(X_test)
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('Improved LR RMSE: %.2f' %(np.sqrt(MSE))) #1671.12

### KNN
grid_params = {
	'n_neighbors': [1, 5, 10],
    'weights': ['uniform', 'distance'],            
}


knn = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=grid_params, cv=2)
grid_results = knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
RMSE_knn = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('KNN RMSE: %.f' %(RMSE_knn)) #1590 

### Polynomial Linear Regression
results = []
for d in range(2,3):
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
print('%.2f' %(r2)) 
MSE = metrics.mean_squared_error(Y_test, Y_pred)
print('Poly RMSE: %.2f' %(np.sqrt(MSE))) #1691.55

### XGBoost
grid_params = {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [500, 1000],
            'max_depth': [3, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'early_stopping_rounds': [10],
            #'min_child_weight': [1, 10],            
            #'gamma': [0, 1]
            }
      
xgb = GridSearchCV(estimator=XGBRegressor(seed=42), param_grid=grid_params, cv=2, verbose=1)
xgb.fit(X_train, Y_train)


# <bound method XGBModel.get_params of XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.8, early_stopping_rounds=10,
#              gamma=0, importance_type='gain', learning_rate=0.1,
#              max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
#              n_estimators=500, n_jobs=1, nthread=None, objective='reg:linear',
#              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#              seed=42, silent=None, subsample=1.0, verbosity=1)>



#model = XGBRegressor(learning_rate = 0.1, n_estimators=1000, max_depth=3, subsample=0.8, colsample_bytree=1, gamma= 1, seed=42)         
#model.fit(X_train, Y_train, eval_metric="rmse", eva_lset=[(X_train, Y_train), (X_valid, Y_valid)],  verbose=True, early_stopping_rounds = 10)

Y_pred = xgb.predict(X_test)
RMSE_xgb = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('XGB RMSE: %.2f' %(RMSE_xgb)) #1528.02

idx = np.argsort(xgb.best_estimator_.feature_importances_)[-5:]

labels = X_train.columns[idx]
values = np.sort(xgb.best_estimator_.feature_importances_)[-5:]
plt.bar(labels, values)
plt.show()

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
print('Stacking manual RMSE: %.2f' %(np.sqrt(MSE))) #1480.10

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
	'learning_rate': [0.01, 0.1],
	'n_estimators': [500, 1000],
	'max_depth': [3, 10],
	'subsample': [0.8, 1.0],
	'colsample_bytree': [0.8, 1.0],
	'early_stopping_rounds': [10],
	#'min_child_weight': [1, 10],            
	#'gamma': [0, 1]
}
      
grid = GridSearchCV(estimator=XGBRegressor(seed=42), param_grid=grid_params, cv=2)
grid.fit(X_train_poly, Y_train)


#model = XGBRegressor(learning_rate = 0.1, n_estimators=1000, max_depth=3, subsample=0.8, colsample_bytree=1, gamma= 1, seed=42)         
#model.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_train, Y_train), (X_valid, Y_valid)],  verbose=True, early_stopping_rounds = 10)

Y_pred = grid.predict(X_test_poly)
RMSE_xgb = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE_xgb: %.f' %(RMSE_xgb)) #1666



#http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/
#https://stackoverflow.com/questions/51194303/how-to-run-a-python-script-in-a-py-file-from-a-google-colab-notebook
#https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

#https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114
