# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 01:57:02 2023

@author: Administrator
"""


## Tree Based Models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import time

import gc
gc.collect()

import joblib


from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV


# input data and features name
train_dataset_top = pd.read_csv(r"train_dataset_top.csv", index_col = 0)
val_dataset_top = pd.read_csv(r"val_dataset_top.csv", index_col = 0)
test_dataset_top = pd.read_csv(r"test_dataset_top.csv", index_col = 0)
features_top = pd.read_csv(r"features_top.csv", index_col = 0)


# get 94 original features missing_info and correlation
flat_corr = pd.read_csv(r"flat_corr.csv", index_col = 0)  # pairs correlation
correlation_df_all = pd.read_csv(r"correlation_df_all.csv", index_col = 0)  # correlation with returns, p-values, and missing_percentage
chars = correlation_df_all.index


### Define Functions
# - Huber Loss Functions
# - Out-of-sample R-squared
# - Validation Function Using GridSerach Cross-validation
# - Huber Loss Gradients

from sklearn.model_selection import ParameterGrid

# Huber objective function
def huber_loss(actual, predicted, xi=0.5): 
    residual = actual - predicted
    huber_loss = np.where(np.abs(residual) <= xi, residual**2, 2 * xi * (np.abs(residual) - 0.5 * xi))
    return np.mean(huber_loss)

# Scoring function for out-of-sample R squared
def r2_oos(actual, predicted):
    # Convert input arrays to NumPy arrays
    actual, predicted = np.array(actual), np.array(predicted).flatten()
    
    # Calculate the sum of squared differences between actual and predicted
    ss_residual = np.sum((actual - predicted) ** 2)
    
    # Calculate the total sum of squares (using the mean of actual values as the reference)
    ss_total = np.sum((actual) ** 2)
    
    # Calculate R squared
    r_squared = 1 - (ss_residual / ss_total)
    
    # Return the calculated R squared
    return r_squared

# Validation Function using GridSearchCV
def val_fun_GridSearch(model, params: dict, X_train, y_train, X_val, y_val):
    scorer = make_scorer(huber_loss, greater_is_better=False)
#   grid_search = GridSearchCV(model, params, scoring=scorer, cv=5, verbose=0)
    grid_search = GridSearchCV(model, params, scoring=scorer, verbose=0)
    grid_search.fit(X_train, y_train, eval_set=[X_val, y_val])
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_val)
    best_ros = r2_oos(y_val, y_pred)

    print('\n' + '#' * 60)
    print('Tuning process finished!!!')
    print(f'The best setting is: {best_params}')
    print(f'Out-of-sample R-squared on validation set is: {best_ros * 100:.2f}%')
    print('#' * 60)
    
    return best_model


# Validation Function
def val_fun(model, params: dict, X_train, y_train, X_val, y_val):
    best_score=float('inf')
    # Hyperparameter tuning loop
    for params_sample in ParameterGrid(params):
        print(params_sample)
        model = model.set_params(**params_sample)
        model.fit(X_train, y_train)

        # Evaluate on the validation set
        y_val_pred = model.predict(X_val)
        mse_val = huber_loss(y_val, y_val_pred)

        # Check if this set of hyperparameters is the best so far
        if mse_val < best_score:
            best_score = mse_val
            best_params = params_sample

    # Train the final model with the best hyperparameters on the full training set
    best_model = model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    # Evaluate the final model on the validation set
    y_pred = best_model.predict(X_val)
    best_ros = r2_oos(y_val, y_pred)
        
    print('\n' + '#' * 60)
    print('Tuning process finished!!!')
    print(f'The best setting is: {best_params}')
    print(f'Out-of-sample R-squared on validation set is: {best_ros * 100:.2f}%')
    print('#' * 60)

    return best_model   


def report_result(model, X_train, y_train, X_test, y_test):
    print('\n' + '#' * 60)
    print(f"In-sample R-squared is: {r2_oos(y_train, model.predict(X_train)) * 100:.2f}%")
    print(f"Out-of-sample R-squared is: {r2_oos(y_test, model.predict(X_test)) * 100:.2f}%")
    print('#' * 60)
    
    

# Gradient of Huber objective function with respect to predicted values
def grad_huber_obj(actual, predicted, xi):
    residual = actual - predicted
    gradient = np.where(np.abs(residual) <= xi, -2 * residual, -2 * xi * np.sign(residual))
    return gradient / len(actual)

# Hessian of Huber objective function with respect to predicted values
def hess_huber_obj(actual, predicted, xi):
    residual = actual - predicted
    hessian = np.where(np.abs(residual) <= xi, 2, 2 * xi)
    return hessian / len(actual)

# Example usage in a LightGBM objective
def huber_objective(actual, predicted):
    xi = 0.5
    loss = huber_loss(actual, predicted, xi)
    grad = grad_huber_obj(actual, predicted, xi)
    hess = hess_huber_obj(actual, predicted, xi)
    return grad, hess

    
    
###############################################################################
    # Datasets    
###############################################################################
    
# Extract features and target from the datasets
train_dataset_top = train_dataset_top.dropna()
X_train = train_dataset_top[features_top]
y_train = train_dataset_top["RET"]

val_dataset_top = val_dataset_top.dropna()
X_val = val_dataset_top[features_top]
y_val = val_dataset_top["RET"]

test_dataset_top = test_dataset_top.dropna()
X_test = test_dataset_top[features_top]
y_test = test_dataset_top["RET"]
    
    
    
# Extract features and target from the datasets
train_dataset_top = train_dataset_top.dropna()
X_train_94char = train_dataset_top[chars]
y_train_94char = train_dataset_top["RET"]

val_dataset_top = val_dataset_top.dropna()
X_val_94char = val_dataset_top[chars]
y_val_94char = val_dataset_top["RET"]

test_dataset_top = test_dataset_top.dropna()
X_test_94char = test_dataset_top[chars]
y_test_94char = test_dataset_top["RET"]
    
    
    
###############################################################################
    # Decision Tree with 900+ Features
############################################################################### 
    
from sklearn.tree import DecisionTreeRegressor

params = {
    'max_depth':[1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

DecisionTree_all = val_fun(DecisionTreeRegressor(), params=params, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    
    
report_result(model=DecisionTree_all, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Save the model to a file
joblib.dump(DecisionTree_all, 'model_details/DecisionTree_all_top.joblib')


###############################################################################
    # Decision Tree with 94 Features
############################################################################### 
    
from sklearn.tree import DecisionTreeRegressor

params = {
    'max_depth':[1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}


DecisionTree = val_fun(DecisionTreeRegressor(), params=params, X_train=X_train_94char, y_train=y_train_94char, X_val=X_val_94char, y_val=y_val_94char)

report_result(model=DecisionTree, X_train=X_train_94char, y_train=y_train_94char, X_test=X_test_94char, y_test=y_test_94char)

# Save the model to a file
joblib.dump(DecisionTree, 'model_details/DecisionTree_top.joblib')


###############################################################################
    # Random Forest with 900+ Features
############################################################################### 
    
from sklearn.ensemble import RandomForestRegressor

params = {
    'n_estimators': [50, 100, 300],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'max_features': [30, 50, 100]
}



RamdonForest_all = val_fun(RandomForestRegressor(), params=params, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

report_result(model=RamdonForest_all, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Save the model to a file
joblib.dump(RamdonForest_all, 'model_details/RamdonForest_all_top.joblib')



###############################################################################
    # Random Forest with 94 Features
############################################################################### 

from sklearn.ensemble import RandomForestRegressor

params = {
    'n_estimators': [50, 100, 300],
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    'max_features': [30, 50, 100]
}


RamdonForest = val_fun(RandomForestRegressor(), params=params, X_train=X_train_94char, y_train=y_train_94char, X_val=X_val_94char, y_val=y_val_94char)

report_result(model=RamdonForest, X_train=X_train_94char, y_train=y_train_94char, X_test=X_test_94char, y_test=y_test_94char)

# Save the model to a file
joblib.dump(RamdonForest, 'model_details/RamdonForest_top.joblib')



###############################################################################
    # LightGBM with 900+ Features
###############################################################################

# Gradient of Huber objective function with respect to predicted values
def grad_huber_obj(actual, predicted, xi):
    residual = actual - predicted
    gradient = np.where(np.abs(residual) <= xi, -2 * residual, -2 * xi * np.sign(residual))
    return gradient / len(actual)

# Hessian of Huber objective function with respect to predicted values
def hess_huber_obj(actual, predicted, xi):
    residual = actual - predicted
    hessian = np.where(np.abs(residual) <= xi, 2, 2 * xi)
    return hessian / len(actual)

# Example usage in a LightGBM objective
def huber_objective(actual, predicted):
    xi = 0.5
    loss = huber_loss(actual, predicted, xi)
    grad = grad_huber_obj(actual, predicted, xi)
    hess = hess_huber_obj(actual, predicted, xi)
    return grad, hess

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

params = {
#    'objective':[None, huber_objective],
    'max_depth':[1,2,3,4,5,6,7,8,9,10],
    'num_leaves': [10,30,50,80],
    'n_estimators':[10,50,100,200,500],
    'learning_rate':[0.01,.1]
}

LGBM_all = val_fun(lgb.LGBMRegressor(), params=params, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

report_result(model=LGBM_all, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Save the model to a file
joblib.dump(LGBM_all, 'model_details/LGBM_all_top.joblib')


###############################################################################
    # LightGBM with 94 Features
###############################################################################

params = {
#    'objective':[None, huber_objective],
    'max_depth':[1,2,3,4,5,6,7,8,9,10],
    'num_leaves': [10,30,50,80],
    'n_estimators':[10,50,100,200,500],
    'learning_rate':[0.01,.1]
}
 
LGBM = val_fun(lgb.LGBMRegressor(), params=params, X_train=X_train_94char, y_train=y_train_94char, X_val=X_val_94char, y_val=y_val_94char)

report_result(model=LGBM, X_train=X_train_94char, y_train=y_train_94char, X_test=X_test_94char, y_test=y_test_94char)

# Save the model to a file
joblib.dump(LGBM, 'model_details/LGBM_top.joblib')



###############################################################################
    # XGBoost with 900+ Features
###############################################################################
import xgboost as xgb

params = {
#    'objective': [None, huber_objective],  # Using Huber loss for regression
    'max_depth': [1,2,3,4,5,6,7,8,9,10],  # Adjust the maximum depth of the trees
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'n_estimators': [50, 100, 200]  # Number of boosting rounds
}


XGBoost_all = val_fun(xgb.XGBRegressor(), params=params, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

report_result(model=XGBoost_all, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Save the model to a file
joblib.dump(XGBoost_all, 'model_details/XGBoost_all_top.joblib')


###############################################################################
    # XGBoost with 94 Features
###############################################################################

params = {
#    'objective': [None, huber_objective],  # Using Huber loss for regression
    'max_depth': [1,2,3,4,5,6,7,8,9,10],  # Adjust the maximum depth of the trees
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'n_estimators': [50, 100, 200]  # Number of boosting rounds
}


XGBoost = val_fun(xgb.XGBRegressor(), params=params, X_train=X_train_94char, y_train=y_train_94char, X_val=X_val_94char, y_val=y_val_94char)

report_result(model=XGBoost, X_train=X_train_94char, y_train=y_train_94char, X_test=X_test_94char, y_test=y_test_94char)

# Save the model to a file
joblib.dump(XGBoost, 'model_details/XGBoost_top.joblib')