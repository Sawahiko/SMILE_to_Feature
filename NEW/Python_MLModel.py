# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
import time

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

def RF(x_train,y_train):
    model = RandomForestRegressor()
# =============================================================================
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [5, 10, 15],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 5]
#         }
#     grid_search = GridSearchCV(model, param_grid, cv=5)
#     grid_search.fit(x_train, y_train)
#     best_model = grid_search.best_estimator_
# =============================================================================
    model.fit(x_train, y_train)
    return model

def Ridge_M(x_train,y_train):
    model = Ridge()
# =============================================================================
#     param_grid = {
#         'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
#         }
#     grid_search = GridSearchCV(model, param_grid, cv=5)
#     grid_search.fit(x_train, y_train)
#     best_model = grid_search.best_estimator_
# =============================================================================
    model.fit(x_train, y_train)
    return model

def SVC_R(x_train,y_train):
    model = SVR()
# =============================================================================
#     param_grid = {
#         'C': [0.1, 1, 10, 100],
#         'gamma': [0.01, 0.1, 1, 10]
#         }
#     grid_search = GridSearchCV(model, param_grid, cv=5)
#     grid_search.fit(x_train, y_train)
#     best_model = grid_search.best_estimator_
# =============================================================================
    model.fit(x_train, y_train)
    return model

def XGB(x_train,y_train):
    model = GradientBoostingRegressor()
# =============================================================================
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 6, 9],
#         'learning_rate': [0.05, 0.1, 0.2],
#         'min_child_weight': [1, 10, 100]
#         }
#     grid_search = GridSearchCV(model, param_grid, cv=5)
#     grid_search.fit(x_train, y_train)
#     best_model = grid_search.best_estimator_  
# =============================================================================
    model.fit(x_train, y_train)
    return model

# =============================================================================
# def NN(x_train,y_train):
#     model = Sequential()
#     model.add(Dense(1024, input_dim=x_train.shape[1], activation='relu'))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(x_train, y_train, epochs=100, batch_size=16)
#     return model
# =============================================================================
