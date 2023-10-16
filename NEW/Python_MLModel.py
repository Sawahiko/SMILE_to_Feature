# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
import time

# Machine Learning
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#from keras.models import Sequential
#from keras.layers import Dense
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

def RF(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_features': [None,'sqrt', 'log2'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create a RandomizedSearchCV object
    rf = RandomForestRegressor(random_state=42)
    kfold = KFold(n_splits=8, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    return best_model

def Ridge_M(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'alpha': np.logspace(-3, 3, 7)
    }
    
    # Create a RandomizedSearchCV object
    ridge = Ridge(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(ridge, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    return best_model

def XGB(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400]
    }
    
    # Create a RandomizedSearchCV object
    xgb = XGBRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, 
                                       scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
    return best_model

def NN(x_train, y_train):
    model = Sequential()
    model.add(Dense(1024, input_dim=x_train.shape[1] , activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    return model

def CB(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    # Create a RandomizedSearchCV object
    catboost = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
    kfold = KFold(n_splits=8, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(catboost, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
    return best_model

def DT(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create a RandomizedSearchCV object
    dt = DecisionTreeRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(dt, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
    return best_model

def SVR_M(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': np.logspace(-3, 3, 7),
        'epsilon': [0.1, 0.2, 0.5, 0.01]
    }
    
    # Create a RandomizedSearchCV object
    svr = SVR()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(svr, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
    return best_model