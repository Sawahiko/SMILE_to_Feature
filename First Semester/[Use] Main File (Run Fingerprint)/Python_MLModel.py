# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
import time

# Machine Learning
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def RF(x_train, y_train):
    # Define the parameter grid for GridSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_features': [None,'sqrt', 'log2'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create a GridSearchCV object
    rf = RandomForestRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the GridSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
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
    print(random_search.best_params_)
    
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
def XGB_Default(x_train, y_train):
    # Create a RandomizedSearchCV object
    xgb = XGBRegressor(random_state=42)
    xgb.fit(x_train, y_train)
    
    return xgb

# =============================================================================
# def NN(x_train, y_train):
#     model = Sequential()
# #    model.add(Dense(1024, input_dim=x_train.shape[1]))
#     model.add(Dense(200, input_dim=x_train.shape[1] , activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
#     return model
# =============================================================================

def NN(x_train, y_train):
    model = Sequential()

    # Add BatchNormalization after each dense layer
    model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
    return model

def CB(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    # Create a RandomizedSearchCV object
    catboost = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
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

def DT_Default(x_train, y_train):
    
    # Create a RandomizedSearchCV object
    dt = DecisionTreeRegressor(random_state=42).fit(x_train, y_train)
    
    return dt
# =============================================================================
# def DT(x_train, y_train):
#     model = DecisionTreeRegressor()
#     model_cv =cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
#     model.fit(x_train,y_train)
#     return model
# =============================================================================

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

def KNN(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist_knn = {
        'n_neighbors': range(1, 21),  # Adjust the range as needed
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }
    
    # Create a RandomizedSearchCV object for KNN
    knn = KNeighborsRegressor()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search_knn = RandomizedSearchCV(knn, param_distributions=param_dist_knn, n_iter=5, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object for KNN
    random_search_knn.fit(x_train, y_train)
    
    # Get the best KNN model
    best_knn_model = random_search_knn.best_estimator_
    print(random_search_knn.best_params_)
    
    return best_knn_model

def GP(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'kernel': [1.0 * RBF(length_scale=1.0), 1.0 * RBF(length_scale=0.5)]
    }
    
    # Create a RandomizedSearchCV object
    gp = GaussianProcessRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(gp, param_distributions=param_dist, n_iter=2, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
    return best_model 


def Lasso_R(x_train, y_train):
    # Define the parameter grid for RandomizedSearchCV
    param_dist = {
        'alpha': np.logspace(-3, 3, 7)
    }
    
    # Create a RandomizedSearchCV object
    lasso = Lasso()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed
    
    random_search = RandomizedSearchCV(lasso, param_distributions=param_dist, n_iter=10, cv=kfold, verbose=1, scoring='neg_mean_squared_error')
    
    # Fit the RandomizedSearchCV object
    random_search.fit(x_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    print(random_search.best_params_)
    
    return best_model

