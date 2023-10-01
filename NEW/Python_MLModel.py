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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def RF(x_train,y_train):
    model = RandomForestRegressor()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def Ridge_M(x_train,y_train):
    model = Ridge()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def SVC_R(x_train,y_train):
    model = SVR()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def XGB(x_train,y_train):
    model = GradientBoostingRegressor()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def NN(x_train,y_train):
    model = Sequential()
    model.add(Dense(1024, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=16)
    return model