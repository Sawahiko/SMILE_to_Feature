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

def RF(x_train,y_train):
    model = RandomForestRegressor()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def Ridge(x_train,y_train):
    model = Ridge()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def SVR(x_train,y_train):
    model = SVR()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model

def XGB(x_train,y_train):
    model = GradientBoostingRegressor()
    model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
    model.fit(x_train,y_train)
    return model