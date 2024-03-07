# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import random

# Machine Learning
## Algorithm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
## Tool, Error Metric
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from joblib import dump, load

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


#%% Default
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def Linear_default(x_train, y_train):
  lm = LinearRegression()
  lm.fit(x_train, y_train)
  return lm

def Ridge_default(x_train, y_train):
  ridge = Ridge()
  ridge.fit(x_train, y_train)
  return ridge

def Lasso_default(x_train, y_train):
  lasso = Lasso()
  lasso.fit(x_train, y_train)
  return lasso

def DT_default(x_train, y_train):
  DT = DecisionTreeRegressor()
  DT.fit(x_train, y_train)
  return DT

def RF_default(x_train, y_train):
  RF = RandomForestRegressor()
  RF.fit(x_train, y_train)
  return RF

def XGB_default(x_train, y_train):
  XGB = XGBRegressor()
  XGB.fit(x_train, y_train)
  return XGB

def KNN_default(x_train, y_train):
  KNN = KNeighborsRegressor()
  KNN.fit(x_train, y_train)
  return KNN

def SVM_default(x_train, y_train):
  svr = SVR()
  svr.fit(x_train, y_train)
  return svr

#%% GridSearch
from sklearn.model_selection import GridSearchCV
def RF(x_train, y_train):
    # Parameter grid for Optimized Parameter
    # Default : n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': [None, 1, 'sqrt', 'log2'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust number of splits as needed

    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    return best_model


def XGB(x_train, y_train):
    # Dictionary for GridSearchCV
    # Default Parameter : max_depth=None/6 , learning_rate=None/0.3   , n_estimators=None
    param_grid = {
        'max_depth': [None, 3, 4, 5, 6, 7],
        'learning_rate': [None, 0.01, 0.05, 0.1, 0.2, 0.3],
        'n_estimators': [None, 100, 200, 300, 400]
    }

    xgb = XGBRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=kfold, scoring="neg_root_mean_squared_error")
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    #print(grid_search.cv_results_)
    print(grid_search.best_params_)
    return best_model

def DT(x_train, y_train):
    # Dictionary for GridSearchCV
    # Default Parameter : max_depth=None, min_samples_split=2,  min_samples_leaf=1
    param_grid = {
      'max_depth': [None, 3, 5, 7, 10],
      'min_samples_split': [2, 5, 10, 20],
      'min_samples_leaf': [1, 2, 5, 10]
    }

    dt = DecisionTreeRegressor(random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(dt, param_grid=param_grid, cv=kfold, verbose=2)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    #print(grid_search.cv_results_)
    print(grid_search.best_params_)
    return best_model

def KNN(x_train, y_train):
    # Dictionary for GridSearchCV
    # Default Parameter : n_neighbors=5, weights='uniform',  algorithm='auto'
    param_grid = {
      'n_neighbors': [5, 10, 20, 50],
      'weights': ['uniform', 'distance'],
      'algorithm': ['auto','ball_tree', 'kd_tree', 'brute']
    }

    knn = KNeighborsRegressor()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(knn, param_grid=param_grid, cv=kfold)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    #print(grid_search.cv_results_)
    print(grid_search.best_params_)
    return best_model

def SVM_M(x_train, y_train):
    # Dictionary for GridSearchCV
    # Default Parameter : C=1, kernel='rbf',  degree=3, gamma='scale'
    param_grid = {
      'C': [0.1, 1, 10],
      'kernel': ['linear', 'rbf', 'poly'],
      'degree': [2, 3, 4],
      'gamma': ['scale', 0.01, 0.1, 0.3]
    }

    svr = SVR()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(svr, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    print(grid_search.best_params_)
    return best_model

#%% Import data
x_train = pd.read_csv("csv_02-3 std_x_train.csv").iloc[:,1:]
y_train = pd.read_csv("csv_02-4 std_y_train.csv").iloc[:,1]
x_test  = pd.read_csv("csv_02-5 std_x_test.csv").iloc[:,1:]
y_test  = pd.read_csv("csv_02-6 std_y_test.csv").iloc[:,1]

scaler_x = load("file_02-1 scaler_x.joblib")
scaler_y = load("file_02-2 scaler_y.joblib)

#%% Grid Training

# Specified model need to run
#names_bestpar = ["DT", "RF", "XGB", "KNN"]
names_bestpar = ["DT"]
models_bestpar = [DT]

# Run Training Model
all_result_model_bestpar = []
all_time_fitting_bestpar = []
for iteration in range(len(names_bestpar)) :
    get_model = models_bestpar[iteration]
    time_start = datetime.now()
    result_model = get_model(x_train, y_train)
    time_end = datetime.now()
    duration = (time_end - time_start).total_seconds()
    print(result_model)
    print(f'{duration} seconds\n')
    all_result_model_bestpar.append(result_model)
    all_time_fitting_bestpar.append(duration)

#%%
from datetime import datetime


func_name = ["MAE", "RMSE", "R2"]
rmse_func = lambda x,y : mean_squared_error(x,y, squared=False)
func_list = [mean_absolute_error, rmse_func, r2_score]
def create_result_from_predict_value(table_predict, func_name, func_list):
    list_df = list()
    name_method = pd.Series(table_predict.apply(lambda x : x[0], axis=1), name="Method")
    df_1 = name_method.to_frame();   list_df.append(df_1)
    for i in range(len(func_list)):
      list_error_metric = []

      name_fx = func_name[i]
      func    = func_list[i]
      #name_method = pd.Series(table_predict.apply(lambda x : x[0], axis=1), name="Method")
      value = pd.Series(table_predict.apply(lambda x : func(x[1], x[2]), axis=1),name=name_fx)
      df_2=value.to_frame()
      #print(df_1)
      #print(df_2)
      #df_combine = pd.merge(df_1, df_2, how='inner', left_index=True, right_index=True)
      #df_combine = pd.concat([df_1, df_2], axis=1)
      list_df.append(df_2)

    return pd.concat(list_df, axis=1)



#print(create_result_from_predict_value(result_train, func_name, func_list))
print("\n")
#print(create_result_from_predict_value(result_test, func_name, func_list))

def model_assess(X_train, X_test, y_train, y_test,
                 scaler_x, scaler_y,
                 list_model, name_model,
                 all_time_fitting, title = "Default"):

    func_name = ["MAE", "RMSE", "R2"]
    rmse_func = lambda x,y : mean_squared_error(x,y, squared=False)
    func_list = [mean_absolute_error, rmse_func, r2_score]
    
    model_assess_table = pd.DataFrame(['Method','Training MAE', 'Training RMSE','Training R2','Test MAE', 'Test RMSE','Test R2', 'Time Evaluate']).transpose()
    new_header = model_assess_table.iloc[0] #grab the first row for the header
    model_assess_table.columns = new_header #set the header row as the df header
    model_assess_table.drop(index=model_assess_table.index[0], axis=0, inplace=True)
    
    train_prediction_table = pd.DataFrame(['Method','Training Predict','Training Actual']).transpose()
    new_header = train_prediction_table.iloc[0] #grab the first row for the header
    train_prediction_table.columns = new_header #set the header row as the df header
    train_prediction_table.drop(index=train_prediction_table.index[0], axis=0, inplace=True)
    
    test_prediction_table = pd.DataFrame(['Method','Test Predict','Test Actual']).transpose()
    new_header = test_prediction_table.iloc[0] #grab the first row for the header
    test_prediction_table.columns = new_header #set the header row as the df header
    test_prediction_table.drop(index=test_prediction_table.index[0], axis=0, inplace=True)
    
    
    for iteration in range(len(list_model)):
        time_start = datetime.now()
        model_train = list_model[iteration]
        name = name_model[iteration]
    
        if("DL" not in name):
          model_train.fit(X_train, y_train)
          y_train_pred = model_train.predict(X_train)
          y_test_pred  = model_train.predict(X_test)
        else:
          pass
    
        time_end = datetime.now()
        duration = (time_end - time_start).total_seconds()
        print(name)
        print(duration)
    
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        test_r2 = r2_score(y_test, y_test_pred)
    
        results = pd.DataFrame([name_model[iteration],train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2, duration]).transpose()
        results.columns = ['Method','Training MAE', 'Training RMSE','Training R2','Test MAE', 'Test RMSE','Test R2', 'Time Evaluate']
    
    
        train_prediction_result = pd.DataFrame([name_model[iteration],y_train_pred, y_train]).transpose()
        train_prediction_result.columns = ['Method','Training Predict','Training Actual']
    
        test_prediction_result = pd.DataFrame([name_model[iteration], y_test_pred, y_test]).transpose()
        test_prediction_result.columns = ['Method','Test Predict','Test Actual']
    
        model_assess_table = pd.concat([model_assess_table, results])
        train_prediction_table = pd.concat([train_prediction_table, train_prediction_result])
        test_prediction_table = pd.concat([test_prediction_table, test_prediction_result])
    model_assess_table['Time Training'] = all_time_fitting
    return model_assess_table, train_prediction_table, test_prediction_table










#%% Export

# Export Model
for i in range(len(names_bestpar)):
    i_show = i+1
    path_model = f"file_03-1-{i_show} {names_bestpar[i]}.joblib"
    print(path_model)
    dump(all_result_model_bestpar[i], path_model)
    
# Export Evaluation


# Export Prediction Table