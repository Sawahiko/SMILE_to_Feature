# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:09:55 2023

@author: oomsin
"""

# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
import time

# Machine Learning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# RDKit
import rdkit
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

start_time = time.time()
# %% Setup
MF_bit = 2**10

# %% Import Data : 560 datapoint
#Import Data
df = pd.read_excel("../DataTb.xlsx",sheet_name="AllDataSet")

#Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df["Tb"]

#>>> SHOW X_Data, Y_data

# %% Data Preparation
#Generate Fingerprint from SMILE
X_data_use = X_data_excel.copy()
X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
X_data_use["MACCS"] = X_data_use["molecule"].apply(lambda x: rdkit.Chem.MACCSkeys.GenMACCSKeys(x))

#>>> SHOW X_data_use 

#Transfrom Fingerprint to Column in DataFrame
X_data_fp = []
for i in range(X_data_use.shape[0]):
    array = np.array(X_data_use["MACCS"][i])
    datafram_i = pd.DataFrame(array)
    datafram_i = datafram_i.T
    X_data_fp.append(datafram_i)
X_data_fp = pd.concat(X_data_fp, ignore_index=True)
X_data_ML = pd.concat([X_data_use, X_data_fp], axis=1, join='inner')

Y_data_ML = Y_data.copy()

#>>>  SHOW X_data_fp, Y_data_fp

# %% Train test split

X_train_ML, X_test_ML, y_train_ML, y_test_ML = train_test_split(X_data_ML, Y_data_ML,test_size=0.25,random_state=42)

X_train = X_train_ML.copy().drop(columns = {"SMILES", "molecule", "MACCS"})
X_test = X_test_ML.copy().drop(columns = {"SMILES", "molecule", "MACCS"})
x_total = X_data_ML.copy().drop(columns = {"SMILES", "molecule", "MACCS"})

y_train = y_train_ML.copy()
y_test = y_test_ML.copy()
y_total = Y_data_ML.copy()

# >>> EXAMPLE SHOW {X, Y} of {TRAIN TEST TOTAL}

# %%
# Modeling

param_grid = {
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10]
}

# Initialize the Decision Tree Regressor
dt_reg = DecisionTreeRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=dt_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

md = best_params["max_depth"]
mss = best_params["min_samples_split"]

DT_model = DecisionTreeRegressor(max_depth=md,min_samples_split=mss)
DT_model.fit(X_train,y_train)

# %%   Validation with Error Metrics
mape_train_table = []
rmse_train_table = []
r2_train_table   = []

# Train set
y_predict_train = DT_model.predict(X_train)
mape_train = mean_absolute_percentage_error(y_train, y_predict_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
R2_train = r2_score(y_train, y_predict_train)

mape_train_table.append(mape_train)
rmse_train_table.append(rmse_train)
r2_train_table.append(R2_train)

# Test set
y_predict_test = DT_model.predict(X_test)
mape_test = mean_absolute_percentage_error(y_test, y_predict_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
R2_test = r2_score(y_test, y_predict_test)

mape_train_table.append(mape_test)
rmse_train_table.append(rmse_test)
r2_train_table.append(R2_test)

# Total set
y_predict_total = DT_model.predict(x_total)
mape_total = mean_absolute_percentage_error(y_total, y_predict_total)
rmse_total = np.sqrt(mean_squared_error(y_total, y_predict_total))
R2_total = r2_score(y_total, y_predict_total)

mape_train_table.append(mape_total)
rmse_train_table.append(rmse_total)
r2_train_table.append(R2_total)


# %% Store score y_predict
# Table Score
Score_Table = pd.DataFrame()
data = {
        "MAPE":mape_train_table,
        "RMSE":rmse_train_table,
        "R2"  :r2_train_table
    }
Score_Table = pd.DataFrame(data)

# Train predict Table
Train_Table = pd.DataFrame()
data = {
        "X" :X_train_ML["SMILES"],
        "Y"  :y_train,
        "Y_predict" :y_predict_train
    }
Train_Table = pd.DataFrame(data)

# Test predict Table
Test_Table = pd.DataFrame()
data = {
        "X" :X_test_ML["SMILES"],
        "Y"  :y_test,
        "Y_predict" :y_predict_test
    }
Test_Table = pd.DataFrame(data)

# Total predict Table
Total_Table = pd.DataFrame()
data = {
        "X" :X_data_ML["SMILES"],
        "Y"  :y_total,
        "Y_predict" :y_predict_total
    }
Total_Table = pd.DataFrame(data)

# %%  Export To Excel
with pd.ExcelWriter("ML7_560point.xlsx",mode='a') as writer:  
    Train_Table.to_excel(writer, sheet_name=f'Train_Prediction')
    Test_Table.to_excel(writer, sheet_name=f'Test_Prediction')
    Total_Table.to_excel(writer, sheet_name=f'Total_Prediction')
    
    Score_Table.to_excel(writer, sheet_name=f'Score')