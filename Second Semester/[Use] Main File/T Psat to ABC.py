#%% Import
# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Machine Learning
## Algorithm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
## Tool, Error Metric
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from joblib import dump, load

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
#import pytorch_lightning as L

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
#%% Import Data
df = pd.read_csv("New_Data_Psat_Not_Outliers.csv")
df = df[df['SMILES'] != "None"].reset_index(drop=True)

# Genearate Temp in Tmin-Tmax and expand
df1 = df.copy()
    ## Function to generate equally distributed points
def generate_points(row, amount_point):
    start, end, num_points = row["Tmin"], row["Tmax"], amount_point
    step_size = (end - start) / (num_points - 1)
    return np.linspace(start, end, num_points)
df1["T"] = df1.apply(lambda x : generate_points(x, 10), axis=1)

# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])
#%%
# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]

x = 1/df2["T"]
y = df2["Vapor_Presssure"]

x1 = x[0]
y1 = y[0]
xy_table = pd.DataFrame({
    "SMILES" : df2["SMILES"],
    "x" : x,
    "y" : y})


#%%
from scipy.optimize import curve_fit
def objective(X, a, b, c):
    x,y = X
    # Linearized Equation : y + C * y * x1 = A + B * x1
    return a + b*x - c*y*x
def getABC(row):
    #print(row.x)
    x1 = row.x
    y1 = row.y
    popt, _ = curve_fit(objective, (x1,y1), y1)
    a,b,c = popt
    return [a,b,c]
#z = func((x,y), a, b, c) * 1
xy_table["ABC"] = xy_table.apply(getABC, axis=1)
xy_table[['A', 'B', 'C']] = pd.DataFrame(xy_table['ABC'].tolist())

#%%
df_test = df1.copy()
df_test= df_test.explode('T')
df_test['T'] = df_test['T'].astype('float32')
df_test= df_test.reset_index(drop=True)

# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

df_test["Vapor_Presssure"] = Psat_cal(df_test["T"],
                                      df_test["A"],
                                      df_test["B"],
                                      df_test["C"])
result = df_test.groupby('SMILES')[['T', 'Vapor_Presssure']].agg(list)
# Reset the index to create a DataFrame
result = result.reset_index()


#%%
x = 1/result["T"]
y = result["Vapor_Presssure"]

xy_table = pd.DataFrame({
    "x" : x,
    "y" : y})

result = pd.concat([result, xy_table])
#%%
from scipy.optimize import curve_fit
def objective(X, a, b, c):
    x,y = X
    # Linearized Equation : y + C * y * x1 = A + B * x1
    return a + b*x - c*y*x
def getABC(row):
    #print(row.x)
    x1 = row.x
    y1 = row.y
    popt, _ = curve_fit(objective, (x1,y1), y1)
    a,b,c = popt
    return [a,b,c]
#z = func((x,y), a, b, c) * 1
result["ABC"] = result.apply(getABC, axis=1)
result[['A', 'B', 'C']] = pd.DataFrame(result['ABC'].tolist())