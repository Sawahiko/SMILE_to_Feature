# Python
import numpy as np
import pandas as pd
import itertools

# Machine Learning
## Algorithm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Tool, Error Metric
from sklearn.model_selection import train_test_split

# RDKit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs


#%% Default

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
#%% Import Data
#df = pd.read_csv("csv_01 Psat_[X]_ABCTminTmaxC1-12.csv")
df_original = pd.read_csv("../Refactor Code/csv-01-0 Psat-1800.csv")
filter1 = df_original["SMILES"].str.contains("\+")
#filter2 = df["SMILES"].str.contains("\-")
filter3 = df_original["SMILES"].str.contains("\.")
print(filter1.sum(), filter3.sum())
f = filter1 +filter3 
f.sum()
#df = df_original[~f]
df = df_original.copy()
#%%
# New Train-Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Func. Group"])

train_out = train.groupby("Func. Group").agg({'SMILES': ['count']})
test_out = test.groupby("Func. Group").agg({'SMILES': ['count']})
print(pd.concat([train_out, test_out ], axis=1))
#%% 
def generate_points(row, amount_point):
    start = row["Tmin"]; end = row["Tmax"];
    temp = amount_point
    if end-start == 0:
        amount_point = 1
    else:
        amount_point = temp
    return np.linspace(start, end, amount_point)
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

def generate_FP(MF_bit, MF_radius, SMILES_data, T_data):
    # Generate Fingerprint from SMILE
    X_data_use = SMILES_data.copy()
    X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))    # Create Mol object from SMILES
    X_data_use["count_morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
        x,
        radius=MF_radius,
        nBits=MF_bit,
        useFeatures=True, useChirality=True))         # Create Morgan Fingerprint from Mol object
    
    
    # Transfrom Fingerprint to Datafrme that we can use for training
    X_data_use["arr_count_morgan_fp"] = 0
    X_data_fp = []
    for i in range(X_data_use.shape[0]):
        blank_arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(X_data_use["count_morgan_fp"][i],blank_arr)
        datafram_i = pd.DataFrame(blank_arr)
        datafram_i = datafram_i.T
        X_data_fp.append(datafram_i)
    x_data_fp = pd.concat(X_data_fp, ignore_index=True)
    x_data_fp = x_data_fp.astype(np.float32)
    x_data_fp[MF_bit] = T_data      # Input  = Fingerprint + Temp
    return x_data_fp
#%% Set up Fingerprint
# Parameter for Generate Morgan Fingerprint
all_MF_radius = [2,3,4]
all_MF_bit = [256, 1024, 2048, 2**12]

MF_loop = list(itertools.product(all_MF_radius, all_MF_bit))

#MF_loop = MF_loop[0:2]  ## TEMPORARY ##
#%%
# Genearate Temp in Tmin-Tmax and expand
df1 = train.copy()
df1["T"] = df1.apply(lambda x : generate_points(x, 5), axis=1)
df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)

# Generate VP from Antione Coeff and Temp
df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])

# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
df2 = df2[~df2["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])].reset_index()

X_data= df2[["SMILES"]]                 # feature: SMILE, T
Y_data= df2[["Vapor_Presssure"]]        # Target : Psat

df2_train = df2.copy()
print(df2_train.sort_values(by="Vapor_Presssure"))


#%%
list_x_FP_train = []
for element in MF_loop:
    MF_radius = element[0]; MF_bit= element[1]
    x_data_temp = generate_FP(MF_bit, MF_radius, df2_train, df2_train["T"])
    list_x_FP_train.append(x_data_temp)
    
y_train = df2_train[["Vapor_Presssure"]]
#%%
# Genearate Temp in Tmin-Tmax and expand
df1 = test.copy()
    ## Function to generate equally distributed points
df1["T"] = df1.apply(lambda x : generate_points(x, 5), axis=1)
df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])

# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
df2 = df2[~df2["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])].reset_index()

X_data= df2[["SMILES"]]               # feature: SMILE, T
Y_data= df2[["Vapor_Presssure"]]        # Target : Psat

df2_test = df2.copy()
print(df2_test.sort_values(by="Vapor_Presssure"))
#%%
list_x_FP_test = []
for element in MF_loop:
    MF_radius = element[0]; MF_bit= element[1]
    x_data_temp = generate_FP(MF_bit, MF_radius, df2_test, df2_test["T"])
    list_x_FP_test.append(x_data_temp)
    
y_test = df2_test[["Vapor_Presssure"]]

#%%
list_model = []
list_pred = []
for i in range(len(MF_loop)):
    MF_radius = MF_loop[i][0]; MF_bit= MF_loop[i][1]
    print(f"Training {MF_radius}-{MF_bit}")
    model = XGB_default(list_x_FP_train[i], y_train)
    print(f"Trained {MF_radius}-{MF_bit}")
    list_model.append(model)
    #model_pred = list_model[i]
    x_test_loop = list_x_FP_test[i]
    y_pred = model.predict(x_test_loop)
    list_pred.append(y_pred)
list_model
#%%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
temp = pd.DataFrame({
    "r-Bits set up":MF_loop,
    "Predict":list_pred,
    "Actual":[y_test.values.flatten()]*len(MF_loop)
    }) 
temp["RMSE"]  = temp.apply(lambda x: mean_squared_error(x["Actual"], x["Predict"], squared=False), axis=1)
temp["MAE"]  = temp.apply(lambda x: mean_absolute_error(x["Actual"], x["Predict"]), axis=1)
temp["R2"]  = temp.apply(lambda x: r2_score(x["Actual"], x["Predict"]), axis=1)

temp["MF_radius"] = temp.apply(lambda x: x["r-Bits set up"][0], axis=1)
temp["MF_Bits"] = temp.apply(lambda x: x["r-Bits set up"][1], axis=1)

#%%
temp_export = temp.explode(["Predict", "Actual"])
#%% Visualization

import seaborn as sns
import matplotlib.pyplot as plt

#temp2 = temp[temp["MF_Bits"]!= 8192]
temp2 = temp.copy()
df_log_sns = temp2.pivot_table(index="MF_Bits", columns="MF_radius", values="RMSE")
g = sns.heatmap(df_log_sns, annot=True, fmt=".4f", cmap=sns.color_palette("light:brown_r", as_cmap=True))
g.invert_yaxis()
plt.title("Fingerprint Heatmap with RMSE")
plt.show(g)
#%%
df_log_sns = temp2.pivot_table(index="MF_Bits", columns="MF_radius", values="MAE")
g = sns.heatmap(df_log_sns, annot=True, fmt=".4f", cmap=sns.color_palette("light:brown_r", as_cmap=True))
g.invert_yaxis()
plt.title("Fingerprint Heatmap with MAE")
plt.show(g)
#%%
df_log_sns = temp2.pivot_table(index="MF_Bits", columns="MF_radius", values="R2")
g = sns.heatmap(df_log_sns, annot=True, fmt=".4f", cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))
g.invert_yaxis()
plt.title("Fingerprint Heatmap with R2")
plt.show(g)
#%% Export Section

#temp_export.to_csv("XGB Default Prediction Fingerprint_result.csv")

#temp_inport = temp_export.groupby(['r-Bits set up', 'RMSE', 'MAE', 'R2', 'MF_radius','MF_Bits']).agg(list)
#temp_import = temp_inport.reset_index()

#%% Visualization from temp_import

#%%

# =============================================================================
# import requests
# url = 'https://notify-api.line.me/api/notify'
# token = '3CfMWfczpal9Zye6bD72a8Ud6FWOODnBHQZHIWM1YU4'
# headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}
# 
# msg = f'Inspect Fingerpirnt Done'
# r = requests.post(url, headers=headers, data = {'message':msg})
# print (r.text)
# =============================================================================
