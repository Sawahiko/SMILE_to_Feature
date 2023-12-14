# %% Package
# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Our module
from Python_MLModel import RF, Ridge_M, XGB, NN, CB, DT, SVR_M, KNN
from Python_Scoring_Export import Scoring, Export

def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))
def generateTemp(Tmin, Tmax, amountpoint):
    Trange = Tmax-Tmin
    T_arr =[]
    for i in range(amountpoint):
        Tgen = Tmin+(Trange*random.random())
        T_arr.append(Tgen)
    return T_arr
        
#%% Import Data
df = pd.read_excel("../[Use] Data Preparation/Psat_AllData.xlsx",sheet_name="All")
df = df[df['SMILES'] != "None"].reset_index(drop=True)

df1 = df.copy()
T_Test = generateTemp(df1["Tmin"], df1["Tmax"], 5)
T_all = []
for i in range(len(T_Test[0])):
    T_gen_x_point = [item[i] for item in T_Test]
    T_all.append(T_gen_x_point)

df1["T"] = T_all
df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.drop(columns={"Column1"})
df1 = df1.reset_index(drop=True)



Psat_test = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])          
df1["Vapor_Presssure"] = Psat_test

df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df2[["SMILES"]]
Y_data= df2["Vapor_Presssure"]
        

# %% Fingerprint 
# Generate Fingerprint from SMILE
MF_radius = 3
MF_bit = 1024

X_data_use = X_data_excel.copy()
X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
X_data_use["count_morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
    x, 
    radius=MF_radius, 
    nBits=MF_bit,
    useFeatures=True, useChirality=True))
X_data_use["arr_count_morgan_fp"] = 0
#X_data_use["arr_count_morgan_fp"] = np.zeros((0,), dtype=np.int8)

#X_data_use["arr_count_morgan_fp"] 
#new_df = X_data_use.apply(DataStructs.ConvertToNumpyArray, axis=0, args=('count_morgan_fp',))


# Transfrom Fingerprint to Column in DataFrame
X_data_fp = []
for i in range(X_data_use.shape[0]):
    #print(np.array(X_data_use["morgan_fp"][i]))
    blank_arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(X_data_use["count_morgan_fp"][i],blank_arr)
    datafram_i = pd.DataFrame(blank_arr)
    datafram_i = datafram_i.T
    X_data_fp.append(datafram_i)
x_data_fp = pd.concat(X_data_fp, ignore_index=True)
y_data_fp = Y_data.copy()
# %% MinMaxScaler
#x_data_fp = df2[["Temp_test"]].join(x_data_fp)
x_data_fp[MF_bit] = df2["T"]

from sklearn.preprocessing import MinMaxScaler
# Define the scaler for the last column
scaler_col3 = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to the data
scaler_col3.fit(x_data_fp[MF_bit].values.reshape(-1, 1))

# Transform the data using the fitted scaler
x_data_fp[MF_bit] = scaler_col3.transform(x_data_fp[MF_bit].values.reshape(-1, 1))

#%% Train-test split
x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

#%% Training Model

model = RF(x_train_fp, y_train_fp)

#%% Evaluation
Score_table = Scoring(model , x_train_fp, x_test_fp, x_data_fp, y_train_fp, y_test_fp, y_data_fp)
y_pred_train = model.predict(x_train_fp)
y_pred_test = model.predict(x_test_fp)

df_compare = pd.DataFrame({'Actual': y_test_fp,
                           'Predict': y_pred_test})
df_compare["diff"] = abs(df_compare["Actual"] - df_compare["Predict"])
df_compare_des = df_compare.describe()

df_compare_train = pd.DataFrame({'Actual': y_train_fp,
                           'Predict': y_pred_train})
df_compare_train["diff"] = abs(df_compare_train["Actual"] - df_compare_train["Predict"])
df_compare_train_des = df_compare_train.describe()
#%% Visualization
#x_min = min(min(y_test_fp),min(y_pred_test))
#x_max = max(max(y_test_fp),max(y_pred_test))

x_min = -20; x_max = 25

y_min, y_max = x_min, x_max

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

x = np.linspace(x_min, x_max, 100)
y = x
p1 = plt.plot(x, y, color='black',linestyle='dashed', label='x=y')

plt.scatter(y_train_fp, y_pred_train, label="train", alpha=0.3)
plt.scatter(y_test_fp, y_pred_test, label="test", alpha=0.8)
plt.legend()
