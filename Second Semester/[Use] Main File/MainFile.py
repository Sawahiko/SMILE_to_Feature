# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from Python_Scoring_Export import Scoring, Export

#%%

# Import Data
df = pd.read_csv("Psat_SMILES_sky1.csv")
df = df[df['SMILES'] != "None"].reset_index(drop=True)

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df[["A","B","C"]]
        

# %% Data Preparation
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

#%%
x_train, x_test, y_train, y_test = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

#%%
# Create Nueron Network Model
model = Sequential()
# Add BatchNormalization after each dense layer
model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

#%%
ABC_predict = model.predict(x_test)

A_predict = ABC_predict[:,0]
B_predict = ABC_predict[:,1]
C_predict = ABC_predict[:,2]

A_actual = y_test["A"]
B_actual = y_test["B"]
C_actual = y_test["C"]

def Psat_cal(T,A,B,C):
    return np.log10(A+(B/(T+C)))

Psat_predict = Psat_cal(100, A_predict, B_predict, C_predict)
Psat_antione = Psat_cal(100, A_actual, B_actual, C_actual)

plt.scatter(Psat_predict, Psat_antione)

