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
df = pd.read_excel("../[Use] Data Preparation/Psat_AllData_1.xlsx",sheet_name="CHON")
df = df[df['SMILES'] != "None"].reset_index(drop=True)

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df[["A","B","C"]]
        

# %% Data Preparation
# Generate Fingerprint from SMILE
MF_radius = 3
MF_bit = 4096

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
# =============================================================================
# x_train, x_test, y_train, y_test = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)
# =============================================================================
x_train, x_test, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# created scaler
scaler = MinMaxScaler()
# fit scaler on training dataset
scaler.fit(y_train_fp)
# transform training dataset
y_train = scaler.transform(y_train_fp)
# transform test dataset
y_test = scaler.transform(y_test_fp)
# %%

import keras.backend as K
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.losses import mean_squared_error as mse
import numpy as np

# Add BatchNormalization after each dense layer
model = Sequential()
model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(3))

# Define different weights for each output loss
output_weights = [0.3, 0.6, 0.1]

def custom_loss(y_true, y_pred):
  loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred) * output_weights)
  return loss

# =============================================================================
# model.compile(optimizer='adam', loss='LogCosh')
# =============================================================================
model.compile(optimizer='adam', loss=custom_loss)
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)


#%%
# =============================================================================
# A_actual = y_test["A"]
# B_actual = y_test["B"]
# C_actual = y_test["C"]
# =============================================================================

A_actual = y_test_fp["A"]
B_actual = y_test_fp["B"]
C_actual = y_test_fp["C"]

ABC_predict_raw = model.predict(x_test)
ABC_predict = ABC_predict_raw.copy()
ABC_predict = np.dstack(ABC_predict).reshape(x_test.shape[0],3)
ABC_predict = scaler.inverse_transform(ABC_predict)
                          
A_predict = ABC_predict[:,0]
B_predict = ABC_predict[:,1]
C_predict = ABC_predict[:,2]

Temp = 373
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))
Psat_predict = Psat_cal(Temp , A_predict, B_predict, C_predict)
Psat_antione = Psat_cal(Temp , A_actual, B_actual, C_actual)



df_pow = pd.DataFrame({
    "Psat_antio" : pow(Psat_antione, 10),
    "Psat_pree" : pow(Psat_predict, 10),
     
})

df_compare = pd.DataFrame({
    "Psat_antio" : Psat_antione,
    "Psat_pree" : Psat_predict
})
df_compare["diff"] = abs(df_compare["Psat_antio"]- df_compare["Psat_pree"])
df_compare_des = df_compare.describe()

#%%
#Score_table = Scoring(model , x_train, x_test, x_data_fp, y_train, y_test, y_data_fp)

from sklearn.metrics import  mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score, mean_squared_error as rmse

print(f'mae test: {mae(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
print(f'rmse test: {rmse(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
print(f'mape test: {mape(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
print(f'r2 test: {r2_score(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
#%% Visualization
x_min = min(min(Psat_antione),min(Psat_predict))
x_max = max(max(Psat_antione),max(Psat_predict))

#x_min = -20; x_max = 25

y_min, y_max = x_min, x_max

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

x = np.linspace(x_min, x_max, 100)
y = x
p1 = plt.plot(x, y, color='black',linestyle='dashed', label='x=y')

plt.scatter(Psat_antione, Psat_predict, label="test", alpha=0.3)
plt.legend()
