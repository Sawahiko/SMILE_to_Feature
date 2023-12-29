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
from Python_RemoveO import remove_outliers, remove_outliers_boxplot

#%%

# Import Data
# =============================================================================
# df = pd.read_excel("../[Use] Data Preparation/Psat_AllData_1.xlsx",sheet_name="All")
# df = df[df['SMILES'] != "None"].reset_index(drop=True)
# =============================================================================

columns_ouliers = ["A", "B", "C"]
df = remove_outliers_boxplot("../[Use] Data Preparation/Psat_AllData_1.xlsx", 'All', columns_ouliers)
df = df[df['SMILES'] != "None"].reset_index(drop=True)
#df.to_csv('New_Data_Psat_Not_Outliers.csv')

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df[["A","B","C"]]
        
# Create individual boxplots for each group
plt.figure(figsize=(6, 4))  # Adjust figure size as needed
plt.boxplot(df["A"], labels=["A"], patch_artist=True)
plt.title("Box Plot of A")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.boxplot(df["B"], labels=["B"], patch_artist=True)
plt.title("Box Plot of B")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.boxplot(df["C"], labels=["C"], patch_artist=True)
plt.title("Box Plot of C")
plt.grid(True)
plt.show()

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
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


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

df_compare = pd.DataFrame({
    "A_Actual" : A_actual,
    "A_Predict" : A_predict,
    "B_Actual" : B_actual,
    "B_Predict" : B_predict,
    "C_Actual" : C_actual,
    "C_Predict" : C_predict})
     
plt.scatter(C_actual, C_predict)
plt.plot(C_actual, C_actual)
