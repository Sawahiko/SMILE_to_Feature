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
#from Python_Scoring_Export import Scoring, Export

#%%

# Import Data
df = pd.read_excel("../[Use] Data Preparation/Psat_AllData.xlsx",sheet_name="All")
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

# %%

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_logarithmic_error as msle
import numpy as np

# Some random training data
labels_1 = y_train["A"]
labels_2 = y_train["B"]
labels_3 = y_train["C"]

# Input layer, one hidden layer
input_layer = Input((x_train.shape[1],))
dense_1 = Dense(500, "relu")(input_layer)
dense_2 = Dense(100, "relu")(dense_1)
# Two outputs
output_1 = Dense(1)(dense_2)
output_2 = Dense(1)(dense_2)
output_3 = Dense(1)(dense_2)

# Two additional 'inputs' for the labels
label_layer_1 = Input((1,))
label_layer_2 = Input((1,))
label_layer_3 = Input((1,))

# Instantiate model, pass label layers as inputs
model = Model(inputs=[input_layer, label_layer_1, label_layer_2, label_layer_3], outputs=[output_1, output_2, output_3])

# Construct your custom loss as a tensor
# =============================================================================
# def loss(output_1, label_layer_1, output_2, label_layer_2) :
#     return K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2))
# =============================================================================


def Psat_cal_TF(T,A,B,C):
    return A-(B/(T+C))

Temp = 373
#loss_fun = K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2) * mse(output_3, label_layer_3))
loss_fun = K.mean(msle(Psat_cal_TF(Temp, output_1, output_2, output_3), Psat_cal_TF(Temp, label_layer_1, label_layer_2, label_layer_3)))

# Add loss to model
model.add_loss(loss_fun)

# Compile without specifying a loss
model.compile(optimizer='adam')

dummy = np.zeros(x_train.shape[0])
model.fit([x_train, labels_1, labels_2, labels_3], dummy, epochs=100)


#%%
A_actual = y_test["A"]
B_actual = y_test["B"]
C_actual = y_test["C"]
ABC_predict = model.predict([x_test, A_actual, B_actual, C_actual])
ABC_predict = np.dstack(ABC_predict).reshape(x_test.shape[0],3)
                          
A_predict = ABC_predict[:,0]
B_predict = ABC_predict[:,1]
C_predict = ABC_predict[:,2]

def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))
Psat_predict = Psat_cal(Temp , A_predict, B_predict, C_predict)
Psat_antione = Psat_cal(Temp , A_actual, B_actual, C_actual)


x_min, x_max = -5, 20
y_min, y_max = -5, 20

# =============================================================================
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# =============================================================================

x = np.linspace(x_min, x_max, 100)
y = x
plt.plot(x, y, color='black',linestyle='dashed', label='x=y')

plt.scatter(Psat_predict, Psat_antione)

from sklearn.metrics import  mean_absolute_percentage_error as mape
mape(Psat_antione, Psat_predict)

df_pow = pd.DataFrame({
    "Psat_antio" : pow(Psat_antione, 10),
    "Psat_pree" : pow(Psat_predict, 10)
})

df_compare = pd.DataFrame({
    "Psat_antio" : Psat_antione,
    "Psat_pree" : Psat_predict
})
df_compare["ABS"] = abs(df_compare["Psat_antio"]- df_compare["Psat_pree"])
df_compare.describe()
