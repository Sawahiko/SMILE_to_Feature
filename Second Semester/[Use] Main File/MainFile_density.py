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
df = pd.read_csv("Density.csv")
df = df[df['SMILES'] != "None"].reset_index(drop=True)

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df[["C1","C2","C3", "C4"]]
        

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
x_train, x_test, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
# created scaler
scaler = StandardScaler()
# fit scaler on training dataset
scaler.fit(y_train_fp)
# transform training dataset
y_train = scaler.transform(y_train_fp)
# transform test dataset
y_test = scaler.transform(y_test_fp)


# %%

import keras.backend as K
from keras.layers import Input, Dense, Normalization
from keras.models import Model
#from keras.losses import mean_squared_logarithmic_error as msle
from keras.losses import mean_squared_error as mse
import numpy as np

# Some random training data
# =============================================================================
# labels_1 = y_train["C1"]
# labels_2 = y_train["C2"]
# labels_3 = y_train["C3"]
# labels_4 = y_train["C4"]
# =============================================================================

labels_1 = y_train[:,0]
labels_2 = y_train[:,1]
labels_3 = y_train[:,2]
labels_4 = y_train[:,3]

# Input layer, one hidden layer
input_layer = Input((x_train.shape[1],))
#norm_input_layer = Normalization(axis=1, )(input_layer)
dense_1 = Dense(500, "relu")(input_layer)
dense_2 = Dense(100, "relu")(dense_1)
# Two outputs
output_1 = Dense(1)(dense_2)
output_2 = Dense(1)(dense_2)
output_3 = Dense(1)(dense_2)
output_4 = Dense(1)(dense_2)

# Two additional 'inputs' for the labels
label_layer_1 = Input((1,))
label_layer_2 = Input((1,))
label_layer_3 = Input((1,))
label_layer_4 = Input((1,))

# Instantiate model, pass label layers as inputs
model = Model(inputs=[input_layer, label_layer_1, label_layer_2, label_layer_3, label_layer_4], outputs=[output_1, output_2, output_3, output_4])

# Construct your custom loss as a tensor
# =============================================================================
# def loss(output_1, label_layer_1, output_2, label_layer_2) :
#     return K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2))
# =============================================================================


def rho_cal_TF(T,C1,C2,C3,C4):
    top = C1
    frac_exp = K.pow(C4,1+(1-(T/C3)))
    frac = K.pow(frac_exp,C2)
    return top/frac
def rho_cal_TF2(T,C1,C2,C3,C4):
    #top = K.log(C1)
    top = C1
    #print(C1)
    
    #frac_exp = K.pow(C4,1+(1-(T/C3)))
    frac_exp = C3+C4
    
    #frac_base = K.log(C2)
    frac_base = C2
    return top-frac_exp*frac_base

Temp = 373
#loss_fun = K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2) * mse(output_3, label_layer_3))
loss_fun = K.mean(mse(rho_cal_TF2(Temp, output_1, output_2, output_3, output_4), rho_cal_TF2(Temp, label_layer_1, label_layer_2, label_layer_3, label_layer_4)))

# Add loss to model
model.add_loss(loss_fun)

# Compile without specifying a loss
model.compile(optimizer='adam')

dummy = np.zeros(x_train.shape[0])
model.fit([x_train, labels_1, labels_2, labels_3, labels_4], dummy, epochs=100)


#%%
C1_actual= y_test[:,0]
C2_actual= y_test[:,1]
C3_actual= y_test[:,2]
C4_actual= y_test[:,3]

C1234_predict = model.predict([x_test, C1_actual, C2_actual, C3_actual, C4_actual])
C1234_predict = np.dstack(C1234_predict).reshape(x_test.shape[0],4)
C1234_predict  = scaler.inverse_transform(C1234_predict)
C1_predict = C1234_predict[:,0]
C2_predict = C1234_predict[:,1]
C3_predict = C1234_predict[:,2]
C4_predict = C1234_predict[:,3]

def rho_cal(T,C1,C2,C3,C4):
    top = C1
    frac_exp = pow(C4,1+(1-(T/C3)))
    frac = pow(frac_exp,C2)
    return top/frac
rho_predict = rho_cal(Temp , C1_predict, C2_predict, C3_predict, C4_predict)
rho_cal     = rho_cal(Temp , C1_actual, C2_actual, C3_actual, C4_actual)

# %%
x_min = min(rho_cal)
x_max = max(rho_cal)
#x_max = 4500
y_min, y_max = x_min, x_max

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

x = np.linspace(x_min, x_max, 100)
y = x
plt.plot(x, y, color='black',linestyle='dashed', label='x=y')

plt.scatter(rho_cal, rho_predict)

df_compare = pd.DataFrame({
    "rho_cal" : rho_cal,
    "rho_pre" : rho_predict
})
df_compare["rho Diff"]= abs(df_compare["rho_cal"]-df_compare["rho_pre"])
df_compare.describe()
from sklearn.metrics import  mean_absolute_error as mae
#print(f'mae : {mae(df_compare["rho_cal"], df_compare["rho_pre"])}')