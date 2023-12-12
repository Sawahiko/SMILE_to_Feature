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
import keras
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_logarithmic_error as msle
from keras.losses import mean_squared_error as mse
import numpy as np

# Some random training data
# =============================================================================
# labels_1 = y_train["A"]
# labels_2 = y_train["B"]
# labels_3 = y_train["C"]
# =============================================================================

labels_1 = y_train[:,0]
labels_2 = y_train[:,1]
labels_3 = y_train[:,2]

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
# =============================================================================
#     outputs = model.layers[1].output_mask
#     print(B)
#     print(type(A))
#     print(dir(A))
#     print(A.dtype)
#     print(A.__array__)
#     print(A.get_shape())
#     #print(K.eval(outputs))
# =============================================================================
    y_transform = tf.reshape([A,B,C], (1,-1))
    
    #A_num = A.
    A = A* K.constant(scaler.scale_[0]) + K.constant(scaler.mean_[0])
    B = B* K.constant(scaler.scale_[1]) + K.constant(scaler.mean_[1])
    C = C* K.constant(scaler.scale_[2]) + K.constant(scaler.mean_[2])
    return A-(B/(T+C))

Temp = 373
#loss_fun = K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2) * mse(output_3, label_layer_3))
loss_fun = K.mean(mse(Psat_cal_TF(Temp, output_1, output_2, output_3), Psat_cal_TF(Temp, label_layer_1, label_layer_2, label_layer_3)))

# Add loss to model
model.add_loss(loss_fun)

# Compile without specifying a loss
model.compile(optimizer='adam')

dummy = np.zeros(x_train.shape[0])
model.fit([x_train, labels_1, labels_2, labels_3], dummy, epochs=100)
a = K.eval(np.array([0.1,0.3]))

#%%
model2 = model
A_actual = y_test_fp["A"]
B_actual = y_test_fp["B"]
C_actual = y_test_fp["C"]


ABC_predict = model.predict([x_test, A_actual, B_actual, C_actual])
ABC_predict = np.dstack(ABC_predict).reshape(x_test.shape[0],3)
ABC_predict = scaler.inverse_transform(ABC_predict )
                          
A_predict = ABC_predict[:,0]
B_predict = ABC_predict[:,1]
C_predict = ABC_predict[:,2]

def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))
Psat_predict = Psat_cal(Temp , A_predict, B_predict, C_predict)
Psat_antione = Psat_cal(Temp , A_actual, B_actual, C_actual)

plt.xscale("linear")
plt.yscale("linear")

x_min = min(Psat_antione)
x_max = 25 #max(Psat_antione)
x_min = -15; x_max = 19
y_min, y_max = x_min, x_max

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

x = np.linspace(x_min, x_max, 100)
y = x
plt.plot(x, y, color='black',linestyle='dashed', label='x=y')

plt.scatter(Psat_antione, Psat_predict)



df_pow = pd.DataFrame({
    "Psat_antio" : pow(10, Psat_antione),
    "Psat_pree" : pow(10, Psat_predict),
     
})

df_compare = pd.DataFrame({
    "Psat_antio" : Psat_antione,
    "Psat_pree" : Psat_predict
})
df_compare["Log(Psat) Diff"]= abs(df_compare["Psat_antio"]-df_compare["Psat_pree"])
df_compare.describe()
from sklearn.metrics import  mean_absolute_error as mae
from sklearn.metrics import  mean_absolute_percentage_error as mape
print(f'mae log(Psat): {mae(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
print(f'msle Psat: {msle(df_pow["Psat_antio"], df_pow["Psat_pree"])}')
#%%
plt.xscale("log")
plt.yscale("log")
Psat_predict_new = np.float64(pow(10, Psat_predict))
Psat_antione_new= np.float64(pow(10, Psat_antione))
val_min = min(min(Psat_predict_new), min(Psat_antione_new))
val_max = max(max(Psat_predict_new), max(Psat_antione_new))

val_max = pow(10, 19)
val_min = pow(10,-15)
plt.xlim(val_min , val_max )
plt.ylim(val_min , val_max )

x = np.linspace(val_min, val_max, 20)
y = x
plt.plot(x, y, color='black',linestyle='dashed', label='x=y')
plt.scatter(Psat_antione_new, Psat_predict_new)
