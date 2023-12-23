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
from Python_Scoring_Export import Scoring, Export

def generateTemp(Tmin, Tmax, amountpoint):
    Trange = Tmax-Tmin
    T_arr =[]
    for i in range(amountpoint):
        Tgen = Tmin+(Trange*random.random())
        T_arr.append(Tgen)
    return T_arr
#%%
# Import Data
df = pd.read_excel("../[Use] Data Preparation/Psat_AllData_1.xlsx",sheet_name="All")
df = df[df['SMILES'] != "None"].reset_index(drop=True)

df1 = df.copy()
T_Test = generateTemp(df1["Tmin"], df1["Tmax"], 1)
T_all = []
for i in range(len(T_Test[0])):
    T_gen_x_point = [item[i] for item in T_Test]
    T_all.append(T_gen_x_point)

df1["T"] = T_all
df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.drop(columns={"Column1"})
df1 = df1.reset_index(drop=True)

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df1[["SMILES"]]
Y_data= df1[["A","B","C"]]
        

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
x_data_fp[MF_bit] = df1["T"]
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
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mean_squared_logarithmic_error as msle
import numpy as np

labels_0 = x_train[MF_bit]
T_actual = x_test[MF_bit]
#%%
labels_1 = y_train[:,0]
labels_2 = y_train[:,1]
labels_3 = y_train[:,2]

x_train = x_train.drop(columns=MF_bit)
x_test = x_test.drop(columns=MF_bit)

# Input layer, one hidden layer
input_layer = Input((x_train.shape[1],))
dense_1 = Dense(500, "relu")(input_layer)
dense_2 = Dense(100, "relu")(dense_1)
# Two outputs
output_1 = Dense(1)(dense_2)
output_2 = Dense(1)(dense_2)
output_3 = Dense(1)(dense_2)

# Two additional 'inputs' for the labels
label_layer_0 = Input((1,))
label_layer_1 = Input((1,))
label_layer_2 = Input((1,))
label_layer_3 = Input((1,))

# Instantiate model, pass label layers as inputs
model = Model(inputs=[input_layer, label_layer_0, label_layer_1, label_layer_2, label_layer_3], outputs=[output_1, output_2, output_3])

# Construct your custom loss as a tensor
# =============================================================================
# def loss(output_1, label_layer_1, output_2, label_layer_2) :
#     return K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2))
# =============================================================================
#%%
def Psat_cal_TF(T,A,B,C):
    A = A* K.constant(scaler.scale_[0]) + K.constant(scaler.mean_[0])
    B = B* K.constant(scaler.scale_[1]) + K.constant(scaler.mean_[1])
    C = C* K.constant(scaler.scale_[2]) + K.constant(scaler.mean_[2])
    return A-(B/(T+C))

#loss_fun = K.mean(mse(output_1, label_layer_1) * mse(output_2, label_layer_2) * mse(output_3, label_layer_3))
loss_fun = K.mean(msle(Psat_cal_TF(label_layer_0, output_1, output_2, output_3), Psat_cal_TF(label_layer_0, label_layer_1, label_layer_2, label_layer_3)))

# Add loss to model
#model.add_loss(loss_fun)

# Compile without specifying a loss
#model.compile(optimizer='adam')
model.compile(optimizer='adam', loss="mse")

dummy = np.zeros(x_train.shape[0])
model.fit([x_train, labels_0, labels_1, labels_2, labels_3], dummy, epochs=25)

#%%
A_actual = y_test_fp["A"]
B_actual = y_test_fp["B"]
C_actual = y_test_fp["C"]

dummy = np.zeros(x_test.shape[0])
ABC_predict = model.predict([x_test, T_actual, dummy, dummy, dummy])
ABC_predict = np.dstack(ABC_predict).reshape(x_test.shape[0],3)
ABC_predict = scaler.inverse_transform(ABC_predict )
                          
A_predict = ABC_predict[:,0]
B_predict = ABC_predict[:,1]
C_predict = ABC_predict[:,2]

#%%
def Psat_cal(T,A,B,C):
    return A-(B/(T+C))
Psat_predict = Psat_cal(T_actual , A_predict, B_predict, C_predict)
Psat_antione = Psat_cal(T_actual , A_actual, B_actual, C_actual)

# =============================================================================
# df_pow = pd.DataFrame({
#     "Psat_antio" : pow(10, Psat_antione),
#     "Psat_pree" : pow(10, Psat_predict),
#      
# })
# =============================================================================

df_compare = pd.DataFrame({
    "Psat_antio" : Psat_antione,
    "Psat_pree" : Psat_predict
})
df_compare["Log(Psat) Diff"]= abs(df_compare["Psat_antio"]-df_compare["Psat_pree"])
df_compare.describe()
#%% Evaluation
from sklearn.metrics import  mean_absolute_error as mae
from sklearn.metrics import  mean_squared_error as mse
from sklearn.metrics import  r2_score
print(f'mae log(Psat): {mae(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
print(f'rmse log(Psat): {mse(df_compare["Psat_antio"], df_compare["Psat_pree"])}')
print(f'r2 log(Psat): {r2_score(df_compare["Psat_antio"], df_compare["Psat_pree"])}')


#%% Visual1
plt.xscale("linear")
plt.yscale("linear")

x_min = min(min(Psat_antione), min(Psat_predict)); x_max = max(max(Psat_antione), max(Psat_predict))
#x_min = -15; x_max = 25
y_min, y_max = x_min, x_max

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Define desired figure width and height in inches
width = 6
height = 6

# Create the figure with specified size
plt.figure(figsize=(width, height))

x = np.linspace(x_min, x_max, 100)
y = x
p1 = plt.plot(x, y, color='black',linestyle='dashed', label='x=y')

plt.scatter(Psat_antione, Psat_predict, label="test", alpha=0.4)    
plt.legend()


#%% Visual2
# =============================================================================
# plt.xscale("log")
# plt.yscale("log")
# Psat_predict_new = np.float64(pow(10, Psat_predict))
# Psat_antione_new= np.float64(pow(10, Psat_antione))
# val_min = min(min(Psat_predict_new), min(Psat_antione_new))
# val_max = max(max(Psat_predict_new), max(Psat_antione_new))
# 
# val_max = pow(10, 19)
# val_min = pow(10,-15)
# plt.xlim(val_min , val_max )
# plt.ylim(val_min , val_max )
# 
# x = np.linspace(val_min, val_max, 20)
# y = x
# p2 = plt.plot(x, y, color='black',linestyle='dashed', label='x=y')
# plt.scatter(Psat_antione_new, Psat_predict_new)
# =============================================================================
