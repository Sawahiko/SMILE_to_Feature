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
T_Test = generateTemp(df1["Tmin"], df1["Tmax"], 2)
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
x_train, x_test, y_train, y_test = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

# %%

import keras.backend as K
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.losses import mean_squared_error as mse
import numpy as np

# Some random training data
labels_0 = x_train[MF_bit]
T_actual = x_test[MF_bit]

labels_1 = y_train["A"]
labels_2 = y_train["B"]
labels_3 = y_train["C"]

x_train = x_train.drop(columns=MF_bit)
x_test = x_test.drop(columns=MF_bit)

model = Sequential()

# Add BatchNormalization after each dense layer
model.add(Dense(500, input_dim=x_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_split=0.2)


#%%

A_actual = y_test["A"]
B_actual = y_test["B"]
C_actual = y_test["C"]

dummy = np.zeros(x_test.shape[0])
ABC_predict = model.predict(x_test)
ABC_predict = np.dstack(ABC_predict).reshape(x_test.shape[0],3)
                          
A_predict = ABC_predict[:,0]
B_predict = ABC_predict[:,1]
C_predict = ABC_predict[:,2]

def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))
Psat_predict = Psat_cal(T_actual , A_predict, B_predict, C_predict)
Psat_antione = Psat_cal(T_actual , A_actual, B_actual, C_actual)



df_compare = pd.DataFrame({
    "Actual" : Psat_antione,
    "Predict" : Psat_predict
})
df_compare["diff"] = abs(df_compare["Actual"]- df_compare["Predict"])
df_compare = df_compare.query("Actual >-10")
df_compare_des = df_compare.describe()

#%%
#Score_table = Scoring(model , x_train, x_test, x_data_fp, y_train, y_test, y_data_fp)

from sklearn.metrics import  mean_absolute_error as mae, mean_absolute_percentage_error as mape, r2_score, mean_squared_error as mse

print(f'mae test: {mae(df_compare["Actual"], df_compare["Predict"])}')
print(f'rmse test: {mse(df_compare["Actual"], df_compare["Predict"], squared = False)}')
print(f'r2 test: {r2_score(df_compare["Actual"], df_compare["Predict"])}')

#%% Visual1
plt.xscale("linear")
plt.yscale("linear")

#x_min = min(min(df_compare["Actual"]), min(df_compare["Predict"])); 
#x_max = max(max(df_compare["Actual"]), max(df_compare["Predict"]))
x_min = -20; x_max = 25
y_min, y_max = x_min, x_max



# Define desired figure width and height in inches
width = 6
height = 6

# Create the figure with specified size
plt.figure(figsize=(width, height))

x = np.linspace(x_min, x_max, 100)
y = x
p1 = plt.plot(x, y, color='black',linestyle='dashed', label='x=y')
plt.scatter(df_compare["Actual"], df_compare["Predict"], label="test", alpha=0.4)    

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("$P_{sat}$  |Input = C-MF,T  |ML=DL|  Output=ABC")
plt.xlabel("Actual $LogP_{sat}$ [Pa]")
plt.ylabel("Predict $LogP_{sat}$ [Pa]")
plt.legend()