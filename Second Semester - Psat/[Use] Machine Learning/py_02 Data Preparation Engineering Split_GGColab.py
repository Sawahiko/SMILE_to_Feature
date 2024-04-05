# Python
import numpy as np
import pandas as pd

# Tool, Error Metric
from sklearn.model_selection import train_test_split

# RDKit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs


#%% Import Data
df_original = pd.read_csv("/content/SMILE_to_Feature/Second Semester/Refactor Code/csv-01-0 Psat-1800.csv")
filter1 = df_original["SMILES"].str.contains("\+")
#filter2 = df["SMILES"].str.contains("\-")
filter3 = df_original["SMILES"].str.contains("\.")
print(filter1.sum(), filter3.sum())
f = filter1 +filter3 
f.sum()
#df = df[~f]
#df = df_original[~f]
df = df_original.copy()
#%%
# New Train-Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Func. Group"])

train_out = train.groupby("Func. Group").agg({'SMILES': ['count']})
test_out = test.groupby("Func. Group").agg({'SMILES': ['count']})
print(pd.concat([train_out, test_out ], axis=1))
#%% 
# Genearate Temp in Tmin-Tmax and expand
df1 = train.copy()
    ## Function to generate equally distributed points
def generate_points(row, amount_point):
    start = row["Tmin"]; end = row["Tmax"];
    range_temp = end-start
    if range_temp>0:
        return np.linspace(start, end, amount_point)
    else:
        return np.linspace(start, end, 1)
df1["T"] = df1.apply(lambda x : generate_points(x, 5), axis=1)

df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)

# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])
#test = df1[df1["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])]
#print(test[['SMILES','Vapor_Presssure']].sort_values(by="Vapor_Presssure").head(5))
#print(f"\n {test[['SMILES','Vapor_Presssure']].index} ")
# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
df2 = df2[~df2["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])].reset_index()
#print(df2[['SMILES','Vapor_Presssure']].sort_values(by="Vapor_Presssure").head(5))
X_data= df2[["SMILES"]]               # feature: SMILE, T
Y_data= df2[["Vapor_Presssure"]]        # Target : Psat

df2_train = df2.copy()
print(df2_train.sort_values(by="Vapor_Presssure"))

# Fingerprint
# Parameter for Generate Morgan Fingerprint
MF_radius = 4;   MF_bit = 4096

# Generate Fingerprint from SMILE
X_data_use = X_data.copy()
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

# Final Data for Training
x_data_fp[MF_bit] = df2["T"]      # Input  = Fingerprint + Temp
y_data_fp = Y_data.copy()         # Output = Vapor Pressure

x_train_notz = x_data_fp.copy()
y_train_notz = np.ravel(y_data_fp.copy())

#%%
# Genearate Temp in Tmin-Tmax and expand
df1 = test.copy()
    ## Function to generate equally distributed points
df1["T"] = df1.apply(lambda x : generate_points(x, 5), axis=1)

df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)
#print(test)
# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])

# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
df2 = df2[~df2["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])].reset_index()
X_data= df2[["SMILES"]]               # feature: SMILE, T
Y_data= df2[["Vapor_Presssure"]]        # Target : Psat
#df[df["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])]
df2_test = df2.copy()
print(df2_test.sort_values(by="Vapor_Presssure"))

# Fingerprint
# Parameter for Generate Morgan Fingerprint

# Generate Fingerprint from SMILE
X_data_use = X_data.copy()
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

# Final Data for Training
x_data_fp[MF_bit] = df2["T"]      # Input  = Fingerprint + Temp
y_data_fp = Y_data.copy()         # Output = Vapor Pressure

x_test_notz = x_data_fp.copy()
y_test_notz = np.ravel(y_data_fp.copy())

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scale_x= StandardScaler()
scale_x.fit(x_train_notz)
print(scale_x)
scale_y= StandardScaler()
scale_y.fit(y_train_notz.reshape(-1,1))
print(scale_y)
x_train_fp = scale_x.transform(x_train_notz)
x_test_fp  = scale_x.transform(x_test_notz)

y_train_fp = scale_y.transform(y_train_notz.reshape(-1,1)).flatten()
y_test_fp  = scale_y.transform(y_test_notz.reshape(-1,1)).flatten()

#%% Export Section
from joblib import dump, load
#df2_train.to_csv("csv_02-1 df_train.csv")
#df2_test.to_csv("csv_02-2 df_test.csv")

pd.DataFrame(x_train_fp).to_csv("/content/SMILE_to_Feature/Second Semester/Refactor Code/csv_02-3 std_x_train.csv")
#pd.DataFrame(y_train_fp).to_csv("csv_02-4 std_y_train.csv")
pd.DataFrame(x_test_fp).to_csv("/content/SMILE_to_Feature/Second Semester/Refactor Code/csv_02-5 std_x_test.csv")
#pd.DataFrame(y_test_fp).to_csv("csv_02-6 std_y_test.csv")

##dump(scale_x, "file_02-1 scaler_x.joblib")
#dump(scale_y, "file_02-2 scaler_y.joblib")
