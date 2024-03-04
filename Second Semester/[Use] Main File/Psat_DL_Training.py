# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import random

## Tool, Error Metric
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from joblib import dump, load

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
#import pytorch_lightning as L

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

#Deep Learning Add
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint

#%% Import Data
df = pd.read_csv("./Psat_NO_ABCTminTmaxC1-12.csv")
df = df[df['SMILES'] != "None"]
df = df.drop_duplicates(subset='SMILES').reset_index(drop=True)
df.sort_values(by="No.C")
#len(df["SMILES"].unique())

# New Train-Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42)
# =============================================================================
# random.seed(42)
# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]
# =============================================================================

# Genearate Temp in Tmin-Tmax and expand
df1 = train.copy()
    ## Function to generate equally distributed points
def generate_points(row, amount_point):
    start = row["Tmin"]; end = row["Tmax"];
    return np.linspace(start, end, amount_point)
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
print(df2.sort_values(by="Vapor_Presssure"))

# Fingerprint
# Parameter for Generate Morgan Fingerprint
MF_radius = 3;   MF_bit = 2048

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

# Genearate Temp in Tmin-Tmax and expand
df1 = test.copy()
    ## Function to generate equally distributed points
def generate_points(row, amount_point):
    start = row["Tmin"]; end = row["Tmax"];
    return np.linspace(start, end, amount_point)
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
print(df2.sort_values(by="Vapor_Presssure"))
#%%
# Fingerprint
# Parameter for Generate Morgan Fingerprint
MF_radius = 3;   MF_bit = 2048

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

#%% Setup Data for DL
inputs = torch.tensor(x_train_fp, dtype=torch.float64)
labels = torch.tensor(y_train_fp, dtype=torch.float64)
trainloader = TensorDataset(inputs, labels)
train_loader = DataLoader(trainloader, batch_size=32, shuffle=True)
inputs_test = torch.tensor(x_test_fp, dtype=torch.float64)
labels_test = torch.tensor(y_test_fp, dtype=torch.float64)
testloader = TensorDataset(inputs_test, labels_test)
test_loader = DataLoader(trainloader, batch_size=32, shuffle=False)

trainloader_id = ray.put(trainloader)
testloader_id = ray.put(testloader)

#Size=torch.tensor(x_train_fp).size(1)
#%% Tunning
class PSAT_DL(nn.Module):
    def __init__(self, N_Input, N_Output, N_Hidden, N_Layer, dropout_rate=0.2):
        super().__init__()
        activation = nn.ReLU
        self.fcs = nn.Sequential(
            nn.Linear(N_Input, N_Hidden),
            activation()
        )
        self.fch = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(N_Hidden, N_Hidden),
                nn.Dropout(p=dropout_rate),
                activation()
            ) for _ in range(N_Layer - 1)]
        )
        self.fce = nn.Linear(N_Hidden, N_Output)

        self.double()

    def forward(self, input):
        x = self.fcs(input)
        x = self.fch(x)
        x = self.fce(x)
        return x

# Define hyperparameters
N_Output = 1
dropout_rate = 0.2
learning_rate = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# for epoch in range(100):  # loop over the dataset multiple times
# #    trainloader = ray.get(trainloader_id)
# #    testloader = ray.get(testloader_id)
#     model.train()
#     train_loss = 0.0
#     for i, data in enumerate(train_loader):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)
# 
#         # zero the parameter gradients
#         optimizer.zero_grad()
# 
#         # forward + backward + optimize
#         outputs = model(inputs)
#         outputs = torch.reshape(outputs,(-1,))
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
# 
#         # print statistics
#         train_loss += loss.item()
#         
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():  # Disable gradient calculation during validation
#         for i, data in enumerate(test_loader):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
# 
#             output = model(inputs)
#             output = torch.reshape(output,(-1,))
# # =============================================================================
# #             print(labels.shape)
# #             print(torch.reshape(output,(-1,)).shape)
# # =============================================================================
#             val_loss += criterion(output, labels).item()
# 
#     # Logging training/validation performance
#     train_loss /= len(trainloader)
#     val_loss /= len(testloader)
#     training_log["train_loss"].append(train_loss)
#     training_log["val_loss"].append(val_loss)
#     print(f'Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
#     
# print("Finished Training")
# 
# =============================================================================

# Create a loop to vary the number of hidden layers and nodes
N_Output = 1
N_Hidden = 1000
N_Layer = 2
dropout_rate = 0.2
learning_rate = 0.0001
training_log = {"train_loss": [], "val_loss": [], "N_Hidden": [], "N_Layer": []}
model = PSAT_DL(N_Input=(MF_bit + 1), N_Output=N_Output, N_Hidden=N_Hidden, N_Layer=N_Layer, dropout_rate=dropout_rate)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(500):  # loop over the dataset multiple times
    model.train()
    train_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.reshape(outputs, (-1,))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation during validation
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            output = torch.reshape(output, (-1,))
            val_loss += criterion(output, labels).item()

    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    training_log["train_loss"].append(train_loss)
    training_log["val_loss"].append(val_loss)
    training_log["N_Hidden"].append(N_Hidden)
    training_log["N_Layer"].append(N_Layer)
    print(f'N_Layer: {N_Layer}, N_Hidden: {N_Hidden}, Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

print(f"Finished Training for N_Layer: {N_Layer}, N_Hidden: {N_Hidden}")

def plot_graph(history):
    # Sample data (replace with your actual data)
    epochs = range(len(history["train_loss"]))
    # Create the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.plot(epochs, history["train_loss"], label='Training Loss')
    plt.plot(epochs, history["val_loss"], label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Evaluation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_graph(training_log)
# =============================================================================
# train_df = pd.DataFrame(training_log)
# train_df.to_csv("Training_log.csv")
# =============================================================================
# Save Deep Learning Model
save_path = "Psat_H1000_L2_D2_500epoch.pth"
torch.save(model.state_dict(), save_path)
