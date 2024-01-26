# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

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
df = pd.read_csv("./RDKit_CHON_New_Data_Psat_Not_Outliers.csv")
df = df[df['SMILES'] != "None"]
df = df.drop_duplicates(subset='SMILES').reset_index(drop=True)
df.sort_values(by="No.C")
#len(df["SMILES"].unique())

# Genearate Temp in Tmin-Tmax and expand
df1 = df.copy()
    ## Function to generate equally distributed points
def generate_points(row, amount_point):
    start, end, num_points = row["Tmin"], row["Tmax"], amount_point
    step_size = (end - start) / (num_points - 1)
    return np.linspace(start, end, num_points)
df1["T"] = df1.apply(lambda x : generate_points(x, 2), axis=1)

df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)

# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])

# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
X_data= df2[["SMILES"]]               # feature: SMILE, T
Y_data= df2[["Vapor_Presssure"]]        # Target : Psat

# %% Fingerprint
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

# Normailzation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()                             # created scaler
y_data_fp = scaler.fit_transform(y_data_fp)         # fit scaler on training dataset

#%% Train-test split Data
x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

# Setup Data for DL
inputs = torch.tensor(x_train_fp.values, dtype=torch.float64)
labels = torch.tensor(y_train_fp, dtype=torch.float64)
trainloader = TensorDataset(inputs, labels)
inputs_test = torch.tensor(x_test_fp.values, dtype=torch.float64)
labels_test = torch.tensor(y_test_fp, dtype=torch.float64)
testloader = TensorDataset(inputs_test, labels_test)

trainloader_id = ray.put(trainloader)
testloader_id = ray.put(testloader)

Size=torch.tensor(x_train_fp.values).size(1)

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

        #self.learning_rate = tune.sample_from(lambda spec: tune.loguniform(1e-4, 1e-1))
        self.double()

    def forward(self, input):
        x = self.fcs(input)
        x = self.fch(x)
        x = self.fce(x)
        return x

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSAT_DL(N_Input=(MF_bit+1), N_Output=1, N_Hidden=config["N_Hidden"], N_Layer=config["N_Layer"], dropout_rate=config["dropout_rate"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        trainloader = ray.get(trainloader_id)
        testloader = ray.get(testloader_id)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
#        total = 0
#        correct = 0
        for i, data in enumerate(testloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
#                _, predicted = torch.max(outputs.data, 1)
#                total += labels.size(0)
#                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
#            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps)},
#                checkpoint=checkpoint,
            )
    print("Finished Training")

# Define the search space
search_space = {
    "N_Hidden": tune.choice([1000, 1500, 2000]),
    "N_Layer": tune.choice([2, 3, 4]),
    "dropout_rate": tune.uniform(0.2, 0.3),
    "learning_rate": tune.uniform(1e-1, 1e-2),
}

# Start the hyperparameter tuning

tuner = tune.Tuner(
    train_model,
    #tune.with_parameters(train_model),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        num_samples=6,
    ),
)

results = tuner.fit()
best_result = results.get_best_result("loss", "min")
best_result.config

import requests
url = 'https://notify-api.line.me/api/notify'
token = '3CfMWfczpal9Zye6bD72a8Ud6FWOODnBHQZHIWM1YU4'
headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}

msg = f'Psat Deep Learning run เสร็จแล้ว (10 epoch) \n config = {best_result.config}'
r = requests.post(url, headers=headers, data = {'message':msg})
print (r.text)
