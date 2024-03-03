# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Machine Learning
from sklearn.model_selection import train_test_split

#Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader, random_split

# Our module
from Python_Scoring_Export import Scoring, Export
from Python_RemoveO import remove_outliers, remove_outliers_boxplot

# Tuning Model
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch

# Another
import os
import tempfile
from filelock import FileLock

# %% Function

def Psat_cal(T,A,B,C):
    return A-(B/(T+C))
def generateTemp(Tmin, Tmax, amountpoint):
    Trange = Tmax-Tmin
    T_arr =[]
    for i in range(amountpoint):
        Tgen = Tmin+(Trange*random.random())
        T_arr.append(Tgen)
    return T_arr

#%%
df = pd.read_csv("./Psat_NO_ABCTminTmaxC1-12.csv")
df = df[df['SMILES'] != "None"].reset_index(drop=True)
#df.to_csv('New_Data_Psat_Not_Outliers.csv')

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df[["A","B","C"]]
        
df1 = df.copy()
T_Test = generateTemp(df1["Tmin"], df1["Tmax"], 5)
T_all = []
for i in range(len(T_Test[0])):
    T_gen_x_point = [item[i] for item in T_Test]
    T_all.append(T_gen_x_point)

df1["T"] = T_all
df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1.drop(df.columns[0], axis='columns', inplace=True)
df1 = df1.reset_index(drop=True)

Psat_test = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])
df1["Vapor_Presssure"] = Psat_test

df2 = df1[["SMILES", "T", "A", "B", "C", "Vapor_Presssure"]]
# Select feature for data: X=SMILE, Y=Tb
#X_data_excel= df2[["SMILES","T"]]
#Y_data= df2[["A", "B", "C"]]

X_data_excel= df[["SMILES"]]
Y_data= df[["A", "B", "C"]]
T_X_data_excel = df2[["T"]]

# %% Fingerprint
# Generate Fingerprint from SMILE
MF_radius = 3
MF_bit = 2048

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
x_data_fp[MF_bit] = df2["T"]
x_data_fp = x_data_fp.astype(np.float32)

y_data_fp = Y_data.copy()

#%% Train-test split
# Spilt Data
x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,test_size=0.2,random_state=42)

T_x_train = x_train_fp[[MF_bit]]; x_train = x_train_fp.drop(columns=MF_bit).astype(np.float64)
T_x_test  = x_test_fp[[MF_bit]]; x_test = x_test_fp.drop(columns=MF_bit).astype(np.float64)

# Normailzation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()                     # created scaler
scaler.fit(y_train_fp)                      # fit scaler on training dataset
y_train = scaler.transform(y_train_fp)      # transform training dataset : y
y_test = scaler.transform(y_test_fp)        # transform test dataset : y

# Setup Data for DL
inputs = torch.tensor(x_train.values, dtype=torch.float64)
labels = torch.tensor(y_train, dtype=torch.float64)
trainloader = TensorDataset(inputs, labels)
train_loader = DataLoader(trainloader, batch_size=16, shuffle=True)
inputs_test = torch.tensor(x_test.values, dtype=torch.float64)
labels_test = torch.tensor(y_test, dtype=torch.float64)
testloader = TensorDataset(inputs_test, labels_test)
test_loader = DataLoader(trainloader, batch_size=16, shuffle=False)

#trainloader_id = ray.put(train_loader)
#testloader_id = ray.put(test_loader)

#%%
class ABC_2(nn.Module):
    def __init__(self, N_Input, N_Output, N_Hidden, N_Layer, dropout_rate=0.5):
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
    model = ABC_2(N_Input=MF_bit, N_Output=3, N_Hidden=config["N_Hidden"], N_Layer=config["N_Layer"], dropout_rate=config["dropout_rate"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
# =============================================================================
#         train_loader = DataLoader(trainloader,batch_size=config["batch_size"])
#         test_loader = DataLoader(testloader,batch_size=config["batch_size"])
# =============================================================================
        for i, data in enumerate(train_loader):
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
            if i % 1000 == 999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps)},
#                checkpoint=checkpoint,
            )
    print("Finished Training")
        
# Define the search space
search_space = {
    "N_Hidden": tune.choice([128, 256, 512, 1024]),
    "N_Layer": tune.choice([2, 3, 4]),
    "dropout_rate": tune.quniform(0.2, 0.5, 0.1),
    "learning_rate": tune.choice([1e-2, 1e-3, 1e-4]),
#    "batch_size": tune.choice([16, 32, 64])
}

# Start the hyperparameter tuning

tuner = tune.Tuner(
    train_model,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        search_alg=OptunaSearch(),
        metric="loss",
        mode="min",
        num_samples=100,
    ),
)

results = tuner.fit()
best_result = results.get_best_result("loss", "min")
best_result.config

# =============================================================================
# dfs = {result.path: result.metrics_dataframe for result in results}
# 
# # Plot by epoch
# ax = None  # This plots everything on the same plot
# for d in dfs.values():
#     ax = d.loss.plot(ax=ax, legend=False)
# 
# =============================================================================
import requests
url = 'https://notify-api.line.me/api/notify'
token = 'dU86jOuWu70g5gT43y1ICB5fZcKANukkLqOFNLxRKjk'
headers = {'content-type':'application/x-www-form-urlencoded','Authorization':'Bearer '+token}

msg = f'run เสร็จแล้ว จากเทรน 100 แบบ 100 epoch (batch_size=16)\n(StandardScaler)\nconfig = {best_result.config}'
r = requests.post(url, headers=headers, data = {'message':msg})
print (r.text)
