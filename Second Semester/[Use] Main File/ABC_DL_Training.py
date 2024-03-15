# Python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# RDKit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

#Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
df = pd.read_csv("../Refactor Code/csv-01-0 Psat-1800.csv")

df = df[df['SMILES'] != "None"].reset_index(drop=True)

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df[["A","B","C"]]

# %% Fingerprint
# Generate Fingerprint from SMILE
MF_radius = 3
MF_bit = 4096
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Func. Group"])

X_train = train[['SMILES']]
Y_train = train[['A','B','C']]

X_test = test[['SMILES']]
Y_test = test[['A','B','C']]

X_train_use = X_train.copy()
X_train_use["molecule"] = X_train_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
X_train_use["count_morgan_fp"] = X_train_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
    x,
    radius=MF_radius,
    nBits=MF_bit,
    useFeatures=True, useChirality=True))
X_train_use["arr_count_morgan_fp"] = 0

X_test_use = X_test.copy()
X_test_use["molecule"] = X_test_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
X_test_use["count_morgan_fp"] = X_test_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
    x,
    radius=MF_radius,
    nBits=MF_bit,
    useFeatures=True, useChirality=True))
X_train_use["arr_count_morgan_fp"] = 0

X_train_use = X_train_use.reset_index(drop=True)
X_test_use  =X_test_use.reset_index(drop=True)

# Transfrom Fingerprint to Column in DataFrame
X_data_train_fp = []
for i in range(X_train_use.shape[0]):
    #print(np.array(X_data_use["morgan_fp"][i]))
    blank_arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(X_train_use["count_morgan_fp"][i],blank_arr)
    datafram_i = pd.DataFrame(blank_arr)
    datafram_i = datafram_i.T
    X_data_train_fp.append(datafram_i)
X_data_train_fp = pd.concat(X_data_train_fp, ignore_index=True)
X_data_train_fp = X_data_train_fp.astype(np.float32)

X_data_test_fp = []
for i in range(X_test_use.shape[0]):
    #print(np.array(X_data_use["morgan_fp"][i]))
    blank_arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(X_test_use["count_morgan_fp"][i],blank_arr)
    datafram_i = pd.DataFrame(blank_arr)
    datafram_i = datafram_i.T
    X_data_test_fp.append(datafram_i)
X_data_test_fp = pd.concat(X_data_test_fp, ignore_index=True)
X_data_test_fp = X_data_test_fp.astype(np.float32)

x_train = X_data_train_fp.copy()
y_train = Y_train.copy()

x_test = X_data_test_fp.copy()
y_test = Y_test.copy()

# Normailzation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()                     # created scaler
scaler.fit(y_train)                      # fit scaler on training dataset
y_train = scaler.transform(y_train)      # transform training dataset : y
y_test = scaler.transform(y_test)        # transform test dataset : y

#%%
## Setup Data for DL
# Train
inputs = torch.tensor(x_train.values, dtype=torch.float64)
labels = torch.tensor(y_train, dtype=torch.float64)
trainloader = TensorDataset(inputs, labels)
train_loader = DataLoader(trainloader, batch_size=32, shuffle=True)
# Test
inputs_test = torch.tensor(x_test.values, dtype=torch.float64)
labels_test = torch.tensor(y_test, dtype=torch.float64)
testloader = TensorDataset(inputs_test, labels_test)
test_loader = DataLoader(trainloader, batch_size=32, shuffle=False)

#%% Training for prediction
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABC_2(N_Input=MF_bit, N_Output=3, N_Hidden=1000, N_Layer=3, dropout_rate=0.25)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
model.to(device)

training_log = {"train_loss": [], "val_loss": []}
for epoch in range(200):  # loop over the dataset multiple times
#    trainloader = ray.get(trainloader_id)
#    testloader = ray.get(testloader_id)
    model.train()
    train_loss = 0.0
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
        train_loss += loss.item()
        
    model.eval()
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation during validation
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = model(inputs)
            val_loss += criterion(output, labels).item()

    # Logging training/validation performance
    train_loss /= len(trainloader)
    val_loss /= len(testloader)
    training_log["train_loss"].append(train_loss)
    training_log["val_loss"].append(val_loss)
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

print("Finished Training")

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

# Save Deep Learning Model
#save_path = "ABC_D2_H1000_N2_BS32_final.pth"
#torch.save(model.state_dict(), save_path)

#%%
model.to('cpu')
df_name_smile = df[['Name','SMILES']]
df_T = (df["Tmin"]+df["Tmax"])/2
train_T = pd.concat([df_name_smile, Y_train, df_T], axis=1, join="inner")
test_T = pd.concat([df_name_smile, Y_test ,df_T], axis=1, join="inner")

pred0 = model(torch.DoubleTensor(x_train.values)) #Train
pred1 = model(torch.DoubleTensor(x_test.values)) #Test
inv_pred0 = scaler.inverse_transform(pred0.detach().numpy())
inv_pred1 = scaler.inverse_transform(pred1.detach().numpy())

#Train
df_com0 = pd.DataFrame({
    "Name": train_T["Name"],
    "SMILES": train_T["SMILES"],
    "A_act" : Y_train["A"],
    "A_pred": inv_pred0[:,0],
    "B_act" : Y_train["B"],
    "B_pred": inv_pred0[:,1],
    "C_act" : Y_train["C"],
    "C_pred": inv_pred0[:,2],
    "T": train_T.iloc[:,-1],
})

df_com0["lnPsat_Act"] = df_com0["A_act"]-(df_com0["B_act"]/(df_com0["T"]+df_com0["C_act"]))
df_com0["lnPsat_Cal"] = df_com0["A_pred"]-(df_com0["B_pred"]/(df_com0["T"]+df_com0["C_pred"]))

#Test
df_com1 = pd.DataFrame({
    "Name": test_T["Name"],
    "SMILES": test_T["SMILES"],
    "A_act" : Y_test["A"],
    "A_pred": inv_pred1[:,0],
    "B_act" : Y_test["B"],
    "B_pred": inv_pred1[:,1],
    "C_act" : Y_test["C"],
    "C_pred": inv_pred1[:,2],
    "T": test_T.iloc[:,-1],
})

df_com1["lnPsat_Act"] = df_com1["A_act"]-(df_com1["B_act"]/(df_com1["T"]+df_com1["C_act"]))
df_com1["lnPsat_Cal"] = df_com1["A_pred"]-(df_com1["B_pred"]/(df_com1["T"]+df_com1["C_pred"]))

#%% Metrics Test
mae_test = mean_absolute_error(df_com1["A_act"], df_com1["A_pred"]), mean_absolute_error(df_com1["B_act"], df_com1["B_pred"]), mean_absolute_error(df_com1["C_act"], df_com1["C_pred"]), mean_absolute_error(df_com1["lnPsat_Act"], df_com1["lnPsat_Cal"])
mape_test = mean_absolute_percentage_error(df_com1["A_act"], df_com1["A_pred"]), mean_absolute_percentage_error(df_com1["B_act"], df_com1["B_pred"]), mean_absolute_percentage_error(df_com1["C_act"], df_com1["C_pred"]), mean_absolute_percentage_error(df_com1["lnPsat_Act"], df_com1["lnPsat_Cal"])
mse_test = mean_squared_error(df_com1["A_act"], df_com1["A_pred"]), mean_squared_error(df_com1["B_act"], df_com1["B_pred"]), mean_squared_error(df_com1["C_act"], df_com1["C_pred"]), mean_squared_error(df_com1["lnPsat_Act"], df_com1["lnPsat_Cal"])
R2_test = r2_score(df_com1["A_act"], df_com1["A_pred"]),r2_score(df_com1["B_act"], df_com1["B_pred"]),r2_score(df_com1["C_act"], df_com1["C_pred"]), r2_score(df_com1["lnPsat_Act"], df_com1["lnPsat_Cal"])

print("MAE","\nA = ",mae_test[0],"\nB = ",mae_test[1],"\nC = ",mae_test[2],"\nLogVP = ", mae_test[3])
print("MAPE","\nA = ",mape_test[0],"\nB = ",mape_test[1],"\nC = ",mape_test[2],"\nLogVP = ", mape_test[3])
print("RMSE","\nA = ",np.sqrt(mse_test[0]),"\nB = ",np.sqrt(mse_test[1]),"\nC = ",np.sqrt(mse_test[2]),"\nLogVP = ", np.sqrt(mse_test[3]))
print("R2","\nA = ",R2_test[0],"\nB = ",R2_test[1],"\nC = ",R2_test[2],"\nLogVP = ", R2_test[3])

#%% Visualization

x_min = min(min(df_com1["A_pred"]),min(df_com1["A_act"]))
x_max = max(max(df_com1["A_pred"]),max(df_com1["A_act"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com1["A_act"], df_com1["A_pred"], alpha=0.5, color='black')
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("A")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%%
x_min = min(min(df_com1["B_pred"]),min(df_com1["B_act"]))
x_max = max(max(df_com1["B_pred"]),max(df_com1["B_act"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com1["B_act"], df_com1["B_pred"], alpha=0.5, color='black')
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("B")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%%
x_min = min(min(df_com1["C_pred"]),min(df_com1["C_act"]))
x_max = max(max(df_com1["C_pred"]),max(df_com1["C_act"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com1["C_act"], df_com1["C_pred"], alpha=0.5, color='black')
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("C")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%%
x_min = min(min(df_com1["lnPsat_Cal"]),min(df_com1["lnPsat_Act"]))
x_max = max(max(df_com1["lnPsat_Cal"]),max(df_com1["lnPsat_Act"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com1["lnPsat_Act"], df_com1["lnPsat_Cal"], alpha=0.5, color='black')
plt.xlabel("Actual")
plt.ylabel("Calculation")
plt.title("lnVP")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%%
df_com1["Vapor_Pressure_Act"] = np.exp(df_com1['lnPsat_Act'])/10**5
df_com1["Vapor_Pressure_Cal"] = np.exp(df_com1['lnPsat_Cal'])/10**5

#%%
x_min = min(min(df_com1["Vapor_Pressure_Cal"]),min(df_com1["Vapor_Pressure_Act"]))
x_max = max(max(df_com1["Vapor_Pressure_Cal"]),max(df_com1["Vapor_Pressure_Act"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com1["Vapor_Pressure_Act"], df_com1["Vapor_Pressure_Cal"], alpha=0.5, color='black')
plt.xlabel("Vapor Pressure (atm) Actual")
plt.ylabel("Vapor Pressure (atm) Cal")
plt.title("Vapor Pressure (atm)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%%
df_com0.to_csv("Train_Act_Pred_Cal.csv")
df_com1.to_csv("Test_Act_Pred_Cal.csv")