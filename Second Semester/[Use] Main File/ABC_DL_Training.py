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

#Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import ray

# Our module
from Python_Scoring_Export import Scoring, Export
from Python_RemoveO import remove_outliers, remove_outliers_boxplot

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
#Y_data= df[["A"]]
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
# =============================================================================
# y_train = y_train_fp.values      # transform training dataset : y
# y_test = y_test_fp.values        # transform test dataset : y
# =============================================================================

# Setup Data for DL
inputs = torch.tensor(x_train.values, dtype=torch.float64)
labels = torch.tensor(y_train, dtype=torch.float64)
trainloader = TensorDataset(inputs, labels)
train_loader = DataLoader(trainloader, batch_size=32, shuffle=True)
inputs_test = torch.tensor(x_test.values, dtype=torch.float64)
labels_test = torch.tensor(y_test, dtype=torch.float64)
testloader = TensorDataset(inputs_test, labels_test)
test_loader = DataLoader(trainloader, batch_size=32, shuffle=False)

trainloader_id = ray.put(train_loader)
testloader_id = ray.put(test_loader)

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
model = ABC_2(N_Input=MF_bit, N_Output=3, N_Hidden=20, N_Layer=2, dropout_rate=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
model.to(device)

patience = 10  # Number of epochs to wait before stopping if no improvement

best_val_loss = float('inf')
epochs_no_improvement = 0
training_log = {"train_loss": [], "val_loss": []}
for epoch in range(500):  # loop over the dataset multiple times
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

    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improvement = 0
    else:
        epochs_no_improvement += 1

    if epochs_no_improvement >= patience:
        print('Early stopping triggered.')
        break

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
save_path = "ABC_New_Tune_BS32_2.pth"
torch.save(model.state_dict(), save_path)