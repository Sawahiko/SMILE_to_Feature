# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Tool, Error Metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from joblib import dump, load

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#%% Import data

x_train = pd.read_csv("csv_02-3 std_x_train.csv").iloc[:,1:]
y_train = pd.read_csv("csv_02-4 std_y_train.csv").iloc[:,1]
x_test  = pd.read_csv("csv_02-5 std_x_test.csv").iloc[:,1:]
y_test  = pd.read_csv("csv_02-6 std_y_test.csv").iloc[:,1]

scaler_x = load("file_02-1 scaler_x.joblib")
scaler_y = load("file_02-2 scaler_y.joblib")

#%% Setup Data for DL

#Training Set
inputs_train = torch.tensor(x_train.values, dtype=torch.float64)
labels_train = torch.tensor(y_train.values, dtype=torch.float64)
trainloader = TensorDataset(inputs_train, labels_train)
train_loader = DataLoader(trainloader, batch_size=32, shuffle=True)

#Test Set
inputs_test = torch.tensor(x_test.values, dtype=torch.float64)
labels_test = torch.tensor(y_test.values, dtype=torch.float64)
testloader = TensorDataset(inputs_test, labels_test)
test_loader = DataLoader(trainloader, batch_size=32, shuffle=False)

#%% Create Neural Network

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Check GPU
N_Output = 1
N_Layer = 2
N_Hidden = 1000
dropout_rate = 0.2
learning_rate = 0.0001

#Create Model
MF_bit = 2048
model = PSAT_DL(N_Input=(MF_bit + 1), N_Output=N_Output, N_Hidden=N_Hidden, N_Layer=N_Layer, dropout_rate=dropout_rate)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.001)
criterion = nn.MSELoss()
training_log = {"train_loss": [], "val_loss": [], "N_Hidden": [], "N_Layer": []}

for epoch in range(200):  # loop over the dataset multiple times
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

#%% Plot graph check training and validation loss (training)
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

#%% Predict
#inputs_test
model.to("cpu")
pred0 = model(inputs_train)
inv_pred0 = scaler_y.inverse_transform(pred0.detach().numpy())

pred1 = model(inputs_test)
inv_pred1 = scaler_y.inverse_transform(pred1.detach().numpy())

#Train Dataframe
df_com0 = pd.DataFrame({
    "ln_Psat_act" : scaler_y.inverse_transform(np.array(y_train).reshape(-1, 1)).flatten(),
    "ln_Psat_pred": inv_pred0.flatten(),
})
df_com0["Psat_act (atm)"] = np.exp(df_com0["ln_Psat_act"])/10**5
df_com0["Psat_pred (atm)"] = np.exp(df_com0["ln_Psat_pred"])/10**5

#Test Dataframe
df_com1 = pd.DataFrame({
    "ln_Psat_act" : scaler_y.inverse_transform(np.array(y_test).reshape(-1,1)).flatten(),
    "ln_Psat_pred": inv_pred1.flatten(),
})
df_com1["Psat_act (atm)"] = np.exp(df_com1["ln_Psat_act"])/10**5
df_com1["Psat_pred (atm)"] = np.exp(df_com1["ln_Psat_pred"])/10**5

#%% Visualization

x_min = min(min(df_com0["ln_Psat_act"]),min(df_com0["ln_Psat_pred"]))
x_max = max(max(df_com0["ln_Psat_act"]),max(df_com0["ln_Psat_pred"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com0["ln_Psat_act"], df_com0["ln_Psat_pred"], alpha=0.5)
plt.xlabel("Actual")    
plt.ylabel("Predictions")
plt.title("Psat")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%% Visualization

x_min = min(min(df_com1["ln_Psat_act"]),min(df_com1["ln_Psat_pred"]))
x_max = max(max(df_com1["ln_Psat_act"]),max(df_com1["ln_Psat_pred"]))
y_min, y_max = x_min, x_max

x = np.linspace(x_min, x_max, 100)
y = x

# PyPlot
plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
plt.scatter(df_com1["ln_Psat_act"], df_com1["ln_Psat_pred"], alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predictions")
plt.title("Psat")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%% Visualization

# =============================================================================
# x_min = -20
# x_max = 20
# y_min, y_max = x_min, x_max
# 
# x = np.linspace(x_min, x_max, 100)
# y = x
# 
# # PyPlot
# plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
# plt.scatter(df_com1["ln_Psat_act"], df_com1["ln_Psat_pred"], alpha=0.5)
# plt.xlabel("Actual")
# plt.ylabel("Predictions")
# plt.title("Psat")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# =============================================================================
