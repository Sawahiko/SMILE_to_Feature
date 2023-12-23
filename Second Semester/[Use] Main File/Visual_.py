#%% Import
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

#%% Import Data

df_compare_raw = pd.read_csv("../[Use] Data Preparation/Psat_XGB df_compare.csv")
df_fil_sort = df_compare_raw.sort_values(by='diff')
df_fil = df_fil_sort.tail(len(df_fil_sort) - 0)
df_compare = df_fil.copy()
df_compare = df_compare.query("Actual >-10")

act = df_compare["Actual"]; pre = df_compare["Predict"]

#%% Evaluation
from sklearn.metrics import  mean_absolute_error as mae
from sklearn.metrics import  mean_squared_error as mse
from sklearn.metrics import  r2_score
print(f'mae log(Psat): {mae(df_compare["Actual"], df_compare["Predict"])}')
print(f'rmse log(Psat): {mse(df_compare["Actual"], df_compare["Predict"], squared=False)}')
print(f'r2 log(Psat): {r2_score(df_compare["Actual"], df_compare["Predict"])}')
#%% Visual
#x_min = min(min(df_compare["Actual"]),min(df_compare["Predict"]))
#x_max = max(max(df_compare["Actual"]),max(df_compare["Predict"]))

x_min = -20; x_max = 25

y_min, y_max = x_min, x_max

#plt.yscale("linear")

# Define desired figure width and height in inches
width = 6
height = 6

# Create the figure with specified size
plt.figure(figsize=(width, height))

x = np.linspace(x_min, x_max, 100); y = x
plt.plot(x, y, color='black',linestyle='dashed', label='x=y')
plt.scatter(df_compare["Actual"], df_compare["Predict"], label="test", alpha=0.4)

plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
plt.title("$P_{sat}$  |Input = C-MF,T  |ML=XGB")
plt.xlabel("Actual $LogP_{sat}$ [Pa]")
plt.ylabel("Predict $LogP_{sat}$ [Pa]")

plt.legend()
#Score_table = Scoring(model , x_train_fp, x_test_fp, x_data_fp, y_train_fp, y_test_fp, y_data_fp)

#%%
x = np.linspace(x_min, x_max, 100)
y = np.zeros(len(x))


# Define desired figure width and height in inches
width = 6
height = 6

# Create the figure with specified size
plt.figure(figsize=(width, height))
plt.plot(x, y, color='black',linestyle='dashed', label='x=y')
#plt.scatter(df_filtered2["Actual"], df_filtered2["Predict"], label="train", alpha=0.3)
plt.scatter(df_compare["Actual"], df_compare["ARD"], label="test", alpha=0.4)


plt.xlim(x_min, x_max)
plt.title("%Error $P_{sat}$  |Input = C-MF,T  |ML=XGB")
plt.xlabel("Actual $LogP_{sat}$ [Pa]")
plt.ylabel("ARD $LogP_{sat}$ [%]")

#plt.yscale("log")
plt.legend()
