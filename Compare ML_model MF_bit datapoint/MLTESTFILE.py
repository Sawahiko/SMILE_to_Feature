# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn
import time


# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# RDKit
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Our module
from Python_Scoring_Export import Scoring, Export
from MLModel import RF, Ridge, XGB
from Xdata_2_FP import X2FP

# %% Option
MF_bit = 2**10
MF_radius = 4

# %% Import Data : 560 datapoint
# Import Data
df = pd.read_excel("../DataTb.xlsx",sheet_name="AllDataSet")

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df["Tb"]

# %% Data Preparation
# Generate Fingerprint from SMILE

X_data_use = X_data_excel.copy()
X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
X_data_use["morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(
        x, 
        radius=MF_radius, 
        nBits=MF_bit, 
        useFeatures=True, useChirality=True))

# Transfrom Fingerprint to Column in DataFrame
X_data_fp = []
for i in range(X_data_use.shape[0]):
    #print(np.array(X_data_use["morgan_fp"][i]))
    array = np.array(X_data_use["morgan_fp"][i])
    datafram_i = pd.DataFrame(array)
    datafram_i = datafram_i.T
    X_data_fp.append(datafram_i)
X_data_fp = pd.concat(X_data_fp, ignore_index=True)

Y_data_fp = Y_data.copy()

#Decrase feature with PCA
pca = PCA(n_components=256)
X_pca = pca.fit_transform(X_data_fp)

# %%
# Train-test_Modeling & Cross Validation Modeling
X_train_fp, X_test_fp, y_train_fp, y_test_fp = train_test_split(X_pca, Y_data_fp,
                                                                test_size=0.25,
                                                                random_state=42)
#RF_model = RF(X_train_fp, y_train_fp)
#Ridge_model = Ridge(X_train_fp, y_train_fp)
XGB_model = XGB(X_train_fp, y_train_fp)
# %%
# Scoring & Export
Score_table = Scoring(XGB_model , X_train_fp, X_test_fp, X_data_fp, y_train_fp, y_test_fp, Y_data_fp)
#Export(Score_table, "TESTFILE.csv")
