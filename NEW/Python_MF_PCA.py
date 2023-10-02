# Python
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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
from Python_MLModel import RF, Ridge_M, XGB, SVC_R, NN

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
x_data_fp = pd.concat(X_data_fp, ignore_index=True)

y_data_fp = Y_data.copy()

# %%
# Train-test_Modeling & Cross Validation Modeling
#Decrase feature with PCA
pca = PCA(n_components=512)
x_pca = pca.fit_transform(x_data_fp)
exp_var = pca.explained_variance_ratio_
# =============================================================================
# print("Total variation explained : {0} = {1:.2f} %".format(exp_var, sum(pca.explained_variance_ratio_*100)))
# print("Original shape (#instances, #features):   ",x_data_fp.shape)
# print("Transformed shape (#instances, #features):", x_pca.shape)
# =============================================================================
# %%

x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_pca, y_data_fp,
                                                                test_size=0.25,
                                                                random_state=42)
RF_model = RF(x_train_fp, y_train_fp)


# %%
# Scoring & Export
Score_table = Scoring(RF_model , x_train_fp, x_test_fp, x_pca, y_train_fp, y_test_fp, y_data_fp)
Export(Score_table, "MF1024_PCA512_RF.csv")
