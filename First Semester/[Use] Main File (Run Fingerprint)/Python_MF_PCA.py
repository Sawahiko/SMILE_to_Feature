# Python
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

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
#from Python_MLModel import RF, Ridge_M, XGB, NN

# %% Option
MF_bit = 2**13
MF_radius = 2

# %% Import Data : 560 datapoint
# Import Data
df = pd.read_excel("../Data.xlsx",sheet_name="Load_AllDataSetC12")

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
n_components = 2048
pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x_data_fp)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.yticks(np.arange(0, 1.2, step=0.2))
plt.xticks(np.arange(0, 2048, step=200))
plt.plot([0,n_components], [0.75,0.75], '--')
plt.plot([0,n_components], [0.95,0.95], '--')
plt.plot([0,n_components], [1,1], '-')

plt.xlim([0, 2048])
plt.title(f" B-MF PCA From {MF_bit} bit, r={MF_radius} to {n_components} comp")
plt.xlabel('components')
plt.ylabel('cumulative explained variance');

# =============================================================================
# print("Total variation explained : {0} = {1:.2f} %".format(exp_var, sum(pca.explained_variance_ratio_*100)))
# print("Original shape (#instances, #features):   ",x_data_fp.shape)
# print("Transformed shape (#instances, #features):", x_pca.shape)
# =============================================================================
# %%

x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_pca, y_data_fp,
                                                                test_size=0.20,
  
                                                                random_state=42)
start_time = time.time()
RF_model = RF(x_train_fp, y_train_fp)
end_time = time.time()
print("Elasped Time : ", end_time-start_time, " seconds")

# %%
# Scoring & Export
Score_table = Scoring(RF_model , x_train_fp, x_test_fp, x_pca, y_train_fp, y_test_fp, y_data_fp)
Export(Score_table, "2023-10-09 MF4096_PCA512_RF r=6.csv")
