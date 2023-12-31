# Python
import numpy as np
import pandas as pd
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
from rdkit import DataStructs

# Our module
from Python_Scoring_Export import Scoring, Export
from Python_MLModel import RF, Ridge_M, XGB, KNN
from Python_RemoveO import remove_outliers

# %% Option
MF_bit = 2**11
MF_radius = 2

# %% Import Data : 560 datapoint
# Import Data
df = remove_outliers("../Data.xlsx", "Load_AllDataSetC12", 2)
#df = pd.read_excel("../DataTb.xlsx",sheet_name="AllDataSet")
#df = pd.read_excel("../Data.xlsx",sheet_name="Load_AllDataSetC12")
#df = pd.read_excel("../Data.xlsx",sheet_name="Load_CHO")

# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df["Tb"]

# %% Data Preparation
# Generate Fingerprint from SMILE

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

y_data_fp = Y_data.copy()

# %%
# Train-test_Modeling & Cross Validation Modeling
pca = PCA(n_components=1000)
x_pca = pca.fit_transform(x_data_fp)
# =============================================================================
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.plot([0,4096], [0.75,0.75], '--')
# plt.plot([0,4096], [0.95,0.95], '--')
# plt.xlabel('components')
# plt.ylabel('cumulative explained variance');
# =============================================================================
# %%
x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_pca, y_data_fp,
                                                                test_size=0.2,
                                                                random_state=42)
model = KNN(x_train_fp, y_train_fp)


# %%
# Scoring & Export
Score_table = Scoring(model , x_train_fp, x_test_fp, x_pca, y_train_fp, y_test_fp, y_data_fp)
#Export(Score_table, "C_MF4096_RF_PCA1000.csv")
