# Python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

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
from Python_MLModel import RF, Ridge_M, XGB, NN, CB, DT, SVR_M, KNN
from Python_RemoveO import remove_outliers

MF_bit = 2**11
MF_radius = 2
Name_model = "RF"
j=0

df = remove_outliers("../Data.xlsx", "New_Data", 2)
# Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df["Tb"] 

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

#k_best = SelectKBest(f_regression, k=500) # Select the top 200 features
#x_new = k_best.fit_transform(x_data_fp, y_data_fp)

x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp, test_size=0.1, random_state=42)
out_list = []
for column in x_train_fp.columns:
    corr_tuple = pearsonr(x_train_fp[column], y_train_fp)
    out_list.append([column, corr_tuple[0], corr_tuple[1]])

corr_df = pd.DataFrame(out_list, columns=["Features", "Correlation", "P-Value"])
    
#scores = k_best.scores_

#selected_features = k_best.get_support()
#feature_names = x_data_fp.columns
        
# Print the scores of the selected features
#for i, feature_name in enumerate(feature_names):
#    if selected_features[i]:
#        print(f"Feature {feature_name}: {scores[i]:.2f}")
        
