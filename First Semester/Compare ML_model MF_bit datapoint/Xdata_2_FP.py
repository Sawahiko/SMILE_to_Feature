import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw

def X2FP(X_data_excel, MF_radius, MF_bit) :
    X_data_use = X_data_excel.copy()
    #X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    
    m1 = X_data_use["molecule"][50]
    img1 = Draw.MolToImage(m1)
    
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
    X_data_fp_new = pd.concat(X_data_fp, ignore_index=True)
    return X_data_fp_new, X_data_use, i, datafram_i, array, img1