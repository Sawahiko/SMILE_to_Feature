# Python
import numpy as np
import pandas as pd
import time

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
from Python_MLModel import RF, Ridge_M, XGB, NN
from Python_RemoveO import remove_outliers


old_df = pd.DataFrame({
    'MAE':[0], 'MAPE(%)':[0], 'RMSE':[0], 'R2':[0], 'Radius':[0], 'nBits':[0], 'Model':[0]
    })

# =============================================================================
# # %% Option
# MF_bit = 2**5
# MF_radius = 6
# Name_model = "CB"
# =============================================================================

# %% Option Many Bit
MF_bit_s = [2**11]
MF_radius_s = [3]

#MF_bit_s = [2**5-1, 2**6-1]
#MF_radius_s = [3]
Name_model = "RF"
j=0
for MF_radius in MF_radius_s:
    for MF_bit in MF_bit_s :
        
        # %% 
        # Import Data
        df = pd.read_excel("../Data.xlsx",sheet_name="560point")
        #df = remove_outliers("../Data.xlsx", "New_Data", 2)

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
        
        
        x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,
                                                                        test_size=0.20,
                                                                        random_state=42)
        start_time = time.time()
        model = RF(x_train_fp, y_train_fp)
        
        end_time = time.time()
        print("Training Elasped Time : ", end_time-start_time, " seconds")
        
        
        # %%
        # Scoring & Export
        Score_table = Scoring(model , x_train_fp, x_test_fp, x_data_fp, y_train_fp, y_test_fp, y_data_fp)
        #Export(Score_table, "2023-10-10 MF1024_XGB r=6.csv")
        
        
        # %%
        
        df2 = pd.DataFrame({'Radius': [f"r= {MF_radius}", f"r= {MF_radius}", f"r= {MF_radius}"], 
                            'nBits': [f"n= {MF_bit}", f"n= {MF_bit}", f"n= {MF_bit}"],
                            'Model': [Name_model, Name_model, Name_model]
                                      })
        
        df = pd.concat([Score_table, df2], axis=1)
# %%
        if(j>0):
            old_df = df_combine.copy()
        new_df = df.copy()
        j+=1

        df_combine = pd.concat([old_df, new_df], ignore_index=True)
        
# %%
Export(df_combine, "B-MF 2023-10-23/RF.csv")
