# Python
import numpy as np
import pandas as pd
import time

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
from Python_MLModel import RF, Ridge_M, XGB, NN, CB, DT, SVR_M, KNN, XGB_Default
from Python_RemoveO import remove_outliers

old_df = pd.DataFrame({
    'MAE':[0], 'MAPE(%)':[0], 'RMSE':[0], 'R2':[0], 'Radius':[0], 'nBits':[0], 'Model':[0]
    })

# %% Option Many Bit
MF_bit_s = [2**8, 2**9, 2**10, 2**11, 2**12, 2**13]
MF_radius_s = [2, 3, 4, 5, 6]

# =============================================================================
# MF_bit_s = [2**12]
# MF_radius_s = [3]
# =============================================================================
Name_model = "XGB"
j=0

for MF_radius in MF_radius_s:
    for MF_bit in MF_bit_s :

        # %% Import Data : 560 datapoint
        # Import Data
        #df = remove_outliers("../Data.xlsx", "New_Data", 2)
        #df = remove_outliers("../Data.xlsx", "CHO", 2)
        df = pd.read_excel("../Data.xlsx",sheet_name="560point")
        #df = pd.read_excel("../Data.xlsx",sheet_name="Load_AllDataSetC12")
        #df = pd.read_excel("../Data.xlsx",sheet_name="Load_CHO")
        #df = pd.read_excel("../Data.xlsx",sheet_name="New_Data")
        
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
        sel_f = x_data_fp.sort_values(by=1, ascending=False, inplace=False)
        print(sel_f[1])
        #print(X_data_excel["SMILES"][3421])
        y_data_fp = Y_data.copy()
# =============================================================================
#         if MF_radius == 2:
#             if MF_bit == 1024:
#                 K = 440
#             elif MF_bit == 2048:
#                 K = 668
#             elif MF_bit == 4096:
#                 K = 893
#         elif MF_radius == 3:
#             if MF_bit == 1024:
#                 K = 556
#             elif MF_bit == 2048:
#                 K = 824
#             elif MF_bit == 4096:
#                 K = 1202
#         elif MF_radius == 4:
#             if MF_bit == 1024:
#                 K = 668
#             elif MF_bit == 2048:
#                 K = 972
#             elif MF_bit == 4096:
#                 K = 1417
#         k_best = SelectKBest(f_regression, k=K) # Select the top 200 features
#         x_new = k_best.fit_transform(x_data_fp, y_data_fp)
# =============================================================================

# =============================================================================
#         scores = k_best.scores_
# 
#         # Get the names of the selected features
#         selected_features = k_best.get_support()
#         feature_names = x_data_fp.columns
# 
#         # Print the scores of the selected features
#         for i, feature_name in enumerate(feature_names):
#             if selected_features[i]:
#                 print(f"Feature {feature_name}: {scores[i]:.2f}")
# =============================================================================
        
        # %%
        # Train-test_Modeling & Cross Validation Modeling
        
        x_train_fp, x_test_fp, y_train_fp, y_test_fp = train_test_split(x_data_fp, y_data_fp,
                                                                        test_size=0.1,
                                                                        random_state=42)
        # %%
        start_time = time.time()
        print(f'start_time : {start_time }')
        model = XGB_Default(x_train_fp, y_train_fp)
        end_time = time.time()
        print("Elasped Time : ", end_time-start_time, "seconds")
        
        # %%
        # Scoring & Export
        #Score_table = Scoring(model , x_train_fp, x_test_fp, x_data_fp, y_train_fp, y_test_fp, y_data_fp)
        Score_table = Scoring(model , x_train_fp, x_test_fp, x_data_fp, y_train_fp, y_test_fp, y_data_fp)
        y_pred_test = model.predict(x_test_fp)
        #Export(Score_table, "C_MF16384_XGB.csv")

        # %%
        
        df2 = pd.DataFrame({'Radius': [f"r= {MF_radius}", f"r= {MF_radius}", f"r= {MF_radius}"], 
                            'nBits': [f"n= {MF_bit}", f"n= {MF_bit}", f"n= {MF_bit}"],
                            'Model': [Name_model, Name_model, Name_model]
                                      })
        
        df = pd.concat([Score_table, df2], axis=1)
        
        df3 = pd.DataFrame({'Actual': y_test_fp,
                            'Predict': y_pred_test})
#        Export(df3, "C-MF 2023-11-11/Ridge_Test_Tb_Value.csv")
# %%
        if(j>0):
            old_df = df_combine.copy()
        new_df = df.copy()
        j+=1

        df_combine = pd.concat([old_df, new_df], ignore_index=True)
        
# %%
#Export(df_combine, "2024-03-15 CHON Fingerprint inspect.csv")
#Export(df_combine, "2024-03-15 CH Fingerprint inspect.csv")
