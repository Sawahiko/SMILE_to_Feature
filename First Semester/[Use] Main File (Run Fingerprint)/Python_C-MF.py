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
from Python_MLModel import RF, Ridge_M, XGB, NN, CB, DT, SVR_M, KNN, DT_Default
from Python_RemoveO import remove_outliers


df_func = pd.read_csv("Tb_Data_CHON_Func_Group.csv")
train, test= train_test_split(df_func, test_size=0.2,
                              random_state=42, stratify=df_func["Atom2"])
#%% For Modeling  

old_df = pd.DataFrame({
    'MAE':[0], 'MAPE(%)':[0], 'RMSE':[0], 'R2':[0], 'Radius':[0], 'nBits':[0], 'Model':[0]
    })

import itertools
#MF_bit_s = [2**9, 2**10, 2**11, 2**12] ; MF_radius_s = [2,3,4,5]
MF_bit_s = [2**12] ; MF_radius_s = [2]

#Name_model = ["XGB"] ; model_func = [XGB]
#Name_model = ["DT"] ; model_func = [DT_Default]  ## TEMPORARY ##
Name_model = ["Ridge", "KNN", "RF", "XGB"] ; model_func = [Ridge_M, KNN, RF, XGB]

MF_list = list(itertools.product(MF_bit_s, MF_radius_s ))
model_list = list(zip(Name_model, model_func))
MF_model_list = list(itertools.product(MF_list, model_list ))
j=0
for i in MF_model_list:
    (MF_bit, MF_radius),(model_name, model_func) = i
    print(i)

    #%% Data Preparation - Train
    X_data_excel= train[["SMILES"]]
    Y_data= train["Tb"]
    
    # Generate Fingerprint from SMILE
    X_data_use = X_data_excel.copy()
    X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    X_data_use["count_morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
        x, 
        radius=MF_radius, 
        nBits=MF_bit,
        useFeatures=True, useChirality=True))
    X_data_use["arr_count_morgan_fp"] = 0
    
    # Transfrom Fingerprint to Column in DataFrame
    X_data_fp = []
    def cb_convert_FP(row):
        blank_row = row
        blank_arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(row["count_morgan_fp"],blank_arr)
        datafram_i = pd.DataFrame(blank_arr)
        datafram_i = datafram_i.T
        return datafram_i
    X_data_fp = list(X_data_use.apply(lambda x: cb_convert_FP(x), axis=1))
    
    x_train_fp = pd.concat(X_data_fp)        
    y_train_fp = Y_data.copy()
    #%% Data Preparation - Test
    X_data_excel= test[["SMILES"]]
    Y_data= test["Tb"]
    
    # Generate Fingerprint from SMILE
    X_data_use = X_data_excel.copy()
    X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    X_data_use["count_morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
        x, 
        radius=MF_radius, 
        nBits=MF_bit,
        useFeatures=True, useChirality=True))
    X_data_use["arr_count_morgan_fp"] = 0
    
    # Transfrom Fingerprint to Column in DataFrame
    X_data_fp = []
    X_data_fp = list(X_data_use.apply(lambda x: cb_convert_FP(x), axis=1))
    
    x_test_fp = pd.concat(X_data_fp)
    y_test_fp = Y_data.copy()
    #%% Select ML Algorithm
    start_time = time.time()
    model = model_func(x_train_fp, y_train_fp)
    end_time = time.time()
    print("Elasped Time for Modeling: ", end_time-start_time, "seconds")
    
    #%% Scoring 
    x_data_fp = pd.concat([x_train_fp, x_test_fp], ignore_index=True)
    y_data_fp = pd.concat([y_train_fp, y_test_fp], ignore_index=True)
    Score_table = Scoring(model , x_train_fp, x_test_fp, x_data_fp, y_train_fp, y_test_fp, y_data_fp)
    y_pred_test = model.predict(x_test_fp)

    #%% Prepare Export for each MF_FP
    # Score Table 
    df2_r = pd.DataFrame({'Radius': [f"r= {MF_radius}", f"r= {MF_radius}", f"r= {MF_radius}"], 
                        'nBits': [f"n= {MF_bit}", f"n= {MF_bit}", f"n= {MF_bit}"],
                        'Model': [model_name, model_name, model_name]
                                  })
    df_r = pd.concat([Score_table, df2_r], axis=1)
    
    # Prediction Table 
    df3_r = pd.DataFrame({'SMIELS':test["SMILES"],  'Atom2':test["Atom2"],
                          'Func. Group':test["Func. Group"],
                          'Actual': y_test_fp, 'Predict': y_pred_test})
    Export(df3_r, f"Result & Visual/CHON 2024-03-23/{model_name}_Test_Tb_ValueP2.csv")
    #%% Prepare Export 2 - FP Insepction
    if(j>0):
        old_df = df_combine.copy()
    new_df = df_r.copy()
    j+=1

    df_combine = pd.concat([old_df, new_df], ignore_index=True)
        
#%% Prepare Export 3 - FP Insepction
Export(df_combine, f"Result & Visual/CHON 2024-03-23/{Name_model}_2.csv")


