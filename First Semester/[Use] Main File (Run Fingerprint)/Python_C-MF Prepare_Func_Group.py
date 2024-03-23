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
from Python_MLModel import RF, Ridge_M, XGB, NN, CB, DT, SVR_M, KNN
from Python_RemoveO import remove_outliers_boxplot

start_time = time.time()
# %% Import Data : 560 datapoint
# Import Data
df_original = remove_outliers_boxplot("../Data.xlsx", "New_Data", ["Tb"])
#df_original = pd.read_excel("../Data.xlsx",sheet_name="560point")

# %% Get Atom Num in Molecule [1,6,7,8] 
df2 = df_original[["SMILES"]].copy()
def get_all_atomic_number(SMILES):
    def composition(molecule):
        """Get the composition of an RDKit molecule:
        Atomic counts, including hydrogen atoms, and any charge.
        For example, fluoride ion (chemical formula F-, SMILES string [F-])
        returns {9: 1, 0: -1}.
    
        :param molecule: The molecule to analyze
        :type some_input: An RDKit molecule
        :rtype: A dictionary.
        """
        #print(SMILES)
        set_atom = set()
        # Check that there is a valid molecule
        if molecule:
    
            # Add hydrogen atoms--RDKit excludes them by default
            molecule_with_Hs = Chem.AddHs(molecule)
    
            # Get atomic number
            for atom in molecule_with_Hs.GetAtoms():
                set_atom.add(atom.GetAtomicNum())
            return set_atom

    m = Chem.MolFromSmiles(SMILES)
    try:
        newlist = list(composition(m))
        newlist.sort()
        return newlist
    except:
        return list()

df2["Atom"] = df2["SMILES"].apply(lambda x: get_all_atomic_number(x))
df2["Count Unique Atom"] = df2["Atom"].apply(lambda x: len(x))
def scopeCHON (list_atom_number):
    
    cond1 = all(x in [1,6,7,8] for x in list_atom_number)
    cond2 = sum(x in [1,6] for x in list_atom_number)==2
    return all([cond1, cond2])
    #print(list_atom_number, cond1, cond2)
    #return cond2
df2["Pass Screen"] = df2["Atom"].apply(lambda x: scopeCHON(x))
df3 = df2.loc[df2["Pass Screen"].isin([True])]
df3 = df3.drop(columns="SMILES", axis=0)
df3_index = list(df3.index)

#%% Change Scope Letter
def cb(row):
    #print(row)
    if row == [1,6]:
        return "CH"
    elif row == [1,6,8]:
        return "CHO"
    elif row == [1,6,7]:
        return "CHN"
    elif row == [1,6,7,8]:
        return "CHON"
    elif row == [6,7]:
        return "CN"
    elif row == [6,8]:
        return "CO"
    elif row == [6,7,8]:
        return "CON"
df3["Atom2"] = df3["Atom"].apply(lambda x : cb(x))
#%% Get CHON Scope
df_prepare = df_original.copy()
df_prepare = df_prepare.filter(items = df3_index, axis=0)
df_CHON = pd.concat([df_prepare, df3], axis=1)

#%% Get All Functional Group in Molecule
from thermo.functional_groups import *
def cb(row):
    mol = Chem.MolFromSmiles(row["SMILES"]) 
    func_list = [is_alkane, is_alkene, is_alkyne, is_aromatic,
                 is_alcohol, is_ketone, is_aldehyde, is_carboxylic_acid,
                 is_ether, is_phenol, is_ester, is_amine, is_amide]
    for func in func_list:
        func_name = func.__name__
        result_1 = func(mol)
        if result_1:
            pass
        else:
            result_1= None
        row[func_name] = result_1
        
    
    return row
df_func_group = df_CHON.apply(cb, axis=1)
#%% Split molecule have 1 and more than 1 Group
dffff = pd.DataFrame(df_func_group.iloc[:, 13:].sum(axis=1))
filter01 = dffff[0]>1
df_func_group_1 = df_func_group[~filter01]
df_func_group_more2 = df_func_group[filter01]

#%% get Main Func Group of all Molecule
func_list = [is_alkane, is_alkene, is_alkyne, is_aromatic,
             is_alcohol, is_ketone, is_aldehyde, is_carboxylic_acid,
             is_ether, is_phenol, is_ester, is_amine, is_amide]
func_order = [12, 10, 11, 9, 6, 5, 4, 1, 13, 8, 2, 7, 3]
func_name_list = []
for func in func_list:
    func_name = func.__name__
    func_name_list.append(func_name)
    
# get Func Group of more than 1 Group
f1 = pd.melt(df_func_group_1, id_vars='SMILES', value_vars=func_name_list)
f2 = f1[f1["value"]==True]
f2["Func. Group"] = f2["variable"].str[3:]
f2 = f2.drop(columns=["variable","value"])

# get Func Group of more than 1 Group
f3 = pd.melt(df_func_group_more2, id_vars='SMILES', value_vars=func_name_list)
f4 = f3[f3["value"]==True]
f4 = f4.drop(columns=["value"])
f5 = f4.groupby("SMILES").agg(list)
#%% Order Functional Group for more than 1 Group
def cb2(row):
    for index in range(len(func_order)):
        for idx_fun_sub in range(len(row["variable"])):
            sub_func = row["variable"][idx_fun_sub]
            check_func = func_list[index].__name__
            if sub_func==check_func:
                row["variable"][idx_fun_sub] = func_order[index]
    
        #row =row.sort_values("variable")
    row["variable"] = min(row["variable"])
    sub_order_idx = func_order.index(row["variable"])
    row["variable"] = func_list[sub_order_idx].__name__
    row["variable"] = row["variable"][3:]
    
    return row
f6= f5.apply(lambda x: cb2(x), axis=1)
f6.columns=["Func. Group"]
f6 = f6.reset_index()
#%% Combine Functional Group from 1 and more than 1 func. group
f7 = pd.concat([f2, f6])
df_func = pd.merge(df_CHON, f7, on="SMILES")
#%% Split Data
df_func = df_func.drop_duplicates(subset=['SMILES']).reset_index(drop=True)
df_func.to_csv("Tb_Data_CHON_Func_Group.csv")
