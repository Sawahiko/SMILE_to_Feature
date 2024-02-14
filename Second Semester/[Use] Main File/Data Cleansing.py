# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs

def remove_outliers_boxplot(Excel_path, Excel_sheetname, columns, IQR_factor=1.5, show_result=False):
    df = pd.read_excel(Excel_path, sheet_name=Excel_sheetname)

    for column in columns:
        
        plt.title(column)
        plt.boxplot(df[column])
        plt.show()
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (IQR_factor * IQR)
        upper_bound = Q3 + (IQR_factor * IQR)
        
        df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]
        if show_result:
            name_title = "Extracted "+column 
            plt.title(name_title)
            plt.boxplot(df[column])
            plt.show()
        

    return df.reset_index(drop=True)

#%% Remove A B C Tmin Tmax Outliner
df_original_remove_ABCMinMax = remove_outliers_boxplot("../[Use] Data Preparation/Psat_AllData_1.xlsx",
                                                 "AllDataSet",
                                                 ["A", "B", "C", "Tmin", "Tmax"],
                                                 1.5,True)
#%%
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
    
#%% Get CHON Scope
df2 = df_original_remove_ABCMinMax[["SMILES"]].copy()
df2["Atom"] = df2["SMILES"].apply(lambda x: get_all_atomic_number(x))
df2["Count Unique Atom"] = df2["Atom"].apply(lambda x: len(x))
def scopeCHON (list_atom_number):
    return all( x in [1,6,7,8] for x in list_atom_number)
df2["Pass Screen"] = df2["Atom"].apply(lambda x: scopeCHON(x))
df3 = df2.loc[df2["Pass Screen"].isin([True])]
df3 = df3.drop(columns="SMILES", axis=0)
df3_index = list(df3.index)

#%% Get ABC TMin Tmax CHON
df_prepare = df_original_remove_ABCMinMax.copy()
df_prepare = df_prepare.filter(items = df3_index, axis=0)
#df_prepare.join(df3, on="SMILES")
df_ABCMinMax_CHON = pd.concat([df_prepare, df3], axis=1)
df_ABCMinMax_CHON = df_ABCMinMax_CHON[df_ABCMinMax_CHON["Count Unique Atom"]>0]

#%% Get C1-C12, Unique SMILES name
df_export = df_ABCMinMax_CHON[df_ABCMinMax_CHON["No.C"]<=12]
df_export = df_export[df_export["No.C"]>0]
df_export = df_export.drop_duplicates(subset=['SMILES'])
df_export = df_export.drop_duplicates(subset=['Name'])


#%% Generate 5 Temp
df1 = df_export.copy().reset_index(drop=True)
def generate_points(row, amount_point):
    start = row["Tmin"]; end = row["Tmax"];
    range_temp = end-start
    if range_temp>5:
        return np.linspace(start, end, amount_point)
    else:
        return start
df1["T"] = df1.apply(lambda x : generate_points(x, 5), axis=1)

df1 = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)

# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    #return pow(A-(B/(T+C)),10)/(10^(3))
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])


plt.title("Log(Psat)")
plt.boxplot(df1["Vapor_Presssure"])
plt.show()
#%%
column = "Vapor_Presssure"
Q1 = df1[column].quantile(0.25)
Q3 = df1[column].quantile(0.75)
IQR = Q3 - Q1
IQR_factor = 1.5
lower_bound = Q1 - (IQR_factor * IQR)
upper_bound = Q3 + (IQR_factor * IQR)

df01 = df1[~((df1[column] < lower_bound) | (df1[column] > upper_bound))]

plt.title("Extracted Log(Psat)")
plt.boxplot(df01["Vapor_Presssure"])
plt.show()
#%% Summary Output
df_Psat = df1.copy()
df_Psat_noOut = df01.copy()

#df_export.to_csv("RDKit_CHON_New_Data_Psat_Not_Outliers.csv")
