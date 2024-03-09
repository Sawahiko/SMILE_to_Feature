# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


#%%
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
        
        df = df[(df[column] > lower_bound) & (df[column] < upper_bound)]
        if show_result:
            name_title = "Extracted "+column 
            plt.title(name_title)
            plt.boxplot(df[column])
            plt.show()
        

    return df.reset_index(drop=True)

def remove_outliers_z(Excel_path, Excel_sheetname, columns, z_thereshold, show_result=False):
    df = pd.read_excel(Excel_path, sheet_name=Excel_sheetname)

    for column in columns:
        
        plt.title(column)
        plt.boxplot(df[column])
        plt.show()
        
        
# =============================================================================
#         upper_limit = df_org['cgpa'].mean() + 3*df_org['cgpa'].std()
#         lower_limit = df_org['cgpa'].mean() - 3*df_org['cgpa'].std()
#         
#         df_org['cgpa'] = np.where(
#         df_org['cgpa']>upper_limit,
#         upper_limit,
#         np.where(
#         df_org['cgpa']<lower_limit,
#         lower_limit,
#         df_org['cgpa']
#         )
#         )
# =============================================================================
# =============================================================================
#         SD = df[column].std()
#         mean = df[column].mean()
#         lower_bound = mean - z_thereshold*SD
#         upper_bound = mean + z_thereshold*SD
# =============================================================================
        z_score = np.abs(stats.zscore(df[column]))
        df = df[z_score < z_thereshold]
# =============================================================================
#         df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]
# =============================================================================
        if show_result:
            name_title = "Extracted "+column 
            plt.title(name_title)
            plt.boxplot(df[column])
            plt.show()
        

    return df.reset_index(drop=True)

#%% Remove A B C Tmin Tmax Outliner
# =============================================================================
# df_original_remove_ABCMinMax_1 = remove_outliers_boxplot("../[Use] Data Preparation/Psat_AllData_1.xlsx",
#                                                  "All",
#                                                  #["A", "B", "C", "Tmin", "Tmax"],
#                                                  ["A"],
#                                                  1.5,True)
# print(df_original_remove_ABCMinMax_1.shape)
# df_original_remove_ABCMinMax_2 = remove_outliers_z("../[Use] Data Preparation/Psat_AllData_1.xlsx",
#                                                  "All",
#                                                  #["A", "B", "C", "Tmin", "Tmax"],
#                                                  ["A"],
#                                                  3,True)
# print(df_original_remove_ABCMinMax_2.shape)
# 
# df_original_remove_ABCMinMax = df_original_remove_ABCMinMax_1.copy()
# =============================================================================

df_original = pd.read_excel("../[Use] Data Preparation/Psat_AllData_1.xlsx", sheet_name="All")
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
df2 = df_original[["SMILES"]].copy()
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
#%% Get ABC TMin Tmax CHON
df_prepare = df_original.copy()
df_prepare = df_prepare.filter(items = df3_index, axis=0)
#df_prepare.join(df3, on="SMILES")
df_CHON = pd.concat([df_prepare, df3], axis=1)
df_CHON = df_CHON[df_CHON["Count Unique Atom"]>0]

#%% Get C1-C12, Unique SMILES name
df_CHON = df_CHON[df_CHON["No.C"]<=12]
df_CHON = df_CHON[df_CHON["No.C"]>0]
df_CHON = df_CHON.drop_duplicates(subset=['SMILES'])
df_CHON = df_CHON.drop_duplicates(subset=['Name'])


#%% Generate 5 Temp
df1 = df_CHON.copy().reset_index(drop=True)
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

def Psat_cal(T,A,B,C):
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])


plt.title("Before ln(Psat)")
plt.boxplot(df1["Vapor_Presssure"])
plt.show()
#%% boxplot ABC Tmin




#%%
column = "Vapor_Presssure"
Q1 = df1[column].quantile(0.25)
Q3 = df1[column].quantile(0.75)
IQR = Q3 - Q1
IQR_factor = 1.5
lower_bound = Q1 - (IQR_factor * IQR)
upper_bound = Q3 + (IQR_factor * IQR)


df01 = df1[~((df1[column] < lower_bound) | (df1[column] > upper_bound))]
df01_outliners = df1[((df1[column] < lower_bound) | (df1[column] > upper_bound))]
df_outliner_export = df01_outliners[["SMILES", "Name", "T", "Vapor_Presssure"]]

plt.title("After ln(Psat)")
plt.boxplot(df01["Vapor_Presssure"])
plt.show()

df01["Psat_atm"] = np.exp(df01[column])/(10**5)
plt.title("AFter Psat (atm)")
plt.boxplot(df01["Psat_atm"])
plt.show()

plt.hist(df01["Psat_atm"], bins=5)

#%% Summary Output
df_Psat_noOut = df01.copy()

filter2 = df_CHON["SMILES"].isin(df01["SMILES"].drop_duplicates())
df_VP_export = df_CHON[filter2]
df_VP_outliner_export = df_CHON[~filter2]


filter3 = df01["SMILES"].isin(df01["SMILES"].drop_duplicates())
df_5VP_1 = df01[filter3][["SMILES", "Vapor_Presssure"]]
df_5VP_2 = df_5VP_1.groupby(["SMILES"]).agg({'Vapor_Presssure': lambda x :x.tolist()})
df_5VP_SMILES = df_5VP_1[["SMILES"]].drop_duplicates().reset_index(drop=True)
df_5VP_3 = pd.DataFrame(df_5VP_2['Vapor_Presssure'].to_list(), columns=['VP1','VP2', 'VP3',
                                                                       'VP4', 'VP5'])
df_5VP_all = pd.concat([df_5VP_SMILES, df_5VP_3], axis=1)
df_5VP_dropna = df_5VP_all.dropna()
df_5VP_na = df_5VP_all[df_5VP_all.isna().any(axis=1)]
df_5VP_export1 = df_5VP_dropna.copy()
df_5VP_export2 = df_5VP_na.copy()

filter4 = df_CHON["SMILES"].isin(df_5VP_export1["SMILES"].drop_duplicates())
df_VP_final_export = df_CHON[filter4]
#%% Export Section
df_VP_export.to_csv("csv_01-1 Psat_[X]_ABCTminTmaxC1-12.csv")
df_VP_outliner_export.to_csv("csv_01-2 Psat_Outliner.csv")
df_5VP_export1.to_csv("csv_01-3 Psat_5VP_all_SMILES.csv")
df_5VP_export2.to_csv("csv_01-4 Psat_5VP_nan_SMILES.csv")
df_VP_final_export.to_csv("csv-01-0 Psat-1800.csv")