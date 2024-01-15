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


#%% Import Data

df_import = pd.read_excel("../[Use] Data Preparation/Psat_AllData_1.xlsx",sheet_name="All")
df= df_import[df_import['SMILES'] != "None"].reset_index(drop=True)


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
    newlist = list(composition(m))
    newlist.sort()
    return newlist
#%%
df2 = df[["SMILES"]].copy()
df2["Atom"] = df2["SMILES"].apply(lambda x: get_all_atomic_number(x))
df2["Count Unique Atom"] = df2["Atom"].apply(lambda x: len(x))
def scopeCHON (list_atom_number):
    return all( x in [1,6,7,8] for x in list_atom_number)
df2["Pass Screen"] = df2["Atom"].apply(lambda x: scopeCHON(x))
df3 = df2.loc[df2["Pass Screen"].isin([True])]
df3_index = list(df3.index)

#%%
df_prepare = df.copy()
df_prepare = df_prepare.filter(items = df3_index, axis=0)
#df_prepare.join(df3, on="SMILES")
df_export = pd.concat([df_prepare, df3], axis=1)
df_export.to_csv("RDKit_CHON_New_Data_Psat_Not_Outliers.csv")
