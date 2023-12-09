# %%
# Python
import numpy as np
import pandas as pd
import time

# RDKit
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint


def getSMART2(mol, radius, atomidx):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomidx)
    atomsToUse = set((atomidx, ))
    for b in env:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    enlargedEnv = set()
    for atom in atomsToUse:
        a = mol.GetAtomWithIdx(atom)
        for b in a.GetBonds():
            bidx = b.GetIdx()
            if bidx not in env:
                enlargedEnv.add(bidx)
    enlargedEnv = list(enlargedEnv)
    enlargedEnv += env
    # find all relevant neighbors
    anyAtoms = []
    for a in atomsToUse:
        neighbors = mol.GetAtomWithIdx(a).GetNeighbors()
        for n in neighbors:
            anyIdx = n.GetIdx()
            if anyIdx not in atomsToUse:
                anyAtoms.append(anyIdx)
    # replace atomic number to zero (there is no number for any atom)
    for aA in anyAtoms:
        mol.GetAtomWithIdx(aA).SetAtomicNum(0)
    submol = Chem.PathToSubmol(mol, enlargedEnv)
    # change [0] to *
    MorganBitSmarts = Chem.MolToSmarts(submol).replace('[#0]', '*')
    #print(MorganBitSmarts)
    return MorganBitSmarts


def get_All_SMART_1_mol_v2(mol, radius, nBits):
    bit_info = {}
    fp = rdMolDescriptors.GetMorganFingerprint(
                mol,
                radius=radius,
                bitInfo = bit_info,
                useFeatures=True, useChirality=True)
    #print(get_index_of_array_that_contain_1_in_any_position(list(fp)))
    #print(bit_info.keys())
    string_SMARTS = dict()
    for i,v in enumerate(bit_info.keys()):
        #print(i)
        #print("v=",v)
        atomidx, radius = bit_info[v][0]
        string_SMART = getSMART2(mol=mol, radius=radius, atomidx=atomidx)
        #string_SMART = "TEST"
        #print(v, string_SMART)
        string_SMARTS[v] = string_SMART
        #print(v)
    #print(string_SMARTS)
    #print(string_SMARTS)
    try :
        return string_SMARTS
        #print("HERE")
    except :
        None
 
        
# %%
X_data_excel = pd.read_excel('../Data.xlsx', sheet_name="560point")
radius = 3; nBits = 1024

Dataframe = X_data_excel.copy()
Dataframe["molecule"] = Dataframe["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))

# %%
Dataframe["morgan_fp"] = Dataframe["molecule"].apply(lambda x: rdMolDescriptors.GetMorganFingerprint(x, radius=3, useFeatures=True))

Dataframe_fp = []
for i in range(Dataframe.shape[0]):
    array = np.array(Dataframe["morgan_fp"][i])
    datafram_i = pd.DataFrame(array)
    datafram_i = datafram_i.T
    Dataframe_fp.append(datafram_i)
Dataframe_fp = pd.concat(Dataframe_fp, ignore_index=True)
X_data_ML = pd.concat([Dataframe, Dataframe_fp], axis=1, join='inner')
Table1 = Dataframe.copy()
Table1["All_SMART"] = Table1["molecule"].apply(lambda x : get_All_SMART_1_mol_v2(x, radius, nBits))
Table1

# %%
Dataframe["full_morgan_fp"] = Dataframe["molecule"].apply(lambda x: rdMolDescriptors.GetMorganFingerprint(x, radius=3, useFeatures=True))
Dataframe["morgan_fp"] = Dataframe["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
    x, 
    radius=radius, 
    nBits=nBits,
    useFeatures=True, useChirality=True))

num_mol = 463
m_test = Dataframe["molecule"].iloc[num_mol]
mfp_test = Dataframe["morgan_fp"].iloc[num_mol]
blank_arr = np.zeros((0,), dtype=np.int8)
DataStructs.ConvertToNumpyArray(mfp_test,blank_arr)


full_mfp_test = get_All_SMART_1_mol_v2(m_test,radius,nBits)
num_arr = [key for key in full_mfp_test]
new_num = set([number % nBits for number in num_arr ])

#array = np.zeros((0,), dtype=np.int8)
#temp = DataStructs.ConvertToNumpyArray(test, array )