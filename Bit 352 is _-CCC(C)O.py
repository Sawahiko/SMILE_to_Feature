# RDKit
import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint

from rdkit.Chem.Draw import IPythonConsole
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
#
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from ipywidgets import interact,fixed,IntSlider

def renderFpBit(mol,bitIdx,bitInfo,fn):
    bid = bitIdx
    return(display(fn(mol,bid,bitInfo)))
def get_index_of_array_that_contain_1_in_any_position(list1):
    result = []
    for i, e in enumerate(list1):
        if e > 0:
            result.append(i)
    return result
def getSMART(mol, radius, atomidx):
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
def get_All_SMART_1_mol(mol):
    bit_info = {}
    fp = GetHashedMorganFingerprint(mol, 3, bitInfo=bit_info, useFeatures=True)

    #print(get_index_of_array_that_contain_1_in_any_position(list(fp)))
    #print(bit_info.keys())
    string_SMARTS = []
    for i,v in enumerate(bit_info.keys()):
        #print(i)
        #print("v=",v)
        atomidx, radius = bit_info[v][0]
        string_SMART = getSMART(mol=mol, radius=radius, atomidx=atomidx)
        #print(v, string_SMART)
        string_SMARTS.append((v,string_SMART))
    return string_SMARTS
    
# %% Import Data : 560 datapoint
#Import Data
df = pd.read_excel("DataTb.xlsx",sheet_name="AllDataSet")

#Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["SMILES"]]
Y_data= df["Tb"]

#>>> SHOW X_Data, Y_data
# %%

#def Get_SMART(Dataframe, nBits=1024):

nBits=1024
Dataframe = X_data_excel.copy()
Dataframe["molecule"] = Dataframe["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
Dataframe["morgan_fp"] = Dataframe["molecule"].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x, radius=4, nBits=nBits, useFeatures=True, useChirality=True))

#>>> SHOW X_data_use 

#Transfrom Fingerprint to Column in DataFrame
Dataframe_fp = []
for i in range(Dataframe.shape[0]):
    array = np.array(Dataframe["morgan_fp"][i])
    datafram_i = pd.DataFrame(array)
    datafram_i = datafram_i.T
    Dataframe_fp.append(datafram_i)
Dataframe_fp = pd.concat(Dataframe_fp, ignore_index=True)
X_data_ML = pd.concat([Dataframe, Dataframe_fp], axis=1, join='inner')


# %%
test = X_data_ML["molecule"].apply(lambda x: get_All_SMART_1_mol(x))

# %%
count = set()
for i in test:
    #print(i)
    for idx, val in i:
        count.add(idx)
count = sorted(count)
# %%
len(count)
# %%
# get smart of all bit molecule
check_bit = dict()
for i in count:
     check_bit[str(i)]=set()
    
for v in test:    
    for i in count:
        #check_bit[str(i)]=set()
        print(i,"COUNT")
        for bit in v:
            print(bit[0])
            if i == bit[0]:
              print("PASS")
              check_bit[str(i)].add(bit[1])
