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

# %%  # Interactive
name1 = "CCC#C"
m1 = Chem.MolFromSmiles(name1)

mol = m1
bit_info = {}
fp = GetHashedMorganFingerprint(mol, 3, bitInfo=bit_info, useFeatures=True)

print(get_index_of_array_that_contain_1_in_any_position(list(fp)))


# Select Bit
atomidx, radius = bit_info[352][0]

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
print(MorganBitSmarts)
# %%