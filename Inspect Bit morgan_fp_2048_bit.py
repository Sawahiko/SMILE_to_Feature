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

# %%  # interactive

mol1 = Chem.MolFromSmiles('CO')
mol2 = Chem.MolFromSmiles('CCO')
mol3 = Chem.MolFromSmiles('CCCO')
mol4 = Chem.MolFromSmiles('CCCCO')
bit_info1 = {}
bit_info2 = {}
bit_info3 = {}
bit_info4 = {}
fp1 = GetHashedMorganFingerprint(mol1, 3, bitInfo=bit_info1, useFeatures=True)
fp2 = GetHashedMorganFingerprint(mol2, 3, bitInfo=bit_info2, useFeatures=True)
fp3 = GetHashedMorganFingerprint(mol3, 3, bitInfo=bit_info3, useFeatures=True)
fp4 = GetHashedMorganFingerprint(mol4, 3, bitInfo=bit_info4, useFeatures=True)

print(get_index_of_array_that_contain_1_in_any_position(list(fp1)))
print(get_index_of_array_that_contain_1_in_any_position(list(fp2)))
print(get_index_of_array_that_contain_1_in_any_position(list(fp3)))
print(get_index_of_array_that_contain_1_in_any_position(list(fp4)))

interact(renderFpBit, bitIdx=list(bit_info1.keys()),mol=fixed(mol1),
         bitInfo=fixed(bit_info1),fn=fixed(Draw.DrawMorganBit));
interact(renderFpBit, bitIdx=list(bit_info2.keys()),mol=fixed(mol2),
         bitInfo=fixed(bit_info2),fn=fixed(Draw.DrawMorganBit));
interact(renderFpBit, bitIdx=list(bit_info3.keys()),mol=fixed(mol3),
         bitInfo=fixed(bit_info3),fn=fixed(Draw.DrawMorganBit));
interact(renderFpBit, bitIdx=list(bit_info4.keys()),mol=fixed(mol4),
         bitInfo=fixed(bit_info4),fn=fixed(Draw.DrawMorganBit));

# %%