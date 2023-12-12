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

#from rdkit.Chem.Draw import IPythonConsoleb
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
# %%
# Inspect Bit
name1 = "CCCC(C)O"
m1 = Chem.MolFromSmiles(name1)

mol = m1
bit_info_mol = {}
fp_mol = GetHashedMorganFingerprint(mol, 3, bitInfo=bit_info_mol, useFeatures=True)
all_fragments = [(mol, x, bit_info_mol) for x in get_index_of_array_that_contain_1_in_any_position(list(fp_mol))]
d=Draw.DrawMorganBits(all_fragments[:], 
                      molsPerRow=4, 
                      legends=[str(x) for x in get_index_of_array_that_contain_1_in_any_position(list(fp_mol))][:], 
                      subImgSize=(150,150))

#d.show()