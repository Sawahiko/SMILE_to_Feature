import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn

# https://www.rdkit.org/
#https://github.com/rdkit/rdkit
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# https://pandas.pydata.org
import pandas as pd

# https://numpy.org/doc/stable/release.html
import numpy as np

#https://github.com/mordred-descriptor/mordred
from mordred import Calculator, descriptors
# %%
def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:
        # add hydrogens to molecules
        mol=Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 
# %%
########   Prepare Data     ########
#Import Data
df = pd.read_excel("DataTb.xlsx",sheet_name="AllDataSet")

#Select feature for data
X_data= df.drop(columns ={"Name","Tb","CAS","Type","Formular","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14","C","Double", "Triple", "Bracket", "Cyclic"})
Y_data= df["Tb"]

#
df2= df.drop(columns ={"CAS","Type","Formular","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14","C","Double", "Triple", "Bracket", "Cyclic"})


# %%
# Function call
Mol_descriptors,desc_names = RDkit_descriptors(df2['SMILES'])
df_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

#Data to Array
X_data = df_with_200_descriptors

test = X_data.diff()[1:]
zero_mask = test.eq(0)
zero_mask2=test.drop(columns=zero_mask.columns[(zero_mask == True).any()])

# %%
#df_with_200_descriptors["SMILE"] = df["SMILES"]
#TEST_X_with_200_descriptors.to_csv('output.csv', index=False)
zero_mask2.to_csv('101 remaining feature.csv', index=False)