#%%
import numpy as np
import pandas as pd
import pubchempy as pcp
#%%

#Import Data
df = pd.read_csv("../get_3 DWSIM & ChEDL/Psat_raw2.csv")  # Psat data (A,B,C)

#%%
#Select feature for data: X=SMILE, Y=Tb
X_data_excel= df['Chemical']
#Y_data= df["Tb"]

#pcp.get_synonyms('Aspirin', 'smiles')
names = list()
names = X_data_excel.values.tolist()
#Tb = 
#names = ['Ketene']

#names=names[:20]
List1=list()

for name1 in names:
    print(name1)
    results = pcp.get_compounds(name1, 'name')
    if results == []:
        List1.append(None)
    else:
        print(results[0].isomeric_smiles)
        List1.append(results[0].isomeric_smiles)

data = {
    "Name":names,
    "SMILES":List1,
    "A":df["A"],
    "B":df["B"],
    "C":df["C"],
    "Tmin":df["Tmin"],
    "Tmax":df["Tmax"],
    }

tdf = pd.DataFrame(data).dropna().reset_index(drop=True)

tdf.to_csv("Psat_SMILES_dataset2.csv")
#print(a.cid)
#a.
#pcp.get_properties('IsomericSMILES', 'CC', 'smiles')

#p = pcp.get_properties('IsomericSMILES', 'CC[N+](CC)=C(C)OCC', 'smiles')

#%%
# =============================================================================
# # Density
# 
# #Import Data
# df = pd.read_csv("Liquid_morlar_density_raw1.csv")  # Psat data (A,B,C)
# df_add = df[["C1", "C2", "C3", "C4", "Tmin", "Tmax"]]
# 
# #%%
# #Select feature for data: X=SMILE, Y=Tb
# X_data_excel= df[["Chemical"]].iloc[0:2]
# #Y_data= df["Tb"]
# 
# #pcp.get_synonyms('Aspirin', 'smiles')
# names = list()
# names = X_data_excel.values.tolist()
# #Tb = 
# #names = ['Ketene']
# 
# #names=names[:20]
# List1=list()
# 
# for name1 in names:
#     print(name1)
#     results = pcp.get_compounds(name1, 'name')
#     if results == []:
#         List1.append('None')
#     else:
#         print(results[0].isomeric_smiles)
#         List1.append(results[0].isomeric_smiles)
# data = {
#     "Name":names,
#     "SMILES":List1
#     }
# 
# tdf = pd.DataFrame(data)
# tdf = tdf.join(df_add)
# tdf.to_csv("Density.csv")
# #print(a.cid)
# #a.
# #pcp.get_properties('IsomericSMILES', 'CC', 'smiles')
# 
# #p = pcp.get_properties('IsomericSMILES', 'CC[N+](CC)=C(C)OCC', 'smiles')
# 
# =============================================================================

#