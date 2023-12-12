import numpy as np
import pandas as pd
import pubchempy as pcp

#Import Data
df = pd.read_excel("Data.xlsx",sheet_name="From_8DB_Chemicals")

#Select feature for data: X=SMILE, Y=Tb
X_data_excel= df[["CAS"]]
#Y_data= df["Tb"]

#pcp.get_synonyms('Aspirin', 'smiles')
cas_list = list()
cas_list = X_data_excel.values.tolist()
#Tb = 
#names = ['Ketene']

#names=names[:20]
List1=list()
List2=list()
i = 0

for cas in cas_list[26927:28927]:
    print(str(i) + '. ' + str(cas))
    results = pcp.get_compounds(cas, 'name')
    i += 1
    if results:        
            print(results[0].iupac_name)
            print(results[0].isomeric_smiles)
            List1.append(results[0].iupac_name)
            if results[0].iupac_name == []:    
                List2.append("None")
            else:
                List2.append(results[0].isomeric_smiles)
    else:
        List1.append('None')
        List2.append('None')

data = {
    "CAS":cas_list[26927:28927],
    "NAME":List1
    ,"SMILES":List2
    }

tdf = pd.DataFrame(data)

tdf.to_csv("CAS_to_Name10.csv")
#print(a.cid)
#a.
#pcp.get_properties('IsomericSMILES', 'CC', 'smiles')

#p = pcp.get_properties('IsomericSMILES', 'CC[N+](CC)=C(C)OCC', 'smiles')
# =============================================================================
# test = pcp.get_compounds('286-16-8', 'name')[0].iupac_name
# if test:
#     print(results[0].iupac_name)
#     List1.append(results[0].iupac_name)
# else:
#     print('None')
# =============================================================================
