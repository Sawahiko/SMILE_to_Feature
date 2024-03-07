import glob
import pandas as pd
import numpy as np

# Get File
files = glob.glob('csv_03*test*.csv')
dfs = [pd.read_csv(file).iloc[0:1847] for file in files] ### Temporary  ###

#Change Unit and header
prediction_table = pd.concat(dfs).iloc[:,1:]
prediction_table.columns = ['Method', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']
prediction_table["Psat_Pred (atm)"] = np.exp(prediction_table["ln_Psat_Pred (Pa)"])/(10**5)
prediction_table["Psat_Actual (atm)"] = np.exp(prediction_table["ln_Psat_Actual (Pa)"])/(10**5)
#%%
#Insert SMILES, Temp, CHON
df2_test = pd.read_csv("csv_02-2 df_test.csv").iloc[:,1:]

SMILES_T_table = df2_test.iloc[:,[1,2]]
SMILES_T_table_ex = pd.concat([SMILES_T_table] * len(dfs))
#%%
# Merged
result = pd.concat([prediction_table, SMILES_T_table_ex], axis=1)
