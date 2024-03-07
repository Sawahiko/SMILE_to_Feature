import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Combined Train
# Get File
files_train = glob.glob('csv_03*train*.csv')
files_train = [files_train[0]] ### Temporary  ###
dfs_train = [pd.read_csv(file)for file in files_train] 

#Change Unit and header
prediction_table_train = pd.concat(dfs_train).iloc[:,1:]
prediction_table_train.columns = ['Method', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']
prediction_table_train["Psat_Pred (atm)"] = np.exp(prediction_table_train["ln_Psat_Pred (Pa)"])/(10**5)
prediction_table_train["Psat_Actual (atm)"] = np.exp(prediction_table_train["ln_Psat_Actual (Pa)"])/(10**5)

#Insert SMILES, Temp, CHON
df2_train = pd.read_csv("csv_02-1 df_train.csv").iloc[:,1:]

SMILES_T_table_train = df2_train.iloc[:,[1,2]]
SMILES_T_table_train_ex = pd.concat([SMILES_T_table_train] * len(dfs_train))

# Merged
result_train = pd.concat([prediction_table_train, SMILES_T_table_train_ex], axis=1)

#%% Combined Test
# Get File
files_test = glob.glob('csv_03*test*.csv')
files_test = [files_test[0]] ### Temporary  ###
dfs_test = [pd.read_csv(file)for file in files_test] 

#Change Unit and header
prediction_table_test = pd.concat(dfs_test).iloc[:,1:]
prediction_table_test.columns = ['Method', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']
prediction_table_test["Psat_Pred (atm)"] = np.exp(prediction_table_test["ln_Psat_Pred (Pa)"])/(10**5)
prediction_table_test["Psat_Actual (atm)"] = np.exp(prediction_table_test["ln_Psat_Actual (Pa)"])/(10**5)

#Insert SMILES, Temp, CHON
df2_test = pd.read_csv("csv_02-2 df_test.csv").iloc[:,1:]

SMILES_T_table_test = df2_test.iloc[:,[1,2]]
SMILES_T_table_test_ex = pd.concat([SMILES_T_table_test] * len(dfs_test))

# Merged
result_test = pd.concat([prediction_table_test, SMILES_T_table_test_ex], axis=1)
#%% Visualization-ln(Psat) train

test_prediction_bestpar_visual = result_train.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}  max = {max(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}')

# Specified Range for plot
x_min = -20;  x_max = 25
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)
# Insert Perfect Line
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))

# Add Legend, range of show
#g.add_legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


#%% Visualization-ln(Psat) test

test_prediction_bestpar_visual = result_test.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}  max = {max(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}')

# Specified Range for plot
x_min = -20;  x_max = 25
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)
# Insert Perfect Line
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))

# Add Legend, range of show
#g.add_legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

#%% Visualization- Psat test

test_prediction_bestpar_visual = result_train.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["Psat_Actual (atm)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["Psat_Actual (atm)"])}  max = {max(test_prediction_bestpar_visual["Psat_Actual (atm)"])}')

# Specified Range for plot
x_min = -20;  x_max = 25
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="Psat_Actual (atm)", y="Psat_Pred (atm)", alpha=0.6)
# Insert Perfect Line
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))

# Add Legend, range of show
#g.add_legend()
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)

#%% Visualization- Psat test

test_prediction_bestpar_visual = result_test.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["Psat_Actual (atm)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["Psat_Actual (atm)"])}  max = {max(test_prediction_bestpar_visual["Psat_Actual (atm)"])}')

# Specified Range for plot
x_min = -20;  x_max = 25
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="Psat_Actual (atm)", y="Psat_Pred (atm)", alpha=0.6)
# Insert Perfect Line
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))

# Add Legend, range of show
#g.add_legend()
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)