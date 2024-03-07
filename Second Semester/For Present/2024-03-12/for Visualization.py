# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import random

# Machine Learning
## Algorithm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
## Tool, Error Metric
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from joblib import dump, load

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
#%%

df = pd.read_csv("Training_log.csv")

df_log = df.loc[df.index%100== 99][["N_Hidden", "N_Layer", "val_loss"]]



#df.loc[(df.index < 6) & (df.A == 0), 'C'] = 99

import seaborn as sns

df_log_sns = df_log.pivot_table(index="N_Hidden", columns="N_Layer", values="val_loss")
g = sns.heatmap(df_log_sns, annot=True, fmt=".4f", cmap=sns.color_palette("YlOrBr", as_cmap=True))
g.invert_yaxis()
plt.show(g)
#%%
df['Combined'] = df['N_Hidden'].astype(str) +"-"+ df['N_Layer'].astype(str)

print(df.columns) 
   
df.rename(columns = {'Unnamed: 0':'idx'}, inplace = True) 
   
# After renaming the columns 
print(df.columns) 
#%%
num_repeat = int(len(df)/100)
x = np.tile(np.linspace(1,100,100),num_repeat)
print(x)
df["Index_each"] = x
df_log_sns2 = df.pivot_table(index="Index_each", columns="Combined", values="val_loss")
sns.lineplot(data=df_log_sns2)
