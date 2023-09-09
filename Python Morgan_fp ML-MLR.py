#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn

from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
# https://www.rdkit.org/
#https://github.com/rdkit/rdkit
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors



#%%
########   Prepare Data     ########
#Import Data
df = pd.read_excel("DataTb.xlsx",sheet_name="AllDataSet")

#Select feature for data
X_data= df.drop(columns ={"Name","Tb","CAS","Type","Formular","Unnamed: 11","Unnamed: 12","Unnamed: 13","Unnamed: 14","C","Double", "Triple", "Bracket", "Cyclic"})
Y_data= df["Tb"]

#Generate Fingerprint
X_data_use = X_data
X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
X_data_use["morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x, radius=4, nBits=250, useFeatures=True, useChirality=True))

#Storing Fingerprint in Dataframe
X_data_fp = []
for i in range(X_data_use.shape[0]):
    array = np.array(X_data_use["morgan_fp"][i])
    datafram_i = pd.DataFrame(array)
    datafram_i = datafram_i.T
    X_data_fp.append(datafram_i)
X_data_fp = pd.concat(X_data_fp, ignore_index=True)

#%%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_data_fp, Y_data,test_size=0.3,random_state=42)

# Modeling
Linear = LinearRegression()
Linear.fit(X_train, y_train)

#%%
# Train set
y_predict_train = Linear.predict(X_train)
#from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
mape_train = mean_absolute_percentage_error(y_train, y_predict_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
R2_train = r2_score(y_train, y_predict_train)

# Test set
y_predict_test = Linear.predict(X_test)
mape_test = mean_absolute_percentage_error(y_test, y_predict_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
R2_test = r2_score(y_test, y_predict_test)

# Total set
y_predict_total = Linear.predict(X_data_fp)
mape_total = mean_absolute_percentage_error(Y_data, y_predict_total)
rmse_total = np.sqrt(mean_squared_error(Y_data, y_predict_total))
R2_total = r2_score(Y_data, y_predict_total)

# Score Table
Morgan_fp__ML_MLR = pd.DataFrame()
data = {
        "MAPE":[mape_train, mape_test, mape_total],
        "RMSE":[rmse_train, rmse_test, rmse_total],
        "R2"  :[R2_train, R2_test, R2_total]
    }
Morgan_fp__ML_MLR = pd.DataFrame(data)
Morgan_fp__ML_MLR.to_csv('Morgan_fp ML-MLR.csv', index=False)

#%%
# Plot
p1=sn.regplot(x=y_predict_test, y=y_test,line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
plt.xlabel('Predicted Tb', color='blue')
plt.ylabel('Observed Tb', color ='blue')
plt.title("Test set", color='red')
plt.grid(alpha=0.2)