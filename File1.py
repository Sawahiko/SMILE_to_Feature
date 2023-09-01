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
# %%
def remove_correlated_features(descriptors):
    # Calculate correlation
    correlated_matrix = descriptors.corr().abs()

    # Upper triangle of correlation matrix
    upper_triangle = correlated_matrix.where(np.triu(np.ones(correlated_matrix.shape),k=1).astype(np.bool))

    # Identify columns that have above 0.9 values of correlation
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= 0.9)]
    print(to_drop)
    descriptors_correlated_dropped = descriptors.drop(columns=to_drop, axis=1)
    return descriptors_correlated_dropped    

from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

'''
# %%

descriptors_new = remove_correlated_features(df_with_200_descriptors)
X_data = remove_low_variance(descriptors_new, threshold=0.1)


'''
# %%
'''
import lazypredict
from lazypredict.Supervised import LazyRegressor

# Use Sklearn ML models in 2 lines of codes
# 42 regression ML models

lregs = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models, prediction_tests = lregs.fit(X_train, X_test, y_train, y_test)
'''
# %%
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data,test_size=0.3,random_state=42)

# %%
########   Model  ########
Linear = LinearRegression()
Linear.fit(X_train, y_train)

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
y_predict_total = Linear.predict(X_data)
mape_total = mean_absolute_percentage_error(Y_data, y_predict_total)
rmse_total = np.sqrt(mean_squared_error(Y_data, y_predict_total))
R2_total = r2_score(Y_data, y_predict_total)

# Table Score
MLR_table = pd.DataFrame()
data = {
        "MAPE":[mape_train, mape_test, mape_total],
        "RMSE":[rmse_train, rmse_test, rmse_total],
        "R2"  :[R2_train, R2_test, R2_total]
    }
MLR_table = pd.DataFrame(data)
MLR_table.to_csv('MLR_208_feature.csv', index=False)

# %%
p1=sn.regplot(x=y_predict_test, y=y_test,line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
plt.xlabel('Predicted Tb', color='blue')
plt.ylabel('Observed Tb', color ='blue')
plt.title("Test set", color='red')
plt.grid(alpha=0.2)
#R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))
#MAE = mpatches.Patch(label="MAE={:04.2f}".format(MAE))
#plt.legend(handles=[R2, MAE])

# %%
y_predict_train = Linear.predict(X_train)
p2=sn.regplot(x=y_predict_train, y=y_train,line_kws={"lw":2,'ls':'--','color':'black',"alpha":0.7})
plt.xlabel('Predicted Tb', color='blue')
plt.ylabel('Observed Tb', color ='blue')
plt.title("Train set", color='red')
plt.grid(alpha=0.2)
#R2 = mpatches.Patch(label="R2={:04.2f}".format(R2))
#MAE = mpatches.Patch(label="MAE={:04.2f}".format(MAE))
#plt.legend(handles=[R2, MAE])

# %%

#SMILE_TEST = "c1cccc1"

data = {
  "SMILE": ["CCCC(C)O", "CCC(CC)O"],
  "TB" : [273.15+119.3, 273.15+116]
}
TEST_X = pd.DataFrame(data)

#test = TEST_X["SMILE"]
#test = df2['SMILES']

Mol_descriptors,desc_names = RDkit_descriptors(TEST_X["SMILE"])
TEST_X_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

TEST_y_predict = Linear.predict(TEST_X_with_200_descriptors)

y_predict_table = pd.DataFrame()
y_predict_table["Tb_actual"] = TEST_X["TB"]
y_predict_table["Tb_predict"] = TEST_y_predict

# %%
m = Chem.MolFromSmiles('CCCCC')
from rdkit.Chem import Draw
img = Draw.MolToFile(m,'pic2.png')

# %%
from rdkit.Chem import PandasTools
df4 = df2
PandasTools.AddMoleculeColumnToFrame(df4,'SMILES', 'Structure')


# %%
test = TEST_X_with_200_descriptors.diff()[1:]
#zero_mask = test.eq(0)
zero_mask2=test.drop(columns=zero_mask.columns[(zero_mask == True).any()])

# %%
#df_with_200_descriptors["SMILE"] = df["SMILES"]
#TEST_X_with_200_descriptors.to_csv('output.csv', index=False)
zero_mask2.to_csv('output.csv', index=False)

# %%
m = Chem.MolFromSmiles("CCCC(C)O")
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
desc_names = calc.GetDescriptorNames()

# %%

#SMILE_TEST = "c1cccc1"

data = {
  "SMILE": ["CCCC(C=O)O", "CCC(CC)O"],
  "TB" : [273.15+119.3, 273.15+116]
}
TEST_X = pd.DataFrame(data)

#test = TEST_X["SMILE"]
#test = df2['SMILES']

Mol_descriptors,desc_names = RDkit_descriptors(TEST_X["SMILE"])
TEST_X_with_200_descriptors = pd.DataFrame(Mol_descriptors,columns=desc_names)

TEST_y_predict = Linear.predict(TEST_X_with_200_descriptors)

y_predict_table = pd.DataFrame()
y_predict_table["Tb_actual"] = TEST_X["TB"]
y_predict_table["Tb_predict"] = TEST_y_predict
