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
from sklearn.linear_model import LinearRegression
def Linear_default(x_train, y_train):
  lm = LinearRegression()
  lm.fit(x_train, y_train)
  return lm

from sklearn.linear_model import Ridge
def Ridge_default(x_train, y_train):
  ridge = Ridge()
  ridge.fit(x_train, y_train)
  return ridge

from sklearn.linear_model import Lasso
def Lasso_default(x_train, y_train):
  lasso = Lasso()
  lasso.fit(x_train, y_train)
  return lasso

from sklearn.tree import DecisionTreeRegressor
def DT_default(x_train, y_train):
  DT = DecisionTreeRegressor()
  DT.fit(x_train, y_train)
  return DT

from sklearn.ensemble import RandomForestRegressor
def RF_default(x_train, y_train):
  RF = RandomForestRegressor()
  RF.fit(x_train, y_train)
  return RF

def XGB_default(x_train, y_train):
  XGB = XGBRegressor()
  XGB.fit(x_train, y_train)
  return XGB

from sklearn.neighbors import KNeighborsRegressor
def KNN_default(x_train, y_train):
  KNN = KNeighborsRegressor()
  KNN.fit(x_train, y_train)
  return KNN

from sklearn.svm import SVR
def SVM_default(x_train, y_train):
  svr = SVR()
  svr.fit(x_train, y_train)
  return svr

#%% Import Data
df = pd.read_csv(r"C:\Users\Kan\Documents\GitHub\SMILE_to_Feature\Second Semester\[Use] Main File/Psat_NO_ABCTminTmaxC1-12.csv")
df = df[df['SMILES'] != "None"]
df = df.drop_duplicates(subset='SMILES').reset_index(drop=True)
df.sort_values(by="No.C")

#%%
# Genearate Temp in Tmin-Tmax and expand

df1 = df.copy()
def generate_points(row, amount_point):
    start = row["Tmin"]; end = row["Tmax"];
    return np.linspace(start, end, amount_point)
df1["T"] = df1.apply(lambda x : generate_points(x, 5), axis=1)

df1  = df1.explode('T')
df1['T'] = df1['T'].astype('float32')
df1 = df1.reset_index(drop=True)

# Generate VP from Antione Coeff and Temp
def Psat_cal(T,A,B,C):
    return A-(B/(T+C))

df1["Vapor_Presssure"] = Psat_cal(df1["T"], df1["A"], df1["B"], df1["C"])

# Get Needed Table and split for Training
df2 = df1[["SMILES", "T", "Vapor_Presssure"]]
df2 = df2[~df2["SMILES"].isin(df2[df2["Vapor_Presssure"] <-20]["SMILES"])].reset_index()

X_data= df[["SMILES"]]               # feature: SMILE, T
Y_data_A = df[["A"]]
Y_data_B = df[["B"]]
Y_data_C = df[["C"]]


# %% Fingerprint
# Parameter for Generate Morgan Fingerprint
MF_radius = 3;   MF_bit = 2048

# Generate Fingerprint from SMILE
X_data_use = X_data.copy()
X_data_use["molecule"] = X_data_use["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))    # Create Mol object from SMILES
X_data_use["count_morgan_fp"] = X_data_use["molecule"].apply(lambda x: rdMolDescriptors.GetHashedMorganFingerprint(
    x,
    radius=MF_radius,
    nBits=MF_bit,
    useFeatures=True, useChirality=True))         # Create Morgan Fingerprint from Mol object


# Transfrom Fingerprint to Datafrme that we can use for training
X_data_use["arr_count_morgan_fp"] = 0
X_data_fp = []
for i in range(X_data_use.shape[0]):
    blank_arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(X_data_use["count_morgan_fp"][i],blank_arr)
    datafram_i = pd.DataFrame(blank_arr)
    datafram_i = datafram_i.T
    X_data_fp.append(datafram_i)
x_data_fp = pd.concat(X_data_fp, ignore_index=True)
x_data_fp = x_data_fp.astype(np.float32)

# Final Data for Training
y_data_fp_A = Y_data_A.copy()      # Output = Vapor Pressure
y_data_fp_B = Y_data_B.copy()      # Output = Vapor Pressure
y_data_fp_C = Y_data_C.copy()      # Output = Vapor Pressure

x_notz = x_data_fp.copy()
y_notz_A = np.ravel(y_data_fp_A.copy())
y_notz_B = np.ravel(y_data_fp_B.copy())
y_notz_C = np.ravel(y_data_fp_C.copy())

#%% Standardization
# =============================================================================
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scale_x= StandardScaler()
# scale_x.fit(x_train_notz)
# print(scale_x)
# 
# scale_y= StandardScaler()
# scale_y.fit(y_train_notz.reshape(-1,1))
# print(scale_y)
# x_train_fp = scale_x.transform(x_train_notz)
# x_test_fp  = scale_x.transform(x_test_notz)
# 
# y_train_fp = scale_y.transform(y_train_notz.reshape(-1,1)).flatten()
# y_test_fp  = scale_y.transform(y_test_notz.reshape(-1,1)).flatten()
# =============================================================================

x_train_A, x_test_A, y_train_A, y_test_A = train_test_split(x_notz, y_notz_A, test_size=0.20, random_state=42)
x_train_B, x_test_B, y_train_B, y_test_B = train_test_split(x_notz, y_notz_B, test_size=0.20, random_state=42)
x_train_C, x_test_C, y_train_C, y_test_C = train_test_split(x_notz, y_notz_C, test_size=0.20, random_state=42)

#%% Training Model
from datetime import datetime

def model_assess(X_train, X_test, y_train, y_test, list_model, name_model, title = "Default"):
  model_assess_table = pd.DataFrame(['Method','Training MAE', 'Training RMSE','Training R2','Test MAE', 'Test RMSE','Test R2', 'Time Evaluate']).transpose()
  new_header = model_assess_table.iloc[0] #grab the first row for the header
  model_assess_table.columns = new_header #set the header row as the df header
  model_assess_table.drop(index=model_assess_table.index[0], axis=0, inplace=True)

  train_prediction_table = pd.DataFrame(['Method','Training Predict','Training Actual']).transpose()
  new_header = train_prediction_table.iloc[0] #grab the first row for the header
  train_prediction_table.columns = new_header #set the header row as the df header
  train_prediction_table.drop(index=train_prediction_table.index[0], axis=0, inplace=True)

  test_prediction_table = pd.DataFrame(['Method','Test Predict','Test Actual']).transpose()
  new_header = test_prediction_table.iloc[0] #grab the first row for the header
  test_prediction_table.columns = new_header #set the header row as the df header
  test_prediction_table.drop(index=test_prediction_table.index[0], axis=0, inplace=True)


  for iteration in range(len(list_model)):
      time_start = datetime.now()
      model_train = list_model[iteration]
      name = name_model[iteration]

      if("DL" not in name):
        model_train.fit(X_train, y_train)
        y_train_pred = model_train.predict(X_train)
        y_test_pred  = model_train.predict(X_test)
      else:
        pass

      time_end = datetime.now()
      duration = (time_end - time_start).total_seconds()
      print(name)
      print(duration)

      train_mae = mean_absolute_error(y_train, y_train_pred)
      train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
      train_r2 = r2_score(y_train, y_train_pred)
      test_mae = mean_absolute_error(y_test, y_test_pred)
      test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
      test_r2 = r2_score(y_test, y_test_pred)

      results = pd.DataFrame([name_model[iteration],train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2, duration]).transpose()
      results.columns = ['Method','Training MAE', 'Training RMSE','Training R2','Test MAE', 'Test RMSE','Test R2', 'Time Evaluate']


      train_prediction_result = pd.DataFrame([name_model[iteration],y_train_pred, y_train]).transpose()
      train_prediction_result.columns = ['Method','Training Predict','Training Actual']

      test_prediction_result = pd.DataFrame([name_model[iteration], y_test_pred, y_test]).transpose()
      test_prediction_result.columns = ['Method','Test Predict','Test Actual']

      model_assess_table = pd.concat([model_assess_table, results])
      train_prediction_table = pd.concat([train_prediction_table, train_prediction_result])
      test_prediction_table = pd.concat([test_prediction_table, test_prediction_result])

  return model_assess_table, train_prediction_table, test_prediction_table
#%%
# Specified model need to run
#names = ["XGB", "RF", "DL1", "DL2"]
#models = [XGB, RF, DL, DL2 ]

#names = ["MLR", "Ridge", "Lasso", "DT", "RF", "XGB", "KNN", "SVM"]
names = ["MLR", "Ridge", "Lasso", "DT", "RF", "XGB", "SVM"]
#models = [Linear_default, Ridge_default, Lasso_default, DT_default, RF_default, XGB_default, KNN_default, SVM_default]
models = [Linear_default, Ridge_default, Lasso_default, DT_default, RF_default, XGB_default, SVM_default]

#x_train_notz = x_data_fp.copy()
#y_train_notz_A = np.ravel(y_data_fp_A.copy())

x_train_A, x_test_A, y_train_A, y_test_A = train_test_split(x_notz, y_notz_A, test_size=0.25, random_state=42)
x_train_B, x_test_B, y_train_B, y_test_B = train_test_split(x_notz, y_notz_B, test_size=0.25, random_state=42)
x_train_C, x_test_C, y_train_C, y_test_C = train_test_split(x_notz, y_notz_C, test_size=0.25, random_state=42)

# Run Training Model
all_result_model_A = []
all_result_model_B = []
all_result_model_C = []
all_time_fitting = []
for iteration in range(len(names)) :
    get_model = models[iteration]
    time_start = datetime.now()
    result_model_A = get_model(x_train_A, y_train_A)
    time_end = datetime.now()
    result_model_B = get_model(x_train_B, y_train_B)
    result_model_C = get_model(x_train_C, y_train_C)
    
    
    duration = (time_end - time_start).total_seconds()
    
    print(result_model_A)
    print(result_model_B)
    print(result_model_C)
    print(f'{duration} seconds')
    
    all_result_model_A.append(result_model_A)
    all_result_model_B.append(result_model_B)
    all_result_model_C.append(result_model_C)
    all_time_fitting.append(duration)

#%%
all_time_fitting

#%%
result_evaluation_A, train_prediction_original_A, test_prediction_original_A \
    = model_assess(x_train_A, x_test_A,
                   y_train_A, y_test_A,
                   all_result_model_A,
                   names)
    
result_evaluation_B, train_prediction_original_B, test_prediction_original_B \
    = model_assess(x_train_B, x_test_B,
                   y_train_B, y_test_B,
                   all_result_model_B,
                   names)
    
result_evaluation_C, train_prediction_original_C, test_prediction_original_C \
    = model_assess(x_train_C, x_test_C,
                   y_train_C, y_test_C,
                   all_result_model_C,
                   names)
    
#%%
result_evaluation_A["ABC_result"] = "A"
result_evaluation_B["ABC_result"] = "B"
result_evaluation_C["ABC_result"] = "C"

train_prediction_original_A["ABC_result"] = "A"
train_prediction_original_B["ABC_result"] = "B"
train_prediction_original_C["ABC_result"] = "C"

test_prediction_original_A["ABC_result"] = "A"
test_prediction_original_B["ABC_result"] = "B"
test_prediction_original_C["ABC_result"] = "C"

result_evaluation_all = pd.concat([result_evaluation_A,
                                   result_evaluation_B,
                                   result_evaluation_C])

train_prediction_all = pd.concat([train_prediction_original_A,
                                   train_prediction_original_B,
                                   train_prediction_original_C])

test_prediction_all = pd.concat([test_prediction_original_A,
                                   test_prediction_original_B,
                                   test_prediction_original_C])

#%%
result_evaluation = result_evaluation_all.reset_index(drop=True)
train_prediction = train_prediction_all.reset_index(drop=True).explode(['Training Predict', 'Training Actual'])
test_prediction = test_prediction_all.reset_index(drop=True).explode(['Test Predict', 'Test Actual'])


# Change datatype
def change_data_type(x):
  try :
    #print(x[0])
    try:
      #print(x[0][0])
      return float(x[0][0])
    except:
      return float(x[0])
  except:
    return float(x)

def change_data_type2(x):
  return float(x[0])

test_prediction['Test Predict'] = test_prediction['Test Predict'].apply(
    lambda x: x.reshape(1,-1))
train_prediction['Training Predict'] = train_prediction['Training Predict'].apply(
    lambda x: x.reshape(1,-1))
test_prediction['Test Actual'] = test_prediction['Test Actual'].apply(
    lambda x: x.reshape(1,-1))
train_prediction['Training Actual'] = train_prediction['Training Actual'].apply(
    lambda x: x.reshape(1,-1))

test_prediction['Test Predict'] = test_prediction['Test Predict'].apply(lambda x: change_data_type(x))
test_prediction['Test Actual'] = test_prediction['Test Actual'].apply(lambda x: change_data_type(x))
train_prediction['Training Predict'] = train_prediction['Training Predict'].apply(lambda x: change_data_type(x))
train_prediction['Training Actual'] = train_prediction['Training Actual'].apply(lambda x: change_data_type(x))

#%%

result_test = test_prediction.groupby("Method").agg(lambda x: list(x)).reset_index()
result_train = train_prediction.groupby("Method").agg(lambda x: list(x)).reset_index()
print(result_train)
print(result_test)

#%%

func_name = ["MAE", "RMSE", "R2"]
rmse_func = lambda x,y : mean_squared_error(x,y, squared=False)
func_list = [mean_absolute_error, rmse_func, r2_score]
def create_result_from_predict_value(table_predict, func_name, func_list):
    list_df = list()
    name_method = pd.Series(table_predict.apply(lambda x : x[0], axis=1), name="Method")
    df_1 = name_method.to_frame();   list_df.append(df_1)
    for i in range(len(func_list)):
      list_error_metric = []

      name_fx = func_name[i]
      func    = func_list[i]
      value = pd.Series(table_predict.apply(lambda x : func(x[1], x[2]), axis=1),name=name_fx)
      df_2=value.to_frame()
      list_df.append(df_2)

    return pd.concat(list_df, axis=1)



print(create_result_from_predict_value(result_train, func_name, func_list))
print("\n")
print(create_result_from_predict_value(result_test, func_name, func_list).reset_index(drop=True))

#%%

test_prediction_visual = test_prediction[test_prediction["Method"].isin(["Ridge", "Lasso", "DT", "RF", "XGB", "SVM"])]
#test_prediction_visual = test_prediction.copy()
test_prediction_visual[test_prediction_visual["Test Actual"]<-20]
print(f'min = {min(test_prediction_visual["Test Actual"])}  max = {max(test_prediction_visual["Test Actual"])}')


#%% Visualization
# Specified Range for plot
temp = test_prediction_visual[test_prediction_visual["ABC_result"]=="A"]
x_min = min(min(temp["Test Predict"]),min(temp["Test Actual"]))
x_max = max(max(temp["Test Predict"]),max(temp["Test Actual"]))
y_min, y_max = x_min, x_max

# Plot each metho
# Add Legend, range of show
g = sns.FacetGrid(test_prediction_visual, col="Method", col_wrap=4, hue="Method")
g.map_dataframe(sns.scatterplot, x="Test Actual", y="Test Predict", alpha=0.6)
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('A')

# Add Legend, range of show
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

#%% Visualization
# Specified Range for plot
temp = test_prediction_visual[test_prediction_visual["ABC_result"]=="B"]
x_min = min(min(temp["Test Predict"]),min(temp["Test Actual"]))
x_max = max(max(temp["Test Predict"]),max(temp["Test Actual"]))
y_min, y_max = x_min, x_max

# Plot each metho
# Add Legend, range of show
g = sns.FacetGrid(test_prediction_visual, col="Method", col_wrap=4, hue="Method")
g.map_dataframe(sns.scatterplot, x="Test Actual", y="Test Predict", alpha=0.6)
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('B')

# Add Legend, range of show
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

#%% Visualization
# Specified Range for plot
temp = test_prediction_visual[test_prediction_visual["ABC_result"]=="C"]
x_min = min(min(temp["Test Predict"]),min(temp["Test Actual"]))
x_max = max(max(temp["Test Predict"]),max(temp["Test Actual"]))
y_min, y_max = x_min, x_max

# Plot each metho
# Add Legend, range of show
g = sns.FacetGrid(test_prediction_visual, col="Method", col_wrap=4, hue="Method")
g.map_dataframe(sns.scatterplot, x="Test Actual", y="Test Predict", alpha=0.6)
g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('C')

# Add Legend, range of show
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
#%%
test_prediction_original_A_2 = test_prediction_original_A.copy()
test_prediction_original_B_2 = test_prediction_original_B.copy()
test_prediction_original_C_2 = test_prediction_original_C.copy()

test_prediction_original_A_2.columns = ['Method', 'A Test Predict', 'A Test Actual', 'ABC_result']
test_prediction_original_B_2.columns = ['Method', 'B Test Predict', 'B Test Actual', 'ABC_result']
test_prediction_original_C_2.columns = ['Method', 'C Test Predict', 'C Test Actual', 'ABC_result']

test_prediction_original_A_2 = test_prediction_original_A_2.drop(columns="ABC_result")
test_prediction_original_B_2 = test_prediction_original_B_2.drop(columns="ABC_result")
test_prediction_original_C_2 = test_prediction_original_C_2.drop(columns="ABC_result")
#%%
temp_df = pd.concat([test_prediction_original_A_2, test_prediction_original_B_2, 
           test_prediction_original_C_2], axis=1)
temp_df = temp_df[["Method", 'A Test Predict', 'B Test Predict', 'C Test Predict', 'A Test Actual', 'B Test Actual', 'C Test Actual']]
temp_df = temp_df.iloc[:, [0, 3, 4, 5, 6, 7, 8]]
#temp_df = temp_df[temp_df["Method"].isin(["RF"])]
temp_df["VP_Pred"] = Psat_cal(300, temp_df["A Test Predict"], temp_df["B Test Predict"], temp_df["C Test Predict"])
temp_df["VP_Act"] = Psat_cal(300, temp_df["A Test Actual"], temp_df["B Test Actual"], temp_df["C Test Actual"])

temp_df = temp_df.reset_index(drop=True)

#%%
for i in range(len(temp_df)):
    #print(i+1)
    
    temp = temp_df.copy()
    VP_pred = temp["VP_Pred"][i]
    VP_act = temp["VP_Act"][i]
    name_plot = temp["Method"][i]
    x_min = min(min(VP_pred),min(VP_act))
    x_max = max(max(VP_pred),max(VP_act))
    x_min = -20; x_max= 25
    y_min, y_max = x_min, x_max
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    t1 = mean_absolute_error(VP_act, VP_pred)
    t2 = mean_squared_error(VP_act, VP_pred, squared=False)
    t3 = r2_score(VP_act, VP_pred)
    
    # Add Legend, range of show
    plt.scatter(VP_pred, VP_act)
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    #g.fig.subplots_adjust(top=0.9)
    #g.fig.suptitle('C')
    
    # Add Legend, range of show
    plt.title(name_plot)
    text = f'MAE={t1:.2f}, RMSE={t2:.2f}, R2={t3:.2f}'
    plt.text(-15, 20, text)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()