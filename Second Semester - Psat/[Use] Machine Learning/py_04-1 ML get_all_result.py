import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

#%% Combined Train
# Get File
files_train = glob.glob('csv_03*train*.csv')
#files_train = [x for x in files_train[0:3]] ### Temporary  ###
dfs_train = [pd.read_csv(file)for file in files_train] 

#Change Unit and header
prediction_table_train = pd.concat(dfs_train).iloc[:,1:].reset_index(drop=True)
prediction_table_train.columns = ['Method', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']
prediction_table_train["Psat_Pred (atm)"] = np.exp(prediction_table_train["ln_Psat_Pred (Pa)"])/(10**5)
prediction_table_train["Psat_Actual (atm)"] = np.exp(prediction_table_train["ln_Psat_Actual (Pa)"])/(10**5)

### Temporary  ###
#prediction_table_train=prediction_table_train.iloc[0:int((int(len(prediction_table_train)/4)-105)*4),:]    

#Insert SMILES, Temp, CHON
df2_train = pd.read_csv("csv_02-1 df_train.csv").iloc[:,1:]

SMILES_T_table_train = df2_train.iloc[:,[1,2]]
SMILES_T_table_train_ex = pd.concat([SMILES_T_table_train] * len(dfs_train)).reset_index(drop=True)

# Merged
result_train = pd.concat([prediction_table_train, SMILES_T_table_train_ex], axis=1)

#%% Combined Test
# Get File
files_test = glob.glob('csv_03*test*.csv')
#files_test = [x for x in files_test[0:3]] ### Temporary  ###
dfs_test = [pd.read_csv(file)for file in files_test] 

#Change Unit and header
prediction_table_test = pd.concat(dfs_test).iloc[:,1:].reset_index(drop=True)
prediction_table_test.columns = ['Method', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']
prediction_table_test["Psat_Pred (atm)"] = np.exp(prediction_table_test["ln_Psat_Pred (Pa)"])/(10**5)
prediction_table_test["Psat_Actual (atm)"] = np.exp(prediction_table_test["ln_Psat_Actual (Pa)"])/(10**5)

### Temporary  ###
#prediction_table_test=prediction_table_test.iloc[0:int((int(len(prediction_table_test)/4)-105)*4),:]    
#Insert SMILES, Temp, CHON
df2_test = pd.read_csv("csv_02-2 df_test.csv").iloc[:,1:]

SMILES_T_table_test = df2_test.iloc[:,[1,2]]
SMILES_T_table_test_ex = pd.concat([SMILES_T_table_test] * len(dfs_test)).reset_index(drop=True)

# Merged
result_test = pd.concat([prediction_table_test, SMILES_T_table_test_ex], axis=1)
#%% Visualization-ln(Psat) train
sns.set(font_scale=1.1)
sns.set_style("white")

test_prediction_bestpar_visual = result_train.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}  max = {max(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}')

# Specified Range for plot
x_min = -10;  x_max = 20
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)
# Insert Perfect Line
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    ax = plt.gca()
    count = data["Method"].unique()[0]
    ax.text(.65, 0.05, 'Model = {}'.format(count),
            transform=ax.transAxes)
g.map_dataframe(annotate)
g.set_titles(col_template="")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("$P_{sat}$ Model Prediction")
# Add Legend, range of show
#g.add_legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
g.set(xticks=range(x_min,x_max+5,5))


#%% Visualization-ln(Psat) test

test_prediction_bestpar_visual = result_test.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}  max = {max(test_prediction_bestpar_visual["ln_Psat_Actual (Pa)"])}')

# Specified Range for plot
x_min = -10;  x_max = 20
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)
# Insert Perfect Line
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    ax = plt.gca()
    count = data["Method"].unique()[0]
    ax.text(.65, 0.05, 'Model = {}'.format(count),
            transform=ax.transAxes)
g.map_dataframe(annotate)
g.set_titles(col_template="")
# Add Legend, range of show
#g.add_legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("$P_{sat}$ Model Prediction")
g.set(xticks=range(x_min,x_max+5,5))
#%% Visualization- Psat train


test_prediction_bestpar_visual = result_train.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["Psat_Actual (atm)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["Psat_Actual (atm)"])}  max = {max(test_prediction_bestpar_visual["Psat_Actual (atm)"])}')

# Specified Range for plot
x_min = -5;  x_max = 75
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="Psat_Actual (atm)", y="Psat_Pred (atm)", alpha=0.6)
# Insert Perfect Line
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    ax = plt.gca()
    count = data["Method"].unique()[0]
    ax.text(.05, 0.75, 'Model = {}'.format(count),
            transform=ax.transAxes)
g.map_dataframe(annotate)
g.set_titles(col_template="")
# Add Legend, range of show
#g.add_legend()
#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("$P_{sat}$ Model Prediction")
g.set(xticks=range(0,60+20,20))
#%% Visualization- Psat test

test_prediction_bestpar_visual = result_test.copy()
test_prediction_bestpar_visual[test_prediction_bestpar_visual["Psat_Actual (atm)"]<-20]
print(f'min = {min(test_prediction_bestpar_visual["Psat_Actual (atm)"])}  max = {max(test_prediction_bestpar_visual["Psat_Actual (atm)"])}')

# Specified Range for plot
x_min = -5;  x_max = 75
y_min, y_max = x_min, x_max

# Plot each method
g = sns.FacetGrid(test_prediction_bestpar_visual, col="Method", col_wrap=2, hue="Method")
g.map_dataframe(sns.scatterplot, x="Psat_Actual (atm)", y="Psat_Pred (atm)", alpha=0.6)
# Insert Perfect Line
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    ax = plt.gca()
    count = data["Method"].unique()[0]
    ax.text(.05, 0.75, 'Model = {}'.format(count),
            transform=ax.transAxes)
g.map_dataframe(annotate)
g.set_titles(col_template="")
# Add Legend, range of show
#g.add_legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("$P_{sat}$ Model Prediction")
g.set(xticks=range(0,60+20,20))
#%% Evaluation

temp_train = result_train.groupby(['Method']).agg({'ln_Psat_Pred (Pa)': lambda x: x.tolist(),
                                      'ln_Psat_Actual (Pa)': lambda x: x.tolist()})
temp_test = result_test.groupby(['Method']).agg({'ln_Psat_Pred (Pa)': lambda x: x.tolist(),
                                      'ln_Psat_Actual (Pa)': lambda x: x.tolist()})

func_name = ["MAE", "MAPE", "RMSE", "R2"]
rmse_func = lambda x,y : mean_squared_error(x,y, squared=False)
func_list = [mean_absolute_error, mean_absolute_percentage_error, rmse_func, r2_score]

def create_result_from_predict_value(table_predict, func_name, func_list):
    list_df = list()
    #name_method = pd.Series(table_predict.apply(lambda x : x.index[0], axis=1), name="Method")
    #name_method = pd.Series(table_predict.apply(lambda x : x.index, axis=1), name="Method")
    #df_1 = name_method.to_frame();   list_df.append(df_1)
    for i in range(len(func_list)):
      list_error_metric = []
      #print(table_predict)
      name_fx = func_name[i]
      func    = func_list[i]
      #value = pd.Series(table_predict.apply(lambda x : print(x[0]), axis=1),name=name_fx)
      value = pd.Series(table_predict.apply(lambda x : func(x[0], x[1]), axis=1),name=name_fx)
      df_2=value.to_frame()
      #print(df_1)
      #print(df_2)
      #df_combine = pd.merge(df_1, df_2, how='inner', left_index=True, right_index=True)
      #df_combine = pd.concat([df_1, df_2], axis=1)
      list_df.append(df_2)

    return pd.concat(list_df, axis=1)



train_rrr = create_result_from_predict_value(temp_train, func_name, func_list)
test_rrr= create_result_from_predict_value(temp_test, func_name, func_list)
rrr1 = pd.concat([train_rrr, test_rrr], axis=1)
print(rrr1)


temp_train = result_train.groupby(['Method']).agg({'Psat_Pred (atm)': lambda x: x.tolist(),
                                      'Psat_Actual (atm)': lambda x: x.tolist()})
temp_test = result_test.groupby(['Method']).agg({'Psat_Pred (atm)': lambda x: x.tolist(),
                                      'Psat_Actual (atm)': lambda x: x.tolist()})

train_rrr = create_result_from_predict_value(temp_train, func_name, func_list)
test_rrr= create_result_from_predict_value(temp_test, func_name, func_list)
rrr2 = pd.concat([train_rrr, test_rrr], axis=1)
print(rrr2)

#%% Export
# =============================================================================
# rrr2.to_csv("csv_04-1 test_eval_ln(Psat).csv")
# prediction_table_test.to_csv("csv_04-2 test_pred_ln(Psat).csv")
# 
# prediction_table_train.to_csv("csv_04-3 train_pred_ln(Psat).csv")
# =============================================================================
