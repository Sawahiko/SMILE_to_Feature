import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Import data
df2_test = pd.read_csv("csv_02-2 df_test.csv").iloc[:,1:]
rrr2 = pd.read_csv("csv_04-1 test_eval_ln(Psat).csv")
result_test = pd.read_csv("csv_04-2 test_pred_ln(Psat).csv")


#%%
temp_Psat_actual_file = df2_test.copy()
temp_Psat_actual_file

best_name = rrr2[rrr2["RMSE.1"]==min(rrr2["RMSE.1"])]["Method"].iloc[0]
#%%
prediction_file = result_test[result_test["Method"]==best_name].iloc[:,[2,3]].reset_index(drop=True)
prediction_file

df3 = pd.concat([temp_Psat_actual_file, prediction_file], axis = 1)
df3
#%%
result = df3.groupby('SMILES', sort=False)[['T', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']].agg(list)
# Reset the index to create a DataFrame
result = result.reset_index()
result
#%%
result_temp = df3.groupby('SMILES', sort=False)[['T', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']].agg("count")

#%%
x = result["T"].apply(lambda num: [1/num1 for num1 in num])
y = result["ln_Psat_Actual (Pa)"]

xy_table = pd.DataFrame({
    "x" : x,
    "y" : y})

result2 = result.join(xy_table)
#%%
from scipy.optimize import curve_fit
def objective(X, a, b, c):
    x,y = X
    # Linearized Equation : y + C * y * x1 = A + B * x1
    # return a +(b*x) - (c*y*x)

    # Linearized Equation : logP = A + (AC-B) (1/T) +  (-C) (logP /T)
    a0 = a
    a1 = a*c - b
    a2 = -c
    x1 = x
    x2 = y*x
    return a0 + a1*x1 + a2*x2

def getABC(row):
    #print(row.x)
    x1 = row.x
    y1 = row.y
    popt, _ = curve_fit(objective, (x1,y1), y1, p0=[20, 2000, 0], method="lm")
    a,b,c = popt
    return [a,b,c]
#z = func((x,y), a, b, c) * 1
result2["ABC"] = result2.apply(getABC, axis=1)
result2[['A_Pred', 'B_Pred', 'C_Pred']] = pd.DataFrame(result2['ABC'].tolist())


x_test = result["T"].apply(lambda num: [1/num1 for num1 in num])
y_test = result["ln_Psat_Actual (Pa)"]
xy_test_table = pd.DataFrame({
    "x_test" : x_test,
    "y_test" : y_test})
result2 = result2.join(xy_test_table)
def getABC2(row):
    #print(row.x)
    x1 = row.x_test
    y1 = row.y_test
    popt, _ = curve_fit(objective, (x1,y1), y1, p0=[20, 2000, 60], method="lm")
    a,b,c = popt
    return [a,b,c]
result2["ABC_test"] = result2.apply(getABC2, axis=1)
result2[['A_test', 'B_test', 'C_test']] = pd.DataFrame(result2['ABC_test'].tolist())
result2

result3 = result2[["SMILES", "A_Pred", "B_Pred", "C_Pred", "A_test", "B_test", "C_test"]]
result3