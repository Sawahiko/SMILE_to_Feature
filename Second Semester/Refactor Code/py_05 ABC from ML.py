import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

#%% Import data
df = pd.read_csv("csv_01 Psat_[X]_ABCTminTmaxC1-12.csv")
df2_test = pd.read_csv("csv_02-2 df_test.csv").iloc[:,1:]
rrr2 = pd.read_csv("csv_04-1 test_eval_ln(Psat).csv")
result_test = pd.read_csv("csv_04-2 test_pred_ln(Psat).csv")


#%%
temp_Psat_actual_file = df2_test.copy()
temp_Psat_actual_file

best_name = rrr2[rrr2["RMSE.1"]==min(rrr2["RMSE.1"])]["Method"].iloc[0]

#%% Select Best Algo
data = rrr2
column_headers = data.columns

data = data.applymap(str)
data.round(3) 
cell_text = data.to_numpy()
# =============================================================================
# cell_text = []
# for row in data:
#     try:
#         cell_text.append([f'{x/1000:1.1f}' for x in row])
#     except:
#         cell_text.append(row)
# =============================================================================
fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=cell_text,
                      colLabels=column_headers,
                      loc='center')
fig.tight_layout()

plt.show()
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
re1 = result_temp[result_temp["T"]==1].index

df3 = df3[~df3["SMILES"].isin(re1)]
result = df3.groupby('SMILES', sort=False)[['T', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']].agg(list)
# Reset the index to create a DataFrame
result = result.reset_index()
result
#%%

result_temp = df3.groupby('SMILES', sort=False)[['T', 'ln_Psat_Pred (Pa)', 'ln_Psat_Actual (Pa)']].agg("count")
#%%
x = result["T"].apply(lambda num: [1/num1 for num1 in num])
y = result["ln_Psat_Actual (Pa)"]
#y_flat = y.explode()
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
    popt, _ = curve_fit(objective, (x1,y1), y1, p0=[20, 2000, -5], method="lm")
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
#%%
#df_initial = pd.read_csv(r"C:/Users\Kan\Documents/GitHub\SMILE_to_Feature\Second Semester\[Use] Main File\RDKit_CHON_New_Data_Psat_Not_Outliers.csv")
df_initial = df.copy()
df_for_lookup = df_initial.copy()

final = result3.merge(df_for_lookup, on="SMILES")
#final = result3.join(df_for_lookup, on="SMILES")
final = pd.concat([result3, df_for_lookup[["A", "B", "C", "Atom2"]]], axis=1, join="inner")
#print(final.describe())
final2 = pd.concat([final, result["T"]], axis=1, join="inner")
final2
#%%
rmse_func = lambda x,y : mean_squared_error(x,y, squared=False)
final3 = final2.merge(result[["SMILES", "ln_Psat_Pred (Pa)", "ln_Psat_Actual (Pa)"]], on="SMILES")
def cb2(row):
    #print(row["Test Actual"])
    return  rmse_func(row["ln_Psat_Actual (Pa)"], row["ln_Psat_Pred (Pa)"])
final3["RMSE"] = final3.apply(lambda x : cb2(x), axis=1)
final3
#%%
def do_plot(rmse, is_plot, count, thereshold=1):
    #print(rmse)
    if rmse_predict >= thereshold:
        count+=1
        if is_plot :
            x1 = temp["T"]
            x2 = temp["T"]

            y1 = temp["Test Actual"]
            y2 = temp["Test Predict"]

            smiles = temp["SMILES"]
            
            # Mention x and y limits to define their range
            plt.xlim(0, 700)
            plt.ylim(-5, 20)

            # Plotting graph
            plt.scatter(x1, y1, alpha=0.6)
            plt.plot(x2, y2, alpha=0.6)

            #g.set_xlabels("Actual log($P_{sat}$) [Pa]")
            #g.set_ylabels("Predict log($P_{sat}$) [Pa]")

            #plt.axline((0, 0), slope=1, color='.5', linestyle='--')
            temp_T = temp["T"]
            t = f'T = {min(temp_T):0.1f} - {max(temp_T):0.1f} K'
            abc = f'Actual  : A = {temp["A"]:0.3f}, B = {temp["B"]:0.4f},  C = {temp["C"]:0.2f}'
            p_abc = f'Predict : A = {temp["A_Pred"]:0.3f}, B = {temp["B_Pred"]:0.4f},  C = {temp["C_Pred"]:0.2f}'


            plt.text(10,19,f'sub: {i}')
            plt.text(6,17,smiles)
            plt.text(6,15,t)
            plt.text(6,13,f'rmse = {rmse_predict:0.3f}')
            plt.text(200, 19, abc)
            plt.text(200, 18, p_abc)

            plt.grid()

            #filename = f'2024-02-13 image/figure {i+1}.png'
            #filenames.append(filename)
            #plt.savefig(filename, bbox_inches='tight')

            plt.pause(0.01)

            plt.show()
            plt.xlabel = "Actual log($P_{sat}$) [Pa]"
            plt.ylabel = "Predict log($P_{sat}$) [Pa]"
            plt.close("all")
    return count
#%%
from matplotlib import pyplot as plt

filenames = []
count=0
thereshold_input=2
for i in range(len(result)):
    temp = final3.iloc[i]
    rmse_predict = temp["RMSE"]
    count = do_plot(rmse_predict, False, count, thereshold=thereshold_input)

#%%
(len(result), count)
#%%
final3
#%%
#final3["over"] = final3[
#final3["RMSE"]>=thereshold_input
#final3.loc[final3["RMSE"]>=thereshold_input, "RMSE2"] = "Pass"
final3["RMSE2"] = np.where(final3["RMSE"] <= thereshold_input, "Pass", "NOT")
#final3[["SMILES","RMSE2"]]
final3
#%%
final3_notpass = final3[final3["RMSE2"]=="NOT"]
final3_notpass
#%%
df_join = df[['SMILES', 'Name', 'Atom2']]
df_join
#%%
temporary_df = final3_notpass[["SMILES", "RMSE2"]].merge(df_join, on="SMILES", how='inner')
temporary_df
#.merge(df_SMILES_CHON, on='SMILES', how='inner')
#%%
gp = sns.histplot(final3, x="RMSE", alpha=0.6, binwidth=0.25)
plt.show(gp)
#%%
df_plot = final3.explode(["T", "ln_Psat_Actual (Pa)","ln_Psat_Pred (Pa)"]).reset_index(drop=True)
df_plot

#%%
# Specified Range for plot
x_min = -20;  x_max = 25
y_min, y_max = x_min, x_max

# Plot each method
#Test Predict	Test Actual

#plt.plot(x = final3["Test Predict"], y = final3["Test Predict"])
markers = {"Pass": "o", "NOT": "X"}
gc = sns.scatterplot(df_plot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)",
                     hue="Atom2",
                     #style= "RMSE2", markers=markers
                     alpha=0.6, )
plt.axline((0, 0), slope=1, color='.5', linestyle='--')


# Add Legend, range of show
plt.title(best_name)
plt.xlabel("Actual ln($P_{sat}$)")
plt.ylabel("Predict ln($P_{sat}$)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
#gc.xlabels("Actual log($P_{sat}$) [Pa]")
#plt.ylabel("Predict log($P_{sat}$) [Pa]")
#gc.set_xlabels("Actual log($P_{sat}$) [Pa]")
#gc.set_ylabels("Predict log($P_{sat}$) [Pa]")
plt.show()
#%%
x_min = min(min(df_plot["A"]), min(df_plot["A_Pred"]))-10
x_max = max(max(df_plot["A"]), max(df_plot["A_Pred"]))+10
y_min = x_min; y_max = x_max

sns.scatterplot(df_plot, x="A", y="A_Pred", alpha=0.6)
plt.axline((20, 20), slope=1, color='.5', linestyle='--')

text = "A"
plt.title(text)
plt.xlabel(f"Actual {text}")
plt.ylabel(f"Predict {text}")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()
#%%
x_min = min(min(df_plot["B"]), min(df_plot["B_Pred"]))
x_max = max(max(df_plot["B"]), max(df_plot["B_Pred"]))+1000
#x_min = -10000; x_max = 40000
y_min = x_min; y_max = x_max

sns.scatterplot(df_plot, x="B", y="B_Pred", alpha=0.6)
plt.axline((0, 0), slope=1, color='.5', linestyle='--')

text = "B"
plt.title(text)
plt.xlabel(f"Actual {text}")
plt.ylabel(f"Predict {text}")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()

#%%
x_min = min(min(df_plot["C"]), min(df_plot["C_Pred"]))-100
x_max = max(max(df_plot["C"]), max(df_plot["C_Pred"]))+100
#x_min = -10000; x_max = 40000
y_min = x_min; y_max = x_max

sns.scatterplot(df_plot, x="C", y="C_Pred", alpha=0.6)
plt.axline((-50, -50), slope=1, color='.5', linestyle='--')

text = "C"
plt.title(text)
plt.xlabel(f"Actual {text}")
plt.ylabel(f"Predict {text}")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()