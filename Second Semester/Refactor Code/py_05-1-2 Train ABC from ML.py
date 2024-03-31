import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

#%% Import data
df = pd.read_csv("csv-01-0 Psat-1800.csv")
df2_train = pd.read_csv("csv_02-1 df_train.csv").iloc[:,1:]
rrr2 = pd.read_csv("csv_04-1 test_eval_ln(Psat).csv")
result_train = pd.read_csv("csv_04-3 train_pred_ln(Psat).csv")



#%%
temp_Psat_actual_file = df2_train.copy()
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
prediction_file = result_train[result_train["Method"]==best_name].iloc[:,[2,3]].reset_index(drop=True)
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
from scipy.optimize import minimize

def sse_all(x, name_psat):
    sub = x
    def sse(v, T, logP):
        a,b,c=v
        return np.sum((logP-(a-b/(T+c)))**2)
    #print(sub)
    
    t_input = [np.array(i) for i in sub["T"]]
    logP_input = [np.array(i) for i in sub[name_psat]]
    result_minimize= minimize(sse, (20,2000,-10),args=(t_input, logP_input))
    #print(result_minimize.x)
    return result_minimize.x
result["ABC"] = result.apply(lambda x : sse_all(x, "ln_Psat_Pred (Pa)"), axis=1)

#%%
# =============================================================================
# x = result["T"].apply(lambda num: [1/num1 for num1 in num])
# y = result["ln_Psat_Pred (Pa)"]
# #y_flat = y.explode()
# xy_table = pd.DataFrame({
#     "x" : x,
#     "y" : y})
# 
# result2 = result.join(xy_table)
# 
# from scipy.optimize import curve_fit
# def objective(X, a, b, c):
#     x,y = X
#     # Linearized Equation : y + C * y * x1 = A + B * x1
#     # return a +(b*x) - (c*y*x)
# 
#     # Linearized Equation : logP = A + (AC-B) (1/T) +  (-C) (logP /T)
#     a0 = a
#     a1 = a*c - b
#     a2 = -c
#     x1 = x
#     x2 = y*x
#     return a0 + a1*x1 + a2*x2
# 
# def getABC(row):
#     #print(row.x)
#     x1 = row.x
#     y1 = row.y
#     #print(row.SMILES)
#     popt, _ = curve_fit(objective, (x1,y1), y1, method="dogbox")
#     a,b,c = popt
#     row.a = a
#     row.b = b
#     row.c = c
#     return [row.a, row.b, row.c]
# #z = func((x,y), a, b, c) * 1
# result2["ABC"] = result2.apply(getABC, axis=1)
# =============================================================================
result[['A_Pred', 'B_Pred', 'C_Pred']] = pd.DataFrame(result['ABC'].tolist())

#%%
# =============================================================================
# x_test = result["T"].apply(lambda num: [1/num1 for num1 in num])
# y_test = result["ln_Psat_Actual (Pa)"]
# xy_test_table = pd.DataFrame({
#     "x_test" : x_test,
#     "y_test" : y_test})
# result2 = result2.join(xy_test_table)
# def getABC2(row):
#     #print(row.x)
#     x1 = row.x_test
#     y1 = row.y_test
#     #print(row.SMILES)
#     popt, _ = curve_fit(objective, (x1,y1), y1, method="dogbox")
#     a,b,c = popt
#     row.a = a
#     row.b = b
#     row.c = c
#     return [row.a, row.b, row.c]
# =============================================================================
#%%
result["ABC_test"] = result.apply(lambda x : sse_all(x, "ln_Psat_Actual (Pa)"), axis=1)
result[['A_test', 'B_test', 'C_test']] = pd.DataFrame(result['ABC_test'].tolist())
result

#%%
result3 = result[["SMILES", "A_Pred", "B_Pred", "C_Pred", "A_test", "B_test", "C_test"]]
result3
#%%
#df_initial = pd.read_csv(r"C:/Users\Kan\Documents/GitHub\SMILE_to_Feature\Second Semester\[Use] Main File\RDKit_CHON_New_Data_Psat_Not_Outliers.csv")
df_initial = df.copy()
df_for_lookup = df_initial.copy()

# =============================================================================
# final = result3.set_index('SMILES').join(df_for_lookup.set_index('SMILES'), on="SMILES", validate="one_to_one",
#                      lsuffix='_caller', rsuffix='_other')
# =============================================================================
final = result3.merge(df_for_lookup, on="SMILES", how='inner')
final = final.iloc[:, [0,1,2,3,4,5,6, 15,16,17,24,25]]
#final = pd.concat([result3, df_for_lookup], axis=1, join='inner')

#final = result3.join(df_for_lookup, on="SMILES")
#final = pd.concat([result3, df_for_lookup[["A", "B", "C", "Atom2", "Func. Group"]]], axis=1, join="inner")
#print(final.describe())
final2 = pd.concat([final, result["T"]], axis=1, join="inner")
final2
#%%
rmse_func = lambda x,y : mean_squared_error(x,y, squared=False)
final3 = final2.merge(result[["SMILES", "ln_Psat_Pred (Pa)", "ln_Psat_Actual (Pa)"]],
                      on="SMILES",
                      how='inner')
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
df_plot["T"] = df_plot["T"].astype(float)
df_plot["ln_Psat_Actual (Pa)"] = df_plot["ln_Psat_Actual (Pa)"].astype(float)
df_plot["ln_Psat_Pred (Pa)"] = df_plot["ln_Psat_Pred (Pa)"].astype(float)
df_plot

#%% ln(Psat)
sns.set(font_scale=1.1)
sns.set_style("white")
# All
x_min = -10;  x_max = 20
y_min, y_max = x_min, x_max

sns.scatterplot(df_plot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)
plt.axline((0, 0), slope=1, color='.5', linestyle='--')

plt.title("ln($P^{Sat}$) Prediction from XGB Model, All Substance")
plt.xlabel("Actual ln($P_{sat}$)")
plt.ylabel("Predict ln($P_{sat}$)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()

# functional Group
x_min = -10;  x_max = 20
y_min, y_max = x_min, x_max

g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)

def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    ax = plt.gca()
    count = data["Func. Group"].unique()[0]
    #print(count)
    ax.text(.3, 0.05, 'Group = {}'.format(count), transform=ax.transAxes)
g.map_dataframe(annotate)
g.set_titles(col_template="")

# Add Legend, range of show
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
g.set_xlabels("Actual ln($P_{sat}$)")
g.set_ylabels("Predict ln($P_{sat}$)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('ln($P^{Sat}$) Prediction from XGB Model, Functional Group')
plt.show()

g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)

def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    r2 = r2_score(data['ln_Psat_Actual (Pa)'], data['ln_Psat_Pred (Pa)'])
    count = len(data['ln_Psat_Actual (Pa)'])
    rmse = mean_squared_error(data['ln_Psat_Actual (Pa)'], data['ln_Psat_Pred (Pa)'], squared=False)
    mape = mean_absolute_percentage_error(data['ln_Psat_Actual (Pa)'], data['ln_Psat_Pred (Pa)'])
    mape = mape*100
    ax = plt.gca()
    ax.text(.05, 0.75, 'count={}\nr2={:.3f}\n rmse={:.3f}\n mape={:.2f}(%)'.format(count, r2, rmse, mape),
            transform=ax.transAxes)
g.map_dataframe(annotate)

# Add Legend, range of show
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
g.set_xlabels("Actual ln($P_{sat}$)")
g.set_ylabels("Predict ln($P_{sat}$)")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('ln($P^{Sat}$) Prediction from XGB Model, Functional Group')
plt.show()
#%% Psat - functional Group
df_plot["Psat_Actual (atm)"] = np.exp(df_plot["ln_Psat_Actual (Pa)"])/(10**5)
df_plot["Psat_Pred (atm)"] = np.exp(df_plot["ln_Psat_Pred (Pa)"])/(10**5)

g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="Psat_Actual (atm)", y="Psat_Pred (atm)", alpha=0.6)
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    rmse = r2_score(data['Psat_Actual (atm)'], data['Psat_Pred (atm)'])
    ax = plt.gca()
    ax.text(.05, .8, 'mse={:.4f}'.format(rmse),
            transform=ax.transAxes)
g.map_dataframe(annotate)
    
plt.figure(figsize=(300,300))
g.set_xlabels("Actual $P_{sat}$")
g.set_ylabels("Predict $P_{sat}$")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('$P^{Sat}$ Prediction from XGB Model, Functional Group')
plt.show()
#%%
g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="Psat_Actual (atm)", y="Psat_Pred (atm)", alpha=0.6)
#g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))
g.map_dataframe(annotate)
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.figure(figsize=(300,300))
plt.show()
#%% A
x_min = min(min(df_plot["A"]), min(df_plot["A_Pred"]))-10
x_max = max(max(df_plot["A"]), max(df_plot["A_Pred"]))+10
x_min = 0; x_max = 100
y_min = x_min; y_max = x_max

# A - All
sns.scatterplot(df_plot, x="A", y="A_Pred", alpha=0.6)
plt.axline((0, 0), slope=1, color='.5', linestyle='--')

plt.title("A Prediction from XGB Model, All Substance")
plt.xlabel("Actual A")
plt.ylabel("Calculated A")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()

# A - Functional Group
g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="A", y="A_Pred", alpha=0.6)
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    rmse = r2_score(data['A'], data['A_Pred'])
    ax = plt.gca()
    ax.text(.05, .8, 'mse={:.4f}'.format(rmse),
            transform=ax.transAxes)
g.map_dataframe(annotate)

text = "A"
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
g.set_xlabels(f"Actual {text}")
g.set_ylabels(f"Calculated {text}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(f'{text} Prediction from XGB Model, Functional Group')
plt.show()
#%% B
x_min = min(min(df_plot["B"]), min(df_plot["B_Pred"]))-10000
x_max = max(max(df_plot["B"]), max(df_plot["B_Pred"]))+10000
x_min = -10000; x_max = 130000
y_min = x_min; y_max = x_max


# B - All
sns.scatterplot(df_plot, x="B", y="B_Pred", alpha=0.6)
plt.axline((0, 0), slope=1, color='.5', linestyle='--')

plt.title("B Prediction from XGB Model, All Substance")
plt.xlabel("Actual B")
plt.ylabel("Calculated B")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()

# B - Functional Group

g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="B", y="B_Pred", alpha=0.6)
#g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    rmse = r2_score(data['B'], data['B_Pred'])
    ax = plt.gca()
    ax.text(.05, .8, 'mse={:.4f}'.format(rmse),
            transform=ax.transAxes)
g.map_dataframe(annotate)

text = "B"
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
g.set_xlabels(f"Actual {text}")
g.set_ylabels(f"Calculated {text}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(f'{text} Prediction from XGB Model, Functional Group')
plt.show()

#%% C
x_min = min(min(df_plot["C"]), min(df_plot["C_Pred"]))-100
x_max = max(max(df_plot["C"]), max(df_plot["C_Pred"]))+100
x_min = -1000; x_max = 14000
y_min = x_min; y_max = x_max

# C - All
sns.scatterplot(df_plot, x="C", y="C_Pred", alpha=0.6)
plt.axline((0, 0), slope=1, color='.5', linestyle='--')

plt.title("C Prediction from XGB Model, All Substance")
plt.xlabel("Actual C")
plt.ylabel("Calculated C")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
plt.show()

# C - Functional Group

g = sns.FacetGrid(df_plot, col="Func. Group", col_wrap=4, hue="Func. Group")
g.map_dataframe(sns.scatterplot, x="C", y="C_Pred", alpha=0.6)
#g.map_dataframe(lambda data, **kws: plt.axline((0, 0), slope=1, color='.5', linestyle='--'))
def annotate(data, **kws):
    plt.axline((0, 0), slope=1, color='.5', linestyle='--')
    rmse = r2_score(data['C'], data['C_Pred'])
    ax = plt.gca()
    ax.text(.05, .8, 'mse={:.4f}'.format(rmse),
            transform=ax.transAxes)
g.map_dataframe(annotate)

text = "C"
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.figure(figsize=(300,300))
g.set_xlabels(f"Actual {text}")
g.set_ylabels(f"Calculated {text}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(f'{text} Prediction from XGB Model, Functional Group')
plt.show()
#%% MAPE ABC
mean_absolute_error(final3["A"], final3["A_Pred"])
mean_absolute_error(final3["B"], final3["B_Pred"])
mean_absolute_error(final3["C"], final3["C_Pred"])

mean_absolute_percentage_error(final3["A"], final3["A_Pred"])
mean_absolute_percentage_error(final3["B"], final3["B_Pred"])
mean_absolute_percentage_error(final3["C"], final3["C_Pred"])

r2_score(final3["A"], final3["A_Pred"])
r2_score(final3["B"], final3["B_Pred"])
r2_score(final3["C"], final3["C_Pred"])

#%% Export Section
#final3.to_csv("csv_05-2 Train ABC Calculated.csv")
