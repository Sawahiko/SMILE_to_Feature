import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

#%% Import data
df = pd.read_csv("csv-01-0 Psat-1800.csv")

rrr2 = pd.read_csv("csv_04-1 test_eval_ln(Psat).csv")

df2_train = pd.read_csv("csv_02-1 df_train.csv").iloc[:,1:]
result_train = pd.read_csv("csv_04-3 train_pred_ln(Psat).csv")
result_train["Dataset"] = "Train"

df2_test = pd.read_csv("csv_02-2 df_test.csv").iloc[:,1:]
result_test = pd.read_csv("csv_04-2 test_pred_ln(Psat).csv")
result_test["Dataset"] = "Test"

#%% ln(Psat) train-test compare

result_df = pd.concat([result_train, result_test], axis=0)

best_name = rrr2[rrr2["RMSE.1"]==min(rrr2["RMSE.1"])]["Method"].iloc[0]
result_df = result_df[result_df["Method"]==best_name]
#%%
df_visual = result_df.copy()
g= sns.scatterplot(
    data=df_visual, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)",
    hue="Dataset",
    style="Dataset",
    palette="dark:green", alpha=0.6
)
plt.axline((0, 0), slope=1, color='.5', linestyle='--')

g.set(xlim=(100, 700))
g.set(ylim=(100, 700))
plt.xlim(-10,20)
plt.ylim(-10,20)

plt.title("Model Prediction for $P_{sat}$")
plt.legend(loc='lower right')
plt.show()