import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

#%%
final3 = pd.read_csv("csv_05-2 Train ABC Calculated.csv")


def cb(x):
    x1 = x.strip('][').split(', ')
    x2 = [float(i) for i in x1]
    return x2

final4 = final3[["SMILES","A_Pred", "B_Pred", "C_Pred",
                 "A", "B", "C", "T",
                 "ln_Psat_Actual (Pa)", "ln_Psat_Pred (Pa)", "Func. Group"]]
    
final4["ln_Psat_Actual (Pa)"] = final4["ln_Psat_Actual (Pa)"].apply(lambda x: cb(x))
final4["ln_Psat_Pred (Pa)"] = final4["ln_Psat_Pred (Pa)"].apply(lambda x: cb(x))
#%%

final4["T"] = final4["T"].apply(lambda x: cb(x))
def Psat_cal(T,A,B,C):
    #print(T, type(T))
    #print(A, type(A))
    #final1 =A-(B/(np.array(T)+C)) 
    final2 = A-(B/(np.array(T)+C))
    #return list(final)
    #print(final2)
    return final2

final4["ln_Psat_Actual (Pa)_Cal"] = final4.apply(lambda x: Psat_cal(x["T"], 
                                         x["A"], x["B"], x["C"]), axis=1)
final4["ln_Psat_Pred (Pa)_Cal"] = final4.apply(lambda x: Psat_cal(x["T"], 
                                         x["A_Pred"], x["B_Pred"], x["C_Pred"]), axis=1)
final4["RMSE"] = final4.apply(lambda x: mean_squared_error(x["ln_Psat_Actual (Pa)_Cal"], x["ln_Psat_Pred (Pa)_Cal"],squared=False), axis=1)

#RMSE RMSE2
#%%
def do_plot(rmse, is_plot, count, thereshold=1, is_second_cond=False):
    #print(rmse)
    cond1 = rmse_predict >= thereshold
    selected_cond = not(cond1)  if is_second_cond else cond1
    fig = plt.figure()
    #print(selected_cond)
    if selected_cond:
        
        if is_plot :
            
            x1 = temp["T"]
            #x1 = temp["T"].strip('][').split(', ')
            #x1 = [float(i) for i in x1]
            x2 = x1.copy()

            y1 = temp["ln_Psat_Actual (Pa)_Cal"]
            y2 = temp["ln_Psat_Pred (Pa)_Cal"]
# =============================================================================
#             print(temp["A"], temp["B"], temp["C"])
#             print(temp["A_Pred"], temp["B_Pred"], temp["C_Pred"])
#             print(y1, y2)
# =============================================================================

            smiles = temp["SMILES"]
            
            

            # Plotting graph
            # 1 : Actual - Line
            # 2 : Predict - point (scatter)
            
            plt.plot(x1, y1, alpha=0.6, figure=fig)
            plt.scatter(x2, y2, alpha=0.6)
            # Mention x and y limits to define their range
            plt.xlim(0, 700)
            plt.ylim(-5, 20)

            #g.set_xlabels("Actual log($P_{sat}$) [Pa]")
            #g.set_ylabels("Predict log($P_{sat}$) [Pa]")

            #plt.axline((0, 0), slope=1, color='.5', linestyle='--')
            temp_T = x1
# =============================================================================
#             min_temp_t = min(temp_T)
#             print(temp_T)
#             print(type(temp_T))
# =============================================================================
            t = f'T = {min(temp_T):0.1f} - {max(temp_T):0.1f} K'
            abc = f'Actual  : A = {temp["A"]:0.3f}, B = {temp["B"]:0.3f},  C = {temp["C"]:0.2f}'
            p_abc = f'Predict : A = {temp["A_Pred"]:0.3f}, B = {temp["B_Pred"]:0.3f},  C = {temp["C_Pred"]:0.2f}'


            plt.text(10,19,f'sub: {i}', figure=fig)
            plt.text(6,17,smiles, figure=fig)
            plt.text(6,15,t, figure=fig)
            plt.text(6,13,f'rmse = {rmse_predict:0.3f}', figure=fig)
            plt.text(200, 19, abc, figure=fig)
            plt.text(200, 18, p_abc, figure=fig)

            #plt.grid(figure=fig)

            #filename = f'2024-02-13 image/figure {i+1}.png'
            #filenames.append(filename)
            #plt.savefig(filename, bbox_inches='tight')

            plt.pause(0.1)

            fig.xlabel="Actual ln($P_{sat}$) [Pa]"
            fig.ylabel="Predict ln($P_{sat}$) [Pa]"
            #plt.xlabel("Actual ln($P_{sat}$) [Pa]", figure=fig)
            #plt.ylabel("Predict ln($P_{sat}$) [Pa]", figure=fig)
            plt.show(fig)
            #plt.close("all", fig)
    else:
        fig = None
    count+=1
    return [count,fig]
#%%
from matplotlib import pyplot as plt

filenames = []
count=0
thereshold_input=1
final4['RMSE2'] = np.where(final4['RMSE']<= thereshold_input, True, False)
final4_pass = final4[final4["RMSE2"]==True]
final4_notpass = final4[final4["RMSE2"]==False]

list_pic_not_pass = []
for i in range(len(final4)):
    #print(i)
    temp = final4.iloc[i]
    rmse_predict = temp["RMSE"]
    count, p = do_plot(rmse_predict, True, count, thereshold=thereshold_input, is_second_cond=False)
    list_pic_not_pass.append(p)

list_pic_pass = []    
for i in range(len(final4)):
    #print(i)
    temp = final4.iloc[i]
    rmse_predict = temp["RMSE"]
    count,p = do_plot(rmse_predict, True, count, thereshold=thereshold_input, is_second_cond=True)
    list_pic_pass.append(p)
    
#%%
i=0
for img in list_pic_not_pass:
    i+=1
    if img is not None:
        img.savefig(f"train_img/pic_not_pass_{i}")
#%%
i=0
for img in list_pic_pass:
    i+=1
    if img is not None:
        img.savefig(f"train_img/pic_pass_{i}")
#%%
final5 = final4[["SMILES", "ln_Psat_Actual (Pa)_Cal", "ln_Psat_Pred (Pa)_Cal",
                 "ln_Psat_Actual (Pa)", "ln_Psat_Pred (Pa)"]]
final6 = final5.explode(["ln_Psat_Actual (Pa)_Cal", "ln_Psat_Pred (Pa)_Cal",
                 "ln_Psat_Actual (Pa)", "ln_Psat_Pred (Pa)"])
#%%
print(len(final4_pass))
print(len(final4_notpass ))
sns.scatterplot(final6, x="ln_Psat_Actual (Pa)", y="ln_Psat_Pred (Pa)", alpha=0.6)
plt.xlim(-20, 25)
plt.ylim(-20, 25)
plt.show()
sns.scatterplot(final6, x="ln_Psat_Actual (Pa)_Cal", y="ln_Psat_Pred (Pa)_Cal", alpha=0.6)
plt.xlim(-20, 25)
plt.ylim(-20, 25)
plt.show()

#%%
f41= final4_pass.groupby("Func. Group")["RMSE2"].agg(lambda x: sum(x==True))
f41
f42 = final4_notpass.groupby("Func. Group")["RMSE2"].agg(lambda x: sum(x==False))
f42
f43 = pd.concat([f41, f42 ], axis=1)
f43.columns=["Pass", "Not Pass"]
#f43["Pass"] = f43["Pass"].astype(int)
print(f43)