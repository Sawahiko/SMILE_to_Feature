# importing sys
import sys
import os

data_path = os.path.join(os.path.dirname(__file__), 'P_Pie')
 
# adding Folder_2/subfolder to the system path
sys.path.append(data_path)

from random import randint as rnd
import random
import pandas as pd

from mlmodel_Hansen_dispersion import model, predict_DT_Hansen_dis
from mlmodel_Hansen_Hbondn import model, predict_DT_Hansen_Hbond
from mlmodel_Hansen_polarity import model, predict_DT_Hansen_Polarity
from mlmodel_Hildebrand import model, predict_DT_Hildebrand
from mlmodel_LogP import model, predict_DT_logP
from mlmodel_LogS import model, predict_DT_LogS
from RD2 import check
#from error import errorcheck, errorcheck2
import numpy as np
import time as tm

def pop():
    count = 0
    cc =[]
    while count <  1000 :
        carbon = [rnd(1, 12), rnd(0, 6), rnd(0, 4), rnd(0, 10),rnd(0,2) , rnd(0, 4),rnd(0, 10), rnd(0, 3)]  #มาแก้ตรงนี้ค่าที่สุ่ม
        checkk = check(carbon)
        if checkk == True:
            cc.append(carbon)
            count = count+1
        else:
            count = count

    return pd.DataFrame(cc)

# %%
dataset = pd.read_excel("Clean_LogP.xlsx")
col1=["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]
dataset2 = dataset[col1]
#dataset.to_csv("random.csv")

# %%

dataset_use=dataset2.copy()
dataset_temp = dataset[["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]]
dataset_temp["Predict_LogS"] = predict_DT_LogS(dataset_use)
dataset_temp["Predict_LogP"] = predict_DT_logP(dataset_use)
dataset_temp["Predict_Hansen_Polarity"] = predict_DT_Hansen_Polarity(dataset_use)
dataset_temp["Predict_Hansen_dis"] = predict_DT_Hansen_dis(dataset_use)
dataset_temp["Predict_Hansen_H_bond"] = predict_DT_Hansen_Hbond(dataset_use)