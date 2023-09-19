import pandas as pd
dr = pd.read_excel("Clean_LogP.xlsx")

import numpy as np
x = np.array(dr[["CRe","DoubleCCRe","TripleCC","Bracket","Benzene","CycleRe","SingleCO","DoubleCO"]])
y = np.array(dr.mp)

from sklearn.model_selection import train_test_split, cross_validate
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state =0)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model_cv = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
model.fit(x_train,y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
y_pred = model.predict(x)

def predict_DT_logP(data):
    model.predict(data)
    return model.predict(data)