import pandas as pd
db = pd.read_excel("DataTb.xlsx")
db

import numpy as np
X_data = np.array(db[["C","Double","Triple","Bracket","Cyclic"]])
Y_data = np.array(db.Tb)

## Building ML

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
X_train, X_test, y_train, y_test = train_test_split(X_data,Y_data,test_size=0.3,random_state =0)

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()
DT_cv =cross_validate(DT, X_train, y_train, cv=5, return_train_score=True)
DT.fit(X_train,y_train)

# Train set
y_predict_train = DT.predict(X_train)
#from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
mape_train = mean_absolute_percentage_error(y_train, y_predict_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
R2_train = r2_score(y_train, y_predict_train)

# Test set
y_predict_test = DT.predict(X_test)
mape_test = mean_absolute_percentage_error(y_test, y_predict_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
R2_test = r2_score(y_test, y_predict_test)

# Total set
y_predict_total = DT.predict(X_data)
mape_total = mean_absolute_percentage_error(Y_data, y_predict_total)
rmse_total = np.sqrt(mean_squared_error(Y_data, y_predict_total))
R2_total = r2_score(Y_data, y_predict_total)

# Table Score
DT_table = pd.DataFrame()
data = {
        "MAPE":[mape_train, mape_test, mape_total],
        "RMSE":[rmse_train, rmse_test, rmse_total],
        "R2"  :[R2_train, R2_test, R2_total]
    }
DT_table = pd.DataFrame(data)
DT_table.to_csv('DT_5-flod_5_feature.csv', index=False)


#MODULE

def predict_DT(data):
    DT.predict(data)
    return DT.predict(data)





