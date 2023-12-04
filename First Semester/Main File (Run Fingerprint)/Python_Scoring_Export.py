from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect

def get_name_of_input(func):
  """Returns the name of the input of the function.

  Args:
    func: A function.

  Returns:
    The name of the input of the function.
  """

  argspec = inspect.getfullargspec(func)
  if argspec.args:
    return argspec.args[0]
  else:
    return None

def Scoring(Model, x_train, x_test, x_total, y_train, y_test, y_total) :

    mae_train_table  = []
    mape_train_table = []
    rmse_train_table = []
    r2_train_table   = []
    
    # Train set
    y_predict_train = Model.predict(x_train)
    mae_train = mean_absolute_error(y_train, y_predict_train)
    mape_train = mean_absolute_percentage_error(y_train, y_predict_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_predict_train))
    R2_train = r2_score(y_train, y_predict_train)
    
    mae_train_table.append(mae_train)
    mape_train_table.append(mape_train*100)
    rmse_train_table.append(rmse_train)
    r2_train_table.append(R2_train)
    
    # Test set
    y_predict_test = Model.predict(x_test)
    mae_test = mean_absolute_error(y_test, y_predict_test)
    mape_test = mean_absolute_percentage_error(y_test, y_predict_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_predict_test))
    R2_test = r2_score(y_test, y_predict_test)
    
    mae_train_table.append(mae_test)
    mape_train_table.append(mape_test*100)
    rmse_train_table.append(rmse_test)
    r2_train_table.append(R2_test)
    
    # Total set
    y_predict_total = Model.predict(x_total)
    mae_total = mean_absolute_error(y_total, y_predict_total)
    mape_total = mean_absolute_percentage_error(y_total, y_predict_total)
    rmse_total = np.sqrt(mean_squared_error(y_total, y_predict_total))
    R2_total = r2_score(y_total, y_predict_total)
    
    mae_train_table.append(mae_total)
    mape_train_table.append(mape_total*100)
    rmse_train_table.append(rmse_total)
    r2_train_table.append(R2_total)
    # %% Store score y_predict
    # Table Score
    Score_Table = pd.DataFrame()
    data = {
            "MAE" :mae_train_table,
            "MAPE(%)":mape_train_table,
            "RMSE":rmse_train_table,
            "R2"  :r2_train_table
        }
    Score_Table = pd.DataFrame(data)
    return Score_Table
    
def Export(Score_Table, pathname):
    Score_Table.to_csv(pathname)