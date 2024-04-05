# Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tool, Error Metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from joblib import dump, load

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout

# Ensure GPU is used (If a GPU is available)
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% Import data

x_train = pd.read_csv("csv_02-3 std_x_train.csv").iloc[:,1:]
y_train = pd.read_csv("csv_02-4 std_y_train.csv").iloc[:,1]
x_test  = pd.read_csv("csv_02-5 std_x_test.csv").iloc[:,1:]
y_test  = pd.read_csv("csv_02-6 std_y_test.csv").iloc[:,1]

scaler_x = load("file_02-1 scaler_x.joblib")
scaler_y = load("file_02-2 scaler_y.joblib")

#%%

def build_model(hp):
    
    model = Sequential()
    
    counter = 0
    
    for i in range(hp.Int('num_layers', min_value=1, max_value=4)):
        
        if counter == 0:
            model.add(
                Dense(
                    hp.Int('units' + str(i), min_value=256, max_value=1024, step=32),
                    activation='relu',
                    input_dim=x_train.shape[1])
                )
        else:
            model.add(
                Dense(
                    hp.Int('units' + str(i), min_value=256, max_value=1024, step=32),
                    activation='relu',
                    )
                )
            model.add(
                Dropout(
                    rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1))
                )
        counter+=1
        model.add(Dense(1, activation='linear'))
        
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), 
                      loss="mean_squared_error", 
                      metrics=[keras.metrics.MeanAbsoluteError()])
        
        return model

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_mean_absolute_error", direction="min"),
    factor=3,
    max_epochs=100, #Max epoch to train for each model
    directory='my_directory', 
    project_name='test_f'
)

# Early stopping for efficiency 
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train,
             epochs=100, #Trial
             validation_split=0.1, 
             callbacks=[stop_early])

# Build model best hyperparameter
best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)

loss = pd.DataFrame(history.history)

y_pred_inv = scaler_y.inverse_transform(model.predict(x_test))
y_test_inv = scaler_y.inverse_transform(np.array([y_test]).reshape(-1, 1))

# =============================================================================
# plt.plot(y_test_inv ,y_test_inv, 'k--')
# plt.scatter(y_test_inv, y_pred_inv, alpha=0.3)
# =============================================================================

plt.plot(y_test_inv ,y_test_inv, 'k--')
plt.scatter(y_test_inv, y_pred_inv, alpha=0.3)

#%%

# =============================================================================
# #%% Visualization
# 
# x_min = min(min(df_com0["ln_Psat_act"]),min(df_com0["ln_Psat_pred"]))
# x_max = max(max(df_com0["ln_Psat_act"]),max(df_com0["ln_Psat_pred"]))
# y_min, y_max = x_min, x_max
# 
# x = np.linspace(x_min, x_max, 100)
# y = x
# 
# # PyPlot
# plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
# plt.scatter(df_com0["ln_Psat_act"], df_com0["ln_Psat_pred"], alpha=0.5)
# plt.xlabel("Actual")    
# plt.ylabel("Predictions")
# plt.title("Psat")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# 
# #%% Visualization
# 
# x_min = min(min(df_com1["ln_Psat_act"]),min(df_com1["ln_Psat_pred"]))
# x_max = max(max(df_com1["ln_Psat_act"]),max(df_com1["ln_Psat_pred"]))
# y_min, y_max = x_min, x_max
# 
# x = np.linspace(x_min, x_max, 100)
# y = x
# 
# # PyPlot
# plt.plot([x_min, x_max], [y_min, y_max], color="black", alpha=0.5, linestyle="--")
# plt.scatter(df_com1["ln_Psat_act"], df_com1["ln_Psat_pred"], alpha=0.5)
# plt.xlabel("Actual")
# plt.ylabel("Predictions")
# plt.title("Psat")
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# =============================================================================
