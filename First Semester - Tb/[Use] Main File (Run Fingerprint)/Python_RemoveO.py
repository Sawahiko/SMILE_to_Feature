import numpy as np
import pandas as pd
import pubchempy as pcp


def remove_outliers(Excel_path, Excel_sheetname, Threshold=3):
    df = pd.read_excel(Excel_path, sheet_name=Excel_sheetname)
    data = df['Tb']
    outliers = []
    threshold=Threshold
    mean = np.mean(data)
    std =np.std(data)
    
    for i in data:
        z_score = (i - mean)/std 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    new_df = df[~df['Tb'].isin(outliers)].reset_index(drop=True)
#    print(outliers)
    return new_df

def remove_outliers_boxplot(Excel_path, Excel_sheetname, columns, IQR_factor=1.5):
    df = pd.read_excel(Excel_path, sheet_name=Excel_sheetname)

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (IQR_factor * IQR)
        upper_bound = Q3 + (IQR_factor * IQR)

        df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]

    return df.reset_index(drop=True)
