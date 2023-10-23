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