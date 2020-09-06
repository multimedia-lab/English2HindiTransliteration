import pandas as pd
import os

def read_data(filename,filepath):
    df=pd.read_excel(os.path.join(filepath,filename),encoding="utf-8")
    return df

