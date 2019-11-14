import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Importing the data
df_dir = './data_charts_dataset.xlsx'
df = pd.read_excel(df_dir)

list_of_series = df.ID_Series.unique()
list_of_titles = df.About.unique()

series_list = []
temp_df = pd.DataFrame()

print(df)

for ID_series in list_of_series:
    current_series = pd.DataFrame(df[df["ID_Series"] == ID_series]["Value"])
    print(current_series)