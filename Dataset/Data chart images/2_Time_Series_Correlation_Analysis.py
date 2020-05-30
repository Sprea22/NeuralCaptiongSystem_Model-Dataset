import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Importing the data
df_dir = './Final_Dataset.xlsx'
df = pd.read_excel(df_dir)

list_of_series = df.ID_Series.unique()
list_of_titles = df.About.unique()

series_list = []
temp_df = pd.DataFrame()

for ID_series in list_of_series:
    current_series = pd.DataFrame(df[df["ID_Series"] == ID_series]["Value"])
    current_series = current_series.reset_index(drop=True)
    current_series.index.names = ['index']
    if(temp_df.empty):
        temp_df = current_series
    else:
        temp_df = pd.merge(temp_df, current_series, on='index')

labels_list_of_series = ["series "+ str(s) for s in list_of_series]
temp_df.columns = labels_list_of_series
corr_matrix = temp_df.corr()

def average_corr(df):
    df2 = df.copy()
    df2.values[np.tril_indices_from(df2)] = np.nan
    return df2.unstack().mean()

print(corr_matrix)
x= average_corr(corr_matrix)
print("Average correlation: ", x)

# Display the correlation matrix between the time series
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, interpolation='nearest')
fig.colorbar(cax)  
plt.show()

# Display the correlation matrix between the time series
# Highlighting just the time series with the highest correlation value
corr_matrix[corr_matrix == 1] = 0
corr_matrix[corr_matrix < 0.9] = 0
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix, interpolation='nearest')
fig.colorbar(cax)
plt.show()

lab_cont = 0
high_correlation_counter = 0
for lab1 in labels_list_of_series:
    lab_cont = lab_cont + 1
    for lab2 in labels_list_of_series[lab_cont : ]:
        if(corr_matrix[lab1][lab2] != 0):
            high_correlation_counter = high_correlation_counter + 1
            #print(lab1 + " & " + lab2 + ": " + str(corr_matrix[lab1][lab2]))

## 88 couples over 0.9
## 53 couples over 0.95
## 24 couples over 0.98
## 10 couples over 0.99
print("High correlation counter: ", high_correlation_counter, " couples")