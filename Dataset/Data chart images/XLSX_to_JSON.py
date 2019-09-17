import pandas as pd
import json

# Importing the data
df_dir = './Final_Dataset.xlsx'
df = pd.read_excel(df_dir)

list_of_series = df.ID_Series.unique()

json_data = {}

for time_series_id in list_of_series:
    # Select a single time series from the input dataset
    temp_df = df[df["ID_Series"] == time_series_id]
    title = temp_df["About"].iloc[0]
    year = str(int(temp_df["Year"].iloc[0]))
    geo = temp_df["Geo"].iloc[0]
    title = temp_df["About"].iloc[0]
    unit_of_measure = temp_df["UOM"].iloc[0]
    json_data[str(time_series_id)] = {"title" : title, "year" : year, "geo" : geo, "unit_of_measure" : unit_of_measure}

# Saving the results JSON file
with open("Plots_Information.json", "w") as write_file:
    json.dump(json_data, write_file)

