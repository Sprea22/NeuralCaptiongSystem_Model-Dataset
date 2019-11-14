import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

# Importing the data
df_dir = './Captions collection/v0_captions_collection.xlsx'
df = pd.read_excel(df_dir)

list_of_captions = []
list_of_numb_words = []
list_of_numb_chars = []
stats_df = pd.DataFrame(columns=["ID_Series", "Caption", "Number_words", "Numb_chars"])

for index, row in df.iterrows():
    list_of_captions.append(row["caption_content"])
    list_of_numb_words.append(len(row["caption_content"].split(" ")))
    list_of_numb_chars.append(len(row["caption_content"]))

# Initialize the columns of the dataframe statistics
stats_df["ID_Series"] = df.index.values
stats_df["Caption"] = list_of_captions
stats_df["Number_words"] = list_of_numb_words
stats_df["Numb_chars"] = list_of_numb_chars

# Set the new index
stats_df = stats_df.set_index(["ID_Series"])

# Save the statistics dataframe
# stats_df.to_excel("Data chart images/Statistics_Captions.xlsx")

# Display most frequent words
