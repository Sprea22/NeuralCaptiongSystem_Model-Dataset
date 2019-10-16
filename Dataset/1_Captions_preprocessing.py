import pandas as pd 
import numpy as np
import re

# Importing the dataset about the collected captions and about the generated line charts
v0_captions_collection = pd.read_excel("Captions collection/v0_captions_collection.xlsx")
data_chart_dataset = pd.read_excel("Data chart images/data_charts_dataset.xlsx")

# Initializing the structure of the new dataset
new_columns_list = ["ID_Caption", "ID_Series", "ID_Source", "Year", "Geo", "About", "UOM", 
                    "M1_Jan", "M2_Feb", "M3_Mar", "M4_Apr", "M5_May", "M6_Jun", 
                    "M7_Jul", "M8_Aug", "M9_Sep", "M10_Oct", "M11_Nov", "M12_Dec", "Caption"]

v1_captions_collection = pd.DataFrame(columns = new_columns_list)
IDs = np.unique(v0_captions_collection.index)

# Combining information about captions and about line charts in the same dataset.
for ID in IDs:
    temp_ID_captions_dataset = v0_captions_collection[v0_captions_collection.index == ID]
    temp_ID_captions_dataset = temp_ID_captions_dataset.reset_index()
    temp_ID_dataset = data_chart_dataset[data_chart_dataset["ID_Series"] == temp_ID_captions_dataset["id_caption"][0]]
    temp_ID_dataset = temp_ID_dataset.reset_index()
    
    # Schema and valus of an instance of the new dataset
    new_instance = [ID, temp_ID_captions_dataset["id_caption"][0], temp_ID_dataset["ID_Source"][0], temp_ID_dataset["Year"][0], temp_ID_dataset["Geo"][0],
                temp_ID_dataset["About"][0], temp_ID_dataset["UOM"][0],
                temp_ID_dataset["Value"][0], temp_ID_dataset["Value"][1], temp_ID_dataset["Value"][2],
                temp_ID_dataset["Value"][3], temp_ID_dataset["Value"][4], temp_ID_dataset["Value"][5],
                temp_ID_dataset["Value"][6], temp_ID_dataset["Value"][7], temp_ID_dataset["Value"][8],
                temp_ID_dataset["Value"][9], temp_ID_dataset["Value"][10], temp_ID_dataset["Value"][11], temp_ID_captions_dataset["caption_content"][0]]
    
    # Appending an instance to the new dataset
    v1_captions_collection.loc[ID] = new_instance
    
######################################
# Captions collection pre processing #
######################################

for idx, row in v1_captions_collection.iterrows():
    current_caption = row["Caption"]

    # Check if there are captions shorter than 15 chars or None
    if(len(current_caption) < 15 or current_caption is None):
        v1_captions_collection = v1_captions_collection.drop(idx)

    # Cast everything to string type
    new_caption = new_caption.apply(lambda x: str(x))

    # Lower case all the captions
    new_caption = new_caption.apply(lambda x: x.lower())

    # Check if there are sentences which contain special chars (e.g. \n)
    special_chars_list = ["\n", "\t", u'\u200b']
    for special_char in special_chars_list:
        if(special_char in new_caption):
            new_caption = new_caption.replace(special_char, " ")

    # Check if each sentence is ending with a period char.
    if(new_caption[-1] != "." ):
        new_caption = new_caption + "."       

    # Removing "," within digits e.g. 100,000 -> 1000000 
    splitted_caption = new_caption.split(" ")
    for word in splitted_caption:
        if(bool(re.search(r'\d', word))):
            if(word[-1] == "," or word[-1] == "."  or word[-1] == ":"  or word[-1] == ";"):
                word = word[:-1]
            word = word.replace('(', '')
            word = word.replace(')', '')
            if(',' in word):
                new_word = word.replace(',', '')
                new_caption = new_caption.replace(word, new_word)

    # Updating the current caption with the new caption values
    v1_captions_collection.loc[idx, "Caption"] = new_caption

# Saving the pre-processed original dataset
v1_captions_collection.set_index('ID_Caption')
v1_captions_collection.to_excel("Captions collection/v1_captions_collection.xlsx", index=False) 
