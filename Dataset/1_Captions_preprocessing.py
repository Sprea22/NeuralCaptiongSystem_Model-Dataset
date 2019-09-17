import pandas as pd 
import numpy as np

v0_captions_collection = pd.read_excel("Captions collection/v0_captions_collection.xlsx")
data_chart_dataset = pd.read_excel("Data chart images/data_charts_dataset.xlsx")

new_columns_list = ["ID", "ID_Series", "ID_Source", "Year", "Geo", "About", "UOM", "Scalar_Factor", 
                    "M1_Jan", "M2_Feb", "M3_Mar", "M4_Apr", "M5_May", "M6_Jun", 
                    "M7_Jul", "M8_Aug", "M9_Sep", "M10_Oct", "M11_Nov", "M12_Dec", "caption"]

###############################################
# Captions collection split into subsentences #
###############################################

print("Splitting each caption in subsentences based on the periods..")
v1_captions_collection = pd.DataFrame(columns = new_columns_list)
IDs = np.unique(data_chart_dataset["ID_Series"])
ID_cont = 0

for ID in IDs:
    temp_ID_captions_dataset = v0_captions_collection[v0_captions_collection["id_caption"] == ID]
    temp_ID_captions_dataset = temp_ID_captions_dataset.reset_index()
    temp_ID_dataset = data_chart_dataset[data_chart_dataset["ID_Series"] == ID]
    temp_ID_dataset = temp_ID_dataset.reset_index()
    for cap in temp_ID_captions_dataset["caption_content"]:
        caps_list = cap.split(".")[:-1]
        for idx_s, sentence in enumerate(caps_list):
            # Do not split sentences over periods that have been used for decimal numbers.
            try:
                if(sentence[-1].isdigit() and caps_list[idx_s + 1][0].isdigit()):
                    sentence = sentence + caps_list[idx_s + 1]
                    del caps_list[idx_s + 1]
            except:
                pass

            sentence = sentence + "."
            new_instance = [ID_cont, ID, temp_ID_dataset["ID_Source"][0], temp_ID_dataset["Year"][0], temp_ID_dataset["Geo"][0],
                temp_ID_dataset["About"][0], temp_ID_dataset["UOM"][0], temp_ID_dataset["Scalar_Factor"][0],
                temp_ID_dataset["Value"][0], temp_ID_dataset["Value"][1], temp_ID_dataset["Value"][2],
                temp_ID_dataset["Value"][3], temp_ID_dataset["Value"][4], temp_ID_dataset["Value"][5],
                temp_ID_dataset["Value"][6], temp_ID_dataset["Value"][7], temp_ID_dataset["Value"][8],
                temp_ID_dataset["Value"][9], temp_ID_dataset["Value"][10], temp_ID_dataset["Value"][11]]
            new_instance.append(sentence)
            v1_captions_collection.loc[ID_cont] = new_instance
            ID_cont = ID_cont + 1

print(" \n The captions have been correctly splitted into subsentences!")

######################################
# Captions collection pre processing #
######################################

for idx, row in v1_captions_collection.iterrows():
    # Check if there are sentences which contain special chars (e.g. \n)
    special_chars_list = ["\n, \t"]
    for special_char in special_chars_list:
        if(special_char in row["caption"]):

    # Check if each sentence is ending with a period char.
    if(row["caption"][-1] != "." ):
        row["caption"] = row["caption"] + "."       

    # Check if there are sentences shorter than 15 chars or none value
    if(len(row["caption"]) < 15 or row["caption"] is None):
        v1_captions_collection = v1_captions_collection.drop(idx)

v1_captions_collection.set_index('ID', inplace=True)
v1_captions_collection.to_excel("Captions collection/v1_captions_collection.xlsx") 
