import pandas as pd
import numpy as np

def single_caption_format(data):
    v7_new_dataset = data
    v7_new_dataset = v7_new_dataset.iloc[0:0]

    IDs_captions = np.unique(data["ID_Caption"].values)

    for id_caption in IDs_captions:
        temp_first_row = data[data["ID_Caption"] == id_caption].iloc[0]

        lst = data[data["ID_Caption"] == id_caption]["caption"].values
        orig_caption = ''.join(str(elem + " ") for elem in lst) 
        lst = data[data["ID_Caption"] == id_caption]["tokenized_caption"].values
        tknzed_caption = ''.join(str(elem + " ") for elem in lst) 

        temp_first_row["caption"] = orig_caption
        temp_first_row["tokenized_caption"] = tknzed_caption
        v7_new_dataset = v7_new_dataset.append(temp_first_row, ignore_index=True)
        
        v7_new_dataset.loc[id_caption, "caption"] = orig_caption
        v7_new_dataset.loc[id_caption, "tokenized_caption"] = tknzed_caption

    v7_new_dataset = v7_new_dataset[v7_new_dataset.ID_Caption.notnull()]
    v7_new_dataset = v7_new_dataset.reset_index()
    del v7_new_dataset["index"]
    return v7_new_dataset

def format_structure(dataset):
    try:
        del v7_new_dataset["index"]
    except:
        pass
    captions_list = []
    unique_IDs = dataset["ID_Series"].unique()
    unique_IDs = unique_IDs[~np.isnan(unique_IDs)]
    for idx in unique_IDs:
        temp_df = dataset[dataset["ID_Series"] == idx]
        for index, row in temp_df.iterrows():
            input_seq = ""
            input_seq = input_seq + str(temp_df["Geo"][index]) + "___"
            input_seq = input_seq + str(temp_df["Year"][index]) + "___"
            input_seq = input_seq + str(temp_df["About"][index]) + "___"
            input_seq = input_seq + str(temp_df["UOM"][index]) + "___"
            input_seq = input_seq + str(temp_df["min_time_series"][index]) + "___"
            input_seq = input_seq + str(temp_df["max_time_series"][index]) + "___"
            output_seq =  str(temp_df["tokenized_caption"][index])
            time_series_values = row.iloc[8:20].values
            for value in time_series_values:
                input_seq = input_seq + str(round(value)) + " "
            if(output_seq != "nan"):
                temp_train_seq = input_seq + "___" + output_seq 
                captions_list.append(temp_train_seq)
    return captions_list

def write_text_file(filename, captions_list):
    file = open("0_Final Dataset - Neural Model Input/" + filename, "w", encoding='utf-8') 
    for seq in captions_list:
        try:
            file.write(seq + "\n")
        except:
            print("An exception occurred on the following sentence: ")
            print(seq)
    file.close() 
    
# v7_train_captions_collection = v7_train_captions_collection.reindex(np.random.permutation(v7_train_captions_collection.index))
 
# New format - Full caption test
v5_train_captions_collection = pd.read_excel("captions collection/v5_train_captions_collection.xlsx")
v7_train_captions_collection = single_caption_format(v5_train_captions_collection)
v7_train_captions_collection.to_excel("captions collection/v7_train_captions_collection.xlsx")

# New format - Full caption test
v5_test_captions_collection = pd.read_excel("captions collection/v5_test_captions_collection.xlsx")
v7_test_captions_collection = single_caption_format(v5_test_captions_collection)
v7_test_captions_collection = v7_test_captions_collection.reindex(np.random.permutation(v7_test_captions_collection.index))
v7_test_captions_collection.to_excel("captions collection/v7_test_captions_collection.xlsx")

# Normalization of the time series within the train dataset
v7_train_captions_collection = pd.read_excel("captions collection/v7_train_captions_collection.xlsx")
captions_list = format_structure(v7_train_captions_collection)
write_text_file("v7_final_train_captions_collection.txt", captions_list)

# Normalization of the time series within the test dataset
v7_test_captions_collection = pd.read_excel("captions collection/v7_test_captions_collection.xlsx")
captions_list = format_structure(v7_test_captions_collection)
write_text_file("v7_final_test_captions_collection.txt", captions_list)