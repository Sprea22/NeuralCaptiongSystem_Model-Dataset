import pandas as pd
import numpy as np

def format_structure_train(dataset):
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
            output_seq =  str(temp_df["Tokenized_Caption"][index])
            time_series_values = row.iloc[7:19].values
            for value in time_series_values:
                input_seq = input_seq + str(round(value)) + " "
            if(output_seq != "nan"):
                temp_train_seq = input_seq + "___" + output_seq 
                captions_list.append(temp_train_seq)
    return captions_list


def format_structure_test(dataset):
    captions_list = []
    unique_IDs = dataset["ID_Series"].unique()
    unique_IDs = unique_IDs[~np.isnan(unique_IDs)]
    for idx in unique_IDs:
        temp_df = dataset[dataset["ID_Series"] == idx]
        for index, row in temp_df.iterrows():
            input_seq = ""
            input_seq = input_seq + str(idx) + "___"
            input_seq = input_seq + str(temp_df["Geo"][index]) + "___"
            input_seq = input_seq + str(temp_df["Year"][index]) + "___"
            input_seq = input_seq + str(temp_df["About"][index]) + "___"
            input_seq = input_seq + str(temp_df["UOM"][index]) + "___"
            input_seq = input_seq + str(temp_df["min_time_series"][index]) + "___"
            input_seq = input_seq + str(temp_df["max_time_series"][index]) + "___"
            output_seq =  str(temp_df["Tokenized_Caption"][index])
            time_series_values = row.iloc[7:19].values
            for value in time_series_values:
                input_seq = input_seq + str(round(value)) + " "
            if(output_seq != "nan"):
                temp_train_seq = input_seq + "___" + output_seq 
                captions_list.append(temp_train_seq)
    return captions_list

def write_text_file(filename, captions_list):
    file = open("Captions collection/" + filename, "w", encoding='utf-8') 
    for seq in captions_list:
        try:
            file.write(seq + "\n")
        except:
            print("An exception occurred on the following sentence: ")
            print(seq)
    file.close() 

# Generating the training text file for the subcaption dataset
print("\nGenerating the final tokenized and normalized dataset, in a format adapt to the neural model to train.. \n")

# Normalization of the time series within the train dataset
v3_train_captions_collection = pd.read_excel("Captions collection/v3_train_captions_collection.xlsx")
captions_list = format_structure_train(v3_train_captions_collection)
write_text_file("final_train_captions_collection.txt", captions_list)

# Normalization of the time series within the test dataset
v3_test_captions_collection = pd.read_excel("Captions collection/v3_test_captions_collection.xlsx")
captions_list = format_structure_test(v3_test_captions_collection)
write_text_file("final_test_captions_collection.txt", captions_list)

# Normalization of the time series within the full dataset
v3_captions_collection = pd.read_excel("Captions collection/v3_captions_collection.xlsx")
captions_list = format_structure_test(v3_captions_collection)
write_text_file("final_captions_collection.txt", captions_list)

print("Files have been correctly generated! \n")

