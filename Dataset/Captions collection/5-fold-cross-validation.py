import pandas as pd 
import numpy as np
import random

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

def write_text_file(filename, captions_list):
    file = open(filename, "w", encoding='utf-8') 
    for seq in captions_list:
        try:
            file.write(seq + "\n")
        except:
            print("An exception occurred on the following sentence: ")
            print(seq)
    file.close() 

# Importing the captions dataset
captions_dataset = pd.read_excel("final_captions_collection.xlsx")

# K-fold cross validation iteration
k_fold_cross_iteration = 5

# SPLIT ON "caption" OTHERWISE ON "time_series"
choice = "time_series"

for i in range(1, k_fold_cross_iteration+1):
    # Obtain the indexes list
    if(choice == "caption"):
        param_choice = "ID_Caption"
        train_idx_list = list(captions_dataset[param_choice].values)

    elif(choice == "time_series"):
        param_choice = "ID_Series"
        train_idx_list = list(np.unique(np.array(captions_dataset[param_choice].values)))

    # The test set contains the 20% of the total captions
    num_instances_test = round(len(train_idx_list)/k_fold_cross_iteration)
    test_idxs_list = []
    for n in range(num_instances_test):
        r_num = random.choice(train_idx_list)
        test_idxs_list.append(r_num)
        train_idx_list.remove(r_num) 

    # Selecting the test and train section from the captions collection 
    test_capptions_collection = captions_dataset[captions_dataset[param_choice].astype(int).isin(test_idxs_list)]
    train_captions_collection = captions_dataset[captions_dataset[param_choice].astype(int).isin(train_idx_list)]

    # Resetting the dataframe indexes
    test_capptions_collection = test_capptions_collection.reset_index()
    train_captions_collection = train_captions_collection.reset_index()

    # Dropping the duplicated index column
    del test_capptions_collection['index']
    del train_captions_collection['index']

    # Set the train and test datasets names
    test_name = str(i) + "_test_" + choice + ".xlsx"
    train_name = str(i) + "_train_" + choice + ".xlsx"
    train_txt_name = str(i) + "_train_" + choice + ".txt"

    train_txt_captions = format_structure_train(train_captions_collection)
    write_text_file(train_txt_name, train_txt_captions)

    # Saving the test and train captions collections
    test_capptions_collection.to_excel(test_name) 
    train_captions_collection.to_excel(train_name) 
