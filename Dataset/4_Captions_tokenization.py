import pandas as pd
import numpy as np

# Calculate the similarity between two strings using Jaccard Similarity function
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def find_most_likely_subsentence(about, sentence):
    words = sentence.split()
    len_about = len(about.split())
    sub_sentence = []
    # Create a subset of each sentence based on the about len.
    for n in [len_about-1, len_about, len_about+1, len_about+2, len_about+3]:
        for idx in range(0, len(words) - n):
            sub_sentence.append(words[idx : idx+n])

    # Calculate similarity between each subset of the caption and the about string
    # Print out the most similar sentences
    max_sentence = ""
    max_value = 0
    for sub_sen in sub_sentence:
        sub_sen_temp = ""
        for word in sub_sen:
            sub_sen_temp = sub_sen_temp + " " + word
        similarity_value = get_jaccard_sim(about, sub_sen_temp)
        if(similarity_value > max_value):
            max_sentence = sub_sen_temp
            max_value = similarity_value
    return max_sentence, max_value

def tokenizer (dataset):
    dataset["tokenized_caption"] = ""
    for idx, row in dataset.iterrows():
        temp_cap = row["caption"]
        for field in tkn_col_dict:
            temp_cap = temp_cap.replace(str(row[field]), tkn_col_dict[field])
        temp_about = row["About"]
        max_sentence, max_value = find_most_likely_subsentence(temp_about, temp_cap)
        if(max_sentence != ""):
            new_cap = temp_cap.replace(max_sentence, " TKN_About")
            dataset.iloc[idx, dataset.columns.get_loc("tokenized_caption")] = new_cap
        else:
            dataset.iloc[idx, dataset.columns.get_loc("tokenized_caption")] = temp_cap
    return dataset

########################
# Tokenization process #
########################

tkn_col_dict = {"Year" : "TKN_Year", "Geo" : "TKN_Geo", "UOM" : "TKN_UOM"}

# Tokenization of the train dataset
v3_train_captions_collection = pd.read_excel("Captions collection/v3_train_captions_collection.xlsx")
v4_train_captions_collection = tokenizer(v3_train_captions_collection)

# Tokenization of the validation dataset
v3_val_captions_collection = pd.read_excel("Captions collection/v3_val_captions_collection.xlsx")
v4_val_captions_collection = tokenizer(v3_val_captions_collection)

# Tokenization of the test dataset
v3_test_captions_collection = pd.read_excel("Captions collection/v3_test_captions_collection.xlsx")
v4_test_captions_collection = tokenizer(v3_test_captions_collection)

v4_train_captions_collection.to_excel("Captions collection/v4_train_captions_collection.xlsx") 
v4_val_captions_collection.to_excel("Captions collection/v4_val_captions_collection.xlsx") 
v4_test_captions_collection.to_excel("Captions collection/v4_test_captions_collection.xlsx") 