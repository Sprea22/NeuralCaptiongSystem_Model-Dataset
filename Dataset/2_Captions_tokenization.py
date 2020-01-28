import pandas as pd
import numpy as np

# Calculate the similarity between two strings using Jaccard Similarity function
def get_jaccard_sim(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def find_most_likely_subsentence(about, sentence, mode):
    words = sentence.split()
    len_about = len(about.split())
    sub_sentence = []

    if(mode == "UOM"):  
        # Create a subset of each sentence based on the about len.
        for n in [len_about, len_about+1, len_about+2, len_about+3, len_about+4]:
            for idx in range(0, len(words) - n):
                n_gram = [x.replace(".", "") for x in words[idx : idx+n]]
                n_gram = [x.replace(":", "") for x in n_gram]
                sub_sentence.append(n_gram)
    else:
        # Create a subset of each sentence based on the about len.
        for n in [len_about-1, len_about, len_about+1, len_about+2, len_about+3]:
            for idx in range(0, len(words) - n):
                sub_sentence.append(words[idx : idx+n])

    # Calculate similarity between each subset of the caption and the about string
    # Print out the most similar sentences
    max_sentence = ""
    max_value = 0

    about = about.lower()
    for sub_sen in sub_sentence:
        sub_sen_temp = ""
        for word in sub_sen:
            sub_sen_temp = sub_sen_temp + " " + word
        similarity_value = get_jaccard_sim(about, sub_sen_temp)
        if(similarity_value > max_value):
            max_sentence = sub_sen_temp
            max_value = similarity_value
    if(mode == "UOM" and max_value < 0.150):
        max_sentence = ""
        max_value = 0
    return max_sentence, max_value

def tokenizer (dataset):
    dataset["Tokenized_Caption"] = ""
    for idx, row in dataset.iterrows():
        max_sentence = ""
        temp_cap = row["Caption"]
        for field in tkn_col_dict:
            temp_cap = temp_cap.replace(str(row[field]).lower(), tkn_col_dict[field])
        
        # About token
        temp_about = row["About"]
        max_sentence, max_value = find_most_likely_subsentence(temp_about, temp_cap, "About")
        if(max_sentence != ""):
            new_cap = temp_cap.replace(max_sentence, " TKN_About ")
            dataset.iloc[idx, dataset.columns.get_loc("Tokenized_Caption")] = new_cap
        else:
            dataset.iloc[idx, dataset.columns.get_loc("Tokenized_Caption")] = temp_cap

        # UOM token
        temp_UOM = row["UOM"]
        max_sentence_UOM, max_value_UOM = find_most_likely_subsentence(temp_UOM, temp_cap, "UOM")
        if(max_sentence_UOM != ""):
            temp_cap = temp_cap.replace(max_sentence_UOM, " TKN_UOM ")

    return dataset

########################
# Tokenization process #
########################

tkn_col_dict = {"Year" : " TKN_Year ", "Geo" : " TKN_Geo ", "UOM" : " TKN_UOM "}

# Tokenization of the train dataset
v1_captions_collection = pd.read_excel("Captions collection/v1_captions_collection.xlsx")
v2_captions_collection = tokenizer(v1_captions_collection)
v2_captions_collection.to_excel("Captions collection/v2_captions_collection.xlsx", index=False) 