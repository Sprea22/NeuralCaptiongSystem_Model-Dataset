import numpy as np
import pandas as pd
import random
import rouge
from Utility_functions import print_results, validation_set, set_vocabulary
from Utility_functions import denormalization, prepare_results, rouge_evaluation 

def similarity_model(input_sequence, train_dataset, mode):
    max_correlated_sentence_value = -2
    max_correlated_sentence_id = -2
    
    list_of_series = train_dataset.ID_Series.unique()

    for ID_series in list_of_series:
        temp_dataset = train_dataset[train_dataset["ID_Series"] == ID_series]
        temp_dataset = temp_dataset.reset_index(drop=True)
        temp_series = temp_dataset.iloc[0, 7:19].values.tolist()
        temp_corr = np.corrcoef(input_sequence, temp_series)[0][1]
        if(temp_corr > max_correlated_sentence_value):
            max_correlated_sentence_id = ID_series
            max_correlated_sentence_value = temp_corr

    values = list(train_dataset[train_dataset["ID_Series"] == max_correlated_sentence_id]["Tokenized_Caption"].values)
    if(mode == "caption"):
        r_cap_idx = random.choice(range(0, len(values)))
        output_sentence = train_dataset[train_dataset["ID_Series"] == max_correlated_sentence_id]["Tokenized_Caption"].values[r_cap_idx]
    elif(mode == "sentence"):
        r_cap_idx = random.choice(range(0, len(values)))
        output_sentence = train_dataset[train_dataset["ID_Series"] == max_correlated_sentence_id]["Tokenized_Caption"].values[r_cap_idx]
        output_sentence = output_sentence.split(".")[0]
    return output_sentence, max_correlated_sentence_id, max_correlated_sentence_value

##################
# Initialization # 
##################

k_fold_cross_validatin_iter = "5"

# CHOICE CAN BE BETWEEN "time_series" OR "captions"
choice = "time_series"

train_dataset = pd.read_excel("Dataset/Captions collection/5_Fold_Cross_Validation_" + choice + "/" + k_fold_cross_validatin_iter + "_train_" + choice + ".xlsx")
test_dataset = pd.read_excel("Dataset/Captions collection/5_Fold_Cross_Validation_" + choice + "/"  + k_fold_cross_validatin_iter + "_test_" + choice + ".xlsx")
full_dataset =  pd.read_excel("Dataset/Captions collection/final_captions_collection.xlsx")

if(choice == "time_series"):
    enumerated_idxs = list(np.unique(np.array(test_dataset["ID_Series"].values)))
elif(choice == "captions"):
    enumerated_idxs = list(test_dataset["ID_Series"].values)

orig_sentences, orig_captions = [], []
output_sentences, output_captions = [], []

###################################
# Generating the output sentences #
###################################
for idx, seq_index in enumerate(enumerated_idxs):
    current_df = full_dataset[full_dataset["ID_Series"] == seq_index]
    # Setting the vocabulary for the current time series
    dtkn_vocabulary = set_vocabulary(current_df)
    # Selecting the time series values
    temp_series = test_dataset.iloc[0, 7:19].values.tolist()
    input_sequence = temp_series
    # Generating the output sentence
    output_sentence, max_corr_id, max_corr_val = similarity_model(input_sequence, train_dataset, "sentence")
    # Denormalize the output sentence
    output_dtknzd_sentence = denormalization(output_sentence, current_df["min_time_series"].values[0] , current_df["max_time_series"].values[0] )
    for tkn in dtkn_vocabulary:
         output_dtknzd_sentence = output_dtknzd_sentence.replace(tkn, dtkn_vocabulary[tkn])
    output_dtknzd_sentence = output_dtknzd_sentence.replace("  ", " ") + "."

    # Append the orig and the output sentences, ready for the evaluation     
    output_sentences.append(output_dtknzd_sentence)

    # Retrieving the original sentences associated with ID_Seris == idx
    orig_sent_to_add = []
    for sentence in list(current_df["Caption"].values):
      orig_sent_to_add.extend(sentence.split(".")[:-1])
    orig_sentences.append(orig_sent_to_add)

    #print_results(idx, orig_sent_to_add, output_dtknzd_sentence)

##################################
# Generating the output captions #
##################################
for idx, seq_index in enumerate(enumerated_idxs):
    current_df = full_dataset[full_dataset["ID_Series"] == seq_index]
    current_captions_idxs = np.unique(current_df["ID_Caption"])

    # Retrieving the original captions associated with ID_Seris == idx
    orig_captions.append(list(current_df["Caption"].values))
    
    # Setting the vocabulary for the current time series
    dtkn_vocabulary = set_vocabulary(current_df)

    # Selecting the time series values
    temp_series = test_dataset.iloc[0, 7:19].values.tolist()
    input_sequence = temp_series

    # Generating the output sentence
    output_caption, max_corr_id, max_corr_val = similarity_model(input_sequence, train_dataset, "caption")

    # Denormalize the output sentence
    output_dtknzd_caption = denormalization(output_caption, current_df["min_time_series"].values[0] , current_df["max_time_series"].values[0] )
    for tkn in dtkn_vocabulary:
         output_dtknzd_caption = output_dtknzd_caption.replace(tkn, dtkn_vocabulary[tkn])
    output_dtknzd_sentence = output_dtknzd_sentence.replace("  ", " ") + "."

    # Append the orig and the output sentences, ready for the evaluation     
    output_captions.append(output_dtknzd_caption)
    
    #print_results(idx, list(current_df["Caption"].values), output_dtknzd_caption)

### ### ### ### ####
# MODEL EVALUATION #
### ### ### ### ####

print("\n############################")
print("##### SENTENCE EVALUATION #####")
print("############################\n")
# Rouge metric between list of output detokenized sentences and original sentences
rouge_evaluation(output_sentences, orig_sentences)

print("\n############################")
print("#### CAPTION EVALUATION ####")
print("############################\n")

# Rouge metric between list of output detokenized captions and original captions
rouge_evaluation(output_captions, orig_captions)