import numpy as np
import pandas as pd
import random
import rouge
import string
import re
import nltk

from Utility_functions import print_results, validation_set, set_vocabulary
from Utility_functions import denormalization, prepare_results, rouge_evaluation 

def ngram_model(n_grams, article_text, output_sentence_len):
    ngrams = {}
    words = n_grams

    words_tokens = nltk.word_tokenize(article_text)
    for i in range(len(words_tokens)-words):
        seq = ' '.join(words_tokens[i:i+words])
        if  seq not in ngrams.keys():
            ngrams[seq] = []
        ngrams[seq].append(words_tokens[i+words])

    curr_sequence = ' '.join(words_tokens[0:words])
    output = curr_sequence

    for i in range(output_sentence_len):
        if curr_sequence not in ngrams.keys():
            break
        possible_words = ngrams[curr_sequence]
        next_word = possible_words[random.randrange(len(possible_words))]
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words)-words:len(seq_words)])

    sent_output = output.split(".")[:-1]
    final_output = ''
    for sent in sent_output:
        final_output = final_output + sent + " . "

    return final_output

##################
# Initialization # 
##################

k_fold_cross_validatin_iter = "4"

# CHOICE CAN BE BETWEEN "time_series" OR "captions"
choice = "time_series"

train_dataset = pd.read_excel("Dataset/Captions collection/5_Fold_Cross_Validation_" + choice + "/" + k_fold_cross_validatin_iter + "_train_" + choice + ".xlsx")
test_dataset = pd.read_excel("Dataset/Captions collection/5_Fold_Cross_Validation_" + choice + "/"  + k_fold_cross_validatin_iter + "_test_" + choice + ".xlsx")
full_dataset =  pd.read_excel("Dataset/Captions collection/final_captions_collection.xlsx")

if(choice == "time_series"):
    enumerated_idxs = list(np.unique(np.array(test_dataset["ID_Series"].values)))
elif(choice == "captions"):
    enumerated_idxs = list(test_dataset["ID_Series"].values)

article_text = ''
for caption in train_dataset["Tokenized_Caption"]:
    article_text += caption

orig_sentences, orig_captions = [], []
output_sentences, output_captions = [], []

###################################
# Generating the output sentences #
###################################
for idx, seq_index in enumerate(enumerated_idxs):
    current_df = full_dataset[full_dataset["ID_Series"] == seq_index]
    orig_sent_to_add = []

    # Add the original sentences to the hypothesis
    for sentence in list(current_df["Caption"].values):
      orig_sent_to_add.extend(sentence.split(".")[:-1])
    orig_sentences.append(orig_sent_to_add)
    
    # Setting the vocabulary for the current time series
    dtkn_vocabulary = set_vocabulary(current_df)
    # Generating the output sentence
    output_sentence = ngram_model(2, article_text, 25)
    # Denormalize the output sentence
    output_dtknzd_sentence = denormalization(output_sentence, current_df["min_time_series"].values[0] , current_df["max_time_series"].values[0] )
    for tkn in dtkn_vocabulary:
         output_dtknzd_sentence = output_dtknzd_sentence.replace(tkn, dtkn_vocabulary[tkn])
    # Append the orig and the output sentences, ready for the evaluation     
    output_sentences.append(output_dtknzd_sentence)

    print_results(seq_index, orig_sent_to_add, output_dtknzd_sentence)


##################################
# Generating the output captions #
##################################
for idx, seq_index in enumerate(enumerated_idxs):
    current_df = full_dataset[full_dataset["ID_Series"] == seq_index]

    # Add the original captions to the hypothesis
    orig_capt_to_add = []
    for capt in list(current_df["Caption"].values):
        orig_capt_to_add.append(capt)
    orig_captions.append(orig_capt_to_add)

    # Setting the vocabulary for the current time series
    dtkn_vocabulary = set_vocabulary(current_df)
    # Generating the output sentence
    output_caption = ngram_model(2, article_text, 53)
    # Denormalize the output sentence
    output_dtknzd_caption = denormalization(output_caption, current_df["min_time_series"].values[0] , current_df["max_time_series"].values[0] )
    for tkn in dtkn_vocabulary:
         output_dtknzd_caption = output_dtknzd_caption.replace(tkn, dtkn_vocabulary[tkn])
    # Append the orig and the output sentences, ready for the evaluation     
    output_captions.append(output_dtknzd_caption)
    print_results(seq_index, orig_capt_to_add, output_dtknzd_caption)
    
### ### ### ### ####
# MODEL EVALUATION #
### ### ### ### ####

print("\n############################")
print("### SENTENCE EVALUATION ####")
print("############################\n")
# Rouge metric between list of output detokenized sentences and original sentences
rouge_evaluation(output_sentences, orig_sentences)

print("\n############################")
print("#### CAPTION EVALUATION ####")
print("############################\n")
# Rouge metric between list of output detokenized captions and original captions
rouge_evaluation(output_captions, orig_captions)