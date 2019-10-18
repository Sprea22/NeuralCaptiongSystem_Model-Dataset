import numpy as np
import pandas as pd
import random
import rouge
import nltk
import string
import bs4 as bs
import urllib.request
import re

def print_results(idx, input_sequence, output_sentence):
    print('-- #', idx, '------------------------------------------------------------------------------------------------')
    print("Input values sequence: ", input_sequence)
    print("Output tokenized sequence: ", output_sentence)
    print('\n')

def validation_set(dataset, numb):
    numb_list = list(dataset["ID_Series"].values)
    choosen_list = []

    for n in range(numb):
        r_num = random.choice(numb_list)
        choosen_list.append(r_num)
        numb_list.remove(r_num) 
    return choosen_list

def set_vocabulary(current_df):
    dtkn_vocabulary = {}
    dtkn_vocabulary["TKN_Year"] = str(current_df["Year"].values[0])
    dtkn_vocabulary["TKN_Geo"] = current_df["Geo"].values[0] 
    dtkn_vocabulary["TKN_About"] = current_df["About"].values[0] 
    dtkn_vocabulary["TKN_UOM"] = current_df["UOM"].values[0] 
    return dtkn_vocabulary

def denormalization(input_sequence, input_min, input_max):
    input_sequence = input_sequence[:-1]
    new_caption = input_sequence
    for word in input_sequence.split(" "):
            try:
                if(word[-1] == "."):
                    val = int(word[:-1])
                else:
                    # Check if the word is a float
                    val = int(word)
                # Normalize the value
                #N = val - input_min
                #D = input_max - input_min
                val_to_substitute = (val/100 * (input_max - input_min)) + input_min
                # Substitute the normalized value with the original value in the tokenized caption
                new_caption = new_caption.replace(str(val), str(round(val_to_substitute, 2)))
            except:
                pass
    return new_caption
                  
def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def rouge_evaluation(all_hypothesis, all_references):

    # it's possible to add also 'Individual' to check the evaluation between
    # each single hypothesis and each single reference.
    for aggregator in ['Avg', 'Best']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=apply_avg,
                            apply_best=apply_best,
                            alpha=0.5, # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

        # Evaluating the input hypothesis and references
        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(metric, results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))
        print()

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

# Selecting a list of 10 random time series  from the train set to evaluate through rouge 
dataset = pd.read_excel("Dataset/Captions collection/final_train_captions_collection.xlsx")
#choosen_list = validation_set(dataset, 10)

article_text = ''
for caption in dataset["Tokenized_Caption"]:
    article_text += caption

# Decomment the following rows to execute the evaluation on the test set
dataset = pd.read_excel("Dataset/Captions collection/final_test_captions_collection.xlsx")
choosen_list = list(dataset["ID_Series"].values)

orig_sentences, orig_captions = [], []
output_sentences, output_captions = [], []

###################################
# Generating the output sentences #
###################################
for idx, seq_index in enumerate(choosen_list):
    current_id = dataset.iloc[idx]["ID_Series"]
    current_df = dataset[dataset["ID_Series"] == current_id]
    # Setting the vocabulary for the current time series
    dtkn_vocabulary = set_vocabulary(current_df)
    # Generating the output sentence
    output_sentence = ngram_model(2, article_text, 15)
    # Denormalize the output sentence
    output_dtknzd_sentence = denormalization(output_sentence, current_df["min_time_series"].values[0] , current_df["max_time_series"].values[0] )
    for tkn in dtkn_vocabulary:
         output_dtknzd_sentence = output_dtknzd_sentence.replace(tkn, dtkn_vocabulary[tkn])
    # Append the orig and the output sentences, ready for the evaluation     
    output_sentences.append(output_dtknzd_sentence)
    orig_sentences.append(list(current_df["Caption"].values))
    #print_results(idx, input_sequence, output_sentence)

##################################
# Generating the output captions #
##################################
for idx, seq_index in enumerate(choosen_list):
    current_df = dataset[dataset["ID_Series"] == seq_index]
    current_captions_idxs = np.unique(current_df["ID_Caption"])
    current_captions = []
    # Merging sentences back to the original caption
    for temp_id_caption in current_captions_idxs:
        t_caption = ""
        for sentence in current_df[current_df["ID_Caption"] == temp_id_caption]["Caption"].values:
            t_caption = t_caption + " " + sentence
        current_captions.append(t_caption)
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
    orig_captions.append(current_captions)
    #print_results(idx, input_sequence, output_sentence)
    
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