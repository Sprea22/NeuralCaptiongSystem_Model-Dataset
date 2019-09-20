import numpy as np
import pandas as pd
import random
import rouge
  
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

def similarity_model(input_sequence, dataset):
    max_correlated_sentence_value = -2
    max_correlated_sentence_id = -2
    
    list_of_series = dataset.ID_Series.unique()

    for ID_series in list_of_series:
        temp_dataset = dataset[dataset["ID_Series"] == ID_series]
        temp_dataset = temp_dataset.reset_index(drop=True)
        temp_series = temp_dataset.iloc[0, 8:20].values.tolist()
        temp_corr = np.corrcoef(input_sequence, temp_series)[0][1]
        if(temp_corr > max_correlated_sentence_value):
            max_correlated_sentence_id = ID_series
            max_correlated_sentence_value = temp_corr

    numb_values = len(dataset[dataset["ID_Series"] == max_correlated_sentence_id]["tokenized_caption"].values)
    r_cap_idx = random.choice(range(0,numb_values))

    return dataset[dataset["ID_Series"] == max_correlated_sentence_id]["tokenized_caption"].values[r_cap_idx], max_correlated_sentence_id, max_correlated_sentence_value
# Choose one of the caption from the dataset and give it as output

dataset = pd.read_excel("Dataset/Captions collection/v5_test_captions_collection.xlsx")
dataset_train = pd.read_excel("Dataset/Captions collection/v5_train_captions_collection.xlsx")

orig_captions = []
orig_tknzed_captions = []

output_captions = []
output_tknzed_captions = []

IDs = dataset["ID_Series"].values

for idx, seq_index in enumerate(IDs):
    current_id = dataset.iloc[seq_index]["ID_Series"]
    current_df = dataset[dataset["ID_Series"] == current_id]

    # Detokenization process
    dtkn_vocabulary = {}
    dtkn_vocabulary["TKN_Year"] = str(current_df["Year"].values[0])
    dtkn_vocabulary["TKN_Geo"] = current_df["Geo"].values[0] 
    dtkn_vocabulary["TKN_About"] = current_df["About"].values[0] 
    dtkn_vocabulary["TKN_UOM"] = current_df["UOM"].values[0] 
  
    temp_dataset = dataset[dataset["ID_Series"] == seq_index]
    temp_dataset = temp_dataset.reset_index(drop=True)
    temp_series = temp_dataset.iloc[0, 8:20].values.tolist()
    input_sequence = temp_series

    decoded_sentence, max_corr_id, max_corr_val = similarity_model(input_sequence, dataset_train)

    decoded_dtknzd_sentence = decoded_sentence
    for tkn in dtkn_vocabulary:
         decoded_dtknzd_sentence = decoded_dtknzd_sentence.replace(tkn, dtkn_vocabulary[tkn])
         
    '''
    print('-- #', idx, '------------------------------------------------------------------------------------------------')
    print("Correlation idx: ", max_corr_id, " with: ", max_corr_val)
    print("Time series data: ", dtkn_vocabulary)
    print("Input values sequence: ", input_sequence)
    print("Output tokenized sequence: ", decoded_sentence)
    print("Output detokenized sequence: ", decoded_dtknzd_sentence)
    print('\n')
    '''
    output_captions.append(decoded_dtknzd_sentence)
    output_tknzed_captions.append(decoded_sentence)
    
    orig_captions.append(current_df["caption"].values[0])
    orig_tknzed_captions.append(list(current_df["tokenized_caption"].values))
    
### ### ### ### ####
# MODEL EVALUATION #
### ### ### ### ####

print("\n############################")
print("##### MODEL EVALUATION #####")
print("############################\n")

# Rouge metric between list of decoded sentences and orig_tknzed_captions
rouge_evaluation(output_tknzed_captions, orig_tknzed_captions)

# Rouge metric between list of decoded detokenized sentences and orig_captions
rouge_evaluation(output_captions, orig_captions)