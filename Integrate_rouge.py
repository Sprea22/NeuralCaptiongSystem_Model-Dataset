import pandas as pd
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


### ### ### ### ####
# MODEL INFERENCES #
### ### ### ### ####

dataset = pd.read_excel("/content/drive/My Drive/Current Works/UBC Research Period/Training Folder/v5_train_captions_collection.xlsx")


orig_captions = []
orig_tknzed_captions = []

output_captions = []
output_tknzed_captions = []

seq_examples = [1, 100, 200, 300, 400]

for idx, seq_index in enumerate(seq_examples):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    current_id = dataset.iloc[seq_index]["ID_Series"]
    current_df = dataset[dataset["ID_Series"] == current_id]

    # Detokenization process
    dtkn_vocabulary = {}
    dtkn_vocabulary["tkn_year"] = current_df["Year"][0] 
    dtkn_vocabulary["tkn_geo"] = current_df["Geo"][0] 
    dtkn_vocabulary["tkn_about"] = current_df["About"][0] 
    dtkn_vocabulary["tkn_uom"] = current_df["UOM"][0] 

    decoded_dtknzd_sentence = decoded_sentence
    for tkn in dtkn_vocabulary:
         decoded_dtknzd_sentence = decoded_dtknzd_sentence.replace(tkn, dtkn_vocabulary[tkn])

    print("Input values sequence: ", input_seq)
    print("Output tokenized sequence: ", decoded_sentence)
    print("Output detokenized sequence: ", decoded_dtknzd_sentence)
    
    output_captions.append(decoded_dtknzd_sentence)
    output_tknzed_captions.append(decoded_sentence)
    
    related_caption = current_df["caption"]#check out the values types.. it has to be a list of strings
    orig_tknzed_captions = current_df["tokenized_caption"] #check out the values types.. it has to be a list of strings

### ### ### ### ####
# MODEL EVALUATION #
### ### ### ### ####

# Rouge metric between list of decoded sentences and orig_tknzed_captions
rouge_evaluation(output_tknzed_captions, orig_tknzed_captions)

# Rouge metric between list of decoded detokenized sentences and orig_captions
rouge_evaluation(output_captions, orig_captions)


          