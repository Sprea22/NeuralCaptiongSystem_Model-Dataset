import rouge

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
    return str(round(100.0 * p, 2)) + "\t" + str(round(100.0 * r, 2)) + "\t" + str(round(100.0 * f, 2))
    #return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def rouge_evaluation(all_hypothesis, all_references):
    # it's possible to add also 'Individual' to check the evaluation between
    # each single hypothesis and each single reference.
    for aggregator in ['Avg', 'Best']:
        print()
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
