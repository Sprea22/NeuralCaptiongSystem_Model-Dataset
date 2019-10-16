def caption_generator(input_seq, mode):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of linpth 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['CAP_START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_caption = ''
    captions_len = 0
    cont = 0
    
    bs_num = 3

    while not stop_condition:
        # Predicting the probability for the next output tokens
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Creating a list of tokens ordered by their probability values.
        indexed = list(enumerate(output_tokens[0, -1, :]))
        top_values = list(reversed(sorted(indexed, key=operator.itemgetter(1))))
        top_idxs = [i for i, v in top_values]
          
        # Randomness factor force the system to random pick the first word of a caption
        # within the "randomness factor" most probable words of the list.
        
        # Save the token
        sampled_token_index = [top_idxs[0], top_idxs[1], top_idxs[2]]
        # Save the probability
        probabilities_list = [top_values[0], top_values[1], top_values[2]]
        
        bs_dictionary[cont] = {"sampled_words" : sampled_token_index, "probabilities" : probabilities_list}
        
        max_prob = 0
        max_tokens = []
        for stok_idx, stok_value in enumerate(sampled_token_index):
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = stok_value
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            indexed = list(enumerate(output_tokens[0, -1, :]))
            top_values = list(reversed(sorted(indexed, key=operator.itemgetter(1))))
            top_idxs = [i for i, v in top_values]
            
            sampled_token_index2.append([top_idxs[0], top_idxs[1], top_idxs[2]]) 
            probabilities_list2.append([top_values[0], top_values[1], top_values[2]])
            for idx, value in enumerate(probabilities_list2):
                if(value * probabilities_list[stok_idx] > max_prob):
                    max_tokens = [stok_value, sampled_token_index2[idx]]
        
        sampled_char = []
        for tok in max_tokens:
            decoded_caption += " " + reverse_target_char_index[0]

        # Exit condition: either hit max linpth
        # or find stop character.
        if (sampled_char == '_END_CAP' or len(decoded_caption) > 700):
          captions_len = captions_len + 1
          stop_condition = True
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = max_tokens[-1]

        # Update states
        states_value = [h, c]
        cont = cont + 1
    
    # Post processing of the output caption
    
    decoded_caption = decoded_caption.replace(" SEQ_START_" , "")
    decoded_caption = decoded_caption.replace(" _SEQ_END" , ". ")
    decoded_caption = decoded_caption.replace(" _CAP_END" , "")
    decoded_caption = decoded_caption.replace(" COMMA" , ",")
    

    return decoded_caption
  

###################################
# Generating the output captions #
###################################
for idx, seq_index in enumerate(choosen_list):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    current_id = dataset.iloc[seq_index]["ID_Series"]
    current_df = dataset[dataset["ID_Series"] == current_id]

    # Generating the output caption
    decoded_caption = caption_generator(input_seq, "caption")

    # Setting the vocabulary for the current time series
    dtkn_vocabulary = set_vocabulary(current_df)
  
    # Denormalize the output caption
    decoded_dtknzd_caption = decoded_caption
    decoded_dtknzd_caption = denormalization(decoded_caption, current_df["min_time_series"].values[0] , current_df["max_time_series"].values[0] )
    for tkn in dtkn_vocabulary:
         decoded_dtknzd_caption = decoded_dtknzd_caption.replace(tkn, dtkn_vocabulary[tkn])

    # Append the orig and the output captions, ready for the evaluation     
    output_captions.append(decoded_dtknzd_caption)
    orig_captions.append(list(current_df["caption"].values))
    
    # Print out the results
    decoded_dtknzd_caption = decoded_dtknzd_caption.capitalize() 
    print_results(idx, decoded_dtknzd_caption, list(current_df["caption"].values))
    