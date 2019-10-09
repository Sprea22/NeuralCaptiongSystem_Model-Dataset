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
    
    while not stop_condition:
        
        # List of target_seq to predict
        
            # predict a target
            # collect 3 best idx
            # collect 3 best prob


        # Predicting the probability for the next output tokens
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Creating a list of tokens ordered by their probability values.
        indexed = list(enumerate(output_tokens[0, -1, :]))
        top_values = list(reversed(sorted(indexed, key=operator.itemgetter(1))))
        top_idxs = [i for i, v in top_values]
          
        # Randomness factor force the system to random pick the first word of a caption
        # within the "randomness factor" most probable words of the list.
        
        # Save the token
        sampled_token_index = [top_idxs[0], top_idxs[1], top_idxs[2], top_idxs[3], top_idxs[4]]
        # Save the probability
        probabilities_list = [top_values[0][1], top_values[1][1], top_values[2][1], top_values[3][1], top_values[4][1]]
        
        max_prob = 0

        sampled_token_index2 = []
        probabilities_list2 = []

        sampled_token_index3 = []
        probabilities_list3 = []

        for stok_idx, stok_value in enumerate(sampled_token_index):

            target_seq = np.zeros((1,1))
            target_seq[0, 0] = stok_value
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            indexed = list(enumerate(output_tokens[0, -1, :]))
            top_values = list(reversed(sorted(indexed, key=operator.itemgetter(1))))
            top_idxs = [i for i, v in top_values]
            
            sampled_token_index2.append(top_idxs[0], top_idxs[1], top_idxs[2], top_idxs[3], top_idxs[4])
            probabilities_list2.append(top_values[0][1], top_values[1][1], top_values[2][1], top_values[3][1], top_values[4][1])
        
        for stok_idx, stok_value in enumerate(sampled_token_index2):

            target_seq = np.zeros((1,1))
            target_seq[0, 0] = stok_value
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            indexed = list(enumerate(output_tokens[0, -1, :]))
            top_values = list(reversed(sorted(indexed, key=operator.itemgetter(1))))
            top_idxs = [i for i, v in top_values]
            
            sampled_token_index3.append(top_idxs[0], top_idxs[1], top_idxs[2], top_idxs[3], top_idxs[4])
            probabilities_list3.append(top_values[0][1], top_values[1][1], top_values[2][1], top_values[3][1], top_values[4][1])

            #print("----------------")
            for idx, value in enumerate(probabilities_list3):
                current_prob = value * probabilities_list2[np.around(idx/5)] * probabilities_list[np.around(idx/5/5)]
                #print(reverse_target_char_index[stok_value], reverse_target_char_index[sampled_token_index2[idx]], current_prob)
                if(current_prob > max_prob):
                    max_tokens = [stok_value, sampled_token_index2[np.around(idx/5)],  sampled_token_index[np.around(idx/5/5)]]
                    max_prob = current_prob
        
        #print("#########################################")
        #print(reverse_target_char_index[max_tokens[0]], reverse_target_char_index[max_tokens[1]])
        #print("#########################################")
        sampled_char = []
        for tok in max_tokens:
            decoded_caption += " " + reverse_target_char_index[tok]

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