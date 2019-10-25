###################################
# CAPTION GENERATION EVALUATION  #
###################################

def caption_generator(input_seq):
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
        # Predicting the probability for the next output tokens
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        
        # Creating a list of tokens ordered by their probability values.
        indexed = list(enumerate(output_tokens[0, -1, :]))
        top_values = sorted(indexed, key=operator.itemgetter(1))
        top_idxs = list(reversed([i for i, v in top_values]))
          
        # Randomness factor force the system to random pick the first word
        if(cont == 0):
          rand_value_range = range(0, 3)
          rand_idx = random.choice(rand_value_range)
          sampled_token_index = top_idxs[rand_idx]
        else:
          sampled_token_index = top_idxs[0]         
        
        # Translating the choosen token
        sampled_char = reverse_target_char_index[sampled_token_index]
        # Appending the translated token to the output caption
        decoded_caption += " " + sampled_char
        cont = cont + 1

        if(sampled_char == "_SEQ_END"):
          cont = 0
        # Exit condition: either hit max linpth
        # or find stop character.
        #if (sampled_char == '_CAP_END' or len(decoded_caption) > 700 or decoded_caption.count("_SEQ_END") == 5):
        if (len(decoded_caption) > 700 or decoded_caption.count("_SEQ_END") == 5):
          stop_condition = True
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    # Post processing of the output caption
    decoded_caption = detokenization(decoded_caption)

    return decoded_caption