def caption_generation_BFS(input_seq, mode, bs, bs_depth):
  
    # MAIN PARAMETERS OF BEAM SEARCH
    bs = bs
    bs_depth = bs_depth
    permutation_number = bs**bs_depth

    # Variables initialization
    decoded_caption = ''
    bs_dictionary = {}
    stop_condition = False
    
    # Encoding the input sequence and saving the states value
    states_value = encoder_model.predict(input_seq)
    new_states = []
    new_states.append(states_value)
    
    # Initializing the first input token
    target_sequence = ['CAP_START_']

    while not stop_condition:
      for idx_total_bs in range(1, bs_depth + 1):
          bs_dictionary[idx_total_bs] = {}
          states_values_list = []
          for idx_cur_target, current_target_seq in enumerate(target_sequence):
              current_target_seq = current_target_seq.replace("#", "")
              # Generate empty target sequence of linpth 1.
              target_seq = np.zeros((1,1))
              target_seq[0, 0] = target_token_index[current_target_seq]
              # Predicting the probability for the next output tokens
              output_tokens, h, c = decoder_model.predict([target_seq] + new_states[idx_cur_target])
              for i in range(0, bs):
                states_values_list.append([h,c])
              # Creating a list of tokens ordered by their probability values.
              indexed = list(enumerate(output_tokens[0, -1, :]))
              top_values = list(reversed(sorted(indexed, key=operator.itemgetter(1))))
              top_idxs = [i for i, v in top_values]
            
              sampled_token_index, char_token_index, probabilities_list = [], [], []
              bs_dict_record = {}
              
              # Save the top tokens, the associated characters and the probabilities.
              for i in range (0, bs):
                sampled_token_index.append(top_idxs[i])
                char_token_index.append(reverse_target_char_index[top_idxs[i]])
                probabilities_list.append(top_values[i])
                bs_dict_record[char_token_index[i]] = top_values[i][1]

              # Method to avoid rewriting same keys in the vocabulary
              try:
                while True:
                  if(bs_dictionary[idx_total_bs][current_target_seq] != None):
                    current_target_seq = current_target_seq + "#"
              except:
                pass
              
              # Adding new record to the BS Dictionary
              bs_dictionary[idx_total_bs][current_target_seq] = bs_dict_record

          # Set up the words to predict for next loop.
          target_sequence = []
          for word_idxs, word_values in bs_dictionary[idx_total_bs].items():
            to_extend = list(word_values.keys())
            target_sequence.extend(to_extend)
          # Update states
          new_states = states_values_list
          
      bs_divider = 1
      cont_word = 0

      # Initializing the dictionary which will contain all the alternative 
      # output sentences.
      values_bs_results = {}
      for idx_permutation in range(1, permutation_number+1):
          values_bs_results[idx_permutation] = ""
      
      # Initializing the dictionary which will contain all the probabilities
      # associated with the alternative output sentences.
      probs_bs_results = {}
      for idx_permutation in range(1, permutation_number+1):
          probs_bs_results[idx_permutation] = 1

      for cont in range(1, len(bs_dictionary)+1):
          idx_permutations = int(permutation_number/bs_divider)
          for idx_bs, value_bs in bs_dictionary[cont].items():
              cont_word = cont_word + 1
              for i in range(1, idx_permutations + 1):
                  # Saving the token values in the values_bs_results dict
                  sequence_idx = i + (cont_word - 1)*idx_permutations
                  if(cont == bs_depth):
                      idx_final_value = (i - 1) % bs
                      values_bs_results[sequence_idx] = values_bs_results[sequence_idx] + " " + idx_bs + " " + list(value_bs.keys())[idx_final_value]
                  else:
                      values_bs_results[sequence_idx] = values_bs_results[sequence_idx] + " " + idx_bs
                  # Saving the token values probabilities in the probs_bs_results dict
                  final_prob_thresh_value = round(permutation_number/(bs*bs_divider))
                  if(i <= final_prob_thresh_value):
                      final_prob_value = 0
                      probs_bs_results[sequence_idx] = probs_bs_results[sequence_idx] * value_bs[list(value_bs.keys())[final_prob_value]]
                  else:
                      final_prob_value = 1
                      probs_bs_results[sequence_idx] = probs_bs_results[sequence_idx] * value_bs[list(value_bs.keys())[final_prob_value]]
                      
          cont_word = 0
          bs_divider = bs_divider * bs
      
      
      # Detecting the most probable sequence of tokens.
      max_idx = -1
      max_value = 0
      for idx, value in probs_bs_results.items():
          if(value > max_value):
              max_idx = idx
              max_value = value

      # Appending the translated token to the output caption
      subseq_to_add = ""
      for word in values_bs_results[max_idx].split()[1:]:
        word = word.replace("#", "")
        subseq_to_add = subseq_to_add + " " + word
       
      # Add the select most probable sequence to the output caption
      decoded_caption += " " + subseq_to_add

      # Exit condition:
      if(mode == "caption"):
        if (decoded_caption.split()[-1] == '_CAP_END' or decoded_caption.count("_SEQ_END") == 4 or len(decoded_caption) > 1000):
          stop_condition = True
      elif(mode == "sentence"):
        if(decoded_caption.count("_SEQ_END") == 1):
          stop_condition = True
          decoded_caption = decoded_caption.split(".")[0]

      # Initializing the new initial target sequence for the next BS and the 
      # new associated states
      target_sequence = [decoded_caption.split()[-1]]
      new_states = [new_states[max_idx]]

    # Post processing of the output caption
    decoded_caption = detokenization(decoded_caption)
    
    # Remove extra words after final "." char during sentence generation
    if(mode == "sentence"):
      decoded_caption = decoded_caption.split(".")[0] + "."
      
    return decoded_caption