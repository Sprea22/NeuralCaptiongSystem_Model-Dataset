def caption_generator_BS(input_seq, mode, bs, bs_depth):

  # Beam Search parameter
  k = 4

  # Encoding the input sequence and saving the states value
  start_state = encoder_model.predict(input_seq)

  # Initializing the first input token
  start_token = 'CAP_START_'

  # Initializing bm dictionary and states vector
  bm = {}
  states_values = []
  for i in range(0, k):
      bm[i, start_token] = 1
      states_values.append(start_state)

  cont = 0 
  while(cont < 65):
      to_predict = []
      temp_search_values = []
      temp_search_probs = []
      temp_search_states = []

      for sent in bm.keys():
          to_predict.append(sent[1].split()[-1]) # or maybe [-2] ???

      for idx, word in enumerate(to_predict): 
          target_seq = np.zeros((1,1))
          target_seq[0, 0] = target_token_index[word]
          # Predicting the probability for the next output tokens
          output_tokens, h, c = decoder_model.predict([target_seq] + states_values[idx])
          
          # Creating a list of tokens ordered by their probability values.
          indexed = list(enumerate(output_tokens[0, -1, :]))
          ordered_predictions = list(reversed(sorted(indexed, key=operator.itemgetter(1))))[0:k]
          top_values = [reverse_target_char_index[i] for i, v in ordered_predictions]
          top_probs = [v for i, v in ordered_predictions]

          for append_idx in range(0, k):
              temp_search_values.append(top_values[append_idx])
              temp_search_probs.append(top_probs[append_idx])
              temp_search_states.append([h,c])
              
      curr_prob_idx = -1
      for bm_key_idx, bm_key_value in enumerate(bm.keys()):
        for prob_idx in range(0, k):
            curr_prob_idx = curr_prob_idx + 1
            temp_search_probs[curr_prob_idx] = bm[bm_key_value] * temp_search_probs[curr_prob_idx]
            temp_search_values[curr_prob_idx] = bm_key_value[1] + " " + temp_search_values[curr_prob_idx]

      temp_search_probs = np.array(temp_search_probs)
      top_indexes = temp_search_probs.argsort()[-3:][::-1]

      bm = {}
      states_values = []
      for subs_idx in range(0, k):
          bm[subs_idx, temp_search_values[subs_idx]] = temp_search_probs[subs_idx]
          states_values.append(temp_search_states[subs_idx])
      cont = cont + 1
     
  max_key = ""
  max_value = 0
  
  # Select the most probable sentence within the bm
  for key, value in bm.items():
    if(value > max_value):
      max_key = key[1]
      max_value = value
    
  decoded_caption = max_key
  
  # Post processing of the output caption
  decoded_caption = detokenization(decoded_caption)

  # Remove extra words after final "." char during sentence generation
  if(mode == "sentence"):
    decoded_caption = decoded_caption.split(".")[0] + "."

  return decoded_caption