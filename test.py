
bs_dictionary = {}

for i in range(1,3):
    print(i)
    
bs_dictionary[1] = {"START" : {"ciao": 1, "hola" : 0.5}}
bs_dictionary[2] = {"ciao" : {"come": 1, "stai" : 0.2}, "hola" : {"eres": 0.7, "chico" : 0.2}}
bs_dictionary[3] = {"come" : {"A": 1, "B" : 1.0}, "stai" : {"C": 1, "D" : 0.2}, "eres" : {"D": 0.7, "E" : 0.2}, "chico" : {"H": 0.7, "F" : 0.2}}

bs = 3
bs_depth = 3
permutation_number = (bs_depth+1) *bs
bs_divider = 1
cont_word = 0

probs_bs_results = {}
values_bs_results = {}

for idx_permutation in range(1, permutation_number+1):
    values_bs_results[idx_permutation] = ""

for idx_permutation in range(1, permutation_number+1):
    probs_bs_results[idx_permutation] = 1

for cont in range(1, len(bs_dictionary)+1):
    idx_permutations = int(permutation_number/bs_divider)
    for idx_bs, value_bs in bs_dictionary[cont].items():
        cont_word = cont_word + 1
        for i in range(1, idx_permutations + 1):
            sequence_idx = i + (cont_word - 1)*idx_permutations

            if(cont == bs_depth):
                idx_final_value = (i - 1) % 2
                values_bs_results[sequence_idx] = values_bs_results[sequence_idx] + " " + idx_bs + " " + list(value_bs.keys())[idx_final_value%2]
            else:
                values_bs_results[sequence_idx] = values_bs_results[sequence_idx] + " " + idx_bs
            
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

print("Most probable sequence of tokens: ", values_bs_results[max_idx])