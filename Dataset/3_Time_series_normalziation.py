import pandas as pd
import numpy as np
import re

def time_series_values_normalizer(input_series):
    min_input_series = np.min(input_series)
    max_input_series = np.max(input_series)
    normalized_input_series = []

    for num in input_series:
        new_num = (num - min_input_series)/(max_input_series - min_input_series)
        normalized_input_series.append(round(new_num, 2)*100)

    return normalized_input_series, min_input_series, max_input_series

def normalizer(dataset):
    dataset["max_time_series"] = ""
    dataset["min_time_series"] = ""

    for row_index, row in dataset.iterrows():
        normalized_time_series, min_input_series, max_input_series = time_series_values_normalizer(row.iloc[7:19].values)
        for col_index, value in enumerate(normalized_time_series):
            dataset.iloc[row_index, 7 + col_index] = value
        dataset.iloc[row_index, dataset.columns.get_loc("min_time_series")] = min_input_series
        dataset.iloc[row_index, dataset.columns.get_loc("max_time_series")] = max_input_series
        
    return dataset

def caption_digits_normalizer(dataset):
    for idx, row in dataset.iterrows():
        current_caption = row["Tokenized_Caption"]
        splitted_caption = current_caption.split(" ")
        new_caption = current_caption
        for word in splitted_caption:
            series_min = dataset.iloc[idx]["min_time_series"]
            series_max = dataset.iloc[idx]["max_time_series"]
            if(bool(re.search(r'\d', word))):
                if(word[-1] == "," or word[-1] == "."  or word[-1] == ":"  or word[-1] == ";"):
                    word = word[:-1]
                word = word.replace('(', '')
                word = word.replace(')', '')
                try:
                    if('.' in word):
                        val = float(word)
                    else:
                        val = int(word)     
                    # Normalize the value
                    N = val - series_min
                    D = series_max - series_min
                    val_to_substitute = int(round(N/D*100))
                    if(val_to_substitute < 150 and val_to_substitute > -50):
                        # Substitute the normalized value with the original value in the tokenized caption
                        new_caption = new_caption.replace(str(val), str(round(val_to_substitute, 2)))
                    else:
                        print("OUT of the range value: ", val, " - compared with range(", series_min, ",", series_max, ")")
                except:
                    print("NOT Substituted: ", word)
                    pass

        # Substitute the new tokenized caption in the dataset
        dataset.iloc[idx, dataset.columns.get_loc("Tokenized_Caption")] = new_caption
    return dataset

#########################
# Normalization process #
#########################

# Normalization of the time series within the dataset
v2_captions_collection = pd.read_excel("Captions collection/v2_captions_collection.xlsx")

v3_captions_collection = normalizer(v2_captions_collection)
v3_captions_collection = caption_digits_normalizer(v3_captions_collection)

v3_captions_collection.to_excel("Captions collection/v3_captions_collection.xlsx", index=False) 