import pandas as pd
import numpy as np

def time_series_values_normalizer(input_series):
    min_input_series = np.min(input_series)
    max_input_series = np.max(input_series)
    normalized_input_series = []

    for num in input_series:
        new_num = (num - min_input_series)/(max_input_series - min_input_series)
        normalized_input_series.append(round(new_num, 2))

    return normalized_input_series, min_input_series, max_input_series

def normalizer(dataset):
    dataset["max_time_series"] = ""
    dataset["min_time_series"] = ""

    for row_index, row in dataset.iterrows():
        normalized_time_series, min_input_series, max_input_series = time_series_values_normalizer(row.iloc[8:20].values)
        for col_index, value in enumerate(normalized_time_series):
            dataset.iloc[row_index, 8 + col_index] = value
        dataset.iloc[row_index, dataset.columns.get_loc("min_time_series")] = min_input_series
        dataset.iloc[row_index, dataset.columns.get_loc("max_time_series")] = max_input_series
        
    return dataset

#########################
# Normalization process #
#########################

# Normalization of the time series within the train dataset
v4_train_captions_collection = pd.read_excel("Captions collection/v4_train_captions_collection.xlsx")
v5_train_captions_collection = normalizer(v4_train_captions_collection)

# Normalization of the time series within the test dataset
v4_test_captions_collection = pd.read_excel("Captions collection/v4_test_captions_collection.xlsx")
v5_test_captions_collection = normalizer(v4_test_captions_collection)


# Normalize the digits within the captions
def caption_digits_normalizer(dataset):
    for idx, row in dataset.iterrows():
        current_caption = row["tokenized_caption"]
        splitted_caption = current_caption.split(" ")
        new_caption = current_caption
        for word in splitted_caption:
            series_min = dataset.iloc[idx]["min_time_series"]
            series_max = dataset.iloc[idx]["max_time_series"]
            try:
                if(word[-1] == "."):
                    val = int(word[:-1])
                else:
                    # Check if the word is a float
                    val = int(word)
                # Normalize the value
                N = val - series_min
                D = series_max - series_min
                val_to_substitute = N/D
                # Substitute the normalized value with the original value in the tokenized caption
                new_caption = new_caption.replace(str(val), str(round(val_to_substitute, 2)))
            except:
                pass
        # Substitute the new tokenized caption in the dataset
        dataset.iloc[idx, dataset.columns.get_loc("tokenized_caption")] = new_caption
    return dataset

v5_train_captions_collection = caption_digits_normalizer(v5_train_captions_collection)
v5_test_captions_collection = caption_digits_normalizer(v5_test_captions_collection)

v5_train_captions_collection.to_excel("Captions collection/v5_train_captions_collection.xlsx") 
v5_test_captions_collection.to_excel("Captions collection/v5_test_captions_collection.xlsx") 