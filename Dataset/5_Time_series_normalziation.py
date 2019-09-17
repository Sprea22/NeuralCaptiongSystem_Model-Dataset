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

# Normalization of the time series within the validation dataset
v4_val_captions_collection = pd.read_excel("Captions collection/v4_val_captions_collection.xlsx")
v5_val_captions_collection = normalizer(v4_val_captions_collection)

# Normalization of the time series within the test dataset
v4_test_captions_collection = pd.read_excel("Captions collection/v4_test_captions_collection.xlsx")
v5_test_captions_collection = normalizer(v4_test_captions_collection)

v5_train_captions_collection.to_excel("Captions collection/v5_train_captions_collection.xlsx") 
v5_val_captions_collection.to_excel("Captions collection/v5_val_captions_collection.xlsx") 
v5_test_captions_collection.to_excel("Captions collection/v5_test_captions_collection.xlsx") 