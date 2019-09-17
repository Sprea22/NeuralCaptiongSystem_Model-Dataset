import pandas as pd
import numpy as np
import random

# Splitting the captions collection into test, validation and train
print("Generating the final dataset with sub captions.. \n")
v1_captions_collection = pd.read_excel("Captions collection/v1_captions_collection.xlsx")
train_idx_list = list(range(1,100))

# Selecting a list of 10 caption idxs for the test split
test_idxs_list = []
for n in range(10):
    r_num = random.choice(train_idx_list)
    test_idxs_list.append(r_num)
    train_idx_list.remove(r_num) 

# Selecting a list of 20 caption idxs for the validation split
val_idxs_list = []
for j in range(20):
    r_num = random.choice(train_idx_list)
    val_idxs_list.append(r_num)
    train_idx_list.remove(r_num) 

# Selecting the test, val and train section from the captions collection 
v3_test_captions_collection = v1_captions_collection[v1_captions_collection["ID_Series"].astype(int).isin(test_idxs_list)]
v3_val_captions_collection = v1_captions_collection[v1_captions_collection["ID_Series"].astype(int).isin(val_idxs_list)]
v3_train_captions_collection = v1_captions_collection[v1_captions_collection["ID_Series"].astype(int).isin(train_idx_list)]

# Resetting the dataframe indexes
v3_test_captions_collection = v3_test_captions_collection.reset_index()
v3_val_captions_collection = v3_val_captions_collection.reset_index()
v3_train_captions_collection = v3_train_captions_collection.reset_index()

# Dropping the duplicated index column
del v3_test_captions_collection['index']
del v3_val_captions_collection['index']
del v3_train_captions_collection['index']

# Displaying the number of captions for each split
print("Test captions collection number: ", len(v3_test_captions_collection["ID_Series"]))
print("Validation captions collection number: ",len(v3_val_captions_collection["ID_Series"]))
print("Train captions collection number: ",len(v3_train_captions_collection["ID_Series"]), "\n")

# Saving the test, val and train captions collections
v3_test_captions_collection.to_excel("Captions collection/v3_test_captions_collection.xlsx") 
v3_val_captions_collection.to_excel("Captions collection/v3_val_captions_collection.xlsx") 
v3_train_captions_collection.to_excel("Captions collection/v3_train_captions_collection.xlsx") 
