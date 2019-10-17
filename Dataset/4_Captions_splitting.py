import pandas as pd
import numpy as np
import random

# Splitting the captions collection into test and train
print("Generating the final dataset with sub captions.. \n")
v3_captions_collection = pd.read_excel("Captions collection/v3_captions_collection.xlsx")
train_idx_list = list(v3_captions_collection["ID_Caption"].values)

# The test set contains the 10% of the total captions
num_instances_test = round(len(train_idx_list)/100*10)
test_idxs_list = []
for n in range(num_instances_test):
    r_num = random.choice(train_idx_list)
    test_idxs_list.append(r_num)
    train_idx_list.remove(r_num) 

# Selecting the test and train section from the captions collection 
v3_test_captions_collection = v3_captions_collection[v3_captions_collection["ID_Series"].astype(int).isin(test_idxs_list)]
v3_train_captions_collection = v3_captions_collection[v3_captions_collection["ID_Series"].astype(int).isin(train_idx_list)]

# Resetting the dataframe indexes
v3_test_captions_collection = v3_test_captions_collection.reset_index()
v3_train_captions_collection = v3_train_captions_collection.reset_index()

# Dropping the duplicated index column
del v3_test_captions_collection['index']
del v3_train_captions_collection['index']

# Displaying the number of captions for each split
print("Test captions collection number: ", len(v3_test_captions_collection["ID_Series"]))
print("Train captions collection number: ",len(v3_train_captions_collection["ID_Series"]), "\n")

# Saving the test and train captions collections
v3_test_captions_collection.to_excel("Captions collection/v3_test_captions_collection.xlsx") 
v3_train_captions_collection.to_excel("Captions collection/v3_train_captions_collection.xlsx") 


# Saving the full captions collections in a separate file
new_final = True
if(True):
    v3_captions_collection.to_excel("Captions collection/final_captions_collection.xlsx")
    v3_test_captions_collection.to_excel("Captions collection/final_test_captions_collection.xlsx") 
    v3_train_captions_collection.to_excel("Captions collection/final_train_captions_collection.xlsx") 
