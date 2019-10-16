import json
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
from collections import Counter

def captions_retrieval():
    # Retrieving the captions from the captions collector web application
    client = MongoClient("mongodb+srv://Admin:ciaociao@captions-collector-cluster-aognl.mongodb.net/test?retryWrites=true&w=majority")
    db = client.get_database('test')
    records = db.captions
    records_list = list(records.find())

    # Collecting the captions
    captions_collection = {}
    for idx in range(0, len(records_list)):
        temp_json_obj = records_list[idx]
        del temp_json_obj["_id"]
        temp_json_obj["date"] = temp_json_obj["date"].strftime("%Y-%m-%d %H:%M:%S")
        captions_collection[idx] = temp_json_obj

    print("Captions correctly collected! \n")

    # Saving the results JSON file
    with open("Captions collection/v0_captions_collection.json", "w") as write_file:
        json.dump(captions_collection, write_file)

    # Converting the dataframe into a JSON file
    captions_df = pd.read_json("Captions collection/ROW_v0_captions_collection.json").T
    captions_df.to_excel("Captions collection/ROW_v0_captions_collection.xlsx")


captions_retrieval()
