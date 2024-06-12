'''This file is used to separate the total data into training, validation, and test sets'''
import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
import csv
import json
import pickle

# create separate csv files for data_handler in encoder
filename = "total.csv"
data = pd.read_csv(filename)
total_data = []
for index, row in data.iterrows():
    total_data.append({"user": row["user"], "item": row["item"]})

# Separate the data into training and test sets with a ratio of 4:1
total_data = pd.DataFrame(total_data)
trn_data, val_tst_data = train_test_split(total_data, test_size=0.2, random_state=42)
val_data, tst_data = train_test_split(val_tst_data, test_size=0.5, random_state=42)
trn_data.to_csv("total_trn.csv", index=False)
val_data.to_csv("total_val.csv", index=False)
tst_data.to_csv("total_tst.csv", index=False)


# create separate pickle files for data_handler in explainer
data = []
with open("data.json", "rb") as file:
    for line in file:
        data.append(json.loads(line))
trn_data, val_tst_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, tst_data = train_test_split(val_tst_data, test_size=0.5, random_state=42)

with open("trn.pkl", "wb") as file:
    pickle.dump(trn_data, file)
with open("val.pkl", "wb") as file:
    pickle.dump(val_data, file)
with open("tst.pkl", "wb") as file:
    pickle.dump(tst_data, file)