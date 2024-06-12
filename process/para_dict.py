'''This file is used to generate the para_dict.pickle file, which is further used for collaborative filtering.'''
import pandas as pd
import pickle

# Load the final mappings
df = pd.read_csv("./total.csv")

# Find the number of unique users and items
user_num = len(df["user"].unique())
item_num = len(df["item"].unique())

# Load the csv files
filename = "./total_trn.csv"
data = pd.read_csv(filename)
trn_user_nb = [[] for _ in range(user_num)]
trn_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    trn_user_nb[row["user"]].append(row["item"])
    trn_item_nb[row["item"]].append(row["user"])

filename = "./total_val.csv"
data = pd.read_csv(filename)
val_user_nb = [[] for _ in range(user_num)]
val_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    val_user_nb[row["user"]].append(row["item"])
    val_item_nb[row["item"]].append(row["user"])

filename = "./total_tst.csv"
data = pd.read_csv(filename)
tst_user_nb = [[] for _ in range(user_num)]
tst_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    tst_user_nb[row["user"]].append(row["item"])
    tst_item_nb[row["item"]].append(row["user"])

para_dict = {
    "trn_user_nb": trn_user_nb,
    "trn_item_nb": trn_item_nb,
    "val_user_nb": val_user_nb,
    "val_item_nb": val_item_nb,
    "tst_user_nb": tst_user_nb,
    "tst_item_nb": tst_item_nb,
    "user_num": user_num,
    "item_num": item_num,
}

with open("./para_dict.pickle", "wb") as handle:
    pickle.dump(para_dict, handle)
print("para_dict saved")
