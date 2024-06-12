'''This file is used to generate total.csv and interaction.json files, which contain the interactions between users and items.'''
import json
import csv
import random
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import pandas as pd

def k_core(data, k):
    G = nx.Graph()
    user_id_mapping = defaultdict(lambda: len(user_id_mapping))
    business_id_mapping = defaultdict(lambda: len(business_id_mapping))

    # Add edges to the graph from interactions
    for temp in data:
        user = temp["user_id"]
        item = temp["business_id"] + "_business"
        G.add_edge(user, item)

    # Find k-core of the graph
    k_core_graph = nx.k_core(G, k)

    # Extract interactions that are in k-core
    k_core_interactions = []
    for temp in data:
        user = temp["user_id"]
        item = temp["business_id"] + "_business"

        if (user, item) in k_core_graph.edges():
            user = user_id_mapping[user]
            item = business_id_mapping[item]
            k_core_interactions.append(
                {
                    "user": user,
                    "item": item,
                    "rating": temp["rating"],
                    "time": temp["time"],
                    "review": temp["review"],
                }
            )

    # remove the "_business" suffix
    business_id_mapping = {k[:-9]: v for k, v in business_id_mapping.items()}
    business_id_mapping_file_path = "./business_id_mapping.json"
    with open(business_id_mapping_file_path, "w") as f:
        json.dump(dict(business_id_mapping), f)

    user_id_mapping_file_path = "./user_id_mapping.json"
    with open(user_id_mapping_file_path, "w") as f:
        json.dump(dict(user_id_mapping), f)

    return k_core_interactions


interaction_path = "./source/review.json"
item_content_path = "./source/business.json"

useless_item = set()
# Read the business.json file line by line and extract the required information
num_lines = sum(1 for line in open(item_content_path, "r"))
with open(item_content_path, "r") as file:
    for line in tqdm(file, total=num_lines, desc="Parsing data", unit=" lines"):
        # Parse each line as a JSON object and append to the data list
        process = json.loads(line)
        if (
            (process["name"] == "")
            or (process["city"] == "")
            or (process["categories"] == [])
        ):
            useless_item.add(process["business_id"])

print("useless item number: ", len(useless_item))
# Read the review.json file line by line and extract the required information
num_lines = sum(1 for line in open(interaction_path, "r"))
user_set = set()
interaction = []
with open(interaction_path, "r") as file:
    for line in tqdm(file, total=num_lines, desc="Parsing data", unit=" lines"):
        # Parse each line as a JSON object and append to the data list
        process = json.loads(line)
        if (
            process["stars"] > 3.0
            and process["business_id"] not in useless_item
            and process["text"] != ""
        ):
            user_set.add(str(process["user_id"]))
            temp = {}
            temp["user_id"] = str(process["user_id"])
            temp["business_id"] = str(process["business_id"])
            temp["rating"] = process["stars"]
            temp["time"] = process["date"]
            temp["review"] = process["text"]
            interaction.append(temp)

unique_user = list(user_set)
unique_user.sort()
# randomly select 40% of the users
sample_size = int(len(unique_user) * 0.4)
random.seed(42)
selected_user = set(random.sample(unique_user, sample_size))
print("selected user: ", len(selected_user))
new_interaction = [temp for temp in interaction if temp["user_id"] in selected_user]

print("Before k-core: ", len(new_interaction))
# Apply k-core to the data, store the mappings
k_core_interactions = k_core(new_interaction, k=8)
print("After k-core: ", len(k_core_interactions))

# convert k-core interactions to a pandas dataframe, and remove duplicate interactions
data = pd.DataFrame(k_core_interactions)
data_unique = data.drop_duplicates(subset=[data.columns[0], data.columns[1]])
# see the number of users, items and interactions
print("Number of users: ", len(data_unique[data.columns[0]].unique()))
print("Number of items: ", len(data_unique[data.columns[1]].unique()))
print("Number of interactions: ", len(data_unique))
data_unique[["user", "item"]].to_csv("total.csv", index=False)
# write data_unique to interaction.json
data_unique.to_json("interaction.json", orient="records", lines=True)