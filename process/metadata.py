'''This file is used to generate the item_content.json file which contains the metadata of the items'''
import json
from tqdm import tqdm

# load the business_mapping.json file
business_mapping = {}
business_mapping_file_path = "./business_id_mapping.json"
with open(business_mapping_file_path, "r") as f:
    business_mapping = json.load(f)

selected_id = list(business_mapping.keys())
prompt = [None] * len(selected_id)

item_content_path = "./source/business.json"
# Read the metadata.json file line by line and extract the required information
with open(item_content_path, "r") as file:
    for line in tqdm(file, desc="Parsing data", unit=" lines"):
        # Parse each line as a JSON object
        entry = json.loads(line)
        business_id = entry.get("business_id", None)
        if business_id in selected_id:
            content = {
                "name": entry.get("name", ""),
                "city": entry.get("city", ""),
                "categories": entry.get("categories", []),
            }
            prompt[business_mapping[business_id]] = {
                "iid": business_mapping[business_id],
                "content": content,
            }

item_content_path = "./item_content.json"
# Write the list of JSON objects to a file
with open(item_content_path, "w") as file:
    for obj in prompt:
        file.write(json.dumps(obj) + "\n")
