'''This file is used to generate data.json file, which will be used in data_handler in explainer.'''
import json

business_title = {}
with open("item_content.json", "r") as file:
    for line in file:
        process = json.loads(line)
        iid = process["iid"]
        title = process["content"]["name"]
        business_title[iid] = title

item_summary = {}
with open("item_profile.json", "r") as file:
    for line in file:
        process = json.loads(line)
        iid = process["iid"]
        summary = json.loads(process["business summary"], strict=False)["summarization"]
        item_summary[iid] = summary

user_summary = {}
with open("user_profile.json", "r") as file:
    for line in file:
        process = json.loads(line)
        uid = process["uid"]
        summary = json.loads(process["user summary"], strict=False)["summarization"]
        user_summary[uid] = summary

data = []
with open("./explanation.json", "r") as file:
    for line in file:
        process = json.loads(line)
        uid = process["uid"]
        iid = process["iid"]
        title = business_title[iid]
        user_sum = user_summary[uid]
        item_sum = item_summary[iid]
        explanation = process["explanation"]
        data.append(
            {
                "uid": uid,
                "iid": iid,
                "title": title,
                "user_summary": user_sum,
                "item_summary": item_sum,
                "explanation": explanation,
            }
        )

# save the data
with open("data.json", "w") as file:
    for d in data:
        file.write(json.dumps(d) + "\n")
