import json
from openai import OpenAI
import numpy as np

"""This file is used to generate the item profile, which contains the summary of the items"""

client = OpenAI(api_key="")  # YOUR OPENAI API_KEY

system_prompt = ""
with open("generation/item_profile/item_system_prompt.txt", "r") as f:
    system_prompt = f.read()

item_prompts = []
with open("generation/item_profile/item_prompts.json", "r") as f:
    for line in f.readlines():
        item_prompts.append(json.loads(line))

def get_gpt_response(input):
    iid = input["iid"]
    prompt = input["prompt"]
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
    )
    response = completion.choices[0].message.content

    result = {"iid": iid, "business summary": response}
    return result

indexs = len(item_prompts)
picked_id = np.random.choice(indexs, size=1)[0]


class Colors:
    GREEN = "\033[92m"
    END = "\033[0m"


print(Colors.GREEN + "Generating Profile for Item" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
print(item_prompts[picked_id])
print("---------------------------------------------------\n")
response = get_gpt_response(item_prompts[picked_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)
