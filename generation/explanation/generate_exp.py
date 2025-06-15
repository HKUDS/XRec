import json
import numpy as np
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ollama_client import get_ollama_response

"""This file is used to generate the explanation of user and item interactions"""

# Configuration for Ollama model
MODEL_NAME = "llama3.1:8b"  # You can change this to other models like "mistral:7b", "codellama:7b", etc.

system_prompt = ""
with open("generation/explanation/exp_system_prompt.txt", "r") as f:
    system_prompt = f.read()

explanation_prompts = []
with open("generation/explanation/exp_prompts.json", "r") as f:
    for line in f.readlines():
        explanation_prompts.append(json.loads(line))


def get_ollama_response_for_explanation(input):
    uid = input["uid"]
    iid = input["iid"]
    prompt = json.dumps(input["feedback"], indent=4)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = get_ollama_response(messages, model_name=MODEL_NAME, temperature=0.7)

    result = {"uid": uid, "iid": iid, "explanation": response}
    return result

indexs = len(explanation_prompts)
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
print(explanation_prompts[picked_id])
print("---------------------------------------------------\n")
response = get_ollama_response_for_explanation(explanation_prompts[picked_id])
print(Colors.GREEN + "Generated Results:\n" + Colors.END)
print(response)
