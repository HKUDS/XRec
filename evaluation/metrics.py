import numpy as np
from bert_score import BERTScorer
import argparse
import pickle
import json
import sys
import os
import concurrent.futures

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generation.utils.ollama_client import get_ollama_response

# Configuration for Ollama model
MODEL_NAME = "llama3.1:8b"  # You can change this to other models like "mistral:7b", "codellama:7b", etc.

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="amazon", help="amazon, yelp or google")
args = parser.parse_args()

with open("evaluation/system_prompt.txt", "r") as f:
    system_prompt = f.read()


class MetricScore:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        # Try the new file names first, then fall back to old names
        pred_path_new = f"data/{args.dataset}/tst_predictions.pkl"
        ref_path_new = f"data/{args.dataset}/tst_references.pkl"
        pred_path_old = f"data/{args.dataset}/tst_pred.pkl"
        ref_path_old = f"data/{args.dataset}/tst_ref.pkl"

        import os
        if os.path.exists(pred_path_new):
            self.pred_input_path = pred_path_new
        else:
            self.pred_input_path = pred_path_old

        if os.path.exists(ref_path_new):
            self.ref_input_path = ref_path_new
        else:
            self.ref_input_path = ref_path_old

        with open(self.pred_input_path, "rb") as f:
            self.data = pickle.load(f)
        with open(self.ref_input_path, "rb") as f:
            self.ref_data = pickle.load(f)

    def get_score(self):
        scores = {}
        (
            bert_precison,
            bert_recall,
            bert_f1,
            bert_precison_std,
            bert_recall_std,
            bert_f1_std,
        ) = BERT_score(self.data, self.ref_data)
        gpt_score, gpt_std = get_gpt_score(self.data, self.ref_data)
        tokens_predict = [s.split() for s in self.data]
        usr, _ = unique_sentence_percent(tokens_predict)

        scores["gpt_score"] = gpt_score
        scores["bert_precision"] = bert_precison
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1
        scores["usr"] = usr

        scores["gpt_std"] = gpt_std
        scores["bert_precision_std"] = bert_precison_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std
        return scores

    def print_score(self):
        scores = self.get_score()
        print(f"dataset: {args.dataset}")
        print("Explanability Evaluation Metrics:")
        print(f"gpt_score: {scores['gpt_score']:.4f}")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"usr: {scores['usr']:.4f}")
        print("-"*30)
        print("Standard Deviation:")
        print(f"gpt_std: {scores['gpt_std']:.4f}")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")
        

def get_ollama_response_for_evaluation(prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = get_ollama_response(messages, model_name=MODEL_NAME, temperature=0.1)

    # Try to extract a float from the response
    try:
        # Look for a number in the response
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            return float(numbers[0])
        else:
            # If no number found, return a default score
            return 3.0
    except:
        return 3.0


def get_gpt_score(predictions, references):
    prompts = []
    for i in range(len(predictions)):
        prompt = {
            "prediction": predictions[i],
            "reference": references[i],
        }
        prompts.append(json.dumps(prompt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Reduced workers for local model
        results = list(executor.map(get_ollama_response_for_evaluation, prompts))

    return np.mean(results), np.std(results)

def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True

def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        # seq is a list of words
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)

def BERT_score(predictions, references):
    # Use BERTScorer with the default roberta-large model
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    precision, recall, f1 = scorer.score(predictions, references)

    # Convert tensors to numpy arrays
    precision = precision.numpy()
    recall = recall.numpy()
    f1 = f1.numpy()

    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1),
        np.std(precision),
        np.std(recall),
        np.std(f1),
    )
