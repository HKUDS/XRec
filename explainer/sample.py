import json, pickle
from utils.parse import args

pred_path = f"data/{args.dataset}/tst_pred.pkl"
ref_path = f"data/{args.dataset}/tst_ref.pkl"

with open(pred_path, "rb") as f:
    predictions = pickle.load(f)
with open(ref_path, "rb") as f:
    references = pickle.load(f)
    
for i in range(len(predictions)):
    print(f"Prediction: {predictions[i]}")
    print(f"Reference: {references[i]}")
    print("-" * 50)
    if i == 5:
        break
