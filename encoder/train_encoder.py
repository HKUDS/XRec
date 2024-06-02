import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pickle
from utils.parse import args
from utils.metrics import Metric
from models.lightgcn import LightGCN
from utils.data_handler import DataHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

class TrainGNN:
    def __init__(self):
        print(f"GNN model: {args.model}")
        print(f"Dataset: {args.dataset}")
        self.trn_loader, self.val_loader, self.tst_loader = DataHandler().load_data()
        self.trn_mat, self.val_mat, self.tst_mat = DataHandler().load_mat()
        self.trn_adj = DataHandler().create_adjacency_matrix(
            f"./data/{args.dataset}/total_trn.csv"
        )
        with open(f"./data/{args.dataset}/para_dict.pickle", "rb") as file:
            self.para_dict = pickle.load(file)
        self.user_num = self.para_dict["user_num"]
        self.item_num = self.para_dict["item_num"]
        self.metric = Metric()
        self.user_embeds_path = f"./data/{args.dataset}/user_emb.pkl"
        self.item_embeds_path = f"./data/{args.dataset}/item_emb.pkl"

    def train(self):
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Initialize model
        model = LightGCN(self.user_num, self.item_num, self.trn_mat)
        model = model.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_value = 0.0
        # Train model
        for epoch in range(args.n_epochs):
            total_loss = 0
            model.train()
            for batch in self.trn_loader:
                for i in batch:
                    i = i.to(device)
                optimizer.zero_grad()
                loss = model.cal_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluation, apply early stop
            model.eval()
            result = self.metric.eval(
                model, self.val_loader, self.para_dict["val_user_nb"]
            )

            val_value = result["recall"].item()
            if val_value > best_val_value:
                patience = 0
                best_val_value = val_value
                recall = result["recall"].item()
                ndcg = result["ndcg"].item()
                precision = result["precision"].item()
                mrr = result["mrr"].item()
                # save the user and item embeddings
                user_embeds, item_embeds = model.forward(self.trn_mat)
                with open(self.user_embeds_path, "wb") as file:
                    pickle.dump(user_embeds, file)
                with open(self.item_embeds_path, "wb") as file:
                    pickle.dump(item_embeds, file)

            print(
                f"Epoch {epoch}, Loss: {total_loss:.4f}, Patience: {patience}, Recall: {val_value:.4f}"
            )
            if patience >= 10:
                break
            patience += 1
        print("Training finished")
        print(
            f"Best Recall: {recall:.4f}, NDCG: {ndcg:.4f}, Precision: {precision:.4f}, MRR: {mrr:.4f}"
        )

def main():
    model = TrainGNN()
    model.train()

main()
