import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.parse import args
from typing import List


class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        return self.input_text[idx]


class DataHandler:
    def __init__(self):
        if args.dataset == "amazon":
            self.system_prompt = "Explain why the user would buy with the book within 50 words."
            self.item = "book"
        elif args.dataset == "yelp" or args.dataset == "google":
            self.system_prompt = "Explain why the user would enjoy the business within 50 words."
            self.item = "business"
        user_path = f"./data/{args.dataset}/user_emb.pkl"
        item_path = f"./data/{args.dataset}/item_emb.pkl"
        with open(user_path, "rb") as file:
            self.user_emb = pickle.load(file)
        with open(item_path, "rb") as file:
            self.item_emb = pickle.load(file)

    def load_data(self):
        # load data from data_loaders in data
        with open(f"./data/{args.dataset}/trn.pkl", "rb") as file:
            trn_data = pickle.load(file)
        with open(f"./data/{args.dataset}/val.pkl", "rb") as file:
            val_data = pickle.load(file)
        with open(f"./data/{args.dataset}/tst.pkl", "rb") as file:
            tst_data = pickle.load(file)

        # convert data into dictionary
        trn_dict = trn_data.to_dict("list")
        val_dict = val_data.to_dict("list")
        tst_dict = tst_data.to_dict("list")

        # combine all information input input string
        trn_input = []
        val_input = []
        tst_input = []
        for i in range(len(trn_dict["uid"])):
            user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {trn_dict['title'][i]} user profile: {trn_dict['user_summary'][i]} {self.item} profile: {trn_dict['item_summary'][i]} <EXPLAIN_POS> {trn_dict['explanation'][i]}"
            trn_input.append(
                (
                    self.user_emb[trn_dict["uid"][i]],
                    self.item_emb[trn_dict["iid"][i]],
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                )
            )
        for i in range(len(val_dict["uid"])):
            user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {val_dict['title'][i]} user profile: {val_dict['user_summary'][i]} {self.item} profile: {val_dict['item_summary'][i]} <EXPLAIN_POS>"
            val_input.append(
                (
                    self.user_emb[val_dict["uid"][i]],
                    self.item_emb[val_dict["iid"][i]],
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    val_dict['explanation'][i],
                )
            )
        for i in range(len(tst_dict["uid"])):
            user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {tst_dict['title'][i]} user profile: {tst_dict['user_summary'][i]} {self.item} profile: {tst_dict['item_summary'][i]} <EXPLAIN_POS>"
            tst_input.append(
                (
                    self.user_emb[tst_dict["uid"][i]],
                    self.item_emb[tst_dict["iid"][i]],
                    f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                    tst_dict["explanation"][i],
                )
            )

        # load training batch
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # load testing batch
        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True)

        return trn_loader, val_loader, tst_loader
