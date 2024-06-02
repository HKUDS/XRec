from utils.parse import args
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, num_user, num_item, adj):
        super(LightGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = args.embedding_dim
        self.n_layers = args.n_layers
        self.adj = adj

        # Initialize user and item embeddings
        self.user_embeds = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_user, self.embedding_dim))
        )
        self.item_embeds = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.num_item, self.embedding_dim))
        )

    def forward(self, adj):
        embeds = torch.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for layer in range(self.n_layers):
            embeddings = torch.spmm(adj.cuda(), embeds_list[-1])
            embeds_list.append(embeddings)

        # Aggregate embeddings from all layers
        all_embeddings = torch.stack(embeds_list, dim=0)
        all_embeddings = torch.sum(all_embeddings, dim=0)
        self.final_embeds = all_embeddings

        return all_embeddings[: self.num_user], all_embeddings[self.num_user :]

    def cal_loss(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # Calculate the difference between the positive and negative item predictions
        pos_scores = torch.sum(anc_embeds * pos_embeds, dim=1)
        neg_scores = torch.sum(anc_embeds * neg_embeds, dim=1)
        diff_scores = pos_scores - neg_scores

        # Compute the BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(diff_scores)))

        # Regularization term (optional)
        reg_loss = args.weight_decay * (
            anc_embeds.norm(2).pow(2)
            + pos_embeds.norm(2).pow(2)
            + neg_embeds.norm(2).pow(2)
        )

        return loss + reg_loss

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj)
        users, _ = batch_data
        pck_user_embeds = user_embeds[users]
        full_preds = pck_user_embeds @ item_embeds.T
        return full_preds
