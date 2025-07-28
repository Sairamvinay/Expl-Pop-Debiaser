import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def utility_relevance_loss(labels, scores):
    relevance_loss = F.binary_cross_entropy_with_logits(scores, labels)
    return relevance_loss

# Stage 1: BPR Explanation Pairwise Loss
def bpr_loss(pos_score, neg_score):
    diff = pos_score - neg_score
    diff = diff.clamp(min=-10, max=10)
    return -torch.mean(F.logsigmoid(diff))


class MatrixFactorization(nn.Module):
    # here we does not consider the bias term 
    def __init__(self, user_num, item_num, embedding_size) -> None:
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size
        # self.padding_index = 0
        # self.user_embedding = nn.Embedding(user_num, embedding_size, padding_idx=self.padding_index)
        # self.item_embedding = nn.Embedding(item_num, embedding_size, padding_idx=self.padding_index)
        self.user_embedding = nn.Embedding(user_num, embedding_size)
        self.item_embedding = nn.Embedding(item_num, embedding_size)
        print("creating MF model, user num:", user_num, "item num:", item_num)

    def user_encoder(self,users,all_users=None):
        return self.user_embedding(users)

    def item_encoder(self,items,all_items=None):
        return self.item_embedding(items)
    
    
    def forward(self,user_ids,item_ids):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        matching = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return matching



