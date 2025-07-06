import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class MatrixFactorization(nn.Module):
    # here we does not consider the bias term 
    def __init__(self, user_num, item_num, embedding_size) -> None:
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size
        self.padding_index = 0
        self.user_embedding = nn.Embedding(user_num, embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(item_num, embedding_size, padding_idx=self.padding_index)
        print("creating MF model, user num:", user_num, "item num:", item_num)

    def user_encoder(self,users,all_users=None):
        # print("user max:", users.max(), users.min())
        return self.user_embedding(users)
    def item_encoder(self,items,all_items=None):
        # print("items max:", items.max(), items.min())
        return self.item_embedding(items)
    
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    
    def forward(self,users,items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        matching = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return matching


