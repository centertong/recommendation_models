import torch
import torch.nn as nn
import torch.sparse as sp
import torch.nn.functional as F
from torch.sparse import mm as smm


class UltraGCN_matrix(nn.Module):
    def __init__(self, config):
        super(UltraGCN_matrix, self).__init__()
        self.user_num = config.user_num
        self.item_num = config.item_num

        self.layers_num = config.layers_num

        self.embs = nn.Embedding(self.user_num + self.item_num, config.hidden_size)
        self.chk = True

    def get_data_from_sparse(self, mat, user, pos, neg):
        pos_beta = torch.tensor([mat[u, v] for u, v in zip(user, pos)])
        neg_beta = torch.tensor([mat[u, v] for u, v in zip(user, neg)])
        return pos_beta, neg_beta
        
    def forward(self, user, pos, neg, pos_beta, neg_beta, weights, neighbor, **kwargs):
        user_emb = self.embs(user)
        pos_emb = self.embs(pos)
        neg_emb = self.embs(neg)

        pos_neighbor = neighbor[pos - self.user_num]
        pos_weight = weights[pos - self.user_num]
        pos_neighbor_emb = self.embs(pos_neighbor)

        
        pos_logits = torch.einsum('id, id->i', user_emb, pos_emb)
        neg_logits = torch.einsum('id, id->i', user_emb, neg_emb)
        pos_neighbor_logits = torch.einsum('id, ikd->ik', user_emb, pos_neighbor_emb)

        pos_neighbor_logits = - torch.log((pos_neighbor_logits).sigmoid())
        LI = (pos_neighbor_logits * pos_weight).sum()

        LO = -torch.log((pos_logits).sigmoid()).sum() -torch.log((-neg_logits).sigmoid()).sum()
        LC = -(torch.log((pos_logits).sigmoid()) * pos_beta).sum() - (torch.log((-neg_logits).sigmoid()) * neg_beta).sum()

        loss = LC + LO + LI * 2.5
        return loss.sum()


