import torch
import torch.nn as nn
import torch.sparse as sp
import torch.nn.functional as F

class LGCN_matrix(nn.Module):
    def __init__(self, config):
        super(LGCN_matrix, self).__init__()
        self.user_num = config.user_num
        self.item_num = config.item_num

        self.layers_num = config.layers_num

        self.emb = nn.Embedding(self.user_num + self.item_num, config.hidden_size)
        
    def forward(self, user, pos, neg, laplacian, **kwargs):
        emb = self.emb.weight

        user_embs = []
        pos_embs = []
        neg_embs = []

        if user is not None:
            user_embs.append(emb[user])
        if pos is not None:
            pos_embs.append(emb[pos])
        if neg is not None:
            neg_embs.append(emb[neg])

        for _ in range(self.layers_num):
            emb = sp.mm(laplacian, emb)
            
            if user is not None:
                user_embs.append(emb[user])
            if pos is not None:
                pos_embs.append(emb[pos])
            if neg is not None:
                neg_embs.append(emb[neg])
        
        user_emb = torch.stack(user_embs, dim=0).mean(dim=0)
        if pos is None:
            return user_emb
        pos_emb = torch.stack(pos_embs, dim=0).mean(dim=0)
        neg_emb = torch.stack(neg_embs, dim=0).mean(dim=0)
        
        pos_logits = torch.einsum('id, id->i', user_emb, pos_emb)
        neg_logits = torch.einsum('id, id->i', user_emb, neg_emb)

        loss = -torch.log((pos_logits - neg_logits).sigmoid())
        return loss.sum()


