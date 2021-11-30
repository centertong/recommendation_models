import torch
import torch.nn as nn
import torch.sparse as sp
import torch.nn.functional as F

class NGCF_matrix(nn.Module):
    def __init__(self, config):
        super(NGCF_matrix, self).__init__()
        self.user_num = config.user_num
        self.item_num = config.item_num

        self.emb = nn.Embedding(self.user_num + self.item_num, config.hidden_size)
        self.w1_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.layers_num)])
        self.w2_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.layers_num)])
        
    def forward(self, user, pos, neg, laplacian, identy):
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

        for w1, w2 in zip(self.w1_layers, self.w2_layers):
            sx = w1(sp.mm(laplacian + identy, emb))
            ox = sp.mm(laplacian, emb) * w2(emb)
            emb = F.leaky_relu(sx + ox)

            if user is not None:
                user_embs.append(emb[user])
            if pos is not None:
                pos_embs.append(emb[pos])
            if neg is not None:
                neg_embs.append(emb[neg])
        
        user_emb = torch.cat(user_embs, dim=-1)
        if pos is None:
            return user_emb
        pos_emb = torch.cat(pos_embs, dim=-1)
        neg_emb = torch.cat(neg_embs, dim=-1)
        
        pos_logits = torch.einsum('id, id->i', user_emb, pos_emb)
        neg_logits = torch.einsum('id, id->i', user_emb, neg_emb)

        loss = -torch.log((pos_logits - neg_logits).sigmoid())
        return loss.sum()


