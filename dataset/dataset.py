import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from torch.sparse import mm as smm
from tqdm.auto import tqdm
import gc

def diagonal_sparse(mat):
    d = torch.sparse.sum(mat, dim=1)
    i = torch.cat([d.indices(), d.indices()], dim=0)
    v = d.values()
    return torch.sparse_coo_tensor(i, v, mat.size())

def topk_diagonal_sparse(mat, k):
    I = mat.indices()
    V = mat.values()
    row, _ = I
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    node_mask = row.new_empty(mat.size(0), dtype=torch.bool)

    start_index = 0
    batch_size = 1000
    end_index = mat.size(0)
    d = []
    while start_index < end_index:
        if start_index + batch_size > end_index:
            batch_size = end_index - start_index
        node_mask.fill_(False)
        node_mask[start_index:start_index+batch_size] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        tmpI = I[:, edge_mask]
        tmpI[0,:] -= start_index
        tmpV = V[edge_mask]
        tmpM = torch.sparse_coo_tensor(tmpI, tmpV, (batch_size, end_index)).to_dense()
        val, _ = tmpM.topk(k, dim=1)
        d += val.sum(dim=1)
        # diag += [tmpM[idx, idx + start_index] for idx in range(batch_size)]

        start_index += batch_size
 
    i = torch.stack([torch.arange(len(d)), torch.arange(len(d))], dim=0)
    v = torch.tensor(d)
    return torch.sparse_coo_tensor(i, v, mat.size())

def extract_diagonal_matrix(mat):
    I = mat.indices()
    V = mat.values()
    row, _ = I
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    node_mask = row.new_empty(mat.size(0), dtype=torch.bool)

    start_index = 0
    batch_size = 1000
    end_index = mat.size(0)
    d = []
    while start_index < end_index:
        if start_index + batch_size > end_index:
            batch_size = end_index - start_index
        node_mask.fill_(False)
        node_mask[start_index:start_index+batch_size] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        tmpI = I[:, edge_mask]
        tmpI[0,:] -= start_index
        tmpV = V[edge_mask]
        tmpM = torch.sparse_coo_tensor(tmpI, tmpV, (batch_size, end_index)).to_dense()
        d.append(torch.tensor([tmpM[idx, idx + start_index] for idx in range(batch_size)]))

        start_index += batch_size
    d = torch.cat(d)
    i = torch.stack([torch.arange(len(d)), torch.arange(len(d))], dim=0)
    v = d

    return torch.sparse_coo_tensor(i, v, mat.size())



class SparseMatrixDataset(Dataset):

    def __init__(self, config):
        super().__init__()
        self.user_num, self.user_feature = self.load_users(config.user_path)
        self.item_num, self.item_feature = self.load_items(config.item_path)
        self.edges = self.load_edges(config.edge_path, self.user_num, self.item_num).coalesce()
        self.n = self.edges.indices().size(1)
        

    def load_edges(self, path, user_num, item_num):
        edges = []
        with open(path, 'r') as f:
            for line in f.readlines():
                datas = line.split()
                user = int(datas[0])
                edges += [ [user, int(item) + user_num] for item in datas[1:]]
                # edges += [ [int(item) + user_num, user] for item in datas[1:]]
        
        indices = torch.tensor(edges).t()
        values = torch.ones((indices.size(1),))
        return torch.sparse_coo_tensor(indices, values, [user_num + item_num, user_num + item_num])

    def load_users(self, path):
        users = pd.read_csv(path, delimiter=' ')
        return len(users), None

    def load_items(self, path):
        items = pd.read_csv(path, delimiter=' ')
        return len(items), None

    def get_laplacian_matrix(self,):
        edges = self.edges + self.edges.t()
        diag_inv = diagonal_sparse(edges) ** -0.5
        return torch.sparse.mm(torch.sparse.mm(diag_inv, edges), diag_inv)

    def get_converge_matrix(self,):
        edges = self.edges + self.edges.t()
        diag_inv = diagonal_sparse(edges) ** -1
        diag_plus = (diagonal_sparse(edges) + self.get_identity_matrix()) ** 0.5
        diag_plus_inv = (diagonal_sparse(edges) + self.get_identity_matrix()) ** -0.5
        diag1 = smm(diag_inv, diag_plus)
        return smm(smm(diag1, edges), diag_plus_inv).coalesce()

    def get_weight_adj_matrix(self, k):
        edges = self.edges
        G = smm(edges.t(), edges).coalesce()
        gk = topk_diagonal_sparse(G, k)
        Gi = extract_diagonal_matrix(G)
        w = smm(smm(gk ** 0.5 , smm((gk-Gi)**-1, G)), gk ** -0.5).coalesce()
        
        I = G.indices()
        V = G.values()
        row, _ = I
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        node_mask = row.new_empty(G.size(0), dtype=torch.bool)

        wI = w.indices()
        wV = w.values()
        w_row, _ = wI
        w_edge_mask = row.new_empty(w_row.size(0), dtype=torch.bool)
        w_node_mask = row.new_empty(G.size(0), dtype=torch.bool)

        
        start_index = self.user_num
        batch_size = 1000
        end_index = G.size(0)
        nm = []
        wm = []
        while start_index < end_index:
            if start_index + batch_size > end_index:
                batch_size = end_index - start_index
            node_mask.fill_(False)
            node_mask[start_index:start_index+batch_size] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            tmpI = I[:, edge_mask]
            tmpI[0,:] -= start_index
            tmpV = V[edge_mask]
            tmpM = torch.sparse_coo_tensor(tmpI, tmpV, (batch_size, end_index)).to_dense()
            _, ind = tmpM.topk(k, dim=1) #ind: batch_size * k
            nm.append(ind)

            w_node_mask.fill_(False)
            w_node_mask[start_index:start_index+batch_size] = True
            torch.index_select(w_node_mask, 0, w_row, out=w_edge_mask)
            tmpI = wI[:, w_edge_mask]
            tmpI[0, :] -= start_index
            tmpV = wV[w_edge_mask]
            tmpM = torch.sparse_coo_tensor(tmpI, tmpV, (batch_size, end_index)).to_dense()
            wm += [tmpM[idx, ind[idx]] for idx in range(batch_size)]
            
            start_index += batch_size
        wm = torch.stack(wm, dim=0)    
        nm = torch.cat(nm, dim=0)
        return wm, nm
    

    def get_identity_matrix(self,):
        size = self.item_num + self.user_num
        i = torch.arange(size)
        i = torch.stack([i, i], dim=0)
        v = torch.ones((size,))
        return torch.sparse_coo_tensor(i, v, [size, size])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        user, item = self.edges.indices()[:, idx]
        while True:
            neg = random.randrange(self.user_num, self.user_num + self.item_num)
            if self.edges[user, neg] == 0:
                break
        
        return user, item, torch.tensor(neg)