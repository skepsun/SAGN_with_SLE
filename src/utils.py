import random

import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

eps = 1e-9

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def entropy(probs):
    res = - probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
    return res

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def to_scipy(tensor):
    """Convert a sparse tensor to scipy matrix"""
    values = tensor._values()
    indices = tensor._indices()
    return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def from_scipy(sparse_mx):
    """Convert a scipy sparse matrix to sparse tensor"""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_spectral_emb(adj, K):
    A = to_scipy(adj.to("cpu"))
    L = from_scipy(sp.csgraph.laplacian(A, normed=True))
    _, spectral_emb = torch.lobpcg(L, K)
    return spectral_emb.to(adj.device)
