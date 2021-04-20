import math
import os
import random
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# form from DGL's implementation of SIGN
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x

# add batchnorm and replace prelu with relu
class MLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, bias=True, residual=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        self.rec_layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats, bias=bias))
        else:
            self.layers.append(nn.Linear(in_feats, hidden, bias=bias))
            self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden, bias=bias))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats, bias=bias))
        if self.n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
        if residual:
            self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
        else:
            self.res_fc = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if isinstance(layer.bias, nn.Parameter):
                nn.init.zeros_(layer.bias)

        for bn in self.bns:
            bn.reset_parameters()

        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, x):
        if self.res_fc is not None:
            res_term = self.res_fc(x)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.relu(self.bns[layer_id](x)))
        if self.res_fc is not None:
            x += res_term
        return x


class SAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, n_layers, num_heads,
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, use_labels=False, use_features=True):
        super(SAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._use_labels = use_labels
        self._use_features = use_features
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.bn = nn.BatchNorm1d(hidden)
        # self.bns = nn.ModuleList([nn.BatchNorm1d(hidden * num_heads) for i in range(num_hops)])
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        # self.position_emb = nn.Embedding(num_hops, hidden * num_heads)
        self.fcs = nn.ModuleList([MLP(in_feats, hidden, hidden * num_heads, n_layers, dropout, bias=True, residual=False) for i in range(num_hops)])
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        # self.res_fc_1 = nn.Linear(hidden, out_feats, bias=False)
        self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        if self._use_labels:
            self.label_fc = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout, bias=True)

        self.mlp = MLP(hidden, hidden, out_feats, n_layers, dropout, bias=True, residual=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs:
            fc.reset_parameters()
        nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self._use_labels:
            self.label_fc.reset_parameters()
        self.mlp.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats, label_emb):
        out = 0
        if self._use_features:
            feats = [self.input_drop(feat) for feat in feats]
            hidden = []
            for i in range(len(feats)):
                hidden.append(self.fcs[i](feats[i]).view(-1, self._num_heads, self._hidden))
            astack_l = [(feat * self.hop_attn_l).sum(dim=-1).unsqueeze(-1) for feat in hidden]
            a_r = (hidden[0] * self.hop_attn_r).sum(dim=-1).unsqueeze(-1)
            astack = torch.cat([(a_l + a_r).unsqueeze(-1) for a_l in astack_l], dim=-1)
            a = self.leaky_relu(astack)
            a = F.softmax(a, dim=-1)
            a = self.attn_dropout(a)
            
            for i in range(a.shape[-1]):
                out += hidden[i] * a[:, :, :, i]
            out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
            out = out.mean(1)
            out = self.dropout(self.relu(self.bn(out)))
            out = self.mlp(out)
        else:
            a = None
        if self._use_labels:
            out += self.label_fc(label_emb)
        return out, a

class PlainSAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, n_layers, num_heads,
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, negative_slope=0.2, use_labels=False, use_features=True):
        super(PlainSAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._use_labels = use_labels
        self._use_features = use_features
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.bn = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, hidden)))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)
        
        if self._use_labels:
            self.label_fc = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout, bias=True)

        self.mlp = MLP(in_feats, hidden, out_feats, n_layers, dropout, bias=True, residual=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
        nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
        if self._use_labels:
            self.label_fc.reset_parameters()
        self.mlp.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats, label_emb):
        out = 0
        if self._use_features:
            feats = [self.input_drop(feat) for feat in feats]
            a = None
            out = feats[-1]
            out = self.mlp(out)
        else:
            a = None
        if self._use_labels:
            out += self.label_fc(label_emb)
        return out, a

class SimpleSAGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, n_layers, num_heads, weight_style = "uniform",
                 dropout=0.5, input_drop=0.0, attn_drop=0.0, use_labels=False, use_features=True):
        super(SimpleSAGN, self).__init__()
        self._num_heads = num_heads
        self._hidden = hidden
        self._out_feats = out_feats
        self._use_labels = use_labels
        self._use_features = use_features
        assert weight_style in ["exponent", "uniform"]
        self.weights = [1 / (num_hops)] * (num_hops) if weight_style == "uniform" \
                        else [0.5**k for k in range(num_hops)]
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.bn = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.fcs = nn.ModuleList([MLP(in_feats, hidden, hidden * num_heads, n_layers, dropout, bias=True, residual=False) for i in range(num_hops)])
        self.res_fc = nn.Linear(in_feats, hidden * num_heads, bias=False)

        
        if self._use_labels:
            self.label_fc = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout, bias=True)
        self.mlp = MLP(hidden, hidden, out_feats, n_layers, dropout, bias=True, residual=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fcs:
            fc.reset_parameters()
        nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if self._use_labels:
            self.label_fc.reset_parameters()
        self.mlp.reset_parameters()

        self.bn.reset_parameters()

    def forward(self, feats, label_emb):
        out = 0
        if self._use_features:
            feats = [self.input_drop(feat) for feat in feats]
            hidden = []
            for i in range(len(feats)):
                hidden.append(self.fcs[i](feats[i]).view(-1, self._num_heads, self._hidden))
            
            for i in range(len(hidden)):
                out += hidden[i] * self.weights[i]
            out += self.res_fc(feats[0]).view(-1, self._num_heads, self._hidden)
            out = out.mean(1)
            out = self.dropout(self.relu(self.bn(out)))
            out = self.mlp(out)
        if self._use_labels:
            out += self.label_fc(label_emb)
        return out

class LPMLP(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout, bias=True, residual=False, input_drop=0., use_labels=False):
        super(LPMLP, self).__init__()
        self.input_dropout = nn.Dropout(input_drop)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.n_layers = n_layers
        self.rec_layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats, bias=bias))
        else:
            self.layers.append(nn.Linear(in_feats, hidden, bias=bias))
            self.bns.append(nn.BatchNorm1d(hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden, bias=bias))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.layers.append(nn.Linear(hidden, out_feats, bias=bias))
        if self.n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
        if residual:
            self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
        else:
            self.res_fc = None
        if use_labels:
            self.label_mlp = MLP(out_feats, hidden, out_feats, n_layers, dropout, bias=True, residual=False)
        else:
            self.label_mlp = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if isinstance(layer.bias, nn.Parameter):
                nn.init.zeros_(layer.bias)

        for bn in self.bns:
            bn.reset_parameters()

        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        
        if self.label_mlp is not None:
            self.label_mlp.reset_parameters()

    def forward(self, x, label_emb):
        x = self.input_dropout(x)
        if self.res_fc is not None:
            res_term = self.res_fc(x)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.relu(self.bns[layer_id](x)))
        if self.res_fc is not None:
            x += res_term
        if self.label_mlp is not None:
            x += self.label_mlp(label_emb)
        return x

# add batchnorm
class LPSIGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, label_in_feats, num_hops, n_layers, dropout=0.5, input_drop=0., use_labels=False, residual=False):
        super(LPSIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_dropout = nn.Dropout(input_drop)
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm1d((num_hops) * hidden)
        self.inception_ffs = nn.ModuleList()
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout))
        # self.linear = nn.Linear(hidden * (R + 1), out_feats)
        self.project = FeedForwardNet((num_hops) * hidden, hidden, out_feats,
                                      n_layers, dropout)
        if residual:
            self.res_fc = nn.Linear(in_feats, (num_hops) * hidden, bias=False)
        else:
            self.res_fc = None
        if use_labels:
            self.label_mlp = MLP(label_in_feats, hidden, out_feats, 2 * n_layers, dropout)
        else:
            self.label_mlp = None
        self.reset_parameters()

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()
        if self.res_fc is not None:
            self.res_fc.reset_parameters()
        if self.label_mlp is not None:
            self.label_mlp.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, feats, label_emb):
        feats = [self.input_dropout(feat) for feat in feats]
        hidden = []
        
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = torch.cat(hidden, dim=-1)
        if self.res_fc is not None:
            out += self.res_fc(feats[0])
        out = self.project(self.dropout(self.prelu(self.bn(out))))
        if self.label_mlp is not None:
            out += self.label_mlp(label_emb)
        return out

