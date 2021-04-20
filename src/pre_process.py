import math
import os
import random
import time
import gc

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from dataset import load_dataset
from utils import compute_spectral_emb, entropy


def neighbor_average_features(g, feat, args, use_norm=False, style="all"):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats", style)
    
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    g = g.to(aggr_device)
    feat = feat.to(aggr_device)

    if style == "all":
        g.ndata['feat_0'] = feat
            
        # print(g.ndata["feat"].shape)
        # print(norm.shape)
        if use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.K + 1):
            g.ndata[f'feat_{hop}'] = g.ndata[f'feat_{hop-1}']
            # g.ndata['pre_label_emb'] = g.ndata['label_emb']
            if use_norm:
                g.ndata[f'feat_{hop}'] = g.ndata[f'feat_{hop}'] * norm
                
                g.update_all(fn.copy_src(src=f'feat_{hop}', out='msg'),
                            fn.sum(msg='msg', out=f'feat_{hop}'))
                g.ndata[f'feat_{hop}'] = g.ndata[f'feat_{hop}'] * norm
            else:
                g.update_all(fn.copy_src(src=f'feat_{hop}', out='msg'),
                            fn.mean(msg='msg', out=f'feat_{hop}'))

                
            # if hop > 1:
            #     g.ndata['label_emb'] = 0.5 * g.ndata['pre_label_emb'] + \
            #                            0.5 * g.ndata['label_emb']
        res = []
        for hop in range(args.K + 1):
            res.append(g.ndata.pop(f'feat_{hop}'))
        gc.collect()

        if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
                target_mask = g.ndata['target_mask']
                target_ids = g.ndata[dgl.NID][target_mask]
                num_target = target_mask.sum().item()
                new_res = []
                for x in res:
                    feat = torch.zeros((num_target,) + x.shape[1:],
                                    dtype=x.dtype, device=x.device)
                    feat[target_ids] = x[target_mask]
                    new_res.append(feat)
                res = new_res

    # del g.ndata['pre_label_emb']
    elif style in ["last", "ppnp"]:
        
        if style == "ppnp": init_feat = feat
        if use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.label_K+1):         
            # g.ndata["f_next"] = g.ndata["f"]
            if use_norm:
                feat = feat * norm
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.sum(msg='msg', out='f'))
                feat = g.ndata.pop('f')
                # degs = g.in_degrees().float().clamp(min=1)
                # norm = torch.pow(degs, -0.5)
                # shp = norm.shape + (1,) * (g.ndata['f'].dim() - 1)
                # norm = torch.reshape(norm, shp)
                feat = feat * norm
            else:
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
                feat = g.ndata.pop('f')
            if style == "ppnp":
                feat = 0.5 * feat + 0.5 * init_feat
            
        res = feat
        gc.collect()

        if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
            target_mask = g.ndata['target_mask']
            target_ids = g.ndata[dgl.NID][target_mask]
            num_target = target_mask.sum().item()
            new_res = torch.zeros((num_target,) + feat.shape[1:],
                                    dtype=feat.dtype, device=feat.device)
            new_res[target_ids] = res[target_mask]
            res = new_res

    
    return res

def prepare_data(device, args, probs_path, stage=0, load_embs=False, load_label_emb=False):
    """
    Load dataset and compute neighbor-averaged node features used by scalable GNN model
    Note that we select only one integrated representation as node feature input for mlp 
    """
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    emb_path = os.path.join("..", "embeddings", args.dataset, 
                args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                "smoothed_emb.pt")

    data = load_dataset(args.dataset, "../../dataset", aggr_device, mag_emb=args.mag_emb)
    t1 = time.time()
    
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    
    label_emb_path = os.path.join("..", "embeddings", args.dataset, 
                args.model if (args.model != "simple_sagn") else (args.model + "_" + args.weight_style),
                "smoothed_label_emb.pt")
    if not os.path.exists(os.path.dirname(emb_path)):
        os.makedirs(os.path.dirname(emb_path))

    feat_averaging_style = "all" if args.model in ["sagn", "plain_sagn", "simple_sagn", "sign"] else "ppnp"
    label_averaging_style = "last"
    in_feats = g.ndata['feat'].shape[1]
    # n_classes = (labels.max() + 1).item() if labels.dim() == 1 else labels.size(1)
    print("in_feats:", in_feats)
    feat = g.ndata.pop('feat')
    if args.model in ["mlp"]:
        spectral_emb_path = os.path.join("..", "embeddings", args.dataset, "spectral.pt")
        if os.path.exists(spectral_emb_path):
            spectral_emb = torch.load(spectral_emb_path).to(aggr_device)
        else:
            spectral_emb = compute_spectral_emb(g.adjacency_matrix(), 128).to(aggr_device)
            if not os.path.exists(os.path.dirname(spectral_emb_path)):
                os.path.makedirs(os.path.dirname(spectral_emb_path))
            torch.save(spectral_emb, spectral_emb_path)
    else:
        spectral_emb = None
    if stage > 0:
        teacher_probs = torch.load(probs_path).to(aggr_device)
        tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)

        # assert len(teacher_probs) == len(feat)
        if args.dataset in ['yelp', 'ppi', 'ppi_large']:
            threshold = - args.threshold * np.log(args.threshold) - (1-args.threshold) * np.log(1-args.threshold)
            entropy_distribution = entropy(teacher_probs)
            print(threshold)
            print(entropy_distribution.mean(1).max().item())
            
            confident_nid_inner = torch.arange(len(teacher_probs))[(entropy_distribution.mean(1) <= threshold)]
        else:
            confident_nid_inner = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > args.threshold]
        extra_confident_nid_inner = confident_nid_inner[confident_nid_inner >= len(train_nid)]
        confident_nid = tr_va_te_nid[confident_nid_inner]
        extra_confident_nid = tr_va_te_nid[extra_confident_nid_inner]
        print(f"pseudo label number: {len(confident_nid)}")
        if args.dataset in ["yelp", "ppi", "ppi_large"]:
            pseudo_labels = teacher_probs
            pseudo_labels[pseudo_labels >= 0.5] = 1
            pseudo_labels[pseudo_labels < 0.5] = 0
            labels_with_pseudos = torch.ones_like(labels)
        else:
            pseudo_labels = torch.argmax(teacher_probs, dim=1).to(labels.device)
            labels_with_pseudos = torch.zeros_like(labels)
        train_nid_with_pseudos = np.union1d(train_nid, confident_nid)
        print(f"enhanced train set number: {len(train_nid_with_pseudos)}")
        labels_with_pseudos[train_nid] = labels[train_nid]
        labels_with_pseudos[extra_confident_nid] = pseudo_labels[extra_confident_nid_inner]
        
        # train_nid_with_pseudos = np.random.choice(train_nid_with_pseudos, size=int(0.5 * len(train_nid_with_pseudos)), replace=False)
    else:
        teacher_probs = None
        pseudo_labels = None
        labels_with_pseudos = labels.clone()
        confident_nid = train_nid
        train_nid_with_pseudos = train_nid
    
    if args.use_labels & ((not args.inductive) or stage > 0):
        print("using label information")
        if args.dataset in ["yelp", "ppi", "ppi_large"]:
            label_emb = 0.5 * torch.ones([feat.shape[0], n_classes]).to(labels.device)
            # label_emb = labels_with_pseudos.mean(0).repeat([feat.shape[0], 1])
            label_emb[train_nid_with_pseudos] = labels_with_pseudos.float()[train_nid_with_pseudos]

        else:
            label_emb = torch.zeros([feat.shape[0], n_classes]).to(labels.device)
            # label_emb = (1. / n_classes) * torch.ones([feat.shape[0], n_classes]).to(device)
            label_emb[train_nid_with_pseudos] = F.one_hot(labels_with_pseudos[train_nid_with_pseudos], num_classes=n_classes).float().to(labels.device)


        if args.dataset == "ogbn-mag":
            # rand_weight = torch.Tensor(n_classes, 128).uniform_(-0.5, 0.5)
            # label_emb = torch.matmul(label_emb, rand_weight.to(device))
            # pca = PCA(n_components=128)
            # label_emb = torch.FloatTensor(pca.fit_transform(label_emb.cpu())).to(device)
            target_mask = g.ndata["target_mask"]
            target_ids = g.ndata[dgl.NID][target_mask]
            num_target = target_mask.sum().item()
            new_label_emb = torch.zeros((len(feat),) + label_emb.shape[1:],
                                dtype=label_emb.dtype, device=label_emb.device)
            new_label_emb[target_mask] = label_emb[target_ids]
            label_emb = new_label_emb
    else:
        label_emb = None
    
    if args.inductive:
        print("inductive setting detected")
        if os.path.exists(os.path.join("../subgraphs",args.dataset, "subgraph_train.pt")):
            print("load train subgraph")
            g_train = torch.load(os.path.join("../subgraphs",args.dataset, "subgraph_train.pt")).to(g.device)
        else:
            print("get train subgraph")
            g_train = dgl.node_subgraph(g, train_nid.to(g.device))
            if not os.path.exists(os.path.join("../subgraphs",args.dataset)):
                os.makedirs(os.path.join("../subgraphs",args.dataset))
            torch.save(g_train, os.path.join("../subgraphs",args.dataset, "subgraph_train.pt"))
        # print("get val/test subgraph")
        # g_val_test = dgl.node_subgraph(g, torch.cat([val_nid, test_nid],dim=0).to(g.device))
        
        train_mask = g_train.ndata[dgl.NID]
        if load_embs and os.path.exists(emb_path):
            pass
        else:
            feats = neighbor_average_features(g, feat, args, use_norm=args.use_norm, style=feat_averaging_style)
            feats_train = neighbor_average_features(g_train, feat[g_train.ndata[dgl.NID]], args, use_norm=args.use_norm, style=feat_averaging_style)
            if args.model in ["sagn", "simple_sagn", "sign"]:
                for i in range(args.K+1):
                    feats[i][train_mask] = feats_train[i]
            else:
                feats[train_mask] = feats_train
            if load_embs:
                if not os.path.exists(emb_path):
                    print("saving smoothed node features to " + emb_path)
                    torch.save(feats, emb_path)
                del feats, feat
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                
        
        if (stage == 0) and load_label_emb and os.path.exists(label_emb_path):
            pass
        else:
            if label_emb is not None:
                label_emb_train = neighbor_average_features(g_train, label_emb[g_train.ndata[dgl.NID]], args, use_norm=args.use_norm if (args.dataset != 'cora') else False, style=label_averaging_style)
            else:
                label_emb_train = None
            
            # del g_train
            # torch.cuda.empty_cache()
            if label_emb is not None:
                label_emb = neighbor_average_features(g, label_emb, args, use_norm=args.use_norm if (args.dataset != 'cora') else False, style=label_averaging_style)
                label_emb[train_mask] = label_emb_train
            if load_label_emb:
                if not os.path.exists(label_emb_path):
                    print("saving initial label embeddings to " + label_emb_path)
                    torch.save(label_emb, label_emb_path)
                del label_emb
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        
    else:
        # for transductive setting
        
        if (stage == 0) and load_label_emb and os.path.exists(label_emb_path):
            pass
        else:
            if label_emb is not None:
                label_emb = neighbor_average_features(g, label_emb, args, use_norm=args.use_norm if (args.dataset != 'cora') else False, style=label_averaging_style)
            if load_label_emb and stage == 0: 
                if (not os.path.exists(label_emb_path)):
                    print("saving initial label embeddings to " + label_emb_path)
                    torch.save(label_emb, label_emb_path)
                del label_emb, g
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

        if load_embs and os.path.exists(emb_path):
            pass
        else:
            feats = neighbor_average_features(g, feat, args, style=feat_averaging_style)
            if load_embs:
                if not os.path.exists(emb_path):
                    print("saving smoothed node features to " + emb_path)
                    torch.save(feats, emb_path)
                del feats, feat
                gc.collect()
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        
        # if args.save_temporal_emb:
        #     torch.save(feats, emb_path)
            
        if spectral_emb is not None:
            # feats = torch.cat([feats, spectral_emb], dim=1)
            in_feats = feats.size(1)
    
    # save smoothed node features and initial smoothed node label embeddings, 
    # if "load" is set true and they have not been saved
 
    if load_embs:
        print("load saved embeddings")
        feats = torch.load(emb_path)
    if load_label_emb and (stage == 0):
        print("load saved label embedding")
        label_emb = torch.load(label_emb_path)

    # label_emb = (label_emb - label_emb.mean(0)) / label_emb.std(0)
    # eval_feats = neighbor_average_features(g, eval_feat, args)
    labels = labels.to(device)
    labels_with_pseudos = labels_with_pseudos.to(device)
    # move to device

    train_nid = train_nid.to(device)
    train_nid_with_pseudos = torch.LongTensor(train_nid_with_pseudos).to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    t2 = time.time()

    return feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, evaluator, t2 - t1
