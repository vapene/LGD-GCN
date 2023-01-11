import torch.nn.functional as fn
import torch.optim as optim
import torch.autograd
import torch.nn as nn
import torch

import scipy.sparse as spsprs
import networkx as nx
import numpy as np
import tempfile
import pickle
import random
import sys
import time
import gc
import os


class SN_DataReader:
    def __init__(self, data_name, data_dir, logfile, config):
        # Loading data
        tvt_nids = pickle.load(open("{}{}_tvt_nids.pkl".format(data_dir, data_name), 'rb'))  # list <-- int64
        adj_org = pickle.load(open("{}{}_adj.pkl".format(data_dir, data_name), 'rb'))  # scipy.sparse.csr.csr_matrix <-- numpy.float64
        feat = pickle.load(open("{}{}_features.pkl".format(data_dir, data_name), 'rb'))  # scipy.sparse.csr.csr_matrix <-- numpy.float32
        targ = pickle.load(open("{}{}_labels.pkl".format(data_dir, data_name), 'rb')).astype(np.int64)  # numpy.int64

        # Splitting the data ...
        trn_idx, val_idx, tst_idx = tvt_nids[0], tvt_nids[1], tvt_nids[2]
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0
        assert len(trn_idx) + len(val_idx) + len(tst_idx) == len(targ)

        # Building the graph ...
        adj_org.setdiag(0)
        graph = nx.from_scipy_sparse_matrix(adj_org)
        assert min(graph.nodes()) == 0
        n = graph.number_of_nodes()
        assert max(graph.nodes()) + 1 == n
        n = max(n, np.max(tst_idx) + 1)
        for u in range(n):
            graph.add_node(u)
        assert graph.number_of_nodes() == n
        assert not graph.is_directed()

        print_both(config, '#instance x #feature ~ #class = %d x %d ~ %d' % (feat.shape[0], feat.shape[1], targ.max() + 1), file=logfile)

        # Storing the data...
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.graph, self.feat, self.targ = graph, feat, targ

        # rand split
        self.rd_trn_idx = config.rd_trn_idx
        self.rd_val_idx = config.rd_val_idx
        self.rd_tst_idx = config.rd_tst_idx

    def get_split(self, is_rand_split=False):
        if is_rand_split:
            return self.rd_trn_idx, self.rd_val_idx, self.rd_tst_idx
        else:
            return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ

    def get_graph_info(self):
        adj_mat = nx.adjacency_matrix(self.graph).todense()
        node_size = adj_mat.shape[0]
        nb_size_list = np.sum(adj_mat, axis=1)
        return node_size, nb_size_list


class CN_DataReader:
    def __init__(self, data_name, data_dir, logfile, config):
        # Reading the data...
        tmp = []
        prefix = os.path.join(data_dir, 'ind.%s.' % data_name)
        for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(prefix + suffix, 'rb') as fin:
                tmp.append(pickle.load(fin, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = tmp
        with open(prefix + 'test.index') as fin:
            tst_idx = [int(i) for i in fin.read().split()]
        assert np.sum(x != allx[:x.shape[0], :]) == 0
        assert np.sum(y != ally[:y.shape[0], :]) == 0

        # split data standardly
        trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
        val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
        tst_idx = np.array(tst_idx, dtype=np.int64)
        assert len(trn_idx) == x.shape[0]
        assert len(trn_idx) + len(val_idx) == allx.shape[0]
        assert len(tst_idx) > 0
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        # build the graph with networkx
        graph = nx.from_dict_of_lists(graph)
        assert min(graph.nodes()) == 0
        n = graph.number_of_nodes()
        assert max(graph.nodes()) + 1 == n
        n = max(n, np.max(tst_idx) + 1)
        for u in range(n):
            graph.add_node(u)
        assert graph.number_of_nodes() == n
        assert not graph.is_directed()

        # build feat-mat and labels
        d, c = x.shape[1], y.shape[1]
        feat_ridx, feat_cidx, feat_data = [], [], []
        allx_coo = allx.tocoo()
        for i, j, v in zip(allx_coo.row, allx_coo.col, allx_coo.data):
            feat_ridx.append(i)
            feat_cidx.append(j)
            feat_data.append(v)
        tx_coo = tx.tocoo()
        for i, j, v in zip(tx_coo.row, tx_coo.col, tx_coo.data):
            feat_ridx.append(tst_idx[i])
            feat_cidx.append(j)
            feat_data.append(v)
        if data_name.startswith('nell.0'):
            isolated = np.sort(np.setdiff1d(range(allx.shape[0], n), tst_idx))
            for i, r in enumerate(isolated):
                feat_ridx.append(r)
                feat_cidx.append(d + i)
                feat_data.append(1)
            d += len(isolated)
        feat = spsprs.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
        targ = np.zeros((n, c), dtype=np.int64)
        targ[trn_idx, :] = y
        targ[val_idx, :] = ally[val_idx, :]
        targ[tst_idx, :] = ty
        targ = dict((i, j) for i, j in zip(*np.where(targ)))
        targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
        print_both(config, '#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c),
                   file=logfile)

        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx[:500], tst_idx
        assert len(self.val_idx) == 500
        self.graph, self.feat, self.targ = graph, feat, targ
        self.rd_trn_idx = config.rd_trn_idx
        self.rd_val_idx = config.rd_val_idx
        self.rd_tst_idx = config.rd_tst_idx

    def get_split(self, is_rand_split=False):
        if is_rand_split:
            return self.rd_trn_idx, self.rd_val_idx, self.rd_tst_idx
        else:
            return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ

    def get_graph_info(self):
        adj_mat = nx.adjacency_matrix(self.graph).todense()
        node_size = adj_mat.shape[0]
        nb_size_list = np.sum(adj_mat, axis=1)
        return node_size, nb_size_list


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))

        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        # pre-init
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias


class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False): # nbsize 50
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps): # 64, 4
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter): # x ([2708, 4, 16]), nbsize: 50, routit: 7
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0) # 2708, 50
        d, k, delta_d = self.d, self.k, self.d // self.k # d 64, k 4, delta_d 16
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d) # 2708,64
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d) # torch.Size([2708, 50, 4, 16])
        u = None
        for clus_iter in range(max_iter): # 7
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k) # 2708, 50, 4
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2) # 가장 높은 채
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d) # torch.Size([2708, 4, 16])
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d) # 2708, 64


class CapsuleNet(nn.Module):
    def __init__(self, nfeat, nclass, hyperpm, ncaps, nhidden, graph_type="knn"):
        super(CapsuleNet, self).__init__()
        ncaps, rep_dim = ncaps, nhidden * ncaps # 4, 'number of channels per layer.'
                                                # 16, 'Number of hidden units per capsule(channel).'
        self.pca = SparseInputLinear(nfeat, rep_dim) # 1433, 4*16
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass) # 64,7
        self.dropout = hyperpm.dropout # 0.0
        self.routit = hyperpm.routit # 7
        self.ncaps = ncaps # 4
        self.rep_dim = rep_dim
        self.nhidden = nhidden # 16
        self.latent_nnb_k = hyperpm.latent_nnb_k # 4

        self.graph_type = graph_type # knn

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        hidden_xs = []
        nb = nb.view(-1)
        x = fn.relu(self.pca(x))
        for idx_l, conv in enumerate(self.conv_ls):
            # nrm-agg with fixed graph
            x = conv(x, nb, self.routit).view(-1, self.ncaps, self.nhidden) # x284 torch.Size([2708, 4, 16])
            if idx_l == len(self.conv_ls) - 1:
                # gcn-agg with latent new-graphs
                hidden_xs.append(x.view(-1, self.rep_dim)) # append(torch.Size([2708, 64]) )
                hidden_disen_x = x.detach().clone() #  torch.Size([2708, 4, 16])
                result = []
                for idx_f in range(self.ncaps): # 4
                    cur_X = hidden_disen_x[:, idx_f, :] # cur_x torch.Size([2708, 16])
                    # knn
                    cur_adj = self.knn_graph(X=cur_X, k=self.latent_nnb_k) #  torch.Size([2708, 2708])
                    cur_output = self.gcn_agg(adj=cur_adj, X=x[:, idx_f, :]) # torch.Size([2708, 16])   # AX
                    result.append(cur_output)
                x = torch.cat(result, dim=-1) # x torch.Size([2708, 64])  # 오른쪽에 가져다 붙임
            x = self._dropout(fn.relu(x)) # x torch.Size([2708, 64])
        return fn.log_softmax(self.mlp(x), dim=1), x, hidden_xs  # label predict, knn으로 만든 adj를 돈거 오른쪽에 붙임, neigh routing을 한 torch.Size([2708, 4, 16])를 2708 64로 바꾼거
        # tensor([[-1.8232, -2.1656, -2.1564,  ..., -2.0916, -2.0104, -1.8053],
        #         [-1.9960, -2.2829, -2.0681,  ..., -1.7627, -1.9243, -1.8685]],
        #        device='cuda:0', grad_fn=<LogSoftmaxBackward0>)
        #        x tensor([[0.5562, 0.6346, 0.3894,  ..., 0.4025, 0.0979, 0.2502],
        #         [0.0990, 0.9306, 0.1899,  ..., 0.5527, 0.1108, 0.3026]]
        #        hidden [tensor([[0.5673, 0.6203, 0.3593,  ..., 0.3743, 0.0804, 0.1786],
        #         [0.0889, 0.8977, 0.2111,  ..., 0.5623, 0.1259, 0.3211]],


    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

    def knn_graph(self, X, k): # torch.Size([2708, 16]) k 4
        assert k < X.shape[0]
        D = self.pairwise_euclidean_distances(X, X)
        D.fill_diagonal_(0.0)
        D_low_k, _ = torch.topk(D, k=k, largest=False, dim=-1) #  torch.Size([2708, 4]), _
        D_ml_k, _ = torch.max(D_low_k, dim=-1) #  tensor([0.1698, 0.1980, 0.1579,  ..., 0.2147, 0.0891, 0.0891], device='cuda:0') torch.Size([2708]), _
        adj = (D - D_ml_k.unsqueeze(dim=-1) <= 0).float().fill_diagonal_(0.0)
        adj = (adj + adj.T) / 2.0
        adj.fill_diagonal_(1.0)
        return adj

    def cknn_graph(self, X, k, delta=1):
        assert k < X.shape[0]
        D = self.pairwise_euclidean_distances(X, X)
        D.fill_diagonal_(0.0)
        D_low_k, _ = torch.topk(D, k=k, largest=False, dim=-1)
        D_low_k = D_low_k[:, -1]
        adj = (D.square() < delta * delta * torch.matmul(D_low_k.view(-1, 1), D_low_k.view(1, -1))).float().fill_diagonal_(0.0)
        adj = (adj + adj.T) / 2.0
        adj.fill_diagonal_(1.0)
        return adj

    def pairwise_euclidean_distances(self, x, y):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf).sqrt()

    def gcn_agg(self, adj, X):
        adj_sp = self.adj_process(adj)
        output = torch.sparse.mm(adj_sp, X)
        return output

    def adj_process(self, adj):
        adj_shape = adj.size()
        adj_indices, adj_values = dense_to_sparse(adj)
        adj_values = self.row_normalize(adj_indices, adj_values, adj_shape)
        return torch.sparse.FloatTensor(adj_indices, adj_values, adj_shape)

    def row_col_normalize(self, adj_indices, adj_values, adj_shape):
        row, col = adj_indices
        deg = scatter_add(adj_values, row, dim=0, dim_size=adj_shape[0])
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_values = deg_inv_sqrt[row] * adj_values * deg_inv_sqrt[col]
        return adj_values

    def row_normalize(self, adj_indices, adj_values, adj_shape):
        row, _ = adj_indices
        deg = scatter_add(adj_values, row, dim=0, dim_size=adj_shape[0])
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_values = deg_inv_sqrt[row] * adj_values
        return adj_values

    def col_normalize(self, adj_indices, adj_values, adj_shape):
        _, col = adj_indices
        deg = scatter_add(adj_values, col, dim=0, dim_size=adj_shape[0])
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        adj_values = deg_inv_sqrt[col] * adj_values
        return adj_values


class EvalHelper:
    def __init__(self, dataset, config, hyperpm, logfile):
        self.config = config
        self.logfile = logfile
        # define data
        use_cuda = torch.cuda.is_available() and not config.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        graph, feat, targ = dataset.get_graph_feat_targ() # 2708, 1433, 7
        targ = torch.from_numpy(targ).to(dev)
        feat = thsprs_from_spsprs(feat).to(dev)

        # standard-split
        trn_idx, val_idx, tst_idx = dataset.get_split()
        print_both(config, "dataset-standard-split", file=self.logfile)

        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)

        ncaps = hyperpm.ncaps # 4 number of channels
        nhidden = hyperpm.nhidden  # 64//4
        graph_type = config.graph_type # knn

        model = CapsuleNet(nfeat, nclass, hyperpm, ncaps, nhidden, graph_type=graph_type).to(dev)

        optmz = optim.Adam(model.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)  # lr 0.05, reg 0.08

        small_covar_diags = 1e-15 * torch.eye(nhidden).float().repeat(ncaps, 1, 1).to(dev)  # offset for inverse-mat computation # torch.Size([4, 16, 16])
        disen_y = torch.arange(ncaps).long().unsqueeze(dim=0).repeat(feat.size(0), 1).flatten().to(dev)  # factor-labels # tensor([0, 1, 2,  ..., 1, 2, 3], device='cuda:0') torch.Size([10832])
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = model, optmz
        self.neib_sampler = NeibSampler(graph, config.nbsz).to(dev) # nighbor size 50
        self.space_lambda = hyperpm.space_lambda # 0.88
        self.div_lambda = hyperpm.div_lambda # 0.033
        self.gm_update_rate = hyperpm.gm_update_rate #  0.35
        self.ncaps = ncaps # 4
        self.nhidden = nhidden # 16
        self.small_covar_diags = small_covar_diags
        self.disen_y = disen_y
        self.epoch_step = 0

        self.det_offset = 1e-6
        self.log_offset = 1e-20

    def compute_mean_from_feat(self, feat, label, nclass):
        means = torch.zeros(nclass, feat.shape[1]).float().to(feat.device)
        for i in range(nclass):
            means[i] = feat.index_select(dim=0, index=torch.where(label == i)[0]).mean(dim=0)
        return means

    def compute_covmat_from_feat(self, feat, label, nclass):
        cov_mats = torch.zeros(nclass, feat.shape[1], feat.shape[1]).float().to(feat.device)
        for i in range(nclass):
            cur_index = torch.where(label == i)[0]
            cur_feat = feat.index_select(dim=0, index=cur_index)
            cur_dfeat = cur_feat - cur_feat.mean(dim=0, keepdim=True)
            cov_mats[i] = torch.mm(cur_dfeat.t(), cur_dfeat) / cur_index.shape[0]
        return cov_mats

    def initialize_mean_cov(self, input_feat):
        means = self.compute_mean_from_feat(input_feat.detach().clone().view(-1, self.nhidden), self.disen_y, self.ncaps)
        cov_mats = self.compute_covmat_from_feat(input_feat.detach().clone().view(-1, self.nhidden), self.disen_y, self.ncaps)
        return means, cov_mats

    def update_mean_cov(self, input_feat, means, cov_mats):
        # compute first, update later
        disen_x_detached = input_feat.view(-1, self.nhidden).detach().clone()
        new_means = means.new_zeros(size=means.size())
        new_cov_mats = cov_mats.new_zeros(size=cov_mats.size())
        for i in range(self.ncaps):
            # index for the current cap
            cur_index = torch.where(self.disen_y == i)[0]
            cur_feat = disen_x_detached.index_select(dim=0, index=cur_index)
            # the new means
            new_means[i] = cur_feat.mean(dim=0)
            # the new cov-mat
            de_cur_feat = cur_feat - new_means[i].unsqueeze(0)  # de-centralized with the new means
            new_cov_mats[i] = torch.mm(de_cur_feat.t(), de_cur_feat) / cur_index.shape[0]
        # (em estimation) update
        means += self.gm_update_rate * (new_means - means)
        cov_mats += self.gm_update_rate * (new_cov_mats - cov_mats)

    def run_epoch(self, config, end='\n'):
        self.epoch_step += 1
        # set training mode
        self.model.train()
        self.optmz.zero_grad()
        # build model
        prob, _, h_pred_feat_list = self.model(self.feat, self.neib_sampler.sample())
        # return fn.log_softmax(self.mlp(x), dim=1), x, hidden_xs  # label predict, knn으로 만든 adj를 돈거 오른쪽에 붙임, neigh routing을 한 torch.Size([2708, 4, 16])를 [2708, 64]로 바꾼거
        # torch.Size([2708, 7]), len(list) 1 list[0] torch.Size([2708, 64]) , torch.Size([2708, 64])

        # initialize the cov-mat using pred-feat in the first epoch
        if self.epoch_step == 1:
            self.h__means_list, self.h__covmats_list = [], []
            for idx, h_pred_feat in enumerate(h_pred_feat_list):
                tp__means, tp__covmats = self.initialize_mean_cov(h_pred_feat) # torch.Size([4, 16]),  torch.Size([4, 16, 16])
                self.h__means_list.append(tp__means)
                self.h__covmats_list.append(tp__covmats)
        else:
            for idx, h_pred_feat in enumerate(h_pred_feat_list):
                self.update_mean_cov(h_pred_feat, self.h__means_list[idx], self.h__covmats_list[idx])
        # build softmax-loss
        loss = fn.nll_loss(prob[self.trn_idx], self.targ[self.trn_idx])
        # build gm-reg-loss for every layer
        h__gm_reg_loss = 0.0
        h__div_reg_loss = 0.0
        for idx, (h_pred_feat, h__means, h__covmats) in enumerate(
                zip(h_pred_feat_list[::-1], self.h__means_list[::-1], self.h__covmats_list[::-1])): # list 뒤에서 부터 출력
            # h_pred torch.Size([2708, 64]), h_mean torch.Size([4, 16]), h_covmats torch.Size([4, 16, 16])
            # lik-reg-loss
            h__gm_reg_loss += (10 ** -idx) * self.compute_gm_reg_loss(
                x=h_pred_feat.view(-1, self.nhidden), y=self.disen_y, means=h__means, cov_mats=h__covmats) # (10 ** -idx) * tensor([0.1975, 0.1245,..., 0.1473], torch.Size([10832])

            # div-reg-loss
            h__div_reg_loss += (10 ** -idx) * self.compute_div_loss(
                disen_likeli=self.compute_gm_likeli_(disen_x=h_pred_feat.view(-1, self.nhidden), means=h__means, inv_covmats=h__covmats))
            # total loss
        total_loss = loss + self.space_lambda * h__gm_reg_loss + self.div_lambda * h__div_reg_loss
        # train
        total_loss.backward()
        self.optmz.step()
        # epoch-visualization
        print_both(config, "epoch-loss: {:.4f}, h-gm-reg-loss: ({:.4f}){:.4f}, h-div-reg-loss: ({:.4f}){:.4f}".format(
            loss.item(),
            self.space_lambda * h__gm_reg_loss.item(), h__gm_reg_loss.item(),
            self.div_lambda * h__div_reg_loss.item(), h__div_reg_loss.item()), file=self.logfile, end=end)
        return loss.item()

    def compute_inv_mat(self, input_mat):
        cov_mats = fn.normalize(input_mat, dim=2, p=2)  # 1st row normalization
        try:
            # inverse
            inv_cov_mats = torch.pinverse(
                cov_mats + self.small_covar_diags * cov_mats.reshape(cov_mats.size()[0], -1).mean(dim=1).unsqueeze(dim=1).unsqueeze(dim=2))
        except:
            inv_cov_mats = 1 / cov_mats.diagonal(dim1=-2, dim2=-1).diag_embed()
        inv_cov_mats = fn.normalize(inv_cov_mats, dim=2, p=2)  # 2nd row normalization
        return inv_cov_mats

    def compute_gm_term(self, disen_x, disen_y, disen_means, disen_cov_mats):# x torch.Size([10832, 16]), y tensor([0, 1, 2,  ..., 1, 2, 3] torch.Size([10832]),  mean torch.Size([4, 16]) torch.Size([4, 16, 16])
        batch_size = disen_x.size()[0]
        inv_cov_mats = self.compute_inv_mat(input_mat=disen_cov_mats)
        # get the batch samples
        means_batch = torch.index_select(disen_means, dim=0, index=disen_y) # torch.Size([10832, 16])
        invcovmat_bath = torch.index_select(inv_cov_mats, dim=0, index=disen_y)
        diff_batch = disen_x - means_batch
        gm_term_batch = torch.matmul(torch.matmul(diff_batch.view(batch_size, 1, -1), invcovmat_bath),
                                     diff_batch.view(batch_size, -1, 1)).squeeze()
        return gm_term_batch # tensor([0.1975, 0.1245,..., 0.1473], torch.Size([10832])

    def compute_gm_reg_loss(self, x, y, means, cov_mats): # x torch.Size([10832, 16]), y tensor([0, 1, 2,  ..., 1, 2, 3] torch.Size([10832]),  mean torch.Size([4, 16]) torch.Size([4, 16, 16])
        return self.compute_gm_term(x, y, means, cov_mats).mean()

    def compute_gm_likeli_(self, disen_x, means, inv_covmats): #  torch.Size([10832, 16]), m torch.Size([4, 16]), inv torch.Size([4, 16, 16])
        batch_diffs = disen_x.unsqueeze(dim=1) - means.unsqueeze(dim=0).repeat(disen_x.shape[0], 1, 1)
        batch_inv_covmatns = inv_covmats.unsqueeze(dim=0).repeat(disen_x.shape[0], 1, 1, 1)
        batch_gm_term = torch.bmm(torch.bmm(batch_diffs.view(-1, 1, self.nhidden), batch_inv_covmatns.view(-1, self.nhidden, self.nhidden)),
                                  batch_diffs.view(-1, self.nhidden, 1)).view(-1, self.ncaps)
        # remove inf caused by exp(89) --> inf
        z = -0.5 * batch_gm_term # torch.Size([10832, 4])
        z = (z.masked_fill(z > 80, 80)).exp()
        gm_likeli_ = fn.normalize(z, dim=-1, p=2)  # l2-norm
        return gm_likeli_ # torch.Size([10832, 4])

    def compute_div_loss(self, disen_likeli): # torch.Size([10832, 4])
        tmp = disen_likeli.view(-1, self.ncaps, self.ncaps) #  torch.Size([2708, 4, 4])
        mat = torch.bmm(tmp, tmp.transpose(dim0=1, dim1=2)) # torch.Size([2708, 4, 4])
        return (-torch.logdet(mat + self.det_offset * torch.eye(self.ncaps).to(mat.device).unsqueeze(dim=0).repeat(mat.shape[0], 1, 1))).mean() # tensor(9.9103, device='cuda:0', grad_fn=<MeanBackward0>)

    def print_trn_acc(self, end="\r\n"):
        trn_acc = self._calculate_acc(self.trn_idx)
        val_acc = self._calculate_acc(self.val_idx)
        print_both(self.config, "trn-acc={:.4f}%, val-acc={:.4f}%".format(trn_acc * 100, val_acc * 100), file=self.logfile, end=end)
        return val_acc

    def print_tst_acc(self, end="\r\n"):
        tst_acc = self._calculate_acc(self.tst_idx)
        print_both(self.config, "(tst) acc={:.4f}%".format(tst_acc * 100), file=self.logfile, end=end)
        return tst_acc

    def _calculate_acc(self, eval_idx):
        self.model.eval()
        prob, _, _ = self.model(self.feat, self.neib_sampler.nb_all)
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        # averaged acc
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        return acc

    def load_checkpoint(self, file, end="\r\n"):
        l_tm = time.time()
        ckp = torch.load(file)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.optmz.load_state_dict(ckp['optimizer_state_dict'])
        print_both(self.config, "load-tm: {:.4f}sec".format(time.time() - l_tm), file=self.logfile, end=end)

    def save_checkpoint(self, file, end="\r\n"):
        s_tm = time.time()
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optmz.state_dict()}, file)
        print_both(self.config, "save-tm: {:.4f}sec".format(time.time() - s_tm), file=self.logfile, end=end)


def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))


def dense_to_sparse(tensor):
    assert tensor.dim() == 2
    index = tensor.nonzero().t().contiguous()
    value = tensor[index[0], index[1]]
    return index, value


def scatter_add(values, index, dim, dim_size):
    output = torch.zeros(size=[dim_size, ]).to(values.device)
    return output.scatter_add(dim=dim, index=index, src=values)


def print_both(config, str, file=None, end="\r\n"):
    if not config.is_print:
        return 0
    print(str, file=sys.stderr, end=end)
    if file is not None:
        print(str, file=file, end=end)
        file.flush()


def create_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
        print("Allocating folder to {} ...".format(folder_dir))
    else:
        print("Folder: {} exists ...".format(folder_dir))


def clean_GPU_memory():
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()


def preparation(config, hyperpm):
    def log_setting(cfg, logfile):
        for p in dir(cfg):
            if not p.startswith("__") and not callable(getattr(cfg, p)):
                print_both(config, "{}: {}".format(p, getattr(cfg, p)), file=logfile)

    create_folder(config.modeldir)
    config.cur_mdir = "{}{}_eval/".format(config.modeldir, config.datname)
    create_folder(config.cur_mdir)
    logfile = open("{}logfile.txt".format(config.cur_mdir), "w+")
    print_both(config, "log hyper-params:", file=logfile)
    log_setting(cfg=config, logfile=logfile)
    log_setting(cfg=hyperpm, logfile=logfile)
    return logfile


# make sure each training in identical setting yields the identical result
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_eval(datadir, datname, config, hyperpm, logfile=None):
    # fix the seed for random initialization
    set_rng_seed(config.rnd_seed)
    # build model

    agent = EvalHelper(CN_DataReader(datname, datadir, logfile, config), config, hyperpm, logfile) # disGCN & neighbor sampler 정의

    best_val_acc, wait_cnt, best_epoch = 0.0, 0, 0
    best_model_sav = tempfile.TemporaryFile()
    neib_sav = torch.zeros_like(agent.neib_sampler.nb_all, device='cpu')
    # epoch training with early-stopping strategy
    for t in range(config.nepoch):
        print_both(config, "epoch: {}/{}".format(t + 1, config.nepoch), file=logfile, end=", ")
        agent.run_epoch(config, end=", ")
        cur_val_acc = agent.print_trn_acc(end=", ")
        if config.record_tst:
            agent.print_tst_acc(end=", ")
        # update the best
        if cur_val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = cur_val_acc
            best_model_sav.close()
            best_model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), best_model_sav)
            neib_sav.copy_(agent.neib_sampler.nb_all)
            best_epoch = t + 1
        else:
            # in case of stopping growth on val-acc
            wait_cnt += 1
            if wait_cnt > config.early:
                break
        # next-epoch
        print_both(config, "", file=logfile)
    # tst
    print_both(config, "load-tst-model ...", file=logfile, end="\r\n")
    best_model_sav.seek(0)
    agent.model.load_state_dict(torch.load(best_model_sav))
    # print final results
    print_both(config, "(val) acc={:.4f}%".format(best_val_acc * 100), file=logfile, end=", ")
    agent.neib_sampler.nb_all.copy_(neib_sav)
    final_tst_acc = agent.print_tst_acc()
    # bk the best model
    if config.is_sav_model:
        agent.save_checkpoint(file=config.cur_mdir + "best_model.pth", end=" ")
    return best_val_acc, final_tst_acc, best_epoch


def LGD(config, hyperpm):
    logfile = preparation(config, hyperpm)
    val_acc, tst_acc, epochs = train_eval(config.cur_ddir, config.datname, config, hyperpm, logfile)
    print_both(config, "val_acc={:.4f}%, tst_acc={:.4f}%, epochs={}".format(val_acc * 100, tst_acc * 100, epochs), file=logfile)
    if logfile is not None:
        logfile.close()
    return val_acc, tst_acc, epochs
