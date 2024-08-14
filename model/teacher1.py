import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, activation = None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features, adj, degs):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        output = torch.mm(degs,output)
        if self.activation != None:
            output = self.activation(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Edge_Attention(nn.Module):
    def __init__(self, head, feature_length):
        super(Edge_Attention, self).__init__()
        self.head = head
        self.feature_length = feature_length
        self.Attention = nn.TransformerEncoderLayer(d_model=feature_length, nhead=head, batch_first=True)

    def forward(self, edges):
        return self.Attention(edges)


class res_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, n_layers, k):
        super(res_GCN, self).__init__()

        self.k = k
        self.n_layers = n_layers
        self.nhid = nhid
        self.nclass = nclass
        self.layers = nn.ModuleList()
        self.total_latent_dim = nhid * n_layers + nclass
        # 输入层
        self.layers.append(GraphConvolution(nfeat, nhid, activation=torch.tanh))
        # 隐层
        for i in range(n_layers - 1):
            self.layers.append(GraphConvolution(nhid, nhid, activation=torch.tanh))
        # 输出层
        self.layers.append(GraphConvolution(nhid, nclass))

    def forward(self, features, graphs, degs, graph_sizes):

        # def a res GCNN here
        feature_list = []
        for i, layer in enumerate(self.layers):
            if i != 0 and i != len(self.layers) - 1:
                features = layer(features, graphs, degs) + features
                feature_list.append(features)
            else:
                features = layer(features, graphs, degs)
                feature_list.append(features)
        #         sort pooling with k row remains

        res = ''
        for i, item in enumerate(feature_list):
            if i == 0:
                res = item
            else:
                res = torch.cat((res, item), 1)

        #         res = features

        #         print(len(feature_list),res.shape)
        sort_channel = res[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(features.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(len(graph_sizes)):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = res.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k - k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(features.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]

        return batch_sortpooling_graphs


class Classifier(nn.Module):
    def __init__(self, classNum, dropout_rate, nfeat, nhid, nclass, n_layers, k, head, features_length):
        super(Classifier, self).__init__()
        self.classNum = classNum
        self.dropout_rate = dropout_rate
        self.resGCNN = res_GCN(nfeat, nhid, nclass, n_layers, k)
        self.edgeAttention = Edge_Attention(head, features_length)

        #         self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm(features_length)
        self.cov1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=nhid * n_layers + nclass,
                              stride=nhid * n_layers + nclass)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=self.dropout_rate)
        self.cov2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        dense = int(k / 2 / 4)

        self.denseLayer1 = nn.Linear(dense * 32, 512)
        self.dropout3 = nn.Dropout(p=self.dropout_rate)
        self.denseLayer2 = nn.Linear(300, 128)
        self.dropout4 = nn.Dropout(p=self.dropout_rate)
        self.denseLayer3 = nn.Linear(512, 300)
        self.dropout5 = nn.Dropout(p=self.dropout_rate)
        self.outputLayer = nn.Linear(128, classNum)

    def forward(self, features, graphs, degs, graph_sizes, edges):
        gcn = self.resGCNN(features, graphs, degs, graph_sizes)
        #         gcn = self.layer_norm1(gcn)
        e_attention = self.edgeAttention(edges)
        e_attention = self.layer_norm2(e_attention)
        #         print(gcn.shape,e_attention.shape)

        gcn = gcn.view(len(graph_sizes), 1, -1)
        e_attention = e_attention.view(len(graph_sizes), 1, -1)

        #         print(,gcn.shape,e_attention.shape)
        res = torch.cat((gcn, e_attention), 1)

        #         res = res.view(len(graph_sizes),1,-1)
        #        1d convolution layer
        res = F.relu(self.cov1(res))
        res = self.dropout1(res)
        res = self.maxpool(res)
        res = F.relu(self.cov2(res))
        res = F.relu(res)
        res = self.dropout2(res)
        #         Dense Layer
        res = res.view(len(graph_sizes), -1)
        res = F.relu(self.denseLayer1(res))
        res = self.dropout3(res)
        features = self.denseLayer3(res)
        res = F.relu(features)
        res = self.dropout5(features)
        res = F.relu(self.denseLayer2(res))
        res = self.dropout4(res)
        output = self.outputLayer(res)
        return features, output.flatten()

