import json
import pickle
from collections import defaultdict
import random
from imblearn.over_sampling import RandomOverSampler
import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from dataset.collator import collate




class GraphDataset(Dataset):
    def __init__(self, dgraphs, labels, idx, graphs, w2v_embeddings, edges, ft_embeddings):
        super(GraphDataset, self).__init__()
        self.dgraphs = dgraphs
        self.labels = labels
        self.idx = idx
        self.graphs = graphs
        self.w2v_embeddings = w2v_embeddings
        self.edges = edges
        self.ft_embeddings = ft_embeddings


    def __getitem__(self, idx):
        return self.dgraphs[idx], self.labels[idx], self.idx[idx], self.graphs[idx], self.w2v_embeddings[
            idx], self.edges[idx], self.ft_embeddings[idx]

    def __len__(self):
        return len(self.dgraphs)

def load_data(files):
    with open(files[0], 'rb') as file:
        data = pickle.load(file)
    w2v_embeddings_dict = {int(data['idx'][i]): data['vector'][i] for i in range(len(data['idx']))}

    with open(files[1], 'rb') as file:
        data = pickle.load(file)
    ft_embeddings_dict = {int(data['idx'][i]): data['vector'][i] for i in range(len(data['idx']))}

    data_list = []
    with open(files[2], 'r') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    combined_data = []

    for data in data_list:
        graph_data = data[0]['graph']
        label = data[0]['label']
        idx = int(data[0]['idx'])
        edge = data[0]['edge']
        edge_pairs = torch.tensor(graph_data['edge_pairs'])
        features = torch.tensor(graph_data['features'])

        src_nodes = edge_pairs[:, 0]
        dst_nodes = edge_pairs[:, 1]
        g = dgl.graph((src_nodes, dst_nodes))
        g.ndata['feat'] = features

        w2v_embedded_data = w2v_embeddings_dict.get(idx)
        ft_embedded_data = ft_embeddings_dict.get(idx)
        if w2v_embedded_data is not None and ft_embedded_data is not None:
            combined_data.append({
                'idx': idx, 'label': label, 'dgraph': g, 'graph': graph_data,
                'w2v_embedding': w2v_embedded_data, 'ft_embedding': ft_embedded_data,
                'edge': edge
            })
        else:
            combined_data.append({'idx': idx, 'graph': graph_data, 'dgraph': g, 'embedding': None})

    dataset = GraphDataset(
        [data['dgraph'] for data in combined_data],
        [data['label'] for data in combined_data],
        [data['idx'] for data in combined_data],
        [data['graph'] for data in combined_data],
        [data['w2v_embedding'] for data in combined_data],
        [data['edge'] for data in combined_data],
        [data['ft_embedding'] for data in combined_data]
    )


    data = {
        'dg': [data[0] for data in dataset],
        'labels': [data[1] for data in dataset],
        'idxo': [data[2] for data in dataset],
        'go': [data[3] for data in dataset],
        'w2vembo': [data[4] for data in dataset],
        'edgeo': [data[5] for data in dataset],
        'ftembo': [data[6] for data in dataset]
    }

    df = pd.DataFrame(data)

    train_df, temp_df = train_test_split(df, test_size=0.8, random_state=42, stratify=df['labels'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['labels'])

    oversampler = RandomOverSampler(random_state=42)

    train_features_over, train_labels_over = oversampler.fit_resample(train_df.drop('labels', axis=1), train_df['labels'])
    val_features_over, val_labels_over = oversampler.fit_resample(val_df.drop('labels', axis=1), val_df['labels'])
    test_features_over, test_labels_over = oversampler.fit_resample(test_df.drop('labels', axis=1), test_df['labels'])

    train_df_over = pd.DataFrame(train_features_over, columns=train_df.drop('labels', axis=1).columns)
    train_df_over['labels'] = train_labels_over

    val_df_over = pd.DataFrame(val_features_over, columns=val_df.drop('labels', axis=1).columns)
    val_df_over['labels'] = val_labels_over

    test_df_over = pd.DataFrame(test_features_over, columns=test_df.drop('labels', axis=1).columns)
    test_df_over['labels'] = test_labels_over

    train_dataset = GraphDataset(*zip(*train_df_over.values.tolist()))
    val_dataset = GraphDataset(*zip(*val_df_over.values.tolist()))
    test_dataset = GraphDataset(*zip(*test_df_over.values.tolist()))

    # # 创建 RandomUnderSampler 实例，专门用于对label为0的样本进行欠采样
    # undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    #
    # # 分别对训练集、验证集和测试集进行欠采样
    # train_features_under, train_labels_under = undersampler.fit_resample(train_df.drop('labels', axis=1), train_df['labels'])
    # val_features_under, val_labels_under = undersampler.fit_resample(val_df.drop('labels', axis=1), val_df['labels'])
    # test_features_under, test_labels_under = undersampler.fit_resample(test_df.drop('labels', axis=1), test_df['labels'])
    #
    # # 创建欠采样后的 DataFrame
    # train_df_under = pd.DataFrame(train_features_under, columns=train_df.drop('labels', axis=1).columns)
    # train_df_under['labels'] = train_labels_under
    #
    # val_df_under = pd.DataFrame(val_features_under, columns=val_df.drop('labels', axis=1).columns)
    # val_df_under['labels'] = val_labels_under
    #
    # test_df_under = pd.DataFrame(test_features_under, columns=test_df.drop('labels', axis=1).columns)
    # test_df_under['labels'] = test_labels_under
    #
    # train_dataset = GraphDataset(
    #     dgraphs=train_df['dg'].tolist(),
    #     labels=train_df['labels'].tolist(),
    #     idx=train_df['idxo'].tolist(),
    #     graphs=train_df['go'].tolist(),
    #     w2v_embeddings=train_df['w2vembo'].tolist(),
    #     edges=train_df['edgeo'].tolist(),
    #     ft_embeddings=train_df['ftembo'].tolist()
    # )
    # # val_dataset = GraphDataset(*zip(*val_df.values.tolist()))
    # # test_dataset = GraphDataset(*zip(*test_df.values.tolist()))


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate)

    return train_loader, test_loader, val_loader