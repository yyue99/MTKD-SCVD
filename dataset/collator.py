# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import dgl


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_1d_unsqueeze_nan(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype).float()
        new_x[:] = float('nan')
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.edge_feature,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.edge_attr,
            item.edge_index,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        edge_attrs,
        edge_indexes,
        xs,
        edge_inputs,
        ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    if ys[0].size(0) == 1:
        y = torch.cat(ys)
    else:
        try:
            max_edge_num = max([y.size(0) for y in ys])
            y = torch.cat([pad_1d_unsqueeze_nan(i, max_edge_num) for i in ys])
        except:
            y = torch.cat([pad_1d_unsqueeze_nan(i, max_node_num) for i in ys])

    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    # max_edge_num = max([edge_attr.shape[0] for edge_attr in edge_attrs])
    # consider the god node
    # edge_index = torch.cat([pad_2d_unsqueeze(i.transpose(-1, -2), max_edge_num) for i in edge_indexes])
    # edge_index = edge_index.transpose(-1, -2)
    return dict(
        idx=torch.LongTensor(idxs),
        edge_index=edge_indexes,
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        # edge_index=torch.LongTensor(edge_index),
        x=x.long(),
        edge_input=edge_input,
        y=y,
    )


def collate(samples):
    # `samples`是一个列表，包含了一批图和标签的元组
    graphs, labels, idxs, graph_list, w2v_embedding, edges, ft_embedding = map(list, zip(*samples))
    # 使用`dgl.batch`将一批图打包成一个大图
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), idxs, graph_list, w2v_embedding, edges, ft_embedding


def collate_teacher(samples):
    # `samples`是一个列表，包含了一批图和标签的元组
    (graphs, labels, idxs, graph_list, edges) = map(list, zip(*samples))
    # 使用`dgl.batch`将一批图打包成一个大图
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), idxs, graph_list, edges


def collate_case(samples):
    # `samples`是一个列表，包含了一批图和标签的元组
    graphs, idxs, graph_list, w2v_embedding, edges, ft_embedding, labels = map(list, zip(*samples))
    # 使用`dgl.batch`将一批图打包成一个大图
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), idxs, graph_list, w2v_embedding, edges, ft_embedding