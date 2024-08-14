import torch

def merge_Graph(graph_list):
    prefix_sum = []
    node_features = []
    node_degs = []
    node_labels = []
    total_num_edges = 0
    total_num_nodes = 0
    edge_pairs = []
    graph_sizes = []
    for i in range(len(graph_list)):
        prefix_sum.append(graph_list[i]['num_nodes'])
        if i != 0:
            prefix_sum[i] += prefix_sum[i - 1]
        node_features.extend(graph_list[i]['features'])
        node_degs.extend(graph_list[i]['node_degs'])
        node_labels.append(graph_list[i]['label'])
        total_num_edges += len(graph_list[i]['edge_pairs'])
        total_num_nodes += graph_list[i]['num_nodes']
        graph_sizes.append(graph_list[i]['num_nodes'])
        edge_pairs.append(graph_list[i]['edge_pairs'])
    # create batch_graph
    n2n_idxes = torch.LongTensor(2, total_num_edges)
    n2n_vals = torch.FloatTensor(total_num_edges)

    for i in range(len(graph_list)):
        prefix_sum[len(graph_list) - i - 1] = prefix_sum[len(graph_list) - i - 2]
    prefix_sum[0] = 0

    for i in range(total_num_edges):
        n2n_vals[i] = 1

    j = 0
    for i in range(len(graph_list)):
        for item in edge_pairs[i]:
            n2n_idxes[0][j] = item[0] + prefix_sum[i]
            n2n_idxes[1][j] = item[1] + prefix_sum[i]

            #             if item[0]+prefix_sum[i] > total_num_nodes:
            #                 print('item0',item[0],prefix_sum[i],total_num_nodes)
            #             if item[1]+prefix_sum[i] > total_num_nodes:
            #                 print('item1',item[1],prefix_sum[i],total_num_nodes)

            #             if item[0]+prefix_sum[i] < 0:
            #                 print('item0',item[0],prefix_sum[i],'position: ',i)
            #             if item[1]+prefix_sum[i] < 0 :
            #                 print('item1',item[1],prefix_sum[i],'position: ',i)

            j += 1
    #     print(j,total_num_edges)
    #     print(n2n_idxes[:,3000:])
    #     print(node_features)
    n2n = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([total_num_nodes, total_num_nodes]))
    node_features = torch.FloatTensor(node_features)
    node_degs = 1 / torch.LongTensor(node_degs)
    degs_index = torch.LongTensor(2, total_num_nodes)

    for i in range(total_num_nodes):
        degs_index[0, i] = i
        degs_index[1, i] = i
    node_degs = torch.sparse.FloatTensor(degs_index, node_degs, torch.Size([total_num_nodes, total_num_nodes]))

    return n2n, node_features, node_degs, graph_sizes