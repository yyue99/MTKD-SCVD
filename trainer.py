import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from dataset.funcs import merge_Graph


def train(loader_t, loader_v, epochs, model, teacher_g=None, teacher_s=None, student=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.train()
    teacher_g.to(device)
    teacher_g.eval()
    teacher_s.to(device)
    teacher_s.eval()
    student.to(device)
    student.eval()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epoch_loss = 0
    num_batches = 0
    best_f1 = 0


    for epoch in range(epochs):
        model.train()

        with tqdm(total=len(loader_t), desc=f"Epoch {epoch}") as pbar:
            for batched_graph, batched_label, _, batched_g, w2v_embedding, batched_edge, ft_embedding in loader_t:
                w2v_embedding = torch.tensor(w2v_embedding, dtype=torch.float32)
                w2v_embedding = w2v_embedding.to(device)
                ft_embedding = torch.tensor(ft_embedding, dtype=torch.float32)
                ft_embedding = ft_embedding.to(device)
                with torch.no_grad():  # 不需要计算梯度
                    # 应用softmax以确保权重和为1
                    weights = torch.softmax(model.w, dim=0)
                    graphs, features, node_degs, graph_sizes = merge_Graph(batched_g)
                    graphs = graphs.cuda()
                    # labels = torch.FloatTensor(labels).cuda()
                    node_degs = node_degs.cuda()
                    features = features.cuda()

                    graphs = Variable(graphs)
                    node_degs = Variable(node_degs)
                    features = Variable(features)
                    batched_edge = torch.tensor([batched_edge], dtype=torch.float32)
                    edges = Variable(batched_edge)
                    edges = edges.squeeze(2).squeeze(0)
                    edges = edges.to(device)
                    feat_g, logits_g = teacher_g(features, graphs, node_degs, graph_sizes, edges)
                    feat_s, logits_s = teacher_s(w2v_embedding, ft_embedding)



                batched_graph = batched_graph.to(device)
                batched_label = batched_label.to(device)

                with torch.no_grad():
                    features = batched_graph.ndata['feat'].to(device)
                    _, _, _, logits_stu = student(w2v_embedding, batched_graph, features)
                logits_stu = logits_stu.squeeze(1)
                logits_stu = F.sigmoid(logits_stu)
                # out = model(batched_embedding).squeeze(1)
                s_out, g_out, _, out = model(w2v_embedding, batched_graph, features)
                out = out.squeeze(1)
                logits = F.sigmoid(out)
                logits_g = logits_g
                logits_g = F.sigmoid(logits_g)
                logits_s = logits_s.squeeze(1)
                logits_s = F.sigmoid(logits_s)
                loss_logits = F.binary_cross_entropy_with_logits(logits, logits_g) + F.binary_cross_entropy_with_logits(logits, logits_s)
                loss_pred = F.binary_cross_entropy_with_logits(logits, batched_label)
                loss_g = F.mse_loss(g_out, feat_g)
                loss_s = F.mse_loss(s_out, feat_s)
                loss_stu = F.binary_cross_entropy_with_logits(logits, logits_stu)
                # loss = 0.1 * loss_g + loss_pred + 0.1 * loss_s
                # loss = 0.1 * loss_g + loss_pred + 0.1 * loss_stu + 0.1 * loss_s + 0.2 * loss_logits
                loss = loss_pred + 0.1 * loss_stu + 0.2 * loss_logits
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                # 更新进度条
                pbar.update(1)
        avg_loss = epoch_loss / num_batches
        print('Epoch %d | Average Loss: %.4f ' % (epoch, avg_loss))
        if (epoch + 1) % 20 == 0:
            # 验证逻辑
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():  # 不需要计算梯度
                tp = tn = fp = fn = 0
                val_loss = 0
                val_batches = 0
                fp_indices = []
                fn_indices = []
                for val_graph, val_label, idx, _, val_embedding, _, _ in loader_v:
                    val_embedding = torch.tensor(val_embedding, dtype=torch.float32)
                    val_embedding = val_embedding.to(device)
                    val_label = val_label.to(device)
                    val_graph = val_graph.to(device)
                    val_features = val_graph.ndata['feat']
                    _, _, _, val_out = model(val_embedding, val_graph, val_features)
                    val_out = val_out.squeeze(1)
                    val_logits = F.sigmoid(val_out)
                    loss = F.binary_cross_entropy_with_logits(val_logits, val_label)
                    val_loss += loss.item()
                    val_batches += 1
                    # 计算FP和FN的索引
                    predictions = torch.round(val_logits)
                    fp_mask = (predictions == 1) & (val_label == 0)
                    fn_mask = (predictions == 0) & (val_label == 1)
                    batch_fp_indices = [idx[i] for i in range(len(idx)) if fp_mask[i]]
                    batch_fn_indices = [idx[i] for i in range(len(idx)) if fn_mask[i]]
                    fp_indices.extend(batch_fp_indices)
                    fn_indices.extend(batch_fn_indices)

                    # 计算TP, TN, FP, FN
                    tp += ((predictions == 1) & (val_label == 1)).sum().item()
                    tn += ((predictions == 0) & (val_label == 0)).sum().item()
                    fp += ((predictions == 1) & (val_label == 0)).sum().item()
                    fn += ((predictions == 0) & (val_label == 1)).sum().item()

                if tp == 0:
                    tp = 0 + 1
                    print('tp == 0 + 1')
                avg_val_loss = val_loss / val_batches
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                acc = (tp + tn) / (tp + tn + fp + fn)
                f1 = (2 * precision * recall) / (precision + recall)
                print('Validation Loss: %.4f' % (avg_val_loss))
                print('TP: %d, TN: %d, FP: %d, FN: %d' % (tp, tn, fp, fn))
                print('acc: %.4f | recall: %.4f | precision: %.4f | f1: %.4f' % (acc, recall, precision, f1))
                if f1 > best_f1:
                    best_f1 = f1
                    filename = './model/ar/student/logits/ar_bestmodel_f1_{:.4f}_re_{:.4f}_pre_{:.4f}.pth'.format(f1,
                                                                                                                recall,
                                                                                                                precision)
                    torch.save(model.state_dict(), filename)
                    print('New best model saved with filename: %s' % (filename))