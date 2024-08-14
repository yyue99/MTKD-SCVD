import torch
import torch.nn.functional as F


def test(model, test_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()  # 设置模型为评估模式
    tp = tn = fp = fn = 0
    test_loss = 0
    test_batches = 0
    with torch.no_grad():  # 不需要计算梯度
        for test_graph, test_label, _, _, test_embedding, _, _ in test_loader:
            test_embedding = torch.tensor(test_embedding, dtype=torch.float32)
            test_embedding = test_embedding.to(device)
            test_label = test_label.to(device)
            test_graph = test_graph.to(device)
            test_features = test_graph.ndata['feat']
            _, _, test_out = model(test_embedding, test_graph, test_features)
            test_out = test_out.squeeze(1)
            test_logits = F.sigmoid(test_out)
            loss = F.binary_cross_entropy_with_logits(test_logits, test_label)
            test_loss += loss.item()
            test_batches += 1
            # 计算预测值
            predictions = torch.round(test_logits)
            # 计算TP, TN, FP, FN
            tp += ((predictions == 1) & (test_label == 1)).sum().item()
            tn += ((predictions == 0) & (test_label == 0)).sum().item()
            fp += ((predictions == 1) & (test_label == 0)).sum().item()
            fn += ((predictions == 0) & (test_label == 1)).sum().item()

    avg_test_loss = test_loss / test_batches
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Test Loss: %.4f' % (avg_test_loss))
    print('TP: %d, TN: %d, FP: %d, FN: %d' % (tp, tn, fp, fn))
    print('acc: %.4f | recall: %.4f | precision: %.4f | f1: %.4f' % (acc, recall, precision, f1))