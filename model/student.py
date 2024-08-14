import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import dgl.nn as dglnn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        self.pool = dglnn.glob.MaxPooling()
        self.classify = nn.Linear(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        hg = self.pool(g, h)  # 在这里，DGL会自动处理每个子图
        return self.classify(hg)

class GCNs(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCNs, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        self.pool = dglnn.glob.MaxPooling()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        hg = self.pool(g, h)  # 在这里，DGL会自动处理每个子图
        return hg


class BLSTM(nn.Module):
    def __init__(self, input_size, time_steps, hidden_size=300, num_class=1, dropout=0.5):
        super(BLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.time_steps = time_steps

        # Define the bidirectional LSTM layer
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Define other layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 300)  # The *2 accounts for bidirectionality
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, num_class)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectionality
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate BLSTM
        out, _ = self.blstm(x, (h0, c0))

        # Process output through fully connected layers
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])  # Consider only the last time step
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

class BLSTMs(nn.Module):
    def __init__(self, input_size, time_steps, hidden_size=300, dropout=0.5):
        super(BLSTMs, self).__init__()

        self.hidden_size = hidden_size
        self.time_steps = time_steps

        # Define the bidirectional LSTM layer
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Define other layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 300)  # The *2 accounts for bidirectionality
        self.fc2 = nn.Linear(300, 300)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectionality
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate BLSTM
        out, _ = self.blstm(x, (h0, c0))

        # Process output through fully connected layers
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])  # Consider only the last time step
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_class):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        out, _ = self.gru(x)
        out = self.relu(out[:, -1, :])  # 取最后一个时间步的输出
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes, dropout):
        super(Classifier, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes
        ])

        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Reshape x to [batch_size, 1, 1, embedding_dim]
        x = x.unsqueeze(1).unsqueeze(2)

        # Apply convolutions and pooling
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(xi, xi.size(2)).squeeze(2) for xi in x]

        # Concatenate and apply dropout
        x = torch.cat(x, dim=1)
        feature_maps = x
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        return feature_maps, x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 300)  # Updated size due to bidirectional output
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, num_classes)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectionality
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)

        # Assuming x shape: [batch_size, seq_len, input_size]
        lstm_output, _ = self.lstm(x, (h0, c0))

        out = self.relu(lstm_output)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

class Student(nn.Module):
    def __init__(self, input_size, time_steps, hidden_size=300, dropout=0.5):
        super(Student, self).__init__()
        self.lstm = BLSTMs(input_size, time_steps)
        self.gcn = GCNs(100, hidden_size)
        self.act = nn.ReLU()
        self.cl = Classifier(600, 100, [1], 1, dropout)
        self.w = nn.Parameter(torch.ones(3))



    def forward(self, emb, g, feat):
        seq_out = self.lstm(emb)
        g_out = self.gcn(g, feat)
        merged_feature = torch.cat([seq_out,g_out], dim=1)
        feature_maps, out = self.cl(merged_feature)
        return seq_out, g_out, feature_maps, out


