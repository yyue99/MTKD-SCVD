import torch.nn as nn
import torch


class CNNComponentPyTorch(nn.Module):
    def __init__(self):
        super(CNNComponentPyTorch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=300, out_channels=32, kernel_size=1, padding='same')
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)  # Assuming dropout value as 0.5

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation1(x)
        x = self.pool2(x)
        x = self.flatten(x)

        return x


class BiGRUComponentPyTorch(nn.Module):
    def __init__(self, input_size, time_steps, dropout=0.5):
        super(BiGRUComponentPyTorch, self).__init__()
        self.bigru = nn.GRU(input_size=input_size, hidden_size=300, num_layers=1, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(600, 300)  # 600 because of bidirectional

    def forward(self, x):
        x, _ = self.bigru(x)
        x = self.relu(x[:, -1, :])  # Using the last sequence output
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class CBGRU(nn.Module):
    def __init__(self, input_size, time_steps, num_classes):
        super(CBGRU, self).__init__()
        self.cnn = CNNComponentPyTorch()
        self.bigru = BiGRUComponentPyTorch(input_size, time_steps)

        # Fully connected layers
        self.fc1 = nn.Linear(1600 + 300, 300)  # 120000 from CNN, 300 from BiGRU
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, num_classes)  # Assuming num_classes is defined somewhere
        self.relu = nn.ReLU()

    def forward(self, x_cnn, x_bigru):
        x_cnn = self.cnn(x_cnn)
        x_bigru = self.bigru(x_bigru)

        x = torch.cat((x_cnn, x_bigru), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        feature = x
        x = self.relu(x)
        x = self.fc3(x)

        return feature, x
