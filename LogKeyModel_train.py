import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
window_size = 10      #滑动窗口大小，序列长度
input_size = 1        #输入矩阵特征数
hidden_size = 64      #输出矩阵特征数
num_layers = 2        #LSTM层数
num_classes = 28      #日志事件类型个数
num_epochs = 300      #指定概率分布的前几个日志事件是正常的
batch_size = 2048
log = 'Adam with batch_size=2048;epoch=300'


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open('data/' + name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    print(inputs)
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
seq_dataset = generate('hdfs_train')
dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
writer = SummaryWriter(log_dir='log/' + log)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
total_step = len(dataloader)
print(total_step)
for epoch in range(num_epochs):  # Loop over the dataset multiple times
    train_loss = 0
    for step, (seq, label) in enumerate(dataloader):
        # Forward pass
        # print(step)
        seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
        output = model(seq)
        loss = criterion(output, label.to(device))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch [{}/{}], Train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(dataloader.dataset)))
    writer.add_scalar('train_loss', train_loss / len(dataloader.dataset), epoch + 1)
torch.save(model.state_dict(), 'model/' + log + '.pt')
writer.close()
print('Finished Training')
