import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
window_size = 6
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 48
num_epochs = 100
batch_size = 2048
log = 'No session;Adam with batch_size=2048;epoch=100;classes=48;window=6'


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    seq = []
    with open(name, 'r') as f:
        for line in f.readlines():
            seq.append(int(line.strip()))
        for i in range(len(seq) - window_size):
            inputs.append(tuple(seq[i:i + window_size]))
            outputs.append(seq[i + window_size])

    print('Number of seqs({}): {}'.format(name, len(inputs)))
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
seq_dataset = generate('E:/logData/test-FP/bd-33-41/train_normal_id.txt')
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
