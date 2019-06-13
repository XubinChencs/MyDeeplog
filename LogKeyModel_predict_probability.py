import torch
import torch.nn as nn

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Hyperparameters
window_size = 10
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 28
num_candidates = 9
model_path = 'model/No session;Adam with batch_size=2048;epoch=300.pt'


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


model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

seq = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
output = model(seq)
predicted = torch.argsort(output, 1)[0][-num_candidates:]
print(predicted)