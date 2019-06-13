import torch
import torch.nn as nn
import time
import json

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Hyperparameters
window_size = 6      #滑动窗口大小，序列长度
input_size = 1
hidden_size = 64
num_layers = 2
num_classes = 48     #日志事件类型个数
num_candidates = 9   #指定概率分布的前几个日志事件是正常的
model_path = 'model/No session;Adam with batch_size=2048;epoch=100;classes=48;window=6.pt'


def generate(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    # hdfs = set()
    hdfs = []
    seq = []
    with open(name, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            seq.append(int(data['eventId']))
        hdfs.append(tuple(seq))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


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


labels = []
with open('E:/logData/test-with label/bd33-40/sequence_20190528.log', 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        labels.append(data['severity_label'])

print(torch.cuda.is_available())
model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print('model_path: {}'.format(model_path))
test_normal_loader = generate('E:/logData/test-with label/bd33-40/sequence_20190528.log')
#test_abnormal_loader = generate('E:/logData/abnormal-authlog/test_abnormal_id.txt')
TP = 0
FP = 0
FN = 0
TN = 0
P = 0
R = 0
F1 = 0
# Test the model
# print(len(test_abnormal_loader))
cnt_normal = 0
cnt_abnormal = 0
start_time = time.time()
with torch.no_grad():
    for line in test_normal_loader:
        i = 0
        while i < len(line) - window_size:
            #print(i)
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            if labels[i + window_size] != 'Informational':
                if label not in predicted:
                    #i += window_size
                    TP += 1
                cnt_abnormal += 1
            else:
                if label not in predicted:
                    # i += window_size
                    print(i + window_size + 1)
                    FP += 1
                cnt_normal += 1
            i += 1


'''
with torch.no_grad():
    cnt = 0
    for line in test_abnormal_loader:
        cnt += 1
        for i in range(len(line) - window_size):
            print(i)
            cnt += 1
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            if label not in predicted:
                TP += 1
'''
# Compute precision, recall and F1-measure
# FN = len(test_abnormal_loader) - TP

FN = cnt_abnormal - TP
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print(
    'normal: {}, abnormal: {}, '
    'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
        cnt_normal,
        cnt_abnormal,
        FP,
        FN,
        P,
        R,
        F1))

print('Finished Predicting')
elapsed_time = time.time() - start_time
print('elapsed_time: {}'.format(elapsed_time))
