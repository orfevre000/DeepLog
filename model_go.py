import torch
import torch.jit
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import torchvision

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


num_classes = 30
num_epochs = 30
batch_size = 2048
input_size = 1
model_dir = 'model'
log = 'Adam_batch_size={}_epoch={}'.format(str(batch_size), str(num_epochs))
parser = argparse.ArgumentParser()
parser.add_argument('-num_layers', default=2, type=int)
parser.add_argument('-hidden_size', default=64, type=int)
parser.add_argument('-window_size', default=10, type=int)
args = parser.parse_args()
num_layers = args.num_layers
hidden_size = args.hidden_size
window_size = args.window_size

model = Model(input_size, hidden_size, num_layers, num_classes)
# 学習済みのPyTorchモデルを読み込む（ここでは例としてResNetを使用）
model = torch.load('model/Adam_batch_size=2048_epoch=30.pt')  # モデルのファイルパスを指定してください

# TorchScriptにコンパイル
jit_model = torch.jit.script(model)


# モデルをトレース
#example_input = torch.randn(1, 3, 224, 224)  # ダミーの入力を生成
#traced_model = torch.jit.trace(torchvision.models.resnet18(), example_input)

# トレースされたモデルをファイルに保存
#jit_model.save('traced_model.pt')
torch.jit.save(jit_model , 'traced_model')