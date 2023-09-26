import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from pprint import pprint
import copy


device = torch.device("cpu")


def average_models(models):
    # 各モデルの重みを取得
    all_weights = [model.state_dict() for model in models]


    # 重みの平均値を計算
    averaged_weights = {}
    for param_name in all_weights[0].keys():
        param_sum = sum(weight[param_name] for weight in all_weights)
        averaged_weights[param_name] = param_sum / len(models)

    # 新しいモデルを作成し、平均化された重みを設定
    averaged_model = copy.deepcopy(models[0])
    averaged_model.load_state_dict(averaged_weights)

    return averaged_model


def show_param(model , file_name):
    model_dict=model.state_dict()
    param_list=[]
    '''
    for param_name in model_dict.keys():
        t =[param_name , model_dict[param_name]]
        param_list.append(t)
    '''
    f=open(file_name+"_param.txt" , "w")
    for param_name in model_dict.keys():
        f.write("################ "+str(param_name)+" ################\n")
        for element in model_dict[param_name]:
            f.writelines(str(element)+"\n")
        f.write("\n")
    f.close()

    #return param_list


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

if __name__ == '__main__':

    # パラメータの定義
    num_classes = 28
    input_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates
    
    #モデル1
    model1_path = 'model/Adam_batch_size=2048_epoch=20.pt'
    model1 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model1_path))
    model1.eval()
    print('model_path: {}'.format(model1_path))
    
    #モデル2
    model2_path = 'model/Adam_batch_size=2048_epoch=3.pt'
    model2 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()
    print('model_path: {}'.format(model2_path))
    
    models_to_average = [model1, model2]
    averaged_model = average_models(models_to_average)
    torch.save(averaged_model.state_dict(), 'fed_model/averaged_model.pt')
    
    
    #averaged_modelをmodel3にコピー
    model3_path="fed_model/averaged_model.pt"
    model3 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model3.load_state_dict(torch.load('fed_model/averaged_model.pt'))
    
    #パラメータをファイルに出力
    model1_param=show_param(model1,"model1")
    model2_param=show_param(model2,"model2")
    '''
    f=open("model_param.txt" ,"w")
    f.writelines(str(model1_param))
    f.write("\n")
    f.write("###########################")
    f.write("\n")
    f.writelines(str(model2_param))
    f.close()
    '''
