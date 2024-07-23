from GCN import GCNConv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch_geometric.nn import MLP,GraphConv,GATConv
import os
import re
from registration import registration
import torch.nn as nn
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(self.folder_path) if f.endswith('.pt')]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = torch.load(os.path.join(self.folder_path, filename))
        print(filename)
        return data
class SelfAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

    def forward(self, x ,kernal_point):
        query = self.W_q(x)
        key = self.W_k(kernal_point)
        value = self.W_v(kernal_point)
        # 矩阵乘法
        with torch.no_grad():
            attention_scores = torch.matmul(query, key.transpose(0, 1))  # (batch_size, seq_length, seq_length)
        # 获取序列长度
        #seq_length = attention_scores.size(1)
        # softmax 操作，这里对最后一个维度进行 softmax 操作
            attention_scores = F.softmax(attention_scores, dim=-1)
        # 注意力加权求和
            output = torch.matmul(attention_scores, value)
        del attention_scores
        return output   

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ###########################构建节点特征：根据空间信息#####################
        self.conv1_1 = GATConv(7, 7)
        self.mlp1_1 = MLP(in_channels=3, hidden_channels=64, out_channels=3, num_layers=3)#自校准的mlp
        self.mlp1_2 = MLP(in_channels=7, hidden_channels=64, out_channels=7, num_layers=3)#
        self.mlp1_3 = MLP(in_channels=16, hidden_channels=64, out_channels=32, num_layers=3)
        self.mlp1_4 = MLP(in_channels=4, hidden_channels=64, out_channels=7, num_layers=3)
        self.conv1_2 = GATConv(7,16)
        self.lin1_1 = torch.nn.Linear(32, 1)
        self.bn1_1 = torch.nn.BatchNorm1d(7)
        self.bn1_2 = torch.nn.BatchNorm1d(7)
        self.bn1_3 = torch.nn.BatchNorm1d(16)
        self.bn1_4 = torch.nn.BatchNorm1d(32)
        self.bn1_5 = torch.nn.BatchNorm1d(1)#edge_weight的归一化
        ####################全局特征部分###########
        self.mlp1_5 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.attention = SelfAttention(3, 3)
        self.mlp1_6 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        ###################第二层对准############
        self.mlp1_7 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
        self.mlp1_8 = MLP(in_channels=10, hidden_channels=32, out_channels=7, num_layers=3)
    def forward(self,data):
        # ######################构建节点特征：根据空间信息########################
        registration_dis1 = self.mlp1_1(data.x[:,0:3])#mlp
        registration_dis1 = registration(data.x, data.edge_index, registration_dis1)
        registration_dis1 = torch.cat((registration_dis1, data.x[:,3:4][data.edge_index[1]]), dim=1)
        registration_dis1 = self.mlp1_4(registration_dis1)#求dege_weight
        registration_dis1 = torch.relu(registration_dis1)
        #######################使用全局特征对空间信息的矫正#################################
        # golbal_feature = self.mlp1_5(data.x[:,0:4])
        # golbal_feature = self.attention(golbal_feature,kernal_point)
        # golbal_feature = self.mlp1_6(golbal_feature)
        # with torch.no_grad():
        #     node_feature1 = data.x[:,0:3]-golbal_feature
        # del golbal_feature
        # node_feature1 = torch.cat((node_feature1, data.x[:,3:7]), dim=1)
        # ##############################传播##############################
        node_feature1 = self.conv1_1(data.x[:,0:7],data.edge_index, edge_attr = registration_dis1)#conv1
        node_feature1 = self.bn1_1(node_feature1)#正则1
        node_feature1 = self.mlp1_2(node_feature1)#mlp2
        node_feature1 = self.bn1_2(torch.relu(node_feature1))#正则2
        del registration_dis1
        ##############################################################
        registration_dis2 = self.mlp1_7(data.x[:,0:3])#mlp
        registration_dis2 = registration(data.x, data.edge_index, registration_dis2)
        registration_dis2 = torch.cat((registration_dis2, node_feature1[data.edge_index[1]]), dim=1)
        registration_dis2 = self.mlp1_8(registration_dis2)#求dege_weight
        registration_dis2 = torch.relu(registration_dis2)
        node_feature1 = self.conv1_2(node_feature1,data.edge_index, edge_attr = registration_dis2)#conv2
        del registration_dis2
        node_feature1 = self.bn1_3(node_feature1)#正则3
        node_feature1 = self.mlp1_3(node_feature1)#mlp3
        node_feature1 = self.bn1_4(torch.relu(node_feature1))#正则4
        node_feature1 = torch.sigmoid(self.lin1_1(node_feature1))#输出
        return node_feature1

def evaluate(data):
    model.eval()

    with torch.no_grad():
        # 从数据集中提取出需要的属性
        # 使用模型进行预测
        node_feature1= model(data)
        pred = node_feature1
        label = data.y
        loss = crit(node_feature1, pred)

    return loss,pred

def node_label_to_cloud(pred, assigment,data_test):
    pred=pred.detach().cpu().numpy()
    assigment=assigment.detach().cpu().numpy()
    pred_label = [0] * len(assigment)
    ass_total = list(set(assigment))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}
    for i in range(len(assigment)):
        index = ass_total_dic.get(assigment[i], None)
        if pred[index].item()>0.2:
            pred_label[i] = 110
    return pred_label


model=Net()
model.load_state_dict(torch.load(f"/root/autodl-tmp/up_net/model_4_17/saved_model_epoch_4w_0.5_6_group12_100.pth"))
crit = torch.nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# folder_path=f"/root/autodl-tmp/4w_0.5_6/yanz/data"#10w测试main
folder_path = f"/root/autodl-tmp/CADC/data" 
files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
files.sort()
loss_all=0
tp_all=0
fp_all=0
fn_all=0
tn_all=0
kk=0
for file in files:
    #print(kk)
    kk=kk+1
    data_path = os.path.join(folder_path, file)
    data_hh=data_path[25:45]
    batch_number = int(re.search(r'\d+', data_hh).group())
    print(batch_number)
    data_test = torch.load(data_path)
    data_test.x = data_test.x.to(torch.float32)
    data_test = data_test.to(device)#导入待测试data数据
    loss, pred = evaluate(data_test)#预测data数据类型及损失
    assigment = torch.load(str(f"/root/autodl-tmp/CADC/ass/assigment_batch{batch_number}.pt"))#10w测试main
    pred_label = node_label_to_cloud(pred, assigment,data_test)
    if batch_number == 3:
        torch.save(pred_label, f"/root/autodl-fs/pred_label_gnn2.pt")
    
    
    
    