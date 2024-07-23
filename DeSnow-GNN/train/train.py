import torch,gc
import importlib
from torch_geometric.nn import MLP,GraphConv,GATConv,TransformerConv
from GCN import GCNConv
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from registration import registration
import torch.nn as nn
import os
import random
import math

def rotate_point_cloud(data, angle_rad):
    # 将角度转换为弧度
    
    # 创建旋转矩阵
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                                     [torch.sin(angle_rad), torch.cos(angle_rad), 0],
                                     [0, 0, 1]])
    rotation_matrix = rotation_matrix.to(device)

    # 应用旋转矩阵
    rotated_data = torch.matmul(rotation_matrix, data.T).T
    
    return rotated_data

# Define your custom dataset class
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

# Modify your Net class and training function to accommodate DataLoader
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
def train(data,kernal_point):
    model.train()
    optimizer.zero_grad()
    # Move input data to GPU
    data = data.to(device)
    with torch.no_grad():
        label = data.y
        rotation_angle = torch.tensor(random.uniform(-math.pi, math.pi))
        data.x[:,0:3] = rotate_point_cloud(data.x[:,0:3], rotation_angle)
        distances = torch.norm(data.x[:,0:3], p=2, dim=1)
        distances = torch.pow(distances, 1/3)
        distances = distances.view(-1,1)
        distances = torch.mul(data.y, distances)
        distances = distances.half()
    node_feature2= model(data)
    distances = torch.mul(node_feature2.half(), distances)
    distances = torch.sum(distances)
    #print(distances)
    del data
    # loss1 = crit1(node_feature1, label)
    # del node_feature1
    # loss1.backward()
    loss2 = crit2(node_feature2, label) - distances*(0.000001)
    del node_feature2,label,distances
    loss2.backward()
    optimizer.step()
    return loss2.item()

# Initialize your dataset and DataLoader
folder_path = "/root/autodl-tmp/1w/data/"
kernal_point = torch.load("/root/autodl-fs/k.pt")
kernal_point = torch.tensor(kernal_point)
dataset = CustomDataset(folder_path)
loader_train = DataLoader(dataset, batch_size=1,shuffle=True)

model = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#pos_weight = torch.tensor([1])
#pos_weight = pos_weight.to(device)
#crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#crit1 = torch.nn.MSELoss()
crit2 = torch.nn.MSELoss()
step_size = 2
gamma = 0.9
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
model.to(device)
save_interval = 10
# Training loop
for epoch in range(101):
    print("进入循环")
    loss_all = 0
    k=0
    for data in loader_train:
        data.x = data.x.to(torch.float32)
        kernal_point = kernal_point.to(torch.float32)
        loss = train(data,kernal_point)
        loss_all += loss
        k=k+1
        print(k)
        gc.collect()
        torch.cuda.empty_cache()
   
    current_learning_rate = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}, Loss: {loss_all}, Learning Rate: {current_learning_rate}")
    scheduler.step()
    if epoch % save_interval == 0:  # 这里的save_interval是您想要保存模型的间隔
        torch.save(model.state_dict(), f"/root/autodl-tmp/up_net/model_5_16/1w1{epoch}.pth")