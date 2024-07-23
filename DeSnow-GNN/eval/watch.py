from GCN import GCNConv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch_geometric.nn import MLP
import os
import re
from registration import registration
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Define your custom dataset class
# Modify your Net class and training function to accommodate DataLoader
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
        self.conv1_1 = GCNConv(7, 7)
        self.mlp1_1 = MLP(in_channels=4, hidden_channels=64, out_channels=3, num_layers=3)#自校准的mlp
        self.mlp1_2 = MLP(in_channels=7, hidden_channels=64, out_channels=16, num_layers=3)#
        self.mlp1_3 = MLP(in_channels=16, hidden_channels=64, out_channels=32, num_layers=3)
        self.mlp1_4 = MLP(in_channels=4, hidden_channels=64, out_channels=1, num_layers=3)
        self.conv1_2 = GCNConv(16,16)
        self.lin1_1 = torch.nn.Linear(32, 1)
        self.bn1_1 = torch.nn.BatchNorm1d(7)
        self.bn1_2 = torch.nn.BatchNorm1d(16)
        self.bn1_3 = torch.nn.BatchNorm1d(16)
        self.bn1_4 = torch.nn.BatchNorm1d(32)
        self.bn1_5 = torch.nn.BatchNorm1d(1)#edge_weight的归一化
        ####################全局特征部分###########
        self.mlp1_5 = MLP(in_channels=4, hidden_channels=32, out_channels=3, num_layers=3)
        self.attention = SelfAttention(3, 3)
        self.mlp1_6 = MLP(in_channels=3, hidden_channels=32, out_channels=3, num_layers=3)
    def forward(self,data):
        # ######################构建节点特征：根据空间信息########################
        registration_dis = self.mlp1_1(data.x[:,0:4])#mlp
        registration_dis = registration(data.x, data.edge_index, registration_dis)
        registration_dis = torch.cat((registration_dis, data.x[:,3:4][data.edge_index[1]]), dim=1)
        registration_dis = self.mlp1_4(registration_dis)#求dege_weight
        registration_dis = torch.relu(registration_dis)
        #######################使用全局特征对空间信息的矫正#################################
        # golbal_feature = self.mlp1_5(data.x[:,0:4])
        # golbal_feature = self.attention(golbal_feature,kernal_point)
        # golbal_feature = self.mlp1_6(golbal_feature)
        # with torch.no_grad():
        #     node_feature1 = data.x[:,0:3]-golbal_feature
        # del golbal_feature
        # node_feature1 = torch.cat((node_feature1, data.x[:,3:7]), dim=1)
        # ##############################传播##############################
        node_feature1 = self.conv1_1(data.x[:,0:7],data.edge_index, edge_weight = registration_dis)#conv1
        node_feature1 = self.bn1_1(node_feature1)#正则1
        #node_feature1 = torch.cat((node_feature1, data.x[:,3:6]), dim=1)
        #################################################################
        node_feature1 = self.mlp1_2(node_feature1)#mlp2
        node_feature1 = self.bn1_2(torch.relu(node_feature1))#正则2
        node_feature1 = self.conv1_2(node_feature1,data.edge_index, edge_weight = registration_dis)#conv2
        del registration_dis
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
    return pred


def Confusion(pred_label, data_test):
    tp_point = []
    fp_point = []
    fn_point = []
    tn_point = []
    fn_data =[]
    tp=0
    fp=0
    fn=0
    tn=0
    a=0.3
    for i in range(len(pred_label)):
        if pred_label[i] > a:
            if data_test.y[i] == 1:
                tp_point.append(data_test.x[i,0:3])
                tp=tp+1
            elif data_test.y[i] != 1:
                fp_point.append(data_test.x[i,0:3])
                fp=fp+1
        elif pred_label[i] <a:
            if data_test.y[i] == 1:
                fn_point.append(data_test.x[i,0:3])
                fn_data.append(data_test.x[i,0:4])
                fn=fn+1
            elif data_test.y[i] != 1:
                tn_point.append(data_test.x[i,0:3])
                tn=tn+1
    return tp_point, fp_point, fn_point, tn_point, fn_data ,tp,fp,fn,tn

watch_index=[1584]
a=0
model=Net()
model.load_state_dict(torch.load(f"/root/autodl-tmp/up_net/model_4_9/saved_model_epoch_4w_0.5_5_group1_20.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data_path=f"/root/autodl-tmp/4w_0.5_6/data/data_batch{watch_index[a]}.pt"
data_test = torch.load(data_path)
data_test.x = data_test.x.to(torch.float32)
data_test = data_test.to(device)
pred = evaluate(data_test)
tp_point, fp_point, fn_point, tn_point, fn_data,tp,fp,fn,tn= Confusion(pred,data_test)

print('tp rate是：', tp)
print('fp rate是：', fp)
print('fn rate是：', fn)
print('tn rate是：', tn)
#################################画不同颜色的点对应tpfntnfp#######################
tp_points = [point.cpu().numpy() for point in tp_point]
fp_points = [point.cpu().numpy() for point in fp_point]
fn_points = [point.cpu().numpy() for point in fn_point]
tn_points = [point.cpu().numpy() for point in tn_point]


x_coords_tp = [point[0] for point in tp_points]
y_coords_tp = [point[1] for point in tp_points]
z_coords_tp = [point[2] for point in tp_points]

x_coords_fp = [point[0] for point in fp_points]
y_coords_fp = [point[1] for point in fp_points]
z_coords_fp = [point[2] for point in fp_points]

x_coords_fn = [point[0] for point in fn_points]
y_coords_fn = [point[1] for point in fn_points]
z_coords_fn = [point[2] for point in fn_points]
# for _ in range(5000):
#     print(fn_data[_])
# x_coords_tn = [point[0] for point in tn_points]
# y_coords_tn = [point[1] for point in tn_points]
# z_coords_tn = [point[2] for point in tn_points]

fig = plt.figure(figsize=(20, 13))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([np.ptp(coord) for coord in [x_coords_tp, y_coords_tp, z_coords_tp]])
ax.scatter(x_coords_tp, y_coords_tp, z_coords_tp, c='green', marker='.', s=1,label='Green Points') 
ax.scatter(x_coords_fp, y_coords_fp, z_coords_fp, c='red', marker='.',s=1, label='Red Points')  
ax.scatter(x_coords_fn, y_coords_fn, z_coords_fn, c='black', marker='.',s=1,label='Black Points') 
# ax.scatter(x_coords_tn, y_coords_tn, z_coords_tn, c='blue', marker='.',s=5,label='Blue Points')  
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()  # 显示图例
# 保存图片到指定地址
plt.savefig('/root/autodl-fs/good_1584.png')
#################################画不同颜色的点对应tpfntnfp#######################


# ###########################对比好坏图的空间位置####################
# fn_points_bad = [point.cpu().numpy() for point in fn_point]
# a=1
# data_path=f"/root/autodl-tmp/4w_0.5_5/yanz/data2/data_batch{watch_index[a]}.pt"
# data_test = torch.load(data_path)
# data_test.x = data_test.x.to(torch.float32)
# data_test = data_test.to(device)
# pred = evaluate(data_test)
# tp_point, fp_point, fn_point, tn_point, fn_data = Confusion(pred,data_test)
# fn_points_good = [point.cpu().numpy() for point in fn_point]
# x_coords_fn_bad = [point[0] for point in fn_points_bad]
# y_coords_fn_bad = [point[1] for point in fn_points_bad]
# z_coords_fn_bad = [point[2] for point in fn_points_bad]
# x_coords_fn_good = [point[0] for point in fn_points_good]
# y_coords_fn_good = [point[1] for point in fn_points_good]
# z_coords_fn_good = [point[2] for point in fn_points_good]
# fig = plt.figure(figsize=(20, 13))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_coords_fn_bad, y_coords_fn_bad, z_coords_fn_bad, c='blue', marker='.', s=1,label='Green bad Points') 
# ax.scatter(x_coords_fn_good, y_coords_fn_good, z_coords_fn_good, c='red', marker='.',s=1, label='Red good Points') 
# ax.set_box_aspect([np.ptp(coord) for coord in [x_coords_fn_good, y_coords_fn_good, z_coords_fn_good]])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_aspect('auto')
# plt.legend()  # 显示图例
# # 保存图片到指定地址
# plt.savefig('/root/autodl-fs/compare1.png')
# ###########################对比好坏图的空间位置####################

##########################画出图像########################
# tp_points = [point.cpu().numpy() for point in data_test.x[:,0:3]]
# x_coords_tp = [point[0] for point in tp_points]
# y_coords_tp = [point[1] for point in tp_points]
# z_coords_tp = [point[2] for point in tp_points]
# fig = plt.figure(figsize=(20, 13))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_coords_tp, y_coords_tp, z_coords_tp, c='green', marker='.', s=1,label='Green Points') 
# ax.set_box_aspect([np.ptp(coord) for coord in [x_coords_tp, y_coords_tp, z_coords_tp]])
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.legend()  # 显示图例
# # 保存图片到指定地址
# plt.savefig('/root/autodl-fs/data_point584.png')














