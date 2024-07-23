import torch
def registration(x,edge_index,registration_dis):#out就是xj
    with torch.no_grad():
        # dis = torch.norm(x[:,0:3], p=2, dim=1)
        # dis = dis.view(-1,1)
        # dis = dis[edge_index[0]]
        x_i = x[:,0:3][edge_index[1]]
        x_j = x[:,0:3][edge_index[0]]
        # x_r = registration_dis[edge_index[1]]
        # result = x_j - x_i + x_r#group3错的！！！！！！

        x_r = registration_dis[edge_index[0]]
        result = x_i - x_j + x_r#group1
        # result = torch.cat((result, dis), dim=1)
    return result
