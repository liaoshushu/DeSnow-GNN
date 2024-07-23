# 生成完全相同的点数的data
# 加入intensity信息
import open3d as o3d
import torch
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import os
from sklearn.cluster import kmeans_plusplus
import subprocess
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

num_clusters = 300  # 聚类的大小
vote = 0.5
radius = 1
voxel_size = 0.4


def remove_column(array, column_to_remove):
    # 使用 np.delete() 删除指定列
    updated_array = np.delete(array, column_to_remove, axis=1)
    return updated_array


def bin_to_pcd(bin_filename, pcd_filename):
    # 读取二进制文件
    with open(bin_filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32, count=-1).reshape([-1, 4])
    intensity_vet = data[:, 3]
    # 将一维数组转换为二维数组，每行包含 XYZ 坐标
    point_cloud = remove_column(data, 3)
    # 创建 Open3D 点云对象
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    # 保存为 PCD 文件
    o3d.io.write_point_cloud(pcd_filename, o3d_cloud)
    return intensity_vet


def node_label(assignments, location_indices, points_labels, le):
    node_labels = np.zeros(le)
    assignments = assignments.cpu().numpy()
    assignments_uni = np.unique(assignments[location_indices])  # 含有110号噪声的节点的ass的值
    ass_total = list(set(assignments))
    ass_total_dic = {value: index for index, value in enumerate(ass_total)}
    for _ in range(len(assignments_uni)):
        i = 0
        indices = np.where(assignments == assignments_uni[_])  # 把所有含有噪声的节点的点坐标值提取出来
        indices = indices[0]
        for a in range(len(indices)):
            if points_labels[indices[a]] == 110:
                i = i + 1
        if i > vote * len(indices):
            index = ass_total_dic.get(assignments_uni[_], None)
            node_labels[index] = 1  # 1是雪
    node_labels = node_labels.reshape(-1, 1)
    return node_labels


# def intensity_to_node_inten_and_cluister_neinum(assignments_cuda_tensor, intensity_vet, le):
#     assignments_cuda_tensor.cpu().tolist()
#     inten_node = [0] * le
#     cluster_number = [0] * le
#     for _ in range(len(assignments_cuda_tensor)):
#         inten_node[assignments_cuda_tensor[_]] = inten_node[assignments_cuda_tensor[_]] + intensity_vet[_]
#         cluster_number[assignments_cuda_tensor[_]] = cluster_number[assignments_cuda_tensor[_]] + 1
#     inten_node = [a / b for a, b in zip(inten_node, cluster_number)]
#     return inten_node, cluster_number


def intensity_to_node_inten_and_cluister_neinum(assignments_cuda_tensor, intensity_vet, le):
    # Convert assignments_cuda_tensor to CPU and NumPy array
    assignments = assignments_cuda_tensor.cpu().numpy()
    # Convert intensity_vet to NumPy array if it is a list
    if isinstance(intensity_vet, list):
        intensity = np.array(intensity_vet)
    else:
        intensity = intensity_vet.cpu().numpy()
    # Initialize arrays
    inten_node = np.zeros(le)
    cluster_number = np.zeros(le)
    # Aggregate intensity and cluster counts
    np.add.at(inten_node, assignments, intensity)
    np.add.at(cluster_number, assignments, 1)
    # Compute average intensity per cluster
    inten_node = np.divide(inten_node, cluster_number, out=np.zeros_like(inten_node), where=cluster_number!=0)
    return inten_node.tolist(), cluster_number.tolist()

def downsample_point_cloud(points, num_points=40000):
    print(num_points)
    num_points_input = points.size(0)  # 获取输入点云的点数
    if num_points_input <= num_points:
        return points
    # 生成随机索引
    indices = torch.randperm(num_points_input, device=points.device)[:num_points]
    # 根据随机索引对输入点云进行下采样
    downsampled_points = points[indices]
    return downsampled_points


def create_ass(cent_cuda_tensor, points_tensor_cuda, num):
    for _ in range(10):
        chunk_size = num // 10  # 将chunk_size转换为整数
        chunk_num = 10
        min_distances = []  # 用于存储各个chunk的最小距离
        min_indices = []  # 用于存储各个chunk的最近索引
        for i in range(chunk_num):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num)  # 确保最后一个chunk的结束索引不超过num
            distances_chunk = torch.cdist(points_tensor_cuda, cent_cuda_tensor[start_idx:end_idx, :])
            min_distance_chunk, min_index_chunk = torch.min(distances_chunk, dim=1)
            min_distances.append(min_distance_chunk)
            min_indices.append(min_index_chunk + start_idx)  # 将相对索引转换为全局索引
        # 将各个chunk的最小距离和索引拼接为张量
        min_distances_tensor = torch.stack(min_distances, dim=1)
        min_indices_tensor = torch.stack(min_indices, dim=1)
        # 在所有chunk中找到最小距离和对应的索引
        min_distances_global, min_indices_global = torch.min(min_distances_tensor, dim=1)
        # torch.save(min_distances_tensor, f"D:\\CADC_dataset\\5w_0.5_5\\min_distances_tensor.pt")
        # 根据全局最小距离找到对应的全局最近索引
        assignment = torch.gather(min_indices_tensor, 1, min_indices_global.unsqueeze(1))
        assignment = assignment.view(-1)
        complete_tensor = torch.arange(num_clusters, device='cuda:0')
        missing_values = torch.masked_select(complete_tensor,
                                             torch.logical_not(torch.isin(complete_tensor, assignment)))
        print("缺少的值:", missing_values)
        print("缺少的值有多少个", len(missing_values))
        if len(missing_values) == 0:
            break
        else:
            if _ == 9:
                return None
            continue

    # tensor_cuda = torch.zeros(num_clusters, 3, device='cuda')
    # for i in range(len(assignment)):
    #     if tensor_cuda[assignment[i],0] != 0:
    #         torch.mean(tensor_cuda[assignment,:],points_tensor_cuda[i,:],dim=0)
    #     else:
    #         tensor_cuda[assignment,:] = points_tensor_cuda[i, :]
    return assignment


def kd_tree_radius_neighbors(data, dis):
    # 构建 KD 树
    kdtree = cKDTree(data)
    # 查询每个点半径范围内的邻居
    neighbors = []
    dis = torch.pow(dis, 1.2)
    for i in range(len(data)):
        # 根据距离动态调整搜索半径
        radius = 0.01 * dis + 1.5  # 可以根据实际需求调整比例系数
        # 查询半径范围内的邻居
        neighbors_in_radius = kdtree.query_ball_point(data[i], r=radius[i])
        neighbors.append([idx for idx in neighbors_in_radius if idx != i])
    distances, indices = kdtree.query(data, k=8)  # 最近的8个邻居
    avg_distances = np.mean(distances, axis=1)
    # 如果某个点半径范围内没有邻居，则返回距离最近的两个邻居的索引，并删除自身的索引
    for i, neigh_list in enumerate(neighbors):
        neighbors[i] = [idx for idx in neighbors[i] if idx != i]
        if not neighbors[i]:
            distances, indices1 = kdtree.query(data[i], k=4)
            indices1 = np.delete(indices1, 0, axis=0)
            neighbors[i] = [idx for idx in indices1]
    return neighbors, avg_distances, indices


def indices_to_edge(result):
    length = 0
    for _ in range(len(result)):
        length = length + len(result[_])
    edge_matrix = np.zeros((2, length))
    k = 0
    for _ in range(len(result)):
        for i in range(len(result[_])):
            edge_matrix[0][k] = _
            edge_matrix[1][k] = result[_][i]
            k = k + 1
    return edge_matrix


def neighbors_num(result):
    nei_number = [0] * len(result)
    for _ in range(len(result)):
        nei_number[_] = len(result[_])
    return nei_number


def remove_duplicates_and_labels(points_tensor_cuda, labels, intensity_vet):
    # 创建一个空集合来存储已经遇到的点
    seen_points = set()
    # 创建一个空列表来存储不重复的点
    unique_points = []
    # 创建一个空列表来存储不重复点对应的标签
    unique_labels = []
    # 创建一个空列表来存储不重复点对应的强度信息
    unique_intensity_vet = []

    # 遍历每一个点和相应的标签以及强度信息
    for point, label, intensity in zip(points_tensor_cuda, labels, intensity_vet):
        # 将点的坐标转换为元组，以便能够在集合中进行比较
        point_tuple = tuple(point.tolist())

        # 如果点未曾存在于集合中，则将其添加到不重复的点列表中，并添加相应的标签和强度信息
        if point_tuple not in seen_points:
            unique_points.append(point)
            unique_labels.append(label)
            unique_intensity_vet.append(intensity)
            seen_points.add(point_tuple)

    return unique_points, unique_labels, unique_intensity_vet


import numpy as np
from sklearn.decomposition import PCA


def min_explained_variance_ratio(indices, pointcloud):
    """
    计算每个点的最小解释方差比率。

    参数:
        indices: 形状为 (N, k) 的二维数组，包含了每个点的最近 k 个邻居的索引。
        pointcloud: 形状为 (N, 3) 的二维数组，包含了每个点的三维坐标。

    返回:
        min_explained_variance_ratios: 形状为 (N, 1) 的二维数组，每个元素是对应点的最小解释方差比率。
    """
    pointcloud = pointcloud.cpu().detach().numpy()

    # 初始化一个空的数组来存储每个点的最小解释方差比率
    min_explained_variance_ratios = np.zeros((len(indices), 1))

    # 逐个计算每个点的最小解释方差比率
    for i, neighbor_indices in enumerate(indices):
        neighbor_coordinates = pointcloud[neighbor_indices]
        # 将邻居坐标展平成一个形状为 (k, 3) 的二维数组，以便用于 PCA
        flattened_neighbor_coordinates = neighbor_coordinates.reshape(-1, 3)
        # 初始化 PCA 模型，并拟合数据
        pca = PCA()
        pca.fit(flattened_neighbor_coordinates)
        # 提取最小解释方差比率
        min_explained_variance_ratios[i] = np.min(pca.explained_variance_ratio_)

    return min_explained_variance_ratios


########################################## 读取数据和标签############
folder_path = f"/root/autodl-tmp/al/velodyne"
i = 0
flag = 0
files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]
files.sort()
for file in files:
    bin_file_path = os.path.join(folder_path, file)
    id = bin_file_path[29:35]
    print(id)
    pcd_file_path = f"/root/autodl-tmp/0.4k/pcb/" + f"{id}" + ".pcd"
    intensity_vet = bin_to_pcd(bin_file_path, pcd_file_path)  # 在文件格式转化函数里加入，提取intensity向量
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points_np = np.asarray(pcd.points)  # 点云坐标转数组
    points_label = np.fromfile(str(f"/root/autodl-tmp/lab/{id}.label"), dtype=np.uint32, count=-1)  # pcl是label
    #####################################删除重复点和label##############################
    points_np, points_label, intensity_vet = remove_duplicates_and_labels(points_np, points_label, intensity_vet)
    points_np = np.array(points_np)
    # print("points lab",points_label)
    location_indices = [index for index, value in enumerate(points_label) if value == 110]  # 噪声所在点的索引
    # print("loc_ind",location_indices)
    points_tensor_cpu = torch.tensor(points_np, dtype=torch.float32)  # 点云数组转张量
    points_tensor_cuda = points_tensor_cpu.cuda()
    ################################体素降采样，生成ass，并且对ass和cent进行调整，保证每个ass都有分配到点云########################
    # cent_cuda_tensor = voxel_grid_downsampling(points_tensor_cpu, voxel_size)
    cent_cuda_tensor = downsample_point_cloud(points_tensor_cuda, num_points=num_clusters)
    assignments_cuda_tensor = create_ass(cent_cuda_tensor, points_tensor_cuda, num=num_clusters)  # 调整过的cent和ass
    if assignments_cuda_tensor is None:  # 如果没有足够的数据
        continue  # 跳过当前文件，处理下一个文件
    cent_cpu_tensor = cent_cuda_tensor.detach().cpu()
    # print(len(set(assignments_cuda_tensor.cpu().tolist())))
    ###################################求每个节点与中心的距离###########################################
    distances = torch.norm(cent_cuda_tensor, p=2, dim=1)
    distances = distances.detach().cpu()
    ###############################获取节点强度和节点邻居数（已矫正）#########################################
    intensity_node_list_cpu, cluster_number_list_cpu = intensity_to_node_inten_and_cluister_neinum(
        assignments_cuda_tensor, intensity_vet, le=len(cent_cpu_tensor))
    node_labels = node_label(assignments_cuda_tensor, location_indices, points_label, le=len(cent_cpu_tensor))
    #########################################建立节点之间的图关系##################################
    result, avg_distances, indices = kd_tree_radius_neighbors(cent_cpu_tensor, distances)  # 寻找节点
    edge_matrix = indices_to_edge(result)  # 建立矩阵
    Dp = avg_distances / distances
    Rp = min_explained_variance_ratio(indices, cent_cpu_tensor)
    ###########################################建立特征向量###############################################
    Rp = torch.tensor(Rp).float()
    Dp = Dp.float()
    cluster_number_cpu_tensor = torch.tensor(cluster_number_list_cpu)
    intensity_node_cpu_tensor = torch.tensor(intensity_node_list_cpu)
    combined_vector = torch.stack((cent_cpu_tensor[:, 0], cent_cpu_tensor[:, 1], cent_cpu_tensor[:, 2],
                                   intensity_node_cpu_tensor, cluster_number_cpu_tensor, Dp), dim=1)
    combined_vector = torch.cat((combined_vector, Rp), dim=1)
    combined_vector = combined_vector.float()
    #################################建立数据集####################
    edge_matrix = torch.tensor(edge_matrix, dtype=torch.long)  # 数据集格式转化
    node_labels = torch.tensor(node_labels, dtype=torch.float32)  # 数据集格式转化
    data = Data(x=combined_vector, edge_index=edge_matrix, y=node_labels)  # 生成数据集
    flag = flag + 1
    ###################################################存储数据#################################
    torch.save(data, f"/root/autodl-tmp/0.3k/data/data_batch{flag}.pt")
    torch.save(assignments_cuda_tensor, f"/root/autodl-tmp/0.3k/ass/assigment_batch{flag}.pt")
    torch.save(points_label, f"/root/autodl-tmp/0.3k/lab/label_batch{flag}.pt")
    torch.save(points_tensor_cpu, f"/root/autodl-tmp/0.3k/point/points_batch{flag}.pt")
    torch.save(intensity_vet, f"/root/autodl-tmp/0.3k/vet/vet_batch{flag}.pt")