import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import torch.linalg as LA


def get_adj(edge_index, num_nodes):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    adj += torch.eye(num_nodes, device=adj.device)

    return adj


def get_M(adj, t=2):
    tran_prob = F.normalize(adj, p=1, dim=0)
    M = sum([LA.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t

    return M


def onehot_encoder(label_list):
    """
    Transform label list to one-hot matrix.
    Arg:
        label_list: e.g. [0, 0, 1]
    Return:
        onehot_mat: e.g. [[1, 0], [1, 0], [0, 1]]
    """
    if isinstance(label_list, np.ndarray):
        labels_arr = label_list
    else:
        try:
            labels_arr = np.array(label_list.cpu().detach().numpy())
        except:
            labels_arr = np.array(label_list)

    num_classes = max(labels_arr) + 1
    onehot_mat = np.zeros((len(labels_arr), num_classes + 1))

    for i in range(len(labels_arr)):
        onehot_mat[i, labels_arr[i]] = 1

    return onehot_mat


def matx2list(adj):
    """
    Transform matrix to list.
    """
    adj_preds = []
    for i in adj:
        if isinstance(i, np.ndarray):
            temp = i
        else:
            temp = i.cpu().detach().numpy()
        for idx, j in enumerate(temp):
            if j == 1:
                adj_preds.append(idx)
                break
            if idx == len(temp) - 1:
                adj_preds.append(-1)

    return adj_preds
