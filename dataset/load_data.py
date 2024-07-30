import json
import codecs
import torch
import pickle
import numpy as np
from os.path import join
from params import set_params
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

_, args = set_params()


def load_data(rfname):
    with open(rfname, 'rb') as rf:
        return pickle.load(rf)


def load_json(rfname):
    with codecs.open(rfname, 'r', encoding='utf-8') as rf:
        return json.load(rf)


def load_dataset(mode):
    if mode == "train":
        data_path = join(args.save_path, "src", "train", "train_author.json")
    elif mode == "valid":
        data_path = join(args.save_path, "src", "valid", "sna_valid_raw.json")
    elif mode == "test":
        data_path = join(args.save_path, "src", "test", "sna_test_raw.json")

    pubs = load_json(data_path)
    names = []
    for name in pubs:
        names.append(name)

    return names, pubs


def load_graph(name, th_a=1, th_o=0.5, th_v=2):
    data_path = join(args.save_path, 'graph')
    datapath = join(data_path, args.mode, name)

    # Load label
    if args.mode == "train":
        p_label = np.load(join(datapath, 'p_label.npy'), allow_pickle=True)
        p_label_list = []
        for pid in p_label.item():
            p_label_list.append(p_label.item()[pid])
        label = torch.LongTensor(p_label_list)

    else:
        label = []

    # Load node feature
    feats = np.load(join(datapath, 'feats_p.npy'), allow_pickle=True)
    ft_list = []
    for idx in feats.item():
        ft_list.append(feats.item()[idx])
    ft_tensor = torch.stack(ft_list)  # size: N * feature dimension

    # Load edge
    temp = set()
    with open(join(datapath, 'adj_attr.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            temp.add(line)

    srcs, dsts = [], []
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 7:
            src, dst = int(toks[0]), int(toks[1])
            val_a, val_o, val_v = int(toks[2]), int(toks[3]), int(toks[5])
            attr_o, attr_v = float(toks[4]), float(toks[6])
        else:
            print('read adj_attr ERROR!\n')

        if val_a >= th_a or attr_o >= th_o or val_v >= th_v:
            srcs.append(src)
            dsts.append(dst)

    temp.clear()

    # Build graph
    edge_index = torch.cat([torch.tensor(srcs).unsqueeze(0), torch.tensor(dsts).unsqueeze(0)], dim=0)
    num_nodes = torch.tensor(len(ft_tensor), dtype=torch.int32)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    return label, ft_tensor, data
