import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
from dataset.enhance_graph import *
from dataset.load_data import load_dataset, load_graph
from dataset.save_results import get_results
from evaluation.eval import evaluate
from model.mccg_model import MCCG, GAT
from .utils import *
from os.path import join
from params import set_params

_, args = set_params()

device = torch.device(("cuda:" + str(args.gpu)) if torch.cuda.is_available() and args.cuda else "cpu")


class MCCG_Trainer:
    def __init__(self) -> None:
        pass

    def fit(self,
            logger,
            mode,
            combin_num,
            layer_shape,
            dim_proj_multiview,
            dim_proj_cluster,
            drop_scheme,
            drop_feature_rate_view1,
            drop_feature_rate_view2,
            drop_edge_rate_view1,
            drop_edge_rate_view2,
            th_a,
            th_o,
            th_v,
            db_eps,
            db_min,
            l2_coef,
            w_cluster,
            t_multiview,
            t_cluster):

        names, pubs = load_dataset(mode)
        results = {}

        p = 0
        total_name = len(names)
        for name in names:
            p += 1
            msg = f"training {p}/{total_name}: {name}"
            logger.info(msg)

            results[name] = []

            # ==== Load data ====
            label, ft_list, data = load_graph(name, th_a, th_o, th_v)
            ft_list = ft_list.float()
            ft_list = ft_list.to(device)
            data = data.to(device)
            adj = get_adj(data.edge_index, data.num_nodes)
            M = get_M(adj, t=2)

            if drop_scheme == 'degree':
                edge_weights = degree_drop_weights(data).to(device)
                node_deg = degree(data.edge_index[1], num_nodes=data.num_nodes)
                feature_weights = feature_drop_weights_dense(ft_list, node_c=node_deg).to(device)
            elif drop_scheme == 'pr':
                edge_weights = pr_drop_weights(data, aggr='sink', k=200).to(device)
                node_pr = compute_pr(data)
                feature_weights = feature_drop_weights_dense(ft_list, node_c=node_pr).to(device)
            elif drop_scheme == 'evc':
                edge_weights = evc_drop_weights(data).to(device)
                node_evc = eigenvector_centrality(data)
                feature_weights = feature_drop_weights_dense(ft_list, node_c=node_evc).to(device)
            else:
                raise ValueError(f'undefined drop scheme: {drop_scheme}.')

            edge_index1 = drop_edge_weighted(data.edge_index, edge_weights, p=drop_edge_rate_view1, threshold=0.7)
            edge_index2 = drop_edge_weighted(data.edge_index, edge_weights, p=drop_edge_rate_view2, threshold=0.7)
            adj1 = get_adj(edge_index1, data.num_nodes)
            adj2 = get_adj(edge_index2, data.num_nodes)
            M1 = get_M(adj1, t=2)
            M2 = get_M(adj2, t=2)

            x1 = drop_feature_weighted_2(ft_list, feature_weights, drop_feature_rate_view1, threshold=0.7)
            x2 = drop_feature_weighted_2(ft_list, feature_weights, drop_feature_rate_view2, threshold=0.7)

            # ==== Init model ====
            encoder = GAT(layer_shape[0], layer_shape[1], layer_shape[2])
            model = MCCG(encoder, dim_hidden=layer_shape[2], dim_proj_multiview=dim_proj_multiview, dim_proj_cluster=dim_proj_cluster)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=l2_coef)

            for epoch in range(1, args.epochs + 1):
                # ==== Train ====
                model.train()
                optimizer.zero_grad()
                embd_multiview, embd_cluster = model(x1, adj1, M1, x2, adj2, M2)

                dis = pairwise_distances(embd_cluster.cpu().detach().numpy(), metric='cosine')
                labels = hdbscan.HDBSCAN(cluster_selection_epsilon=db_eps, min_samples=db_min, min_cluster_size=db_min,
                                         metric='precomputed').fit_predict(dis.astype('double'))
                labels = torch.from_numpy(labels)
                labels = labels.to(device)
                loss_cluster = model.SelfSupConLoss(embd_cluster.unsqueeze(1), labels, contrast_mode='one', temperature=t_cluster)
                loss_multiview = model.SelfSupConLoss(embd_multiview, labels, contrast_mode='all', temperature=t_multiview)

                loss_train = w_cluster * loss_cluster + (1 - w_cluster) * loss_multiview
                if (epoch % 5) == 0:
                    msg = 'epoch: {:3d}, ' \
                          'multiview loss: {:.4f}, ' \
                          'cluster loss: {:.4f}, ' \
                          'ALL loss: {:.4f}'.format(epoch,
                                                    loss_multiview.item(),
                                                    loss_cluster.item(),
                                                    loss_train.item())
                    logger.info(msg)

                loss_train.backward()
                optimizer.step()

            # ==== Evaluate ====
            with torch.no_grad():
                model.eval()
                embd = model.encoder(ft_list, adj, M)
                embd = F.normalize(model.cluster_projector(embd), dim=1)
                lc_dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                labels = hdbscan.HDBSCAN(cluster_selection_epsilon=db_eps, min_samples=db_min, min_cluster_size=db_min,
                                         metric='precomputed').fit_predict(lc_dis.astype('double'))

                pred = []
                # change to one-hot form
                class_matrix = torch.from_numpy(onehot_encoder(labels))
                # get N * N matrix
                labels = torch.mm(class_matrix, class_matrix.t())
                pred = matx2list(labels)

                # Save results
                results[name] = pred

        predict = get_results(names, pubs, results)

        ground_truth = join(args.save_path, 'src', args.mode, args.ground_truth_file)
        pre, rec, f1 = evaluate(predict, ground_truth)

        with open(join('.expert_record', args.predict_result), 'a', encoding='utf-8') as f:
            msg = f"combin_num: {combin_num}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}\n"
            logger.info(msg)
            f.write(msg)
