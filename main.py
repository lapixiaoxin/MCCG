import itertools
import os
import time
import random
import numpy as np
import torch
from os.path import join, dirname
from dataset.preprocess_data import preprocess_data
from dataset.save_results import check_mkdir
from logger import set_log
from params import set_params
from training.mccg_training import MCCG_Trainer

args_dict, args = set_params()


def set_all_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.time()
    set_all_seed(args.seed)

    # Preprocess data, run only once.
    preprocess_data()

    # Training
    param_combin = list(itertools.product(*args_dict.values()))
    param_names = list(args_dict.keys())
    param_dicts = [dict(zip(param_names, combo)) for combo in param_combin]
    count = 0
    for param in param_dicts:
        set_all_seed(args.seed)

        feature_rate1 = param['drop_feature_rate_view1']
        feature_rate2 = param['drop_feature_rate_view2']
        edge_rate1 = param['drop_edge_rate_view1']
        edge_rate2 = param['drop_edge_rate_view2']

        if edge_rate1 == edge_rate2 or feature_rate1 == feature_rate2:
            continue

        count += 1
        check_mkdir(args.log_dir)
        logger = set_log(join(os.path.abspath(dirname(__file__)), f"{args.log_dir}/log_{count}.log"))
        msg = f"\n-----------------param config----------------\n" \
              f"-- dataset:                     {args.dataset}\n" \
              f"-- mode:                        {args.mode}\n" \
              f"-- combin_num:                  {count}\n" \
              f"-- lr:                          {args.lr}\n" \
              f"-- l2_coef:                     {param['l2_coef']}\n" \
              f"-- seed:                        {args.seed}\n" \
              f"-- epochs:                      {args.epochs}\n" \
              f"-- layer_shape:                 {args.layer_shape}\n" \
              f"-- dim_proj_multiview:          {args.dim_proj_multiview}\n" \
              f"-- dim_proj_cluster:            {args.dim_proj_cluster}\n" \
              f"-- drop_scheme:                 {param['drop_scheme']}\n" \
              f"-- drop_feature_rate_view1:     {param['drop_feature_rate_view1']}\n" \
              f"-- drop_feature_rate_view2:     {param['drop_feature_rate_view2']}\n" \
              f"-- drop_edge_rate_view1:        {param['drop_edge_rate_view1']}\n" \
              f"-- drop_edge_rate_view2:        {param['drop_edge_rate_view2']}\n" \
              f"-- th_a:                        {param['th_a']}\n" \
              f"-- th_o:                        {param['th_o']}\n" \
              f"-- th_v:                        {param['th_v']}\n" \
              f"-- db_eps:                      {param['db_eps']}\n" \
              f"-- db_min:                      {param['db_min']}\n" \
              f"-- w_cluster:                   {param['w_cluster']}\n" \
              f"-- t_multiview:                 {param['t_multiview']}\n" \
              f"-- t_cluster:                   {param['t_cluster']}\n"
        logger.info(msg)

        trainer = MCCG_Trainer()
        trainer.fit(logger=logger,
                    mode=args.mode,
                    combin_num=count,
                    layer_shape=args.layer_shape,
                    dim_proj_multiview=args.dim_proj_multiview,
                    dim_proj_cluster=args.dim_proj_cluster,
                    drop_scheme=param['drop_scheme'],
                    drop_feature_rate_view1=param['drop_feature_rate_view1'],
                    drop_feature_rate_view2=param['drop_feature_rate_view2'],
                    drop_edge_rate_view1=param['drop_edge_rate_view1'],
                    drop_edge_rate_view2=param['drop_edge_rate_view2'],
                    th_a=param['th_a'],
                    th_o=param['th_o'],
                    th_v=param['th_v'],
                    db_eps=param['db_eps'],
                    db_min=param['db_min'],
                    l2_coef=param['l2_coef'],
                    w_cluster=param['w_cluster'],
                    t_multiview=param['t_multiview'],
                    t_cluster=param['t_cluster']
                    )

        end_time = time.time()
        total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        message = f"Total running time: {total_time}"
        logger.info(message)
