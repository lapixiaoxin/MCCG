import argparse
import yaml

cfg_path = "config/aminer19_config.yml"


def set_params(config_path=cfg_path):
    parser = argparse.ArgumentParser('argument for training')
    with open(config_path, "r") as setting:
        args_dict = yaml.load(setting, Loader=yaml.FullLoader)

    for key, value in args_dict.items():
        parser.add_argument(f"--{key}", default=value)

    args = parser.parse_args()
    excluded_params = ['cuda', 'gpu', 'seed', 'lr', 'epochs', 'dataset', 'save_path', 'predict_result', 'log_dir',
                       'mode', 'layer_shape', 'dim_proj_multiview', 'dim_proj_cluster', "ground_truth_file"]
    filtered_args_dict = {k: v for k, v in args_dict.items() if k not in excluded_params}

    return filtered_args_dict, args
