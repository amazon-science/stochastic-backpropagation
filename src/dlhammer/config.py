import os
import ast
import yaml

from easydict import EasyDict

BASE_CONFIG = {
    'OUTPUT_DIR': './checkpoints',
    'SESSION': 'base',
    'LOG_FILE': 'log.txt',
}


def update_config(parser, default_cfg=None):
    """Update argparser to args.
    Args:
        parser: argparse.ArgumentParser.
    Returns: config.
    """
    args, _ = parser.parse_known_args()
    if args.config_file == '':
        if default_cfg is not None and 'config_file' in default_cfg:
            args.config_file = default_cfg['config_file']
    else:
        if not os.path.isfile(args.config_file):
            raise ValueError(f'Cannot find config file: {args.config_file}')

    config = EasyDict(BASE_CONFIG.copy())
    config['config_file'] = args.config_file
    if default_cfg is not None:
        config.update(default_cfg)

    # Merge config from yaml
    if os.path.isfile(config.config_file):
        with open(config.config_file, 'r') as f:
            yml_config = yaml.full_load(f)
        config = merge_dict(config, yml_config)

    # Merge config from opts
    config = merge_opts(config, args.opts)

    # Eval dict leaf
    config = eval_dict_leaf(config)

    return config


def merge_dict(a, b, path=None):
    """Merge b into a. The values in b will override values in a.
    Args:
        a (dict): dict to merge to.
        b (dict): dict to merge from.
    Returns: dict.
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def merge_opts(d, opts):
    """Merge opts.
    Args:
        d (dict): The dict.
        opts (list): The opts to merge, whose format is [key1, name1, key2, name2,...].
    Returns: dict.
    """
    assert len(opts) % 2 == 0, f'length of opts must be even. Got: {opts}'
    for i in range(0, len(opts), 2):
        full_k, v = opts[i], opts[i + 1]
        keys = full_k.split('.')
        sub_d = d
        for i, k in enumerate(keys):
            if not hasattr(sub_d, k):
                raise ValueError(f'The key {k} not exist in the dict. Full key: {full_k}')
            if i != len(keys) - 1:
                sub_d = sub_d[k]
            else:
                sub_d[k] = v
    return d


def eval_dict_leaf(d):
    """Eval values of dict leaf.
    Args:
        d (dict): The dict to eval.
    Returns: dict.
    """
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = eval_string(v)
        else:
            eval_dict_leaf(v)
    return d


def eval_string(string):
    """Automatically evaluate string to corresponding types.
    For example:
        not a string -> return the original input
        '0' -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0, 1, 2]
        'eval(1 + 2)' -> 3
        'eval(range(5))' -> [0, 1, 2, 3, 4]
    Args:
        string (string): the string.
    Returns: the corresponding type
    """
    if not isinstance(string, str):
        return string
    if len(string) > 1 and string[0] == '[' and string[-1] == ']':
        return eval(string)
    if string[0:5] == 'eval(':
        return eval(string[5:-1])
    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v
