import torch
import argparse
import yaml
import os
import math
from preprocess.preprocess import DecagonPreprocess
from preprocess.preprocess import printProgress

######################################################################################
# >>> function for arguments
######################################################################################
def parse_args(args):
    parser = argparse.ArgumentParser(description='Arguments for Co-Attention Graph Pooling')

    # >>> data parameters
    parser.add_argument('--dataset', help='Dataset type, must be one of [decagon].')
    parser.add_argument('--phase', help='[train | test]')
    parser.add_argument('--config', help='[binary | multi]')
    parser.add_argument('--model', help='[GCN | SortPool | gPool | CAGPool]')
    parser.add_argument('--init', help='[normal | xavier | kaiming]')
    parser.add_argument('--gcn_type', help='[GCN | SAGE]')
    parser.add_argument('--num_layers', help='number of gcn layers')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from best checkpoint')

    # >>> parser
    parser = parser.parse_args(args)

    # YAML options
    cfg = yaml.load(open('./configs/{}-{}-{}-{}-{}layers.yaml'.format(
        parser.dataset,
        parser.config,
        parser.init,
        parser.gcn_type,
        parser.num_layers)), Loader=yaml.FullLoader)
    cfg['MODEL'] = yaml.load(open(cfg['_MODEL_']), Loader=yaml.FullLoader)
    cfg['MODEL'] = cfg['MODEL']['MULTI_MODEL'] if parser.config=='multi' else cfg['MODEL']['BINARY_MODEL']
    cfg['DATASET'] = yaml.load(open(cfg['_DATASET_']), Loader=yaml.FullLoader)['DATASET']
    cfg['SOLVER'] = yaml.load(open(cfg['_SOLVER_']), Loader=yaml.FullLoader)['SOLVER']

    return parser, cfg

######################################################################################
# >>> function for data preparation
######################################################################################
def prepare_data(cfg):
    if not os.path.exists(cfg['DATASET']['PROCESSED_PATH']) \
           or cfg['DATASET']['RUN_PREPROCESS'] \
           or cfg['DATASET']['RUN_SPLIT'] \
           or (cfg['DATASET']['LABEL_TYPE'] == 'multi' and not os.path.exists(cfg['DATASET']['MULTI_LABEL_DICT'])):
        if cfg['DATASET']['DATA_NAME'] == 'Decagon': DecagonPreprocess(cfg)

######################################################################################
# >>> function for model setup
######################################################################################
def model_setup(args, cfg, device):
    checkpoint_dict = {} # checkpoint dictionary
    save_dir = './checkpoints/{}-{}-{}/'.format(args.dataset, args.config, args.model)
    file_name = '{}-{}-{}layers.t7'.format(
        args.init,
        args.gcn_type,
        args.num_layers)

    if cfg['DATASET']['LABEL_TYPE'] == 'binary':
        from networks.binary_CAG import CAGpool
    elif cfg['DATASET']['LABEL_TYPE'] == 'multi':
        from networks.multi_CAG import CAGpool
        from networks.multi_GCN import GCN
        from networks.multi_SAG import SAGpool
        from networks.multi_TopK import TopKpool
    else:
        raise NotImplementedError

    if (args.resume or args.phase == 'test'):
        assert os.path.isdir('checkpoints'), 'Error: No checkpoint directory found!'
        if args.model == 'CAGPool':
            model = CAGpool(cfg).to(device)
        elif args.model == 'GCN':
            model = GCN(cfg).to(device)
        elif args.model == 'SAGPool':
            model = SAGpool(cfg).to(device)
        elif args.model == 'TopKPool':
            model = TopKpool(cfg).to(device)
        else:
            raise NotImplementedError

        checkpoint_dict = torch.load('{}/{}'.format(save_dir, file_name))
        model.load_state_dict(checkpoint_dict['model'])
        print('| Loading {} from checkpoint {}...'.format(file_name, checkpoint_dict['epoch']))
    else:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if args.model == 'CAGPool':
            model = CAGpool(cfg).to(device)
        elif args.model == 'GCN':
            model = GCN(cfg).to(device)
        elif args.model == 'SAGPool':
            model = SAGpool(cfg).to(device)
        elif args.model == 'TopKPool':
            model = TopKpool(cfg).to(device)
        else:
            raise NotImplementedError

        checkpoint_dict['model']=model.state_dict()
        checkpoint_dict['auroc']=0
        checkpoint_dict['auprc']=0
        checkpoint_dict['ap50']=0
        checkpoint_dict['epoch']=0
        checkpoint_dict['file_name']='{}/{}'.format(save_dir, file_name)

    return model, checkpoint_dict
