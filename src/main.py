# Basic
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
import os, pickle, time, random, math
import torch

# Dataset
from utils import parse_args, model_setup
from utils import prepare_data
from preprocess.dataset_multi import DecagonDataset_multi

# Libraries for training
from torch.optim.lr_scheduler import MultiStepLR

# >>> Main Code
def main(args=None):
    parser, cfg = parse_args(args)
    device = torch.device('cuda')
    dataset={}

    if cfg['DATASET']['LABEL_TYPE'] == 'binary':
        from preprocess.dataset_binary import DecagonDataset_binary as DecagonDataset
        from tools.binary_tools import trainval_binary as trainval
        from tools.binary_tools import test_binary as test
    elif cfg['DATASET']['LABEL_TYPE'] == 'multi':
        from preprocess.dataset_multi import DecagonDataset_multi as DecagonDataset
        from tools.multi_tools import trainval_multi as trainval
        from tools.multi_tools import test_multi as test
    else:
        raise NotImplementedError

    if cfg['SOLVER']['LOSS'] == 'BCE': criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg['SOLVER']['LOSS'] == 'MARGIN': raise NotImplementedError #criterion = torch.nn.MultiLabelSoftMarginLoss()
    else: raise NotImplementedError

    if not os.path.exists(cfg['DATASET']['PROCESSED_DIR']): os.mkdir(cfg['DATASET']['PROCESSED_DIR'])
    prepare_data(cfg)

    phase_list = ['train', 'val', 'test'] if parser.phase=='train' else ['test']

    for split in phase_list:
        print(">>> Preparing {} data...".format(split))
        dataset[split] = DecagonDataset(cfg, split=split)

    if parser.phase == 'train':
        train_loader = DataLoader(dataset['train'], batch_size=cfg['SOLVER']['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(dataset['val'], batch_size=cfg['SOLVER']['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(dataset['test'], batch_size=cfg['SOLVER']['BATCH_SIZE'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint_dict = model_setup(parser, cfg, device)

    if parser.phase == 'train':
        if not os.path.exists(cfg['DATASET']['SAVE_PATH']): os.makedirs(cfg['DATASET']['SAVE_PATH'])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['SOLVER']['BASE_LR'], weight_decay=cfg['SOLVER']['WEIGHT_DECAY'])
        trainval(model, train_loader, val_loader, test_loader, device, criterion, optimizer, cfg['SOLVER']['EPOCHS'], checkpoint_dict)
    elif parser.phase == 'test':
        auroc, auprc, ap = test(model, test_loader, device)
        print("| AUROC (test) : {}".format(auroc))
        print("| AUPRC (test) : {}".format(auprc))
        print("| AP50 (test) : {}".format(ap))

    print(">>> End of program")

if __name__ == "__main__":
    main()
