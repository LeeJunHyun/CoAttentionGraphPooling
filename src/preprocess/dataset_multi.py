# Basic
import networkx as nx
import csv, os, sys
import numpy as np
import torch
import json
import os.path as osp
from pathlib import Path
from ast import literal_eval

# Decoder
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval

# rdkit
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric import data as DATA

# preprocess
from preprocess.molecule_features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, MolFromID
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# print
from utils import printProgress

# >>> Decagon dataset (Multi)
class DecagonDataset_multi(InMemoryDataset):
    def __init__(self, cfg, split, transform=None, pre_transform=None):
        super(DecagonDataset_multi, self).__init__(cfg['DATASET']['DATA_PATH'], transform, pre_transform)
        self.datatype = split
        self.dataset_dir = cfg['DATASET']['PROCESSED_DIR']
        self.total_data_dir = cfg['DATASET']['PROCESSED_PATH']
        with open(cfg['DATASET']['MULTI_LABEL_DICT'], 'r', encoding='utf-8') as f:
            self.json_load = {k:v for k,v in json.load(f).items()}
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0]) # check function

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return ['Decagon-{}-multi.pt'.format(self.datatype)]

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        if osp.exists(os.path.join(self.processed_dir, 'Decagon-{}-multi.pt'.format(self.datatype))):
            return

        data_list = []

        # >>> Obtain One-Hot Encoding for Side-Effects
        json_dict = {literal_eval(k):v for k,v in self.json_load[self.datatype].items()}
        total = len(json_dict)

        for idx, (smiles1, smiles2) in enumerate(json_dict):
            printProgress(idx+1, total, '{} dataset preparation: '.format(self.datatype), ' ', 2, 50)
            mol1 = MolFromSmiles(smiles1)
            mol2 = MolFromSmiles(smiles2)
            label = np.array(json_dict[(smiles1, smiles2)])
            #print(len(label[label == 1]))
            #print(len(label[label == 0]))
            #print("\n{}-[{},{},{}:{}] : {}".format(mode, smiles1, smiles2, se, target_dict[se], label))

            if mol1 is None or mol2 is None:
                print("There is a missing drug from the pair (%s,%s)" %(mol1,mol2))
                continue

            ######################################################################
            # >>> Get pairwise graph G1, G2
            c1_size = mol1.GetNumAtoms()
            c2_size = mol2.GetNumAtoms()

            if c1_size == 0 or c2_size == 0:
                print("There is a size error from pair (%s,%s)" %(mol1,mol2))
                continue

            atoms1 = mol1.GetAtoms(); atoms2 = mol2.GetAtoms()
            bonds1 = mol1.GetBonds(); bonds2 = mol2.GetBonds()

            features, edges = [], []

            for atom in atoms1:
                feature = atom_features(atom)
                features.append(feature/sum(feature)) # normalize
            for atom in atoms2:
                feature = atom_features(atom)
                features.append(feature/sum(feature)) # normalize
            for bond in bonds1:
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            for bond in bonds2:
                edges.append([bond.GetBeginAtomIdx()+c1_size, bond.GetEndAtomIdx()+c1_size])

            if len(edges) == 0:
                continue

            G = nx.Graph(edges).to_directed()
            edge_index = [[e1,e2] for e1,e2 in G.edges]

            GraphSiameseData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1,0), y=torch.Tensor(label).view(1,-1))
            GraphSiameseData.__setitem__('c1_size', torch.LongTensor([c1_size]))
            GraphSiameseData.__setitem__('c2_size', torch.LongTensor([c2_size]))
            data_list.append(GraphSiameseData)
            ###########################################################################

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # check this function
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
