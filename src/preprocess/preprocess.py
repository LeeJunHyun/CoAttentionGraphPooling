# Base
import numpy as np
import csv
import sys
import math
import json
import collections
import random
from pubchempy import get_compounds, Compound
from sklearn.preprocessing import MultiLabelBinarizer

# >>> function for print progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 50):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# >>> function that returns CID dictionary
def __read_CID__(cfg):
    in_file = '{}/chemicals.v5.0.tsv'.format(cfg['DATASET']['DATA_PATH'])
    print(">>> Reading CID dictionary...")
    cid_dict = {}
    with open(in_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        total = sum(1 for row in open(in_file))
        for i, line in enumerate(csv_reader):
            printProgress(i+1, total, '| Preprocessing {} dataset...'.format(cfg['DATASET']['DATA_NAME']), ' ', 1, 50)
            cid = line[0].replace('s', '0').replace('m', '1') # refer pubchem README
            smiles = line[-1]
            cid_dict[cid] = smiles

    return cid_dict

# >>> function that returns a list of labels that have more interactions than NUM_MIN_INTERACTION
def filter_interactions(in_file, cfg):
    print(">>> Filtering Interactions that appear under {}...".format(cfg['DATASET']['NUM_MIN_INTERACTION']))
    label_list, side_effect_list = [], []
    with open(in_file, 'r') as csv_in:
        csv_reader = csv.reader(csv_in, delimiter=','); next(csv_reader, None) # exclude Header

        for line in csv_reader:
            stitch1, stitch2, side_effect, _ = line # Decagon data format
            side_effect_list.append(side_effect)

        counter_dict = collections.Counter(side_effect_list) # create a dictionary with {keys : # of keys in list}

        for cnt in counter_dict:
            if counter_dict[cnt] >= cfg['DATASET']['NUM_MIN_INTERACTION']: label_list.append(cnt)

    return label_list, counter_dict

# >>> function that drops a processed csv file, no returns
def decagon_preprocess(cfg, cid_dict):
    in_file = cfg['DATASET']['COMBO_PATH']
    out_file = cfg['DATASET']['PROCESSED_PATH']

    # get list of labels that have more interactions than NUM_MIN_INTERACTION
    label_list, counter_dict = filter_interactions(in_file, cfg)
    unknown_dict = {}

    with open(in_file, 'r') as csv_in:
        with open(out_file, 'w') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=','); next(csv_reader, None) # ignore Header
            csv_writer = csv.writer(csv_out)

            total = sum(1 for row in open(in_file))
            for line_idx, line in enumerate(csv_reader):
                printProgress(line_idx+1, total, '| Processing {} pairs...'.format(cfg['DATASET']['DATA_NAME']), ' ', 1, 50)
                stitch1, stitch2, side_effect, _ = line # _ : string name for side effects
                if side_effect in label_list:
                    row_string = []
                    for stitch in [stitch1, stitch2]:
                        smiles = None
                        try:
                            smiles = cid_dict[stitch]
                        except KeyError:
                            # Key doesn't exists in dictionary
                            if stitch in unknown_dict:
                                smiles = unknown_dict[stitch] # if already retrieved from pubchempy
                            else:
                                if stitch1[3] == '1' or stitch2[3] == '1':
                                    print("Pair has stereo : [%s,%s]" %(stitch1, stitch2)) # ?
                                print(">>> Adding {} to dictionary...".format(stitch))
                                smiles = Compound.from_cid(int(stitch[4:])).canonical_smiles # Obtain smiles from pubchempy
                                unknown_dict[stitch] = smiles # update unknown dicts for future reference
                        row_string.append(smiles)
                    row_string.append(side_effect)
                    csv_writer.writerow(row_string)

# >>> function for negative sampling, returns a single negative sample for an input triplet (dx, dy, sez)
def negative_sample(line, unique_graphs, unique_pairs, graph_prob, split):
    smiles1, smiles2, side_effect = line

    if split == 'train':
        _smiles1 = smiles1
        _smiles2 = np.random.choice(list(unique_graphs), p=graph_prob) # word2vec random sampling
    else:
        while(True):
            #_smiles1, _smiles2 = np.random.choice(list(unique_graphs), 2, p=graph_prob)
            _smiles1, _smiles2 = random.sample(unique_graphs, 2)
            if (str([_smiles1,_smiles2]) not in unique_pairs and str([_smiles2,_smiles1]) not in unique_pairs):
                break

    return _smiles1, _smiles2, side_effect

# >>> function for train-val-test split
def train_val_test_split(cfg):
    in_file = cfg['DATASET']['PROCESSED_PATH']
    split_dict = {}
    train_lines, val_lines, test_lines = [], [], []

    # >>> count directly from csv files
    side_effect_list = []
    pairs, graphs = [], []
    with open(in_file, 'r') as csv_in:
        csv_reader = csv.reader(csv_in, delimiter=',')
        csv_total = sum(1 for row in open(in_file))
        for i, line in enumerate(csv_reader):
            printProgress(i+1, csv_total, '| Reading pairs from csv...',' ',1,50)
            smiles1, smiles2, side_effect = line
            side_effect_list.append(side_effect)
            pairs.append(str([smiles1,smiles2])) # convert to strings to make a 'set' for pairs
            graphs.append(smiles1)
            graphs.append(smiles2)

    counter_dict = collections.Counter(side_effect_list)
    unique_graphs = set(graphs)
    unique_pairs = set(pairs)

    sigma_sigmoid = sum(list([math.pow(graphs.count(g), 0.75) for g in unique_graphs]))
    graph_prob = [math.pow(graphs.count(g), 0.75)/sigma_sigmoid for g in list(unique_graphs)]
    print("Pre-processing with prior Pr : {}".format(graph_prob))

    for k in counter_dict:
        split_dict[k] = 0

    with open(in_file, 'r') as csv_in:
        csv_reader = csv.reader(csv_in, delimiter=',')

        # >>> define csv
        with open('{}{}-positive-train.csv'.format(cfg['DATASET']['PROCESSED_DIR'], cfg['DATASET']['DATA_NAME']),'w') as train_csv,\
             open('{}{}-positive-val.csv'.format(cfg['DATASET']['PROCESSED_DIR'], cfg['DATASET']['DATA_NAME']),'w') as val_csv,\
             open('{}{}-positive-test.csv'.format(cfg['DATASET']['PROCESSED_DIR'], cfg['DATASET']['DATA_NAME']),'w') as test_csv,\
             open('{}{}-negative-train.csv'.format(cfg['DATASET']['PROCESSED_DIR'], cfg['DATASET']['DATA_NAME']),'w') as neg_train,\
             open('{}{}-negative-val.csv'.format(cfg['DATASET']['PROCESSED_DIR'], cfg['DATASET']['DATA_NAME']),'w') as neg_val,\
             open('{}{}-negative-test.csv'.format(cfg['DATASET']['PROCESSED_DIR'], cfg['DATASET']['DATA_NAME']),'w') as neg_test:

            # >>> define writers
            train_writer = csv.writer(train_csv)
            val_writer = csv.writer(val_csv)
            test_writer = csv.writer(test_csv)
            neg_train_writer = csv.writer(neg_train)
            neg_val_writer = csv.writer(neg_val)
            neg_test_writer = csv.writer(neg_test)

            total = sum(1 for row in open(in_file))
            for i, line in enumerate(csv_reader):
                printProgress(i+1, total, '| Constructing {} train/val/test split...'.format(cfg['DATASET']['DATA_NAME']),' ',1,50)
                smiles1, smiles2, side_effect = line

                if split_dict[side_effect] <= cfg['DATASET']['TRAIN_SPLIT_RATIO'] * counter_dict[side_effect]:
                    train_writer.writerow(line)
                    # >>> negative sampling (train)
                    negative_line = negative_sample(line, unique_graphs, unique_pairs, graph_prob, 'train')
                    neg_train_writer.writerow(negative_line)
                elif split_dict[side_effect] <=\
                        (cfg['DATASET']['TRAIN_SPLIT_RATIO'] + cfg['DATASET']['VAL_SPLIT_RATIO']) * counter_dict[side_effect]:
                    val_writer.writerow(line)
                    # >>> negative sampling (val)
                    negative_line = negative_sample(line, unique_graphs, unique_pairs, graph_prob, 'val')
                    neg_val_writer.writerow(negative_line)
                else:
                    test_writer.writerow(line)
                    # >>> negative sampling (test)
                    negative_line = negative_sample(line, unique_graphs, unique_pairs, graph_prob, 'test')
                    neg_test_writer.writerow(negative_line)

                split_dict[side_effect] += 1 # add count

# >>> function for multi-label composition
def multi_label_processing(cfg):
    print(">>> Extracting Labels...")
    with open(cfg['DATASET']['PROCESSED_PATH'], 'r') as label_csv:
        rdr = csv.reader(label_csv, delimiter=',')
        y = [line[2] for line in rdr]
        labels = list(set(y))
        print(">>> Total Number of Labels : {}".format(len(labels)))

    print(">>> Converting to Multi-Label Configuration...")
    multi_label_dict = {}

    for split in ['train', 'val', 'test']:
        multi_label_dict[split] = {}
        with open('{}{}-positive-{}.csv'.format(cfg['DATASET']['PROCESSED_DIR'],cfg['DATASET']['DATA_NAME'],split),'r') as pos_csv, \
             open('{}{}-negative-{}.csv'.format(cfg['DATASET']['PROCESSED_DIR'],cfg['DATASET']['DATA_NAME'],split),'r') as neg_csv:
            pos_reader = csv.reader(pos_csv, delimiter=',')
            neg_reader = csv.reader(neg_csv, delimiter=',')

            total = sum(1 for row in open('{}{}-positive-{}.csv'.format(cfg['DATASET']['PROCESSED_DIR'],cfg['DATASET']['DATA_NAME'],split),'r'))
            num_overwritten = 0
            for idx, (pos_line, neg_line) in enumerate(zip(pos_reader, neg_reader)):
                printProgress(idx+1, total, '| Binarizing {}-{} dataset...'.format(cfg['DATASET']['DATA_NAME'],split), ' ', 1, 50)
                for if_pos, line in enumerate([neg_line, pos_line]):
                    # if_pos -> 0 for neg_line, 1 for pos_line
                    smiles1, smiles2, cls = line
                    key = str((smiles1, smiles2))

                    if key not in multi_label_dict[split]:
                        multi_label_dict[split][key] = [-1]*len(labels)

                    if multi_label_dict[split][key][labels.index(cls)] == -1:
                        multi_label_dict[split][key][labels.index(cls)] = if_pos
                    else:
                        while  multi_label_dict[split][key][labels.index(cls)] == -1:
                            print("Overwrite occurred on negative sample...")
                            multi_label_dict[split][key][labels.index(cls)] = if_pos
                            num_overwritten += 1

            print(">>> Total {}/{} negative samples have been overwritten".format(num_overwritten, total))

    print(">>> Dumping Multi-Label Dictionary into {}...".format(cfg['DATASET']['MULTI_LABEL_DICT']))
    with open(cfg['DATASET']['MULTI_LABEL_DICT'], 'w') as js:
        #json.dump({str(k):v for k,v in multi_label_dict.items()}, js)
        json.dump(multi_label_dict, js)

# >>> function to read the final preprocessed dataset
def DecagonPreprocess(cfg):
    counter_dict = None
    if cfg['DATASET']['RUN_PREPROCESS']:
        cid_dict = __read_CID__(cfg)
        decagon_preprocess(cfg, cid_dict)
    if cfg['DATASET']['RUN_SPLIT']:
        train_val_test_split(cfg)
    if cfg['DATASET']['LABEL_TYPE'] == 'multi':
        multi_label_processing(cfg)
