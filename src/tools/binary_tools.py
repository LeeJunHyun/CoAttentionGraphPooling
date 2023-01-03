import argparse
import yaml
import os
import math
import time
import numpy as np
from preprocess.preprocess import DecagonPreprocess
from preprocess.preprocess import printProgress
import torch
import pdb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score, accuracy_score, f1_score

######################################################################################
# >>> TRAIN
######################################################################################
def train_iter(model, loader, optimizer, criterion, device):
    model.train()
    loss_all = 0.
    total = len(loader)

    for i, data in enumerate(loader):
        printProgress(i+1, total, '| Train: ', '', 1, 30)
        data = data.to(device)
        optimizer.zero_grad()

        _, _, output_cls = model(data)

        loss = criterion(output_cls, data.y)
        loss.backward() # Backward propagation
        loss_all += data.num_graphs * loss.item()
        optimizer.step() # Update parameters

    return loss_all / len(loader.dataset)

######################################################################################
# >>> VAL
######################################################################################
def val_iter(model, loader, criterion, device):
    model.eval()
    loss_all = 0.

    total = len(loader)
    for i, data in enumerate(loader):
        printProgress(i+1, total, '| Validation: ', '', 1, 30)
        data = data.to(device)

        _, _, output_cls = model(data)
        loss = criterion(output_cls, data.y)
        loss_all += data.num_graphs * loss.item()

    return loss_all / len(loader.dataset)

######################################################################################
# >>> TRAINVAL
######################################################################################
def trainval_binary(model, train_loader, val_loader, device, criterion, optimizer, epochs=100, scheduler=None):
    best_model = None
    patience = 0
    min_loss = math.inf # from Python 3.5

    for epoch in range(epochs):
        print('\nEpoch: #[{:04d}], lr: {:.8f}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        if scheduler is not None: scheduler.step()
        t = time.time()

        loss_train = train_iter(model, train_loader, optimizer, criterion, device)
        loss_val = val_iter(model, val_loader, criterion, device)
        print("| Loss Train: {}\n| Loss Validation: {}".format(loss_train, loss_val))

        if loss_val < min_loss:
            patience = 0
            min_loss = loss_val
            best_model = model
        else:
            patience += 1
            if patience > 10:
                print("| Early stopping model!!!")
                break

######################################################################################
# >>> TEST
######################################################################################
def test_binary(model, test_loader, device):
    model.eval()
    total = len(test_loader)

    prediction, answer = [], []

    for i, data in enumerate(test_loader):
        printProgress(i+1, total, '| Test: ', '', 1, 30)
        data = data.to(device)
        _, _, output_cls = model(data)

        y_pred = torch.sigmoid(output_cls).data.cpu().detach().numpy()
        y_true = data.y.data.cpu().numpy()

        prediction.append(y_pred[:, 0])
        answer.append(y_true[:, 0])

    y_score = np.concatenate(prediction, axis=0)
    y_true = np.concatenate(answer, axis=0)

    # AUROC
    auroc = roc_auc_score(y_score=y_score, y_true=y_true)
    #precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    return auroc
