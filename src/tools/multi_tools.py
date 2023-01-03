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
def train_multi(model, loader, optimizer, criterion, device):
    model.train()
    loss_all = 0.
    total = len(loader)

    for i, data in enumerate(loader):
        printProgress(i+1, total, '| Train: ', '', 1, 30)
        data = data.to(device)
        optimizer.zero_grad()

        _, _, output_cls = model(data)
        # >>> ignore gradients where the label is -1
        mask = (data.y != -1)
        loss = criterion(output_cls[mask], data.y[mask])
        loss.backward() # Backward propagation
        loss_all += data.num_graphs * loss.item()
        optimizer.step() # Update parameters

    return loss_all / len(loader.dataset)

######################################################################################
# >>> VAL
######################################################################################
def val_multi(model, loader, criterion, device):
    model.eval()
    loss_all = 0.

    total = len(loader)
    for i, data in enumerate(loader):
        printProgress(i+1, total, '| Validation: ', '', 1, 30)
        data = data.to(device)
        _, _, output_cls = model(data)
        mask = (data.y != -1)
        loss = criterion(output_cls[mask], data.y[mask]) # >>> ignore where label is -1
        loss_all += data.num_graphs * loss.item()

    return loss_all / len(loader.dataset)

######################################################################################
# >>> TRAINVAL
######################################################################################
def trainval_multi(model, train_loader, val_loader, test_loader, device, criterion, optimizer, epochs, checkpoint_dict):
    best_model = None
    patience = 0
    min_loss = math.inf # from Python 3.5

    for epoch in range(epochs):
        if epoch == 50: optimizer.param_groups[0]['lr'] *= 0.1
        elif epoch == 75: optimizer.param_groups[0]['lr'] *= 0.1
        print('\nEpoch: #[{:04d}], lr: {:.8f}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        #if scheduler is not None: scheduler.step()
        t = time.time()

        loss_train = train_multi(model, train_loader, optimizer, criterion, device)
        #loss_val = val_multi(model, val_loader, criterion, device)
        #print("| Loss Train: {}\n| Loss Validation: {}".format(loss_train, loss_val))
        auroc, auprc, ap = test_multi(model, test_loader, device)
        print("| AUROC (Test) : {}".format(auroc))
        print("| AUPRC (Test) : {}".format(auprc))
        print("| AP@50 (Test) : {}".format(ap))

        #if loss_val < min_loss:
        #print("| Model updated to the last version.")
        save_point = checkpoint_dict['file_name']
        torch.save(checkpoint_dict, save_point) # save model
        patience = 0
        #min_loss = loss_val
        best_model = model
        #else:
        #    patience += 1
        #    if patience > 10:
        #        print("| Early stopping model!!!")
        #        break

######################################################################################
# >>> TEST
######################################################################################
def test_multi(model, test_loader, device):
    model.eval()
    total = len(test_loader)

    prediction, answer = [], []

    for i, data in enumerate(test_loader):
        printProgress(i+1, total, '| Test: ', '', 1, 30)
        data = data.to(device)

        _, _, output_cls = model(data)

        mask = (data.y != -1)
        y_pred = torch.sigmoid(output_cls)[mask].view(-1, 1).data.cpu().detach().numpy()
        y_true = data.y[mask].view(-1, 1).data.cpu().numpy()

        prediction.append(y_pred)
        answer.append(y_true)

    y_score = np.concatenate(prediction, axis=0)
    y_true = np.concatenate(answer, axis=0)

    # AUROC
    auroc = roc_auc_score(y_score=y_score[y_true != -1], y_true=y_true[y_true != -1])
    precision, recall, thresholds = precision_recall_curve(y_true[y_true != -1], y_score[y_true != -1])
    auprc = auc(recall, precision)
    idx_50 = np.argsort(np.absolute(y_score[y_true != -1] - 0.5))[-50:]
    ap50 = average_precision_score(y_true[y_true != -1][idx_50], y_score[y_true != -1][idx_50])

    a = (y_true[y_true != -1])
    b = a[a != 1]
    c = b[b != 0]
    acc_pred = list(map(int, (y_score[y_true != -1] > 0.5)))
    acc = accuracy_score(y_true[y_true != -1], acc_pred)
    print('ACC : {}'.format(acc))

    return auroc, auprc, ap50
