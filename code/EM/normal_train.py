import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset

from read_data import get_data
from read_data_cls import get_data_cls
from sklearn.metrics import f1_score, precision_score, recall_score
from normal_bert import ClassificationBert

parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='Number of labeled data')

parser.add_argument('--mix-option', default=False, type=bool, metavar='N',
                    help='mix option')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='aug for training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--use_cls', default=False, type=bool, metavar='N')
parser.add_argument('--seed_p', default=0, type=int)

parser.add_argument('--cls', default='MSP', type=str,
                    help='one classification of four')
parser.add_argument('--is-cls', default=False, type=bool, metavar='N',
                    help='whether to use cls')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0


def main():
    global best_acc
    '''
    if args.is_cls:
        train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels = get_data_cls(
            args.data_path, args.n_labeled, seed=args.seed, cls=args.cls)
    else:
    '''
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels = get_data(
        args.data_path, args.n_labeled, seed=args.seed)

    SEED = args.seed_p
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=True)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=True)
    in_distribution_test_loader = Data.DataLoader(
        dataset=in_distribution_test_set, batch_size=512, shuffle=True)

    model = ClassificationBert(in_distribution_n_labels, args.use_cls).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    criterion = nn.CrossEntropyLoss()

    test_accs = []
    in_distribution_test_accs = []

    for epoch in range(args.epochs):
        train(labeled_trainloader, model, optimizer, criterion, epoch)

        train_loss, train_acc, _, _, _ = validate(
            labeled_trainloader, model, criterion, epoch, mode='Valid Stats')
        val_loss, val_acc, _, _, _ = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        print("PreTraining: epoch {}, train classify loss {}, train acc{}, val acc {}, val_loss {}".format(
            epoch, train_loss, train_acc, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc, f1, recall, precis = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            in_distribution_test_loss, in_distribution_test_acc, in_f1, in_recall, in_precis = validate(
                in_distribution_test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            in_distribution_test_accs.append(in_distribution_test_acc)
            print("epoch {}, test acc {},test loss {}, test_f1{}, tsst_recall{}, test_precise{}".format(
                epoch, test_acc, test_loss, f1, recall, precis))
            print("epoch {}, test acc {},test loss {}, test_f1{}, tsst_recall{}, test_precise{}".format(
                epoch, in_distribution_test_acc, in_distribution_test_loss, in_f1, in_recall, in_precis))

    print('Best val_acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)

    print('In_distribution test acc:')
    print(in_distribution_test_accs)


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        pred_list = []
        target_list = []

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            # loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            pred_list.extend(np.array(predicted.cpu()))
            target_list.extend(np.array(targets.cpu()))

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            # loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        f1 = f1_score(target_list, pred_list, average='macro')
        recall = recall_score(target_list, pred_list, average='macro')
        precis = precision_score(target_list, pred_list, average='macro')

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, acc_total, f1, recall, precis


def train(labeled_trainloader, model, optimizer, criterion, epoch):
    model.train()

    for batch_idx, (inputs, targets, length) in enumerate(labeled_trainloader):
        inputs, targets = inputs.cuda(), targets.type(torch.LongTensor).cuda(non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        print('epoch {}, step {}, loss {}'.format(
            epoch, batch_idx, loss.item()))
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
