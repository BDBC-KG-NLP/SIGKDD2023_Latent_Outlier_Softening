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

        validate_Q(val_loader, model, criterion, epoch, mode='Valid Stats')


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
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            pred_list.extend(np.array(predicted.cpu()))
            target_list.extend(np.array(targets.cpu()))

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        f1 = f1_score(target_list, pred_list, average='macro')
        recall = recall_score(target_list, pred_list, average='macro')
        precis = precision_score(target_list, pred_list, average='macro')

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, acc_total, f1, recall, precis


def validate_Q(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        for lam in Range(0.5, 1.6, 0.1):
            print("Now the lamda is {}".format(lam))
            all_sum_in = 0
            all_sum_out = 0
            Q_in_0_1 = 0
            Q_in_1_2 = 0
            Q_in_2_3 = 0
            Q_in_3_4 = 0
            Q_in_4_5 = 0
            Q_in_5_6 = 0
            Q_in_6_7 = 0
            Q_in_7_8 = 0
            Q_in_8_9 = 0
            Q_in_9_10 = 0
            Q_out_0_1 = 0
            Q_out_1_2 = 0
            Q_out_2_3 = 0
            Q_out_3_4 = 0
            Q_out_4_5 = 0
            Q_out_5_6 = 0
            Q_out_6_7 = 0
            Q_out_7_8 = 0
            Q_out_8_9 = 0
            Q_out_9_10 = 0

            for batch_idx, (inputs, targets, length) in enumerate(valloader):
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                outputs = model(inputs)
                preds = torch.softmax(outputs, dim=1)
                # preds = torch.index_select(preds, 1, torch.tensor([2, 3, 5, 6, 7, 8]).cuda())
                Q_0, Q_1 = normalize_entropy_Q(preds, lam=lam)
                all_sum_in += targets[torch.where(targets.ne(-1))].size(0)
                all_sum_out += targets[torch.where(targets.eq(-1))].size(0)

                Q_1_in = Q_1[torch.where(targets.ne(-1))]
                Q_1_out = Q_1[torch.where(targets.eq(-1))]
                Q_in_0_1 += Q_1_in[torch.where(Q_1_in.le(0.1))].size(0)
                Q_in_1_2 += Q_1_in[torch.where(Q_1_in.gt(0.1) & Q_1_in.le(0.2))].size(0)
                Q_in_2_3 += Q_1_in[torch.where(Q_1_in.gt(0.2) & Q_1_in.le(0.3))].size(0)
                Q_in_3_4 += Q_1_in[torch.where(Q_1_in.gt(0.3) & Q_1_in.le(0.4))].size(0)
                Q_in_4_5 += Q_1_in[torch.where(Q_1_in.gt(0.4) & Q_1_in.le(0.5))].size(0)
                Q_in_5_6 += Q_1_in[torch.where(Q_1_in.gt(0.5) & Q_1_in.le(0.6))].size(0)
                Q_in_6_7 += Q_1_in[torch.where(Q_1_in.gt(0.6) & Q_1_in.le(0.7))].size(0)
                Q_in_7_8 += Q_1_in[torch.where(Q_1_in.gt(0.7) & Q_1_in.le(0.8))].size(0)
                Q_in_8_9 += Q_1_in[torch.where(Q_1_in.gt(0.8) & Q_1_in.le(0.9))].size(0)
                Q_in_9_10 += Q_1_in[torch.where(Q_1_in.gt(0.9))].size(0)
                Q_out_0_1 += Q_1_out[torch.where(Q_1_out.le(0.1))].size(0)
                Q_out_1_2 += Q_1_out[torch.where(Q_1_out.gt(0.1) & Q_1_out.le(0.2))].size(0)
                Q_out_2_3 += Q_1_out[torch.where(Q_1_out.gt(0.2) & Q_1_out.le(0.3))].size(0)
                Q_out_3_4 += Q_1_out[torch.where(Q_1_out.gt(0.3) & Q_1_out.le(0.4))].size(0)
                Q_out_4_5 += Q_1_out[torch.where(Q_1_out.gt(0.4) & Q_1_out.le(0.5))].size(0)
                Q_out_5_6 += Q_1_out[torch.where(Q_1_out.gt(0.5) & Q_1_out.le(0.6))].size(0)
                Q_out_6_7 += Q_1_out[torch.where(Q_1_out.gt(0.6) & Q_1_out.le(0.7))].size(0)
                Q_out_7_8 += Q_1_out[torch.where(Q_1_out.gt(0.7) & Q_1_out.le(0.8))].size(0)
                Q_out_8_9 += Q_1_out[torch.where(Q_1_out.gt(0.8) & Q_1_out.le(0.9))].size(0)
                Q_out_9_10 += Q_1_out[torch.where(Q_1_out.gt(0.9))].size(0)
            print("Here is the Q distribution of in_distribution part")
            print((Q_in_0_1 + Q_in_1_2 + Q_in_2_3 + Q_in_3_4 + Q_in_4_5) / all_sum_in)
            print((Q_in_5_6 + Q_in_6_7 + Q_in_7_8 + Q_in_8_9 + Q_in_9_10) / all_sum_in)
            print("Here is the Q distribution of out_distribution part")
            print((Q_out_0_1 + Q_out_1_2 + Q_out_2_3 + Q_out_3_4 + Q_out_4_5) / all_sum_out)
            print((Q_out_5_6 + Q_out_6_7 + Q_out_7_8 + Q_out_8_9 + Q_out_9_10) / all_sum_out)


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


def normalize_entropy_Q(probility, lam):  # [batch_size, num_labels]
    # 将概率中的所有0全部换成1，算出来的结果是一样的，至少我这么觉得
    probility = torch.clip(probility, min=1e-20)
    entropy = -probility * torch.log2(probility)
    entropy = torch.sum(entropy, dim=1)
    entropy = entropy / math.log2(probility.size(1))
    entropy = torch.pow(entropy, lam)
    return entropy, 1 - entropy


class Range():
    def __init__(self, start, end, step):
        self.start = start - step
        self.end = end
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        self.start += self.step
        if self.start >= self.end:
            raise StopIteration
        return self.start


if __name__ == '__main__':
    main()
