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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from read_data import *
from read_data_cls import get_data_cls
from mixtext import MixText

parser = argparse.ArgumentParser(description='PyTorch MixText')

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
                    help='number of labeled data')

parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')

parser.add_argument('--val-iteration', type=int, default=200,
                    help='number of labeled data')

parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')
parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')
parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')

parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')
parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--cls', default='MSP', type=str,
                    help='one classification of four')
parser.add_argument('--is-cls', default=False, type=bool, metavar='N',
                    help='whether to use cls')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0
print('Whether mix: ', args.mix_option)
print("Mix layers sets: ", args.mix_layers_set)


def main():
    global best_acc
    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=True, seed=args.seed)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=True)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=True)
    in_distribution_test_loader = Data.DataLoader(
        dataset=in_distribution_test_set, batch_size=512, shuffle=True)

    # Define the model, set the optimizer
    model = MixText(in_distribution_n_labels, args.mix_option).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])

    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration

    scheduler = None
    # WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    test_accs = []

    # Start training
    for epoch in range(args.epochs):
        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, in_distribution_n_labels, args.train_aug)

        # scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        # print("epoch {}, train acc {}".format(epoch, train_acc))

        validate_statistic_Q(val_loader, model, criterion, epoch, mode='Test Stats')


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels,
          train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1
        train_aug = False

        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = next(labeled_train_iter)
        train_aug = True
        if train_aug:
            try:
                (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                    length_u2, length_ori) = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                    length_u2, length_ori) = next(unlabeled_train_iter)

            batch_size = inputs_x.size(0)
            batch_size_2 = inputs_ori.size(0)
            targets_x = torch.zeros(batch_size, n_labels).scatter_(
                1, targets_x.view(-1, 1), 1)

            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            inputs_ori = inputs_ori.cuda()

            mask = []

            with torch.no_grad():
                # Predict labels for unlabeled data.
                outputs_u = model(inputs_u)
                outputs_u2 = model(inputs_u2)
                outputs_ori = model(inputs_ori)

                # Based on translation qualities, choose different weights here.
                # For AG News: German: 1, Russian: 0, ori: 1
                # For DBPedia: German: 1, Russian: 1, ori: 1
                # For IMDB: German: 0, Russian: 0, ori: 1
                # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
                p = (1 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
                                                                             dim=1) + 1 * torch.softmax(outputs_ori,
                                                                                                        dim=1)) / (1)
                # Do a sharpen here.
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()


        else:
            try:
                (inputs_ori, length_ori) = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                (inputs_ori, length_ori) = next(unlabeled_train_iter)

            batch_size = inputs_x.size(0)
            batch_size_2 = inputs_ori.size(0)
            targets_x = torch.zeros(batch_size, n_labels).scatter_(
                1, targets_x.long().view(-1, 1), 1)

            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_ori = inputs_ori.cuda()

            mask = []

            with torch.no_grad():
                # Predict labels for unlabeled data.
                outputs_ori = model(inputs_ori)

                # Based on translation qualities, choose different weights here.
                # For AG News: German: 1, Russian: 0, ori: 1
                # For DBPedia: German: 1, Russian: 1, ori: 1
                # For IMDB: German: 0, Russian: 0, ori: 1
                # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
                p = 1 * torch.softmax(outputs_ori, dim=1) / (1)
                # Do a sharpen here.
                pt = p ** (1 / args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

        mixed = 1
        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1 - l)
        else:
            l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_ori, inputs_ori], dim=0)

            all_lengths = torch.cat(
                [inputs_x_length, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u], dim=0)

        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                   all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            logits = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1 - l))
                    if length1 + length2 > 256:
                        length2 = 256 - length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(),
                                       input_b[i][idx2:idx2 + length2],
                                       torch.tensor([0] * (256 - 1 - length1 - length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]],
                                   torch.tensor([0] * (512 - 1 - int(length_a[i]) - int(length_b[i]))).cuda()),
                                  dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits = model(mixed_input, sent_size=512)

                # mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b) / 2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:],
                                       epoch + batch_idx / args.val_iteration, mixed)

        if mix_ == 1:
            loss = Lx + w * Lu
        else:
            loss = Lx + w * Lu + w2 * Lu2

        # max_grad_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))


def test_out_of_distribution_set(testloader, encoder, classifier, criterion, index_select):
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        correct = 0
        total_sample = 0

        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()
            emb = encoder(inputs)
            if args.cls == 'LMCL':
                predicted, confidence_score = classifier.predict(emb, index_select, targets_tune)
            else:
                predicted, confidence_score = classifier.predict(emb, index_select)
            correct += (np.array(predicted) ==
                        np.array(targets_tune.cpu())).sum()
            total_sample += inputs.shape[0]
        acc_total = correct / total_sample

    return acc_total


def test_all_set(testloader, model, encoder, classifier, criterion, index_select):
    model.eval()
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        pred_list = []
        pred_tune_list = []  # 这里面应该都是0和1
        pred_score_list = []
        target_list = []
        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            emb = encoder(inputs)
            targets_tune = targets_tune.cuda()
            # 注意这里增加了置信度获取，到时候测试是否正确
            if args.cls == 'LMCL':
                predicted_out, confidence_score = classifier.predict(emb, index_select, targets_tune)
            else:
                predicted_out, confidence_score = classifier.predict(emb, index_select)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 下面这是为了把预测结果和真实结果中属于out的部分都标成-1
            for idx, _ in enumerate(predicted):
                if predicted_out[idx] == 0:
                    predicted[idx] = -1
            for idx, _ in enumerate(targets):
                if targets_tune[idx] == 0:
                    targets[idx] = -1
            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])
            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            total_sample += inputs.shape[0]
            pred_list.extend(np.array(predicted.cpu()))
            target_list.extend(np.array(targets.cpu()))
            pred_tune_list.extend(np.array(predicted_out))
            pred_score_list.extend(np.array(confidence_score))

        acc_total = correct / total_sample
        f1 = f1_score(target_list, pred_list, average='macro')
        recall = recall_score(target_list, pred_list, average='macro')
        precis = precision_score(target_list, pred_list, average='macro')
        roc_auc = roc_auc_score(pred_tune_list, pred_score_list, average='macro')
        accuracy = accuracy_score(target_list, pred_list)

    return accuracy, f1, roc_auc


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        preds_sum_all_5 = 0
        preds_sum_all_6 = 0
        preds_sum_all_7 = 0
        preds_sum_all_8 = 0
        preds_sum_all_9 = 0
        all_sum = 0

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            all_sum += inputs.size(0)
            preds_sum_5 = 0
            preds_sum_6 = 0
            preds_sum_7 = 0
            preds_sum_8 = 0
            preds_sum_9 = 0
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1)
            # preds = torch.index_select(preds, 1, torch.tensor([2, 3, 5, 6, 7, 8]).cuda())
            preds_max, _ = torch.max(preds, 1)

            preds_statistic_5 = torch.zeros_like(preds_max).cuda()
            preds_statistic_6 = torch.zeros_like(preds_max).cuda()
            preds_statistic_7 = torch.zeros_like(preds_max).cuda()
            preds_statistic_8 = torch.zeros_like(preds_max).cuda()
            preds_statistic_9 = torch.zeros_like(preds_max).cuda()
            targets_tune = torch.zeros_like(targets).cuda()
            targets_tune[torch.where(targets.eq(0))] = 1
            targets_tune[torch.where(targets.eq(1))] = 1
            targets_tune[torch.where(targets.eq(2))] = 1
            targets_tune[torch.where(targets.eq(4))] = 1
            targets_tune[torch.where(targets.eq(5))] = 1
            targets_tune[torch.where(targets.eq(6))] = 1
            targets_tune[torch.where(targets.eq(8))] = 1
            targets_tune[torch.where(targets.eq(12))] = 1
            preds_statistic_5[torch.where(preds_max.gt(0.5))] = 1
            preds_statistic_6[torch.where(preds_max.gt(0.6))] = 1
            preds_statistic_7[torch.where(preds_max.gt(0.7))] = 1
            preds_statistic_8[torch.where(preds_max.gt(0.8))] = 1
            preds_statistic_9[torch.where(preds_max.gt(0.9))] = 1
            preds_statistic_5 = torch.eq(preds_statistic_5, targets_tune).cuda()
            preds_statistic_6 = torch.eq(preds_statistic_6, targets_tune).cuda()
            preds_statistic_7 = torch.eq(preds_statistic_7, targets_tune).cuda()
            preds_statistic_8 = torch.eq(preds_statistic_8, targets_tune).cuda()
            preds_statistic_9 = torch.eq(preds_statistic_9, targets_tune).cuda()
            preds_sum_5 += torch.sum(preds_statistic_5).item()
            preds_sum_6 += torch.sum(preds_statistic_6).item()
            preds_sum_7 += torch.sum(preds_statistic_7).item()
            preds_sum_8 += torch.sum(preds_statistic_8).item()
            preds_sum_9 += torch.sum(preds_statistic_9).item()
            preds_sum_all_5 += torch.sum(preds_statistic_5).item()
            preds_sum_all_6 += torch.sum(preds_statistic_6).item()
            preds_sum_all_7 += torch.sum(preds_statistic_7).item()
            preds_sum_all_8 += torch.sum(preds_statistic_8).item()
            preds_sum_all_9 += torch.sum(preds_statistic_9).item()

            loss = criterion(outputs, targets.to(torch.int64))

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            print("batch id {}, threshold is 0.5".format(batch_idx))
            print(preds_sum_5 / inputs.size(0))
            print("threshold is 0.6")
            print(preds_sum_6 / inputs.size(0))
            print("threshold is 0.7")
            print(preds_sum_7 / inputs.size(0))
            print("threshold is 0.8")
            print(preds_sum_8 / inputs.size(0))
            print("threshold is 0.9".format(batch_idx))
            print(preds_sum_9 / inputs.size(0))

        print("This epoch {} finished.".format(epoch))
        print("threshold is 0.5")
        print(preds_sum_all_5 / all_sum)
        print("threshold is 0.6")
        print(preds_sum_all_6 / all_sum)
        print("threshold is 0.7")
        print(preds_sum_all_7 / all_sum)
        print("threshold is 0.8")
        print(preds_sum_all_8 / all_sum)
        print("threshold is 0.9")
        print(preds_sum_all_9 / all_sum)


def validate_statistic_Q(valloader, model, criterion, epoch, mode):
    model.eval()
    print("This is epoch {}".format(epoch))
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        for lam in range(2, 9):
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
            print(Q_in_0_1 / all_sum_in)
            print(Q_in_1_2 / all_sum_in)
            print(Q_in_2_3 / all_sum_in)
            print(Q_in_3_4 / all_sum_in)
            print(Q_in_4_5 / all_sum_in)
            print(Q_in_5_6 / all_sum_in)
            print(Q_in_6_7 / all_sum_in)
            print(Q_in_7_8 / all_sum_in)
            print(Q_in_8_9 / all_sum_in)
            print(Q_in_9_10 / all_sum_in)
            print("Here is the Q distribution of out_distribution part")
            print(Q_out_0_1 / all_sum_out)
            print(Q_out_1_2 / all_sum_out)
            print(Q_out_2_3 / all_sum_out)
            print(Q_out_3_4 / all_sum_out)
            print(Q_out_4_5 / all_sum_out)
            print(Q_out_5_6 / all_sum_out)
            print(Q_out_6_7 / all_sum_out)
            print(Q_out_7_8 / all_sum_out)
            print(Q_out_8_9 / all_sum_out)
            print(Q_out_9_10 / all_sum_out)


def normalize_entropy_Q(probility, lam):  # [batch_size, num_labels]
    # 将概率中的所有0全部换成1，算出来的结果是一样的，至少我这么觉得
    probility = torch.clip(probility, min=1e-20)
    entropy = -probility * torch.log2(probility)
    entropy = torch.sum(entropy, dim=1)
    entropy = entropy / math.log2(probility.size(1))
    entropy = torch.pow(entropy, lam)
    return entropy, 1 - entropy


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
