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
from sklearn.neighbors import LocalOutlierFactor

from read_data import *
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
    '''
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled, model=args.model, train_aug=True, seed=args.seed)
    '''
    train_labeled_set, train_unlabeled_set, val_set, test_set, in_distribution_test_set, n_labels, in_distribution_n_labels, out_distribution_test_set, test_set_tune, val_set_tune, train_unlabeled_set_labeled = get_data_tune(
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
    test_tune_loader = Data.DataLoader(
        dataset=test_set_tune, batch_size=512, shuffle=True)
    val_tune_loader = Data.DataLoader(
        dataset=val_set_tune, batch_size=512, shuffle=True)

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
    in_distribution_test_accs = []
    out_distribution_test_errs = []
    out_distribution_test_errs1 = []
    all_test_accs = []
    all_test_f1s = []

    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, in_distribution_n_labels, args.train_aug)

        # scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        # print("epoch {}, train acc {}".format(epoch, train_acc))
        '''
        val_loss, val_acc, _ = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')
        '''

        val_acc, val_f1 = test_all_set(val_tune_loader, model)

        print("epoch {}, val acc {}".format(
            epoch, val_acc))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc, f1 = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            in_distribution_test_loss, in_distribution_test_acc, in_distribution_f1 = validate(
                in_distribution_test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            in_distribution_test_accs.append(in_distribution_test_acc)
            out_test_err, out_test_err1 = test_out_of_distribution_set(test_tune_loader, model)
            out_distribution_test_errs.append(out_test_err)
            out_distribution_test_errs1.append(out_test_err1)
            all_acc, all_f1 = test_all_set(test_tune_loader, model)
            all_test_accs.append(all_acc)
            all_test_f1s.append(all_f1)
            print("epoch {}, test acc {},test loss {} test f1{}".format(
                epoch, test_acc, test_loss, f1))
            print("epoch {}, test acc {},test loss {} test f1{}".format(
                epoch, in_distribution_test_acc, in_distribution_test_loss, in_distribution_f1))
            print("epoch {}, out distribution test err {}".format(epoch, out_test_err))
            print("epoch {}, out distribution test err1 {}".format(epoch, out_test_err1))
            print("epoch {}, all test acc {}".format(epoch, all_acc))
            print("epoch {}, all f1 {}".format(epoch, all_f1))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

        print('In_distribution test acc:')
        print(in_distribution_test_accs)

        print('Out_distribution test err:')
        print(out_distribution_test_errs)

        print('Out_distribution test err1:')
        print(out_distribution_test_errs1)

        print('All test acc:')
        print(all_test_accs)

        print('All test f1:')
        print(all_test_f1s)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)

    print('In_distribution test acc:')
    print(in_distribution_test_accs)

    print('Out_distribution test err:')
    print(out_distribution_test_errs)

    print('Out_distribution test err1:')
    print(out_distribution_test_errs1)

    print('All test acc:')
    print(all_test_accs)

    print('All test f1:')
    print(all_test_f1s)


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
            outputs_u, _ = model(inputs_u)
            outputs_u2, _ = model(inputs_u2)
            outputs_ori, _ = model(inputs_ori)

            # Based on translation qualities, choose different weights here.
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            p = (1 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
                                                                         dim=1) + 1 * torch.softmax(outputs_ori,
                                                                                                    dim=1)) / (1)
            # Do a sharpen here.
            '''
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            '''
            targets_u = p
            targets_u = targets_u.detach()

        mixed = 1

        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        # if mix_ == 1:
        #     l = np.random.beta(args.alpha, args.alpha)
        #     if args.separate_mix:
        #         l = l
        #     else:
        #         l = max(l, 1-l)
        # else:
        l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)

            all_lengths = torch.cat(
                [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

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
            logits, _ = model(input_a, input_b, l, mix_layer)
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

            logits, _ = model(mixed_input)
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
                logits, _ = model(mixed_input, sent_size=512)

                # mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b) / 2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits, _ = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:],
                                       epoch + batch_idx / args.val_iteration, mixed)

        if mix_ == 1:
            # loss = Lx + w * Lu
            loss = Lx + w * Lu
        else:
            # loss = Lx + w * Lu + w2 * Lu2
            loss = Lx + w * Lu

        # max_grad_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))
            print("the mix_method is {}, mixed is {}, mix_ is {}".format(args.mix_method, mixed, mix_))


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
            outputs, _ = model(inputs)
            # loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            pred_list.extend(np.array(predicted.cpu()))
            target_list.extend(np.array(targets.cpu()))
            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            # loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct / total_sample
        loss_total = loss_total / total_sample
        f1 = f1_score(target_list, pred_list, average='macro')

    return loss_total, acc_total, f1


def test_out_of_distribution_set(testloader, model):
    with torch.no_grad():
        correct = 0
        total_sample = 0
        wrong = 0
        wrong1 = 0

        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()  # 这里面都是0和1
            outputs, embedding = model(inputs)
            clf = LocalOutlierFactor(n_neighbors=40, leaf_size=40)
            predicted = clf.fit_predict(embedding.cpu())
            targets_tune[torch.where(targets_tune.eq(0))] = -1
            '''
            correct += (np.array(predicted) ==
                        np.array(targets_tune.cpu())).sum()
            '''
            for idx, data in enumerate(np.array(predicted)):
                if data == 1 and np.array(targets_tune.cpu())[idx] == -1:
                    wrong += 1
            for idx, data in enumerate(np.array(predicted)):
                if data == -1 and np.array(targets_tune.cpu())[idx] == 1:
                    wrong1 += 1
            total_sample += inputs.shape[0]
        err_total = wrong / total_sample
        err_total1 = wrong1 / total_sample

    return err_total, err_total1


def test_all_set(testloader, model):
    model.eval()

    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        pred_list = []
        target_list = []
        for batch_idx, (inputs, targets, targets_tune, length) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets_tune = targets_tune.cuda()
            outputs, embedding = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            clf = LocalOutlierFactor(n_neighbors=40, leaf_size=40)
            predicted_out = clf.fit_predict(embedding.cpu())

            for idx, _ in enumerate(predicted_out):
                if predicted_out[idx] != 1:  # predicted_out
                    predicted[idx] = -1

            '''
            for idx, _ in enumerate(targets):
                if targets_tune[idx] == 0:
                    targets[idx] = -1
            '''
            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])
            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            total_sample += inputs.shape[0]
            pred_list.extend(np.array(predicted.cpu()))
            target_list.extend(np.array(targets.cpu()))

        acc_total = correct / total_sample
        f1 = f1_score(target_list, pred_list, average='macro')

        recall = recall_score(target_list, pred_list, average='macro')
        precis = precision_score(target_list, pred_list, average='macro')
        accuracy = accuracy_score(target_list, pred_list)

    return accuracy, f1


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
