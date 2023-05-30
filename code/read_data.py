import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
import pickle


class Translator:
    """Backtranslation. Here to save time, we pre-processing and save all the translated data into pickle files.
    """

    def __init__(self, path, transform_type='BackTranslation', seed=0, label=10):
        # Pre-processed German data
        with open(path + 'de_seed{}_nlabels{}.pkl'.format(seed, label), 'rb') as f:
            self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + 'ru_seed{}_nlabels{}.pkl'.format(seed, label), 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        try:
            out1 = self.de[idx]
            out2 = self.ru[idx]
            return out1, out2, ori
        except:
            # print('False_idx')
            return ori, ori, ori


def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased',
             train_aug=False, seed=0, use_labels=False):
    """Read data, split the dataset, and build dataset for dataloaders.

    Arguments:
        data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
        n_labeled_per_class {int} -- Number of labeled data per class

    Keyword Arguments:
        unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
        max_seq_len {int} -- Maximum sequence length (default: {256})
        model {str} -- Model name (default: {'bert-base-uncased'})
        train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

    """
    # Load the tokenizer for bert
    np.set_printoptions(threshold=np.inf)
    tokenizer = BertTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path + 'train.csv', header=None)
    test_df = pd.read_csv(data_path + 'test.csv', header=None)
    in_distribution_test_df = pd.read_csv(data_path + 'in_distribution_test.csv', header=None)
    in_distribution_train_df = pd.read_csv(data_path + 'in_distribution_train.csv', header=None)

    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array([v - 1 for v in train_df[0]])
    train_text = np.array([v for v in train_df[2]])
    train_labels_set = set(train_labels)

    test_labels = np.array([u - 1 for u in test_df[0]])
    test_text = np.array([v for v in test_df[2]])

    in_distribution_test_labels = np.array([u - 1 for u in in_distribution_test_df[0]])
    in_distribution_test_text = np.array([v for v in in_distribution_test_df[2]])

    in_distribution_train_labels = np.array([u - 1 for u in in_distribution_train_df[0]])
    in_distribution_train_text = np.array([v for v in in_distribution_train_df[2]])
    in_distribution_train_labels_set = set(in_distribution_train_labels)

    n_labels = max(test_labels) + 1  # 所有类别的总数，因为test不会调整输入的文件，所以这里用test
    in_distribution_n_labels = len(in_distribution_train_labels_set)  # in_distribution部分类别的个数
    print(in_distribution_train_labels_set)
    print(train_labels_set)

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs, in_distribution_train_labeled_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
        in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)

    print(train_labels[in_distribution_train_labeled_idxs])

    in_distribution_train_list = [u for u in in_distribution_train_labels_set]
    in_distribution_train_list.sort()  # 将类别的标号从小到大排列，方便下面做映射
    print(in_distribution_train_list)
    for i, u in enumerate(train_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            train_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            train_labels[i] = -1

    for i, u in enumerate(in_distribution_train_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            in_distribution_train_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            in_distribution_train_labels[i] = -1

    for i, u in enumerate(test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            test_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            test_labels[i] = -1

    for i, u in enumerate(in_distribution_test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            in_distribution_test_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            in_distribution_test_labels[i] = -1

    # Build the dataset class for each set
    # labeled数据只包括in_distribution部分
    train_labeled_dataset = loader_labeled(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs], tokenizer,
        max_seq_len, False)

    if use_labels:
        if not train_aug:
            train_unlabeled_dataset = loader_labeled(train_text[train_unlabeled_idxs],
                                                     train_labels[train_unlabeled_idxs],
                                                     tokenizer, max_seq_len, False)
        else:
            train_unlabeled_dataset = loader_labeled_aug(train_text[train_unlabeled_idxs], train_unlabeled_idxs,
                                                         train_labels[train_unlabeled_idxs],
                                                         tokenizer, max_seq_len,
                                                         Translator(data_path, seed=seed, label=10))
    else:
        if train_aug:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Translator(data_path,
                                                                                                           seed=seed,
                                                                                                           label=10))
        else:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len)
    train_unlabeled_dataset_labeled = loader_labeled(
        train_text[train_unlabeled_idxs], train_labels[train_unlabeled_idxs], tokenizer, max_seq_len)
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len)
    in_distribution_test_dataset = loader_labeled(
        in_distribution_test_text, in_distribution_test_labels, tokenizer, max_seq_len)
    print(train_labels[in_distribution_train_labeled_idxs])

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        train_labels[in_distribution_train_labeled_idxs]), len(train_labels[train_unlabeled_idxs]), len(val_idxs),
        len(in_distribution_test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, in_distribution_test_dataset, n_labels, in_distribution_n_labels


def get_data_tune(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased',
                  train_aug=False, seed=0, use_labels=False):
    """Read data, split the dataset, and build dataset for dataloaders.

        Arguments:
            data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
            n_labeled_per_class {int} -- Number of labeled data per class

        Keyword Arguments:
            unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
            max_seq_len {int} -- Maximum sequence length (default: {256})
            model {str} -- Model name (default: {'bert-base-uncased'})
            train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

        """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path + 'train.csv', header=None)
    test_df = pd.read_csv(data_path + 'test.csv', header=None)
    in_distribution_test_df = pd.read_csv(data_path + 'in_distribution_test.csv', header=None)
    in_distribution_train_df = pd.read_csv(data_path + 'in_distribution_train.csv', header=None)

    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array([v - 1 for v in train_df[0]])
    train_text = np.array([v for v in train_df[2]])
    train_labels_set = set(train_labels)

    test_labels = np.array([u - 1 for u in test_df[0]])
    test_text = np.array([v for v in test_df[2]])

    in_distribution_test_labels = np.array([u - 1 for u in in_distribution_test_df[0]])
    in_distribution_test_text = np.array([v for v in in_distribution_test_df[2]])

    in_distribution_train_labels = np.array([u - 1 for u in in_distribution_train_df[0]])
    in_distribution_train_text = np.array([v for v in in_distribution_train_df[2]])
    in_distribution_train_labels_set = set(in_distribution_train_labels)

    n_labels = max(test_labels) + 1
    in_distribution_n_labels = len(in_distribution_train_labels_set)
    print(in_distribution_train_labels_set)
    print(train_labels_set)

    # In the tune list, 0 means out distribution, 1 means in distribution
    labels_list = []
    for u in train_df[0]:
        if u - 1 in in_distribution_train_labels_set:
            labels_list.append(1)
        else:
            labels_list.append(0)
    train_labels_tune = np.array(labels_list)

    labels_list = []
    for u in test_df[0]:
        if u - 1 in in_distribution_train_labels_set:
            labels_list.append(1)
        else:
            labels_list.append(0)
    test_labels_tune = np.array(labels_list)

    in_distribution_train_list = [u for u in in_distribution_train_labels_set]
    in_distribution_train_list.sort()
    print(in_distribution_train_list)

    # choose the data of out distribution
    labels_list = []
    out_distribution_test_labels = []
    out_distribution_test_text = []
    for i, u in enumerate(test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            continue
        else:
            labels_list.append(0)
            out_distribution_test_labels.append(-1)
            out_distribution_test_text.append(test_text[i])
    out_distribution_test_labels_tune = np.array(labels_list)
    out_distribution_test_text = np.array(out_distribution_test_text)
    out_distribution_test_labels = np.array(out_distribution_test_labels)
    print(out_distribution_test_labels)
    print(out_distribution_test_labels_tune)

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs, in_distribution_train_labeled_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
        in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)
    test_idxs = test_split(test_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
                           in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)

    print(train_labels[in_distribution_train_labeled_idxs])
    for i, u in enumerate(train_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            train_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            train_labels[i] = -1

    for i, u in enumerate(in_distribution_train_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            in_distribution_train_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            in_distribution_train_labels[i] = -1

    for i, u in enumerate(test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            test_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            test_labels[i] = -1

    for i, u in enumerate(in_distribution_test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            in_distribution_test_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            in_distribution_test_labels[i] = -1

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs], tokenizer,
        max_seq_len, False)
    if use_labels:
        if not train_aug:
            train_unlabeled_dataset = loader_labeled(train_text[train_unlabeled_idxs],
                                                     train_labels[train_unlabeled_idxs],
                                                     tokenizer, max_seq_len, False)
        else:
            train_unlabeled_dataset = loader_labeled_aug(train_text[train_unlabeled_idxs], train_unlabeled_idxs,
                                                         train_labels[train_unlabeled_idxs],
                                                         tokenizer, max_seq_len,
                                                         Translator(data_path, seed=seed, label=10))
    else:
        if train_aug:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Translator(data_path,
                                                                                                           seed=seed,
                                                                                                           label=10))
        else:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len)
    train_unlabeled_set = loader_labeled(
        train_text[train_unlabeled_idxs], train_labels[train_unlabeled_idxs], tokenizer,
        max_seq_len)
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    val_dataset_tune = loader_labeled_choose(
        train_text[val_idxs], train_labels[val_idxs], train_labels_tune[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text[test_idxs], test_labels[test_idxs], tokenizer, max_seq_len)
    test_dataset_tune = loader_labeled_choose(
        test_text[test_idxs], test_labels[test_idxs], test_labels_tune[test_idxs], tokenizer, max_seq_len)
    out_distribution_test_dataset = loader_labeled_choose(
        out_distribution_test_text, out_distribution_test_labels, out_distribution_test_labels_tune, tokenizer,
        max_seq_len)
    in_distribution_test_dataset = loader_labeled(
        in_distribution_test_text, in_distribution_test_labels, tokenizer, max_seq_len)
    print(train_labels[in_distribution_train_labeled_idxs])

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        train_labels[in_distribution_train_labeled_idxs]), len(train_labels[train_unlabeled_idxs]),
        len(train_labels[val_idxs]), len(test_labels[test_idxs])))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, in_distribution_test_dataset, n_labels, in_distribution_n_labels, out_distribution_test_dataset, test_dataset_tune, val_dataset_tune, train_unlabeled_set


def get_data_extreme_tune(data_path, extreme_data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256,
                          model='bert-base-uncased', train_aug=False, seed=0, use_labels=False):
    """Read data, split the dataset, and build dataset for dataloaders.

            Arguments:
                data_path {str} -- Path to your dataset folder: contain a train.csv and test.csv
                n_labeled_per_class {int} -- Number of labeled data per class

            Keyword Arguments:
                unlabeled_per_class {int} -- Number of unlabeled data per class (default: {5000})
                max_seq_len {int} -- Maximum sequence length (default: {256})
                model {str} -- Model name (default: {'bert-base-uncased'})
                train_aug {bool} -- Whether performing augmentation on labeled training set (default: {False})

            """
    # Load the tokenizer for bert
    tokenizer = BertTokenizer.from_pretrained(model)

    train_df = pd.read_csv(data_path + 'in_distribution_train.csv', header=None)
    test_df = pd.read_csv(data_path + 'test.csv', header=None)
    in_distribution_test_df = pd.read_csv(data_path + 'in_distribution_test.csv', header=None)
    in_distribution_train_df = pd.read_csv(data_path + 'in_distribution_train.csv', header=None)

    f_train = open(extreme_data_path + 'train.txt', encoding='utf-8')
    f_test = open(extreme_data_path + 'test.txt', encoding='utf-8')
    train_lines = f_train.readlines()
    test_lines = f_test.readlines()
    for i in range(len(train_lines)):
        train_lines[i] = train_lines[i][train_lines[i].index('\t') + 1:-1]
    for i in range(len(test_lines)):
        test_lines[i] = test_lines[i][test_lines[i].index('\t') + 1:-1]
    train_extreme_text = train_lines[:int(len(train_lines)*0.6)] + test_lines[:int(len(test_lines)*0.6)]
    val_extreme_text = train_lines[int(len(train_lines)*0.6):int(len(train_lines)*0.8)] \
                       + test_lines[int(len(test_lines)*0.6):int(len(test_lines)*0.8)]
    test_extreme_text = train_lines[int(len(train_lines)*0.8):] + test_lines[:int(len(test_lines)*0.8):]
    # 按照6:2:2的比例划分extreme数据，这是将train和test混在一起后的

    train_extreme_text = np.array(train_extreme_text)
    val_extreme_text = np.array(val_extreme_text)
    test_extreme_text = np.array(test_extreme_text)

    train_extreme_labels = np.array([-1] * len(train_extreme_text))
    val_extreme_labels = np.array([-1] * len(val_extreme_text))
    test_extreme_labels = np.array([-1] * len(test_extreme_text))

    train_extreme_labels_tune = np.array([0] * len(train_extreme_text))
    val_extreme_labels_tune = np.array([0] * len(val_extreme_text))
    test_extreme_labels_tune = np.array([0] * len(test_extreme_text))

    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array([v - 1 for v in train_df[0]])
    train_text = np.array([v for v in train_df[2]])
    train_labels_set = set(train_labels)

    test_labels = np.array([u - 1 for u in test_df[0]])
    test_text = np.array([v for v in test_df[2]])

    in_distribution_test_labels = np.array([u - 1 for u in in_distribution_test_df[0]])
    in_distribution_test_text = np.array([v for v in in_distribution_test_df[2]])

    in_distribution_train_labels = np.array([u - 1 for u in in_distribution_train_df[0]])
    in_distribution_train_text = np.array([v for v in in_distribution_train_df[2]])
    in_distribution_train_labels_set = set(in_distribution_train_labels)

    n_labels = max(test_labels) + 1
    in_distribution_n_labels = len(in_distribution_train_labels_set)
    print(in_distribution_train_labels_set)
    print(train_labels_set)

    # In the tune list, 0 means out distribution, 1 means in distribution
    labels_list = []
    for u in train_df[0]:
        if u - 1 in in_distribution_train_labels_set:
            labels_list.append(1)
        else:
            labels_list.append(0)
    train_labels_tune = np.array(labels_list)
    train_labels_extreme_tune = np.array([0] * len(train_extreme_text))

    labels_list = []
    for u in test_df[0]:
        if u - 1 in in_distribution_train_labels_set:
            labels_list.append(1)
        else:
            labels_list.append(0)
    test_labels_tune = np.array(labels_list)
    test_labels_extreme_tune = np.array([0] * len(test_labels_tune))

    in_distribution_train_list = [u for u in in_distribution_train_labels_set]
    in_distribution_train_list.sort()
    print(in_distribution_train_list)

    # choose the data of out distribution
    labels_list = []
    out_distribution_test_labels = []
    out_distribution_test_extreme_labels = np.array([-1] * len(test_extreme_text))
    out_distribution_test_text = []

    for i, u in enumerate(test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            continue
        else:
            labels_list.append(0)
            out_distribution_test_labels.append(-1)
            out_distribution_test_text.append(test_text[i])
    out_distribution_test_labels_tune = np.array(labels_list)
    out_distribution_test_extreme_labels_tune = np.array([0] * len(test_extreme_text))
    out_distribution_test_extreme_labels = np.array([-1] * len(test_extreme_text))
    out_distribution_test_text = np.array(out_distribution_test_text)
    out_distribution_test_extreme_text = test_extreme_text  # 在测试的时候，把原有的Out部分换成extreme，也就是把原来的in和extreme放在一起训练
    out_distribution_test_labels = np.array(out_distribution_test_labels)
    print(out_distribution_test_extreme_labels)
    print(out_distribution_test_extreme_labels_tune)

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs, in_distribution_train_labeled_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
        in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)
    test_idxs = test_split_extreme(test_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
                           in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)

    print(train_labels[in_distribution_train_labeled_idxs])
    for i, u in enumerate(train_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            train_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            train_labels[i] = -1

    for i, u in enumerate(in_distribution_train_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            in_distribution_train_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            in_distribution_train_labels[i] = -1

    for i, u in enumerate(test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            test_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            test_labels[i] = -1

    for i, u in enumerate(in_distribution_test_df[0]):
        if u - 1 in in_distribution_train_labels_set:
            in_distribution_test_labels[i] = in_distribution_train_list.index(u - 1)
        else:
            in_distribution_test_labels[i] = -1

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs], tokenizer,
        max_seq_len, False)
    if use_labels:
        if not train_aug:
            train_unlabeled_dataset = loader_labeled_extreme(train_text[train_unlabeled_idxs],
                                                             train_labels[train_unlabeled_idxs],
                                                             tokenizer, max_seq_len, train_extreme_text, False)
        else:
            train_unlabeled_dataset = loader_labeled_aug(train_text[train_unlabeled_idxs], train_unlabeled_idxs,
                                                         train_labels[train_unlabeled_idxs],
                                                         tokenizer, max_seq_len,
                                                         Translator(data_path, seed=seed, label=10))
    else:
        train_unlabeled_dataset = loader_unlabeled_extreme(
            train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, train_extreme_text)
    train_unlabeled_set = loader_labeled_extreme(
        train_text[train_unlabeled_idxs], train_labels[train_unlabeled_idxs], tokenizer,
        max_seq_len, train_extreme_text, train_extreme_labels)
    val_dataset = loader_labeled_extreme(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len, val_extreme_text, val_extreme_labels)
    val_dataset_tune = loader_labeled_choose_extreme(
        train_text[val_idxs], train_labels[val_idxs], train_labels_tune[val_idxs], tokenizer, max_seq_len, val_extreme_text, val_extreme_labels, val_extreme_labels_tune)
    test_dataset = loader_labeled_extreme(
        test_text[test_idxs], test_labels[test_idxs], tokenizer, max_seq_len, test_extreme_text, test_extreme_labels)
    test_dataset_tune = loader_labeled_choose_extreme(
        test_text[test_idxs], test_labels[test_idxs], test_labels_tune[test_idxs], tokenizer, max_seq_len, test_extreme_text, test_extreme_labels, test_extreme_labels_tune)
    out_distribution_test_dataset = loader_labeled_choose_extreme(
        out_distribution_test_extreme_text, out_distribution_test_extreme_labels, out_distribution_test_extreme_labels_tune, tokenizer,
        max_seq_len, [], [], [])
    in_distribution_test_dataset = loader_labeled(
        in_distribution_test_text, in_distribution_test_labels, tokenizer, max_seq_len)
    print(train_labels[in_distribution_train_labeled_idxs])

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        train_labels[in_distribution_train_labeled_idxs]), len(train_labels[train_unlabeled_idxs]) + len(train_extreme_text),
        len(train_labels[val_idxs]) + len(val_extreme_text), len(test_labels[test_idxs]) + len(test_extreme_text)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, in_distribution_test_dataset, n_labels, in_distribution_n_labels, out_distribution_test_dataset, test_dataset_tune, val_dataset_tune, train_unlabeled_set


def train_val_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0, in_dis_labels_set=None,
                    labels_set=None):
    """Split the original training set into labeled training set, unlabeled training set, development set

    Arguments:
        labels {list} -- List of labeles for original training set
        n_labeled_per_class {int} -- Number of labeled data per class
        unlabeled_per_class {int} -- Number of unlabeled data per class
        n_labels {int} -- The number of classes

    Keyword Arguments:
        seed {int} -- [random seed of np.shuffle] (default: {0})

    Returns:
        [list] -- idx for labeled training set, unlabeled training set, development set
    """
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    in_distribution_train_labeled_idxs = []
    print("The n_labels is {}".format(n_labels))

    for i in labels_set:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if n_labels == 2:
            # IMDB
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            train_unlabeled_idxs.extend(
                idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
            if i in in_dis_labels_set:
                in_distribution_train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
        elif n_labels == 14:
            # DBPedia
            train_pool = np.concatenate((idxs[:500], idxs[10500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            if i in in_dis_labels_set:
                train_unlabeled_idxs.extend(idxs[500: 500 + 3750])
            else:
                train_unlabeled_idxs.extend(idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
            if i in in_dis_labels_set:
                in_distribution_train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
        elif n_labels == 10:
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            if i in in_dis_labels_set:
                train_unlabeled_idxs.extend(idxs[500: 500 + 3333])
            else:
                train_unlabeled_idxs.extend(idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
            if i in in_dis_labels_set:
                in_distribution_train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
        else:
            # Yahoo/AG News
            train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
            train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
            if i in in_dis_labels_set:
                train_unlabeled_idxs.extend(idxs[500: 500 + 5000])
            else:
                train_unlabeled_idxs.extend(idxs[500: 500 + 5000])
            val_idxs.extend(idxs[-2000:])
            if i in in_dis_labels_set:
                in_distribution_train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    np.random.shuffle(in_distribution_train_labeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs, in_distribution_train_labeled_idxs


def test_split(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0, in_dis_labels_set=None,
               labels_set=None):
    np.random.seed(seed)
    labels = np.array(labels)
    test_idxs = []
    for i in labels_set:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if n_labels == 2:
            # IMDB
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:1200])
            else:
                test_idxs.extend(idxs[:1200])

        elif n_labels == 14:
            # DBPedia
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:900])
            else:
                test_idxs.extend(idxs[:1200])


        else:
            # Yahoo/AG News
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:2000])
            else:
                test_idxs.extend(idxs[:2000])
    np.random.shuffle(test_idxs)
    return test_idxs


def test_split_extreme(labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=0, in_dis_labels_set=None,
               labels_set=None):
    np.random.seed(seed)
    labels = np.array(labels)
    test_idxs = []
    for i in labels_set:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if n_labels == 2:
            # IMDB
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:2000])
        elif n_labels == 14:
            # DBPedia
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:2000])
        else:
            # Yahoo/AG News
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:2000])
    np.random.shuffle(test_idxs)
    return test_idxs


class loader_labeled(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        if self.aug is not None:
            u, v, ori = self.aug(self.text[idx], self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)),
                    (length_u, length_v, length_ori))
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)


class loader_labeled_extreme(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, extreme_text, extreme_labels, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        np.append(self.text, extreme_text)
        np.append(self.labels, extreme_labels)
        # self.text.append(extreme_text)
        # self.labels.append(extreme_labels)

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled_extreme(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, extreme_text, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len
        np.append(self.text, extreme_text)

        # self.text.append(extreme_text)

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        text = self.text[idx]
        encode_result, length = self.get_tokenized(text)
        return ((torch.tensor(encode_result), torch.tensor(encode_result), torch.tensor(encode_result)),
                (length, length, length))


class loader_labeled_aug(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, dataset_label, tokenizer, max_seq_len, extreme_text, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.labels = dataset_label
        self.aug = aug
        self.max_seq_len = max_seq_len
        self.text.append(extreme_text)

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        text = self.text[idx]
        encode_result, length = self.get_tokenized(text)
        return (torch.tensor(encode_result), length)


class loader_labeled_choose(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, dataset_label_tune, tokenizer, max_seq_len, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.labels_tune = dataset_label_tune
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], self.labels_tune[idx], length)


class loader_labeled_choose_extreme(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, dataset_label_tune, tokenizer, max_seq_len, extreme_text, extreme_labels, extreme_labels_tune, aug=False):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.labels_tune = dataset_label_tune
        self.max_seq_len = max_seq_len
        np.append(self.text, extreme_text)
        np.append(self.labels, extreme_labels)
        np.append(self.labels_tune, extreme_labels_tune)
        # self.text.append(extreme_text)
        # self.labels.append(extreme_labels)
        # self.labels_tune.append(extreme_labels_tune)

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text, sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]),
                    (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], self.labels_tune[idx], length)