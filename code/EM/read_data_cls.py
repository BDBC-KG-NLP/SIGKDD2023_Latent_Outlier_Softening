import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_transformers import *
import torch.utils.data as Data
import pickle

from model_new_intent import *
from model_encoder import *


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


def get_data_cls(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256, model='bert-base-uncased',
                 train_aug=False, seed=0, use_labels=False, cls="MSP"):
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

    in_distribution_train_labels = np.array([u - 1 for u in in_distribution_train_df[0]])
    in_distribution_train_text = np.array([v for v in in_distribution_train_df[2]])
    in_distribution_train_labels_set = set(in_distribution_train_labels)

    in_distribution_test_labels = np.array([u - 1 for u in in_distribution_test_df[0]])
    in_distribution_test_text = np.array([v for v in in_distribution_test_df[2]])

    n_labels = max(test_labels) + 1
    in_distribution_n_labels = len(in_distribution_train_labels_set)

    print(train_labels_set)
    print(in_distribution_train_labels_set)

    index_select = torch.tensor(list(in_distribution_train_labels_set)).cuda()

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

    if cls == "LMCL":
        cls_encoder = LSTMEncoder(128, 128, 1, 128, 128, in_distribution_n_labels, tokenizer.vocab_size).cuda()
    else:
        cls_encoder = LSTMEncoder(128, 128, 1, 128, 128, in_distribution_n_labels, tokenizer.vocab_size).cuda()
    cls_encoder = nn.DataParallel(cls_encoder)

    # Split the labeled training set, unlabeled training set, development set
    train_labeled_idxs, train_unlabeled_idxs, val_idxs, in_distribution_train_labeled_idxs = train_val_split(
        train_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
        in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)
    test_idxs = test_split(test_labels, n_labeled_per_class, unlabeled_per_class, n_labels, seed=seed,
                           in_dis_labels_set=in_distribution_train_labels_set, labels_set=train_labels_set)

    in_distribution_train_list = [u for u in in_distribution_train_labels_set]
    in_distribution_train_list.sort()
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

    train_labeled_dataset_cls = loader_labeled_choose(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs],
        train_labels_tune[in_distribution_train_labeled_idxs],
        tokenizer,
        max_seq_len, False)
    print(train_labels[in_distribution_train_labeled_idxs])

    labeled_trainloader_cls = Data.DataLoader(
        dataset=train_labeled_dataset_cls, batch_size=8, shuffle=True)

    if cls == "MSP":
        classifier = MSP().cuda()
    elif cls == "LSoftmax":
        classifier = LSoftmax(in_distribution_n_labels, in_distribution_n_labels).cuda()
    elif cls == "DOC":
        classifier = DOC().cuda()
    else:
        classifier = LMCL(2, in_distribution_n_labels).cuda()

    # classifier = nn.DataParallel(classifier)
    lr = 1e-4
    optimizer = AdamW(list(cls_encoder.parameters()) + list(classifier.parameters()), lr=lr)

    train(labeled_trainloader_cls, cls_encoder, classifier, optimizer, cls, epochs=10)

    choose_train_unlabeled_idxs = []
    train_labels_map = {}
    loader = loader_labeled_choose(train_text, train_labels, train_labels_tune, tokenizer, 256, False)

    for unlabeled_id in train_unlabeled_idxs:
        result, length = loader.get_tokenized(train_text[unlabeled_id])
        result = torch.tensor(result).unsqueeze(0).cuda()
        emb = cls_encoder(torch.tensor(result))
        label = torch.tensor(train_labels_tune[unlabeled_id]).unsqueeze(0).cuda()
        if cls == 'LMCL':
            outlier, confidence_score = classifier.predict(emb, index_select, label)
        else:
            outlier, confidence_score = classifier.predict(emb, index_select, threshold=0.85)
        if outlier.item() == 1:
            choose_train_unlabeled_idxs.append(unlabeled_id)
            if train_labels[unlabeled_id] not in train_labels_map.keys():
                train_labels_map[train_labels[unlabeled_id]] = 1
            else:
                train_labels_map[train_labels[unlabeled_id]] += 1

    print(train_labels_map)

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs], tokenizer,
        max_seq_len, False)
    if use_labels:
        if not train_aug:
            train_unlabeled_dataset = loader_labeled(train_text[choose_train_unlabeled_idxs],
                                                     train_labels[choose_train_unlabeled_idxs],
                                                     tokenizer, max_seq_len, False)
        else:
            train_unlabeled_dataset = loader_labeled_aug(train_text[choose_train_unlabeled_idxs],
                                                         choose_train_unlabeled_idxs,
                                                         train_labels[choose_train_unlabeled_idxs],
                                                         tokenizer, max_seq_len,
                                                         Translator(data_path, seed=seed, label=10))
    else:
        if train_aug:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[choose_train_unlabeled_idxs], choose_train_unlabeled_idxs, tokenizer, max_seq_len,
                Translator(data_path,
                           seed=seed,
                           label=10))
        else:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[choose_train_unlabeled_idxs], choose_train_unlabeled_idxs, tokenizer, max_seq_len)
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text[test_idxs], test_labels[test_idxs], tokenizer, max_seq_len)
    test_dataset_tune = loader_labeled_choose(
        test_text[test_idxs], test_labels[test_idxs], test_labels_tune[test_idxs], tokenizer, max_seq_len)
    out_distribution_test_dataset = loader_labeled_choose(
        out_distribution_test_text, out_distribution_test_labels, out_distribution_test_labels_tune, tokenizer,
        max_seq_len)
    in_distribution_test_dataset = loader_labeled(
        in_distribution_test_text, in_distribution_test_labels, tokenizer, max_seq_len)
    print(out_distribution_test_labels)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        in_distribution_train_labeled_idxs), len(choose_train_unlabeled_idxs), len(val_idxs),
        len(test_labels[test_idxs])))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, in_distribution_test_dataset, n_labels, in_distribution_n_labels, cls_encoder, classifier, index_select, out_distribution_test_dataset, test_dataset_tune


def get_data_cls_tune(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256,
                      model='bert-base-uncased',
                      train_aug=False, seed=0, use_labels=False, cls="MSP"):
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

    in_distribution_train_labels = np.array([u - 1 for u in in_distribution_train_df[0]])
    in_distribution_train_text = np.array([v for v in in_distribution_train_df[2]])
    in_distribution_train_labels_set = set(in_distribution_train_labels)

    in_distribution_test_labels = np.array([u - 1 for u in in_distribution_test_df[0]])
    in_distribution_test_text = np.array([v for v in in_distribution_test_df[2]])

    n_labels = max(test_labels) + 1
    in_distribution_n_labels = len(in_distribution_train_labels_set)

    print(train_labels_set)
    print(in_distribution_train_labels_set)

    index_select = torch.tensor(list(in_distribution_train_labels_set)).cuda()

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

    if cls == "LMCL":
        cls_encoder = LSTMEncoder(128, 128, 1, 128, 128, in_distribution_n_labels, tokenizer.vocab_size).cuda()
    else:
        cls_encoder = LSTMEncoder(128, 128, 1, 128, 128, in_distribution_n_labels, tokenizer.vocab_size).cuda()
    cls_encoder = nn.DataParallel(cls_encoder)

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

    # This dataset is to train the classifier and encoder
    train_labeled_dataset_cls = loader_labeled_choose(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs],
        train_labels_tune[in_distribution_train_labeled_idxs],
        tokenizer,
        max_seq_len, False)

    labeled_trainloader_cls = Data.DataLoader(
        dataset=train_labeled_dataset_cls, batch_size=8, shuffle=True)

    if cls == "MSP":
        classifier = MSP().cuda()
    elif cls == "LSoftmax":
        classifier = LSoftmax(in_distribution_n_labels, in_distribution_n_labels).cuda()
    elif cls == "DOC":
        classifier = DOC().cuda()
    else:
        classifier = LMCL(2, in_distribution_n_labels).cuda()

    # classifier = nn.DataParallel(classifier)
    lr = 1e-4
    optimizer = AdamW(list(cls_encoder.parameters()) + list(classifier.parameters()), lr=lr)

    train(labeled_trainloader_cls, cls_encoder, classifier, optimizer, cls, epochs=10)

    choose_train_unlabeled_idxs = []
    train_labels_map = {}
    loader = loader_labeled_choose(train_text, train_labels, train_labels_tune, tokenizer, 256, False)
    # We just use the get_tokenized() of loader
    if n_labeled_per_class == 10:
        threshold = 0.5
    elif n_labeled_per_class == 50:
        threshold = 0.75
    else:
        threshold = 0.85
    for unlabeled_id in train_unlabeled_idxs:
        result, length = loader.get_tokenized(train_text[unlabeled_id])
        result = torch.tensor(result).unsqueeze(0).cuda()
        emb = cls_encoder(torch.tensor(result))
        label = torch.tensor(train_labels_tune[unlabeled_id]).unsqueeze(0).cuda()
        if cls == 'LMCL':
            outlier, confidence_score = classifier.predict(emb, index_select, label)
        else:
            outlier, confidence_score = classifier.predict(emb, index_select, threshold=threshold)
        if outlier.item() == 1:
            choose_train_unlabeled_idxs.append(unlabeled_id)
            if train_labels[unlabeled_id] not in train_labels_map.keys():
                train_labels_map[train_labels[unlabeled_id]] = 1
            else:
                train_labels_map[train_labels[unlabeled_id]] += 1

    print(train_labels_map)

    # Build the dataset class for each set
    train_labeled_dataset = loader_labeled(
        train_text[in_distribution_train_labeled_idxs], train_labels[in_distribution_train_labeled_idxs], tokenizer,
        max_seq_len, False)
    if use_labels:
        if not train_aug:
            train_unlabeled_dataset = loader_labeled(train_text[choose_train_unlabeled_idxs],
                                                     train_labels[choose_train_unlabeled_idxs],
                                                     tokenizer, max_seq_len, False)
        else:
            train_unlabeled_dataset = loader_labeled_aug(train_text[choose_train_unlabeled_idxs],
                                                         choose_train_unlabeled_idxs,
                                                         train_labels[choose_train_unlabeled_idxs],
                                                         tokenizer, max_seq_len,
                                                         Translator(data_path, seed=seed, label=10))
    else:
        if train_aug:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[choose_train_unlabeled_idxs], choose_train_unlabeled_idxs, tokenizer, max_seq_len,
                Translator(data_path,
                           seed=seed,
                           label=10))
        else:
            train_unlabeled_dataset = loader_unlabeled(
                train_text[choose_train_unlabeled_idxs], choose_train_unlabeled_idxs, tokenizer, max_seq_len)
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
        in_distribution_train_labeled_idxs), len(choose_train_unlabeled_idxs), len(val_idxs),
        len(test_labels[test_idxs])))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, in_distribution_test_dataset, n_labels, in_distribution_n_labels, cls_encoder, classifier, index_select, out_distribution_test_dataset, test_dataset_tune, val_dataset_tune


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

        elif n_labels == 10:
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:800])
            else:
                test_idxs.extend(idxs[:1200])


        else:
            # Yahoo/AG News
            if i in in_dis_labels_set:
                test_idxs.extend(idxs[:1200])
            else:
                test_idxs.extend(idxs[:1200])
    np.random.shuffle(test_idxs)
    return test_idxs


def train(labeled_trainloader, encoder, classifier, optimizer, cls, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        for (result, labels, labels_tune, length) in labeled_trainloader:
            result = result.cuda()
            emb = encoder(result)
            logits = classifier(emb, labels.long().cuda(), device=device)
            loss = classifier.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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


class loader_labeled_aug(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, dataset_label, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.labels = dataset_label
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
                    self.labels[idx], (length_u, length_v, length_ori))
        else:
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
