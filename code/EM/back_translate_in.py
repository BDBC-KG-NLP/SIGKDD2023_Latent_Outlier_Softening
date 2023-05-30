import pandas as pd
import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lan', default='de')
    parser.add_argument('--n_labels', type=int, default=10)
    parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                        help='path to data folders')
    parser.add_argument('--gpu', default='0,1,2,3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    en2ru = torch.hub.load('/home/LAB/chenjc/.cache/torch/hub/pytorch_fairseq_main', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe',source='local')
    ru2en = torch.hub.load('/home/LAB/chenjc/.cache/torch/hub/pytorch_fairseq_main', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe',source='local')
    en2de = torch.hub.load('/home/LAB/chenjc/.cache/torch/hub/pytorch_fairseq_main', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe',source='local')
    de2en = torch.hub.load('/home/LAB/chenjc/.cache/torch/hub/pytorch_fairseq_main', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe',source='local')
    en2ru.cuda()
    ru2en.cuda()
    en2de.cuda()
    de2en.cuda()
    path = args.data_path
    train_df = pd.read_csv(path+'in_distribution_train.csv', header=None)
    test_df = pd.read_csv(path+'test.csv', header=None)
    train_df.head()
    train_labels = [v-1 for v in train_df[0]]
    train_text = [v for v in train_df[2]]
    train_labels_set = set(train_labels)
    test_labels = [u-1 for u in test_df[0]]

    def train_val_split(labels, n_labeled_per_class, n_labels, seed = 0, labels_set = None):
        np.random.seed(seed)
        labels = np.array(labels)
        train_labeled_idxs = []
        train_unlabeled_idxs = []
        val_idxs = []

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
            elif n_labels == 10:
                # DBPedia
                train_pool = np.concatenate((idxs[:500], idxs[10500:-2000]))
                train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
                train_unlabeled_idxs.extend(
                    idxs[500: 500 + 5000])
                val_idxs.extend(idxs[-2000:])
            else:
                # Yahoo/AG News
                train_pool = np.concatenate((idxs[:500], idxs[5500:-2000]))
                train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
                train_unlabeled_idxs.extend(
                    idxs[500: 500 + 5000])
                val_idxs.extend(idxs[-2000:])

        np.random.shuffle(train_labeled_idxs)
        np.random.shuffle(train_unlabeled_idxs)
        np.random.shuffle(val_idxs)
        return train_labeled_idxs, train_unlabeled_idxs, val_idxs

    seed = args.seed
    total_labels = len(train_labels_set)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_labels, args.n_labels, total_labels, seed=seed, labels_set=train_labels_set)
    print('Finish Split !{}'.format(seed))
    idxs = train_unlabeled_idxs
    # back translate using Russian as middle language
    def translate_ru(start, end, file_name):
        print('Translate-Ru')
        trans_result = {}
        for id in tqdm(range(start, end)):
            trans_result[idxs[id]] = ru2en.translate(en2ru.translate(train_text[idxs[id]][:1024],  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
            if id % 500 == 0:
                print('Processing :{}'.format(id/end))
                with open(file_name, 'wb') as f:
                    pickle.dump(trans_result, f)
        with open(file_name, 'wb') as f:
            pickle.dump(trans_result, f)

    # back translate using German as middle language
    def translate_de(start, end, file_name):
        print('Translate-De')
        trans_result = {}
        for id in tqdm(range(start, end)):
            trans_result[idxs[id]] = de2en.translate(en2de.translate(train_text[idxs[id]][:1024],  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
            if id % 500 == 0:
                print('Processing :{}'.format(id/end))
                with open(file_name, 'wb') as f:
                    pickle.dump(trans_result, f)

        with open(file_name, 'wb') as f:
            pickle.dump(trans_result, f)

    max_pos = 5000 * total_labels
    if args.lan == 'de':
        translate_de(0, max_pos, path + 'de_seed{}_nlabels{}.pkl'.format(seed, args.n_labels))
    else:
        translate_ru(0, max_pos, path + 'ru_seed{}_nlabels{}.pkl'.format(seed, args.n_labels))





