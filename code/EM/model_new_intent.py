import torch
import torch.nn as nn
import numpy as np
import model_encoder as encoder
import math
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, d_emb, d_hidden, n_layer, d_a, feat_dim, num_classes,
                 n_vocab=None, dropout_lstm=0, dropout_fc=0, embedding=None,
                 key_pretrained='bert-base-uncased', sep_outlier=False):
        super(LSTMEncoder, self).__init__()
        # self.encoder = LstmLayer(n_layer, d_hidden, d_emb, dropout_lstm,
        #                         n_vocab=n_vocab, embedding=embedding)
        self.encoder = encoder.TransformersLayer(key_pretrained, d_ctx=d_hidden)
        self.aggregation = encoder.SelfAttLayer(d_hidden, d_a)
        self.reduction = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(2 * d_hidden, num_classes),
            # nn.ReLU(),
            # nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        emb_ctx = self.encoder(x)  # emb_ctx[bsz, max_len, d_ctx*2]
        emb_aggregate = self.aggregation(emb_ctx)  # [bsz, 2*d_ctx]
        emb_reduction = self.reduction(emb_aggregate)  # [bsz, dim_feature]
        return emb_reduction


class MSP(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MSP, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, *args, **kwargs):
        logits = feat
        if labels is not None:
            self.loss = self.CE(logits, labels)
        return logits

    def predict(self, feat, index_select, threshold=0.6):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: outlier_np: numpy array [n_test] in {-1, 1}
        prop = self.softmax(feat)
        # prop = torch.index_select(prop, 1, index_select)
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        outlier_np = np.zeros(confidence_np.shape)
        outlier_np[confidence_np < threshold] = -1
        outlier_np[confidence_np >= threshold] = 1
        confidence_score_np = np.zeros(confidence_np.shape)
        for idx, data in enumerate(outlier_np):
            if data == 1:
                confidence_score_np[idx] = confidence_score[idx]
            else:
                confidence_score_np[idx] = confidence_score[idx]
        return outlier_np, confidence_score_np

    def score_samples(self, feat):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: confidence_np: numpy array [n_test]
        prop = self.softmax(torch.from_numpy(feat))
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        return confidence_np


class LSoftmax(nn.Module):
    def __init__(self, num_classes, feat_dim, **kwargs):
        super(LSoftmax, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, *args, **kwargs):
        logits = self.fc(feat)
        if labels is not None:
            self.loss = self.CE(logits, labels)
        return logits

    def predict(self, feat, index_select, threshold=0.5):
        prop = self.fc(feat)
        # prop = torch.index_select(prop, 1, index_select)
        # prop = torch.index_select(prop, 1, index_select)
        confidence_score = torch.max(prop, dim=1)[1]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        outlier_np = np.zeros(confidence_np.shape)
        outlier_np[confidence_np < threshold] = -1
        outlier_np[confidence_np >= threshold] = 1
        confidence_score_np = np.zeros(confidence_np.shape)
        for idx, data in enumerate(outlier_np):
            if data == 1:
                confidence_score_np[idx] = confidence_score[idx]
            else:
                confidence_score_np[idx] = 1 - confidence_score[idx]
        return outlier_np, confidence_score_np


class DOC(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DOC, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, device=None, *args, **kwargs):
        logits = self.sigmoid(feat)
        # logits = feat
        if labels is not None:
            batch_size, n_c = logits.shape
            onehot = torch.zeros(batch_size, n_c).to(device).scatter_(1, labels.unsqueeze(1), 1)
            onehot_neg = torch.ones(batch_size, n_c).to(device).scatter_(1, labels.unsqueeze(1), 0)
            p = logits.mul(onehot) - logits.mul(onehot_neg) + onehot_neg + 0.0001
            p_log = p.log()
            self.loss = torch.sum(-p_log)
            if self.loss.item() != self.loss.item():
                # print('nan')
                pass
            # self.loss = self.CE(logits, labels)
        return logits

    def predict(self, feat, index_select, threshold=0.75):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: outlier_np: numpy array [n_test] in {-1, 1}
        prop = self.sigmoid(feat)
        # prop = torch.index_select(prop, 1, index_select)
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        outlier_np = np.zeros(confidence_np.shape)
        outlier_np[confidence_np < threshold] = -1
        outlier_np[confidence_np >= threshold] = 1
        confidence_score_np = np.zeros(confidence_np.shape)
        for idx, data in enumerate(outlier_np):
            if data == 1:
                confidence_score_np[idx] = confidence_score[idx]
            else:
                confidence_score_np[idx] = 1 - confidence_score[idx]
        return outlier_np, confidence_score_np

    def score_samples(self, feat):
        # input: feat: numpy array [n_test, n_seen_class]
        # output: confidence_np: numpy array [n_test]
        prop = self.sigmoid(torch.from_numpy(feat))
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        return confidence_np


class LMCL(nn.Module):
    def __init__(self, num_classes, feat_dim, s=30, m=0.35, **kwargs):
        super(LMCL, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.weights)
        self.CE = nn.CrossEntropyLoss()

    def forward(self, feat, labels=None, *args, **kwargs):
        assert feat.size(1) == self.feat_dim, 'embedding size wrong'
        logits = F.linear(F.normalize(feat), F.normalize(self.weights))
        if labels is not None:
            margin = torch.zeros_like(logits)
            index = labels.view(-1, 1).long()
            margin.scatter_(1, index, self.m)
            m_logits = self.s * (logits - margin)
            self.loss = self.CE(m_logits, labels)
            return m_logits
        return logits

    def predict(self, feat, index_select, labels=None, threshold=0.5):
        logits = F.linear(F.normalize(feat), F.normalize(self.weights))
        if labels is not None:
            margin = torch.zeros_like(logits)
            index = labels.view(-1, 1).long()
            margin.scatter_(1, index, self.m)
            m_logits = self.s * (logits - margin)
            logits = m_logits
        prop = logits
        confidence_score = torch.max(prop, dim=1)[0]
        confidence_np = confidence_score.detach().cpu().clone().numpy()
        outlier_np = np.zeros(confidence_np.shape)
        outlier_np[confidence_np < threshold] = -1
        outlier_np[confidence_np >= threshold] = 1
        confidence_score_np = np.zeros(confidence_np.shape)
        for idx, data in enumerate(outlier_np):
            if data == 1:
                confidence_score_np[idx] = confidence_score[idx]
            else:
                confidence_score_np[idx] = 1 - confidence_score[idx]
        return outlier_np, confidence_score_np
