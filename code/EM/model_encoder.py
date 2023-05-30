import torch
import torch.nn as nn
import transformers
from pytorch_transformers import *


class LstmLayer(nn.Module):
    def __init__(self, n_layer, d_ctx, d_emb, dropout=0,
                 if_freeze=False, n_vocab=None, embedding=None):
        super(LstmLayer, self).__init__()

        self.embed = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.encoder = nn.LSTM(d_emb, d_ctx, n_layer, dropout=dropout,
                               bidirectional=True, batch_first=True)

    def forward(self, x):
        emb = self.embed(x)  # x[bsz, max_len] -> emb[bsz, max_len, dim_emb]
        emb_ctx, _ = self.encoder(emb)  # emb_ctx[bsz, max_len, 2*d_ctx]
        return emb_ctx


class SelfAttLayer(nn.Module):
    def __init__(self, d_ctx, d_att):
        super(SelfAttLayer, self).__init__()
        self.layer_att = nn.Sequential(
            nn.Linear(2 * d_ctx, d_att),
            nn.Tanh(),
            nn.Linear(d_att, 1),
            nn.Softmax(dim=-2),
        )

    def forward(self, emb_ctx):  # [bsz, T, 2*d_ctx]
        attention = self.layer_att(emb_ctx)  # [bsz, T, 1]
        emb_att = torch.mul(emb_ctx, attention)  # [bsz, T, 2*d_ctx]
        emb_aggregate = torch.sum(emb_att, dim=1)  # [bsz, 2*d_ctx]
        return emb_aggregate


def get_info_transfomer(key_pretrained):
    transfomer_info = dict()
    transfomer_info['distilbert-base-uncased'] = dict(
        architecture='DistilBert',
        d_hidden=768,
    )
    transfomer_info['bert-base-uncased'] = dict(
        architecture='Bert',
        d_hidden=768,
    )
    transfomer_info['bert-base-chinese'] = dict(
        architecture='Bert',
        d_hidden=768,
    )
    return transfomer_info[key_pretrained]


class TransformersLayer(nn.Module):
    def __init__(self, key_pretrained, d_ctx=64, dropout=0):
        super(TransformersLayer, self).__init__()
        info = get_info_transfomer(key_pretrained)
        key_architecture = info['architecture']
        d_hidden = info['d_hidden']
        key_model = key_architecture + "Model"
        self.encoder = BertModel.from_pretrained(key_pretrained)
        self.finetune = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 2 * d_ctx),
            nn.Tanh(),
            nn.Linear(2 * d_ctx, 2 * d_ctx),
        )
        print(self.encoder)

    def forward(self, x):
        with torch.no_grad():
            emb, _ = self.encoder(x)
        emb_ctx = self.finetune(emb)
        return emb_ctx
