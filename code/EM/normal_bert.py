import torch
import torch.nn as nn
from transformers import *


class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2, use_cls=False):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))
        self.use_cls = use_cls

    def forward(self, x, length=256):
        # Encode input text
        all_hidden, pooler = self.bert(x)
        if not self.use_cls:
            pooled_output = torch.mean(all_hidden, 1)
        else:
            pooled_output = all_hidden[:,0,:]
        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)

        return predict
