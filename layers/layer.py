from torch import nn
import torch
import math
import numpy as np


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        # self.hidden = nn.Linear()

        self.gamma_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = nn.Linear(hidden_size, hidden_size, bias=False)

        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)

    def forward(self, x, condition):
        '''

        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        condition = condition.unsqueeze(1).expand_as(x)
        gamma = self.gamma + self.gamma_dense(condition)
        beta = self.beta + self.beta_dense(condition)
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x


class AttentionAddition(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0):
        super(AttentionAddition, self).__init__()
        self.hidden = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.xavier_normal_(self.hidden.weight)

    def forward(self, query, context, mask):
        '''

        :param query: [b, e]
        :param context: [b, t, e]
        :param mask: [b, t], 0 if masked
        :return: [b, e]
        '''
        mask = (mask < 1)
        query = query.unsqueeze(1).expand_as(context)  # [b, t, e]
        tmp = torch.cat([query, context], dim=-1)  # [b, t, 2e]
        tmp = torch.tanh(tmp)  # [b, t, 2e]
        scores = self.v(tmp).squeeze(-1)  # [b, t]
        scores = self.dropout(scores)
        scores = scores.masked_fill_(mask, -1e10)
        scores = torch.softmax(scores, dim=-1)
        scores = scores.unsqueeze(1)  # [b, 1, t]
        context = torch.bmm(scores, context).squeeze(1)  # [b, e]; [b, 1, e] = [b, 1, t] * [b, t, e]
        return context


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size  x seq_length]
            mask is 0 if it is masked

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        mask = mask. \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output
