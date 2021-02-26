#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: layers.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_hidden_size: int = 768,
                 key_value_hidden_size: int = 768,
                 num_attention_heads: int = 1,
                 attention_head_size: int = 200,
                 dropout_prob: float = 0.1,
                 ):
        super().__init__()
        self.query_hidden_size = query_hidden_size
        self.key_value_hidden_size = key_value_hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = num_attention_heads * attention_head_size

        self.query = nn.Linear(query_hidden_size, self.all_head_size)
        self.key = nn.Linear(key_value_hidden_size, self.all_head_size)
        self.value = nn.Linear(key_value_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(self.all_head_size, self.all_head_size)

    def forward(self, input_query, input_key_value, key_value_mask):
        if key_value_mask is None:
            key_value_mask = input_query.new_ones(input_key_value.size()[:2]).to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = key_value_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0
        key_value_mask = extended_attention_mask

        mixed_query_layer = self.query(input_query)
        mixed_key_layer = self.key(input_key_value)
        mixed_value_layer = self.value(input_key_value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if key_value_mask is not None:
            attention_scores = attention_scores + key_value_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        reshaped_context_layer = context_layer.view(*new_context_layer_shape)
        reshaped_context_layer = self.dropout(self.dense(reshaped_context_layer))
        # [b, max_seq_len, all_head_size]
        return reshaped_context_layer, attention_probs, value_layer

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


class BiMatchGate(nn.Module):
    def __init__(self, hidden_size, source_length, query_length):
        super(BiMatchGate, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.map_linear = nn.Linear(query_length, 1)
        self.bias = nn.Parameter(torch.FloatTensor(source_length))
        torch.nn.init.zeros_(self.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, source, query):
        scores = self.W(source)
        scores = torch.bmm(scores, query.transpose(2, 1))
        scores = self.tanh(scores)
        scores = self.map_linear(scores)
        scores = scores.squeeze()
        gates = self.sigmoid(scores + self.bias)
        return gates


class InGate(nn.Module):
    def __init__(self,
                 hidden_size
                 ):
        super().__init__()
        self.attr_weight = nn.Linear(hidden_size, 1, bias=False)
        self.mm_weight = nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, attr_feats, mm_feats):
        attr_feats = torch.max(attr_feats, 1)[0]
        attr_feats = self.attr_weight(attr_feats)
        mm_feats = self.mm_weight(mm_feats).squeeze()
        regional_gate = self.sigmoid(attr_feats + mm_feats)
        return regional_gate


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32):
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = nn.functional.softmax(masked_vector, dim=dim)
    return result


class GlobalGate(nn.Module):
    def __init__(self,
                 image_hidden_size: int = 2048,
                 txt_hidden_size: int = 768,
                 max_seq_length: int = 46
                 ):
        super().__init__()
        self.txt_weight = nn.Linear(txt_hidden_size, 1, bias=False)
        self.image_weight = nn.Linear(image_hidden_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(max_seq_length))
        torch.nn.init.zeros_(self.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq_txt_features, global_image_features):
        txt_feats = self.txt_weight(seq_txt_features).squeeze()
        img_feats = self.image_weight(global_image_features)
        global_gate = self.sigmoid(txt_feats + img_feats + self.bias)
        return global_gate


class RegionalGate(nn.Module):
    def __init__(self,
                 image_hidden_size: int = 2048,
                 vocab_label_size: int = 28
                 ):
        super().__init__()
        self.label_weight = nn.Linear(vocab_label_size, 1, bias=False)
        self.image_weight = nn.Linear(image_hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred_labels, regional_img_features):
        labels = self.label_weight(pred_labels)
        images = self.image_weight(regional_img_features).squeeze()
        regional_gate = self.sigmoid(labels + images)
        return regional_gate


class LabelProjection(nn.Module):
    def __init__(self,
                 txt_hidden_size: int = 768,
                 all_head_size: int = 200,
                 vocab_label_size: int = 28,
                 dropout_prob: float = 0.0):
        super().__init__()
        self.txt_weight = nn.Linear(txt_hidden_size, all_head_size, bias=False)
        self.mm_weight = nn.Linear(all_head_size, all_head_size, bias=False)
        self.cls_weight = nn.Linear(txt_hidden_size, all_head_size, bias=False)
        self.label_proj = nn.Linear(all_head_size, vocab_label_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, txt_feats, mm_feats, cls_feats):
        txt = self.dropout(self.txt_weight(torch.sum(txt_feats, dim=1)))
        mm = self.dropout(self.mm_weight(torch.sum(mm_feats, dim=1)))
        cls = self.dropout(self.cls_weight(cls_feats))
        logits_label = self.label_proj(txt + mm + cls)
        return logits_label
