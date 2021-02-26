#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: mjave.py
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import torch
from torch import nn

from src.models.layers import MultiHeadAttention, GlobalGate, RegionalGate, LabelProjection


class MJAVEInteraction(nn.Module):
    def __init__(self,
                 txt_hidden_size: int = 768,
                 image_hidden_size: int = 2048,
                 max_seq_length: int = 46,
                 vocab_label_size: int = 28,
                 vocab_bio_size: int = 55,
                 num_attention_head: int = 1,
                 attention_head_size: int = 200,
                 dropout_prob: float = 0.0):
        super().__init__()

        self.all_head_size = num_attention_head * attention_head_size
        self.vocab_label_size = vocab_label_size

        self.txt_attention = MultiHeadAttention(query_hidden_size=txt_hidden_size,
                                                num_attention_heads=num_attention_head,
                                                key_value_hidden_size=txt_hidden_size,
                                                attention_head_size=attention_head_size,
                                                dropout_prob=dropout_prob
                                                )

        self.cross_modal = MultiHeadAttention(query_hidden_size=txt_hidden_size,
                                              key_value_hidden_size=image_hidden_size,
                                              num_attention_heads=num_attention_head,
                                              attention_head_size=attention_head_size,
                                              dropout_prob=dropout_prob,
                                              )

        self.global_gate = GlobalGate(image_hidden_size=image_hidden_size,
                                      txt_hidden_size=txt_hidden_size,
                                      max_seq_length=max_seq_length
                                      )

        self.regional_gate = RegionalGate(image_hidden_size=image_hidden_size,
                                          vocab_label_size=vocab_label_size
                                          )

        self.label_proj = LabelProjection(txt_hidden_size=txt_hidden_size,
                                          all_head_size=self.all_head_size,
                                          vocab_label_size=vocab_label_size,
                                          dropout_prob=dropout_prob
                                          )

        self.seq_txt_proj = nn.Linear(txt_hidden_size, self.all_head_size, bias=False)
        self.seq_mm_proj = nn.Linear(self.all_head_size, self.all_head_size, bias=False)
        self.label_injection = nn.Linear(vocab_label_size, self.all_head_size, bias=False)
        self.seq_proj = nn.Linear(self.all_head_size, vocab_bio_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                global_img_features,
                regional_img_features,
                cls_txt_features,
                seq_txt_features,
                seq_attention_mask,
                seq_length):
        txt_attn, _, _ = self.txt_attention(seq_txt_features, seq_txt_features, seq_attention_mask)

        img_attn, attn_score, value_layer = self.cross_modal(seq_txt_features, regional_img_features, None)

        global_gate = self.global_gate(seq_txt_features, global_img_features)
        img_attn = torch.mul(global_gate.unsqueeze(-1), img_attn)
        mm_attn = txt_attn + img_attn

        logits_label = self.label_proj(seq_txt_features, mm_attn, cls_txt_features)
        pred_label = self.sigmoid(logits_label)
        label_injection = self.label_injection(pred_label)

        regional_gate = self.regional_gate(pred_label, regional_img_features)
        logit_regional_gate = torch.matmul(regional_gate.unsqueeze(1).unsqueeze(2) * attn_score, value_layer)
        logit_regional_gate = logit_regional_gate.permute(0, 2, 1, 3).contiguous()
        new_logit_regional_gate = logit_regional_gate.size()[:-2] + (self.all_head_size,)
        reshaped_logit_regional_gate = logit_regional_gate.view(*new_logit_regional_gate)

        logits_seq = self.dropout(self.seq_txt_proj(seq_txt_features)) \
                    + self.dropout(self.seq_mm_proj(mm_attn)) \
                    + label_injection.unsqueeze(1) \
                    + reshaped_logit_regional_gate
        logits_seq = self.seq_proj(logits_seq)
        pred_seq = self.softmax(logits_seq)

        return logits_label, pred_label, logits_seq, pred_seq, seq_attention_mask, seq_length

