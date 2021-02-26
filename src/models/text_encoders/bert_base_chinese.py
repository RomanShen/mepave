#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: bert_base_chinese.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import logging

from transformers import BertModel
from torch import nn

logger = logging.getLogger(__name__)


class BertBaseChinese(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        logger.info("currently using bert-base-chinese for text encoding.")

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        return output
