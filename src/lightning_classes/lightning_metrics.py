#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: lightning_metrics.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import logging

from pytorch_lightning.metrics import Metric
import torch

logger = logging.getLogger(__name__)


class SpanF1Measure(Metric):
    """ metric for bio tagging"""
    def __init__(self, vocab, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_positives", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.vocab = vocab

    def update(self, pred_bios, gold_bios, seq_lens):
        temp_preds = []
        temp_golds = []
        for pred_bio, gold_bio, l in zip(pred_bios, gold_bios, seq_lens):
            pred_bio = torch.argmax(pred_bio, -1)
            temp_preds.append([self.vocab[int(i.detach())] for i in pred_bio[:int(l.detach())]])
            temp_golds.append([self.vocab[int(i.detach())] for i in gold_bio[:int(l.detach())]])
        for i in range(2):
            logger.info("\npreds: {}. golds: {}".format(temp_preds[i], temp_golds[i]))
        true_positives, predicted_positives, actual_positives = compute_f1_score(temp_golds, temp_preds)
        self.true_positives += true_positives
        self.predicted_positives += predicted_positives
        self.actual_positives += actual_positives

    def compute(self):
        if self.predicted_positives > 0:
            precision = 100 * self.true_positives / self.predicted_positives
        else:
            precision = torch.tensor(0, dtype=torch.float)

        if self.actual_positives > 0:
            recall = 100 * self.true_positives / self.actual_positives
        else:
            recall = torch.tensor(0, dtype=torch.float)

        if (precision + recall) > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = torch.tensor(0, dtype=torch.float)

        return precision, recall, f1


def __startOfChunk(prevTag, tag, prevTagType, tagType):
    if prevTag == 'B' and tag == 'B':
        return True
    if prevTag == 'I' and tag == 'B':
        return True
    if prevTag == 'O' and tag == 'B':
        return True
    if prevTag == 'O' and tag == 'I':
        return True
    if tag != 'O' and prevTagType != tagType:
        return True
    return False


def __endOfChunk(prevTag, tag, prevTagType, tagType):
    if prevTag == 'B' and tag == 'B':
        return True
    if prevTag == 'B' and tag == 'O':
        return True
    if prevTag == 'I' and tag == 'B':
        return True
    if prevTag == 'I' and tag == 'O':
        return True
    if prevTag != 'O' and prevTagType != tagType:
        return True
    return False


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def compute_f1_score(gold_slots, pred_slots):
    correctChunkCnt = 0
    goldChunkCnt = 0
    predChunkCnt = 0
    for gold_slot, pred_slot in zip(gold_slots, pred_slots):
        in_correcting = False
        lastGoldTag = 'O'
        lastGoldType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(gold_slot, pred_slot):
            goldTag, goldType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if in_correcting == True:
                if __endOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) == True and \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (lastGoldType == lastPredType):
                    in_correcting = False
                    correctChunkCnt += 1
                elif __endOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (goldType != predType):
                    in_correcting = False

            if __startOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) == True and \
                    __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                    (goldType == predType):
                in_correcting = True

            if __startOfChunk(lastGoldTag, goldTag, lastGoldType, goldType) == True:
                goldChunkCnt += 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                predChunkCnt += 1

            lastGoldTag = goldTag
            lastGoldType = goldType
            lastPredTag = predTag
            lastPredType = predType

        if in_correcting == True:
            correctChunkCnt += 1

    correctChunkCnt = torch.tensor(correctChunkCnt, dtype=torch.float)
    predChunkCnt = torch.tensor(predChunkCnt, dtype=torch.float)
    goldChunkCnt = torch.tensor(goldChunkCnt, dtype=torch.float)

    return correctChunkCnt, predChunkCnt, goldChunkCnt