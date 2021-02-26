#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: lightning_mjave_bert.py 
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

import logging
import os

import pytorch_lightning as pl
from torch import nn
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from src.datasets import load_vocabulary
from src.models.text_encoders import BertBaseChinese
from src.models.image_encoders import ResNet101
from src.models.mjave import MJAVEInteraction
from src.lightning_classes import SpanF1Measure

logger = logging.getLogger(__name__)


class MjaveBert(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.bert = BertBaseChinese()
        self.resnet = ResNet101()

        self.w2i, self.i2w = load_vocabulary(os.path.join(self.hparams.vocab_path, "vocab.bio"))
        self.vocab_size = len(self.w2i)

        self.mjave = MJAVEInteraction(
            txt_hidden_size=self.hparams.txt_hidden_size,
            image_hidden_size=self.hparams.image_hidden_size,
            max_seq_length=self.hparams.max_seq_length_no_pad,
            vocab_label_size=self.hparams.vocab_label_size,
            vocab_bio_size=self.hparams.vocab_bio_size,
            num_attention_head=self.hparams.num_attention_heads,
            attention_head_size=self.hparams.attention_head_size,
            dropout_prob=self.hparams.dropout_prob
        )

        self.a_loss = nn.BCEWithLogitsLoss()
        self.v_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="mean")

        self.valid_score = SpanF1Measure(self.i2w)
        self.test_score = SpanF1Measure(self.i2w)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        region_feats, global_feats = self.resnet(inputs[0])

        outputs = self.bert(input_ids=inputs[1],
                            attention_mask=inputs[3],
                            token_type_ids=inputs[2])
        seq_txt_features = outputs[0].squeeze()
        cls_txt_features = outputs[1].squeeze()
        new_attention_mask = seq_txt_features.new_zeros(
            (seq_txt_features.size(0), seq_txt_features.size(1) - 2),
        ).to(dtype=next(self.parameters()).dtype)
        new_seq_txt_features = seq_txt_features[:, 1:-1, :]
        sequence_length = torch.sum(inputs[3], -1) - 2
        for i in range(seq_txt_features.size(0)):
            new_attention_mask[i, :int(sequence_length[i])] = seq_txt_features.new_ones((1, int(sequence_length[i])))

        return self.mjave(global_feats,
                          region_feats,
                          cls_txt_features,
                          new_seq_txt_features,
                          new_attention_mask,
                          sequence_length)

    def training_step(self, batch, batch_index):
        logits_label, pred_label, logits_seq, pred_seq, seq_attention_mask, seq_length = self(batch)
        loss_a, loss_v, loss_kl = self.calculate_loss(pred_label,
                                                      logits_label,
                                                      batch[4],
                                                      pred_seq,
                                                      logits_seq,
                                                      batch[5],
                                                      seq_attention_mask)
        loss = loss_a + loss_v + 0.5 * loss_kl
        self.log("training/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index):
        logits_label, pred_label, logits_seq, pred_seq, seq_attention_mask, seq_length = self(batch)
        loss_a, loss_v, loss_kl = self.calculate_loss(pred_label,
                                                      logits_label,
                                                      batch[4],
                                                      pred_seq,
                                                      logits_seq,
                                                      batch[5],
                                                      seq_attention_mask)
        loss = loss_a + loss_v + 0.5 * loss_kl

        p, r, f1 = self.valid_score(pred_seq, batch[5], torch.sum(seq_attention_mask, -1))

        self.log("validation/loss", loss)

    def validation_epoch_end(self, outputs):
        p, r, f1 = self.valid_score.compute()
        self.log("validation/f1", f1, prog_bar=True)

    def test_step(self, batch, batch_index):
        logits_label, pred_label, logits_seq, pred_seq, seq_attention_mask, seq_length = self(batch)
        loss_a, loss_v, loss_kl = self.calculate_loss(pred_label,
                                                      logits_label,
                                                      batch[4],
                                                      pred_seq,
                                                      logits_seq,
                                                      batch[5],
                                                      seq_attention_mask)
        loss = loss_a + loss_v + 0.5 * loss_kl

        p, r, f1 = self.test_score(pred_seq, batch[5], torch.sum(seq_attention_mask, -1))

        self.log("test/loss", loss)

    def test_epoch_end(self, outputs):
        p, r, f1 = self.test_score.compute()
        self.log("test/f1", f1, prog_bar=True)

    def configure_optimizers(self):
        if self.current_epoch >= self.hparams.freeze_bert_epoch:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.adam_weight_decay,
                    "lr": self.hparams.lr_bert
                },
                {
                    "params": [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": self.hparams.lr_bert
                },
                {
                    "params": self.mjave.parameters(),
                    "weight_decay": self.hparams.adam_weight_decay,
                    "lr": self.hparams.lr_classifier
                },
            ]
            optimizer = AdamW(
                optimizer_grouped_parameters,
                eps=self.hparams.adam_epsilon,
                weight_decay=self.hparams.adam_weight_decay
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=111,
                num_training_steps=1112
            )
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        else:
            optimizer = AdamW(
                self.parameters(),
                lr=self.hparams.lr_classifier,
                eps=self.hparams.adam_epsilon,
                weight_decay=self.hparams.adam_weight_decay
            )
            return optimizer

    def on_epoch_start(self):
        if self.current_epoch >= self.hparams.freeze_bert_epoch:
            for param in self.bert.parameters():
                param.requires_grad = True

    def calculate_loss(self,
                       pred_label,
                       logits_label,
                       output_labels,
                       pred_seq,
                       logits_seq,
                       output_seqs,
                       attention_mask):
        loss_a = self.a_loss(logits_label, output_labels.type_as(logits_label))

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits_seq.view(-1, self.hparams.vocab_bio_size)
        active_seqs = torch.where(
            active_loss, output_seqs.view(-1), torch.tensor(self.v_loss.ignore_index).type_as(output_seqs)
        )
        loss_v = self.v_loss(active_logits, active_seqs)

        # KL
        prob1 = pred_label[:, 2:]
        prob2 = torch.max(pred_seq, 1)[0]
        prob2 = prob2[:, 3:]
        index1 = torch.tensor(([2 * i for i in range(self.hparams.vocab_label_size - 2)]), device=self.device)
        index2 = torch.tensor(([2 * i + 1 for i in range(self.hparams.vocab_label_size - 2)]), device=self.device)
        prob2_agg_b = torch.index_select(prob2, 1, index1)
        prob2_agg_i = torch.index_select(prob2, 1, index2)
        prob2_agg = (prob2_agg_b + prob2_agg_i) / 2
        prob1 = torch.clamp(prob1, 1e-8, 1.0)
        prob2_agg = torch.clamp(prob2_agg, 1e-8, 1.0)
        loss_kl = self.kl_loss(torch.log(prob1), prob2_agg)

        return loss_a, loss_v, loss_kl
