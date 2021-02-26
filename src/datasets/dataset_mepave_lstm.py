#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: romanshen 
@file: dataset_mepave.py
@time: 2021/02/26
@contact: xiangqing.shen@njust.edu.cn
"""

from PIL import Image
from dataclasses import dataclass
import os
import logging
from enum import Enum

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def load_vocabulary(path: str):
    """Load vocabulary from path."""
    with open(path, "r", encoding="utf-8") as f:
        vocab = f.read().strip().split('\n')
    logger.info("loading vocab from: {}, containing words: {}.".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w


@dataclass(frozen=True)
class InputExample:
    """ A single training/test example for MEPAVE """
    image_path: str
    input_seq: str
    output_label: str
    output_seq: str


@dataclass(frozen=True)
class InputFeatures:
    """ A single set of features of data"""
    image: torch.Tensor
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    attention_mask: torch.Tensor
    output_label: torch.Tensor
    output_seq: torch.Tensor


class Split(Enum):
    train = "train"
    valid = "valid"
    test = "test"


class MEPAVEDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 tokenizer: BertTokenizer,
                 transform: transforms,
                 max_seq_length: int,
                 vocab_path: str,
                 mode: Split = Split.train
                 ):
        super().__init__()
        processor = MEPAVEProcessor()
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = max_seq_length

        self.w2i_label, self.i2w_label = load_vocabulary(os.path.join(vocab_path, "vocab.label"))
        self.w2i_bio, self.i2w_bio = load_vocabulary(os.path.join(vocab_path, "vocab.bio"))
        self.w2i_word, self.i2w_word = load_vocabulary(os.path.join(vocab_path, "vocab.word"))

        if mode == Split.valid:
            self.examples = processor.get_valid_examples(data_dir)
        elif mode == Split.test:
            self.examples = processor.get_test_examples(data_dir)
        else:
            self.examples = processor.get_train_examples(data_dir)
        logger.info("Current examples: %s", len(self.examples))

    def __getitem__(self, index):
        item = convert_examples_to_features([self.examples[index]],
                                            self.tokenizer,
                                            self.transform,
                                            self.max_seq_length,
                                            self.w2i_label,
                                            self.i2w_label,
                                            self.w2i_bio,
                                            self.i2w_bio,
                                            self.w2i_word,
                                            self.i2w_word)
        return (item[0].image,
                item[0].input_ids,
                item[0].token_type_ids,
                item[0].attention_mask,
                item[0].output_label,
                item[0].output_seq
                )

    def __len__(self):
        return len(self.examples)


class DataProcessor:
    """Base class for data converters for MEPAVE."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()


class MEPAVEProcessor(DataProcessor):
    """Processor for the MEPAVE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}/train".format(data_dir))
        index = os.path.join(data_dir, "train/indexs")
        input_seq = os.path.join(data_dir, "train/input.seq")
        output_label = os.path.join(data_dir, "train/output.label")
        output_seq = os.path.join(data_dir, "train/output.seq")

        return self._create_examples(os.path.join(data_dir, "product_images"),
                                     index,
                                     input_seq,
                                     output_label,
                                     output_seq)

    def get_valid_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}/valid".format(data_dir))
        index = os.path.join(data_dir, "valid/indexs")
        input_seq = os.path.join(data_dir, "valid/input.seq")
        output_label = os.path.join(data_dir, "valid/output.label")
        output_seq = os.path.join(data_dir, "valid/output.seq")

        return self._create_examples(os.path.join(data_dir, "product_images"),
                                     index,
                                     input_seq,
                                     output_label,
                                     output_seq)

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}/test".format(data_dir))
        index = os.path.join(data_dir, "test/indexs")
        input_seq = os.path.join(data_dir, "test/input.seq")
        output_label = os.path.join(data_dir, "test/output.label")
        output_seq = os.path.join(data_dir, "test/output.seq")

        return self._create_examples(os.path.join(data_dir, "product_images"),
                                     index,
                                     input_seq,
                                     output_label,
                                     output_seq)

    def _create_examples(self, image_dir, index, input_seq, output_label, output_seq):
        examples = []
        raw_index = open(index, "r", encoding="utf-8").read().strip().split("\n")
        raw_input_seq = open(input_seq, "r", encoding="utf-8").read().strip().split("\n")
        raw_output_label = open(output_label, "r", encoding="utf-8").read().strip().split("\n")
        raw_output_seq = open(output_seq, "r", encoding="utf-8").read().strip().split("\n")
        for i in range(len(raw_index)):
            cid, sid = raw_index[i].split("\t")
            input_seq = raw_input_seq[i]
            output_label = raw_output_label[i]
            output_seq = raw_output_seq[i]
            image_path = os.path.join(image_dir, cid.split("_")[1] + ".jpg")
            examples.append(
                InputExample(
                    image_path=image_path,
                    input_seq=input_seq,
                    output_label=output_label,
                    output_seq=output_seq
                )
            )
        return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        transform,
        max_seq_length,
        w2i_label,
        i2w_label,
        w2i_bio,
        i2w_bio,
        w2i_word,
        i2w_word
):
    features = []
    # for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="converting examples to features"):
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        image = Image.open(example.image_path).convert("RGB")
        image = transform(image)

        output_label = [0] * len(w2i_label)
        for w in example.output_label.split(" "):
            if w != "[PAD]":
                output_label[w2i_label[w]] = 1
        output_label = torch.tensor(output_label, dtype=torch.long)

        output_seq = [w2i_bio[w] for w in example.output_seq.split(" ")]
        output_seq.extend([w2i_bio["O"]] * (max_seq_length - len(example.input_seq.split(" ")) - 2))
        output_seq = torch.tensor(output_seq, dtype=torch.long)

        input_embed = [w2i_word[w] if w in w2i_word else w2i_word["[UNK]"] for w in example.input_seq.split(" ")]
        attention_mask = [1] * len(input_embed)
        input_embed.extend([w2i_word["[PAD]"]] * (max_seq_length - len(example.input_seq.split(" ")) - 2))
        attention_mask.extend([0] * (max_seq_length - len(example.input_seq.split(" ")) - 2))
        input_embed = torch.tensor(input_embed, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        features.append(
            InputFeatures(
                image=image,
                input_ids=input_embed,
                token_type_ids=input_embed,
                attention_mask=attention_mask,
                output_label=output_label,
                output_seq=output_seq
            )
        )
    return features
