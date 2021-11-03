import collections

import numpy as np
import pandas as pd
import re

import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
from random import shuffle
from utils import cuda,preprocessing

from transformers import AutoTokenizer, AutoModel, AutoConfig


PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `CommentDataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}


    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `CommentDataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (comment, _) in samples:
            for token in np.itertools.chain(comment):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words

    def __len__(self):
        return len(self.words)

class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]

class CommentDataset():

    def __init__(self, path):
        self.samples = preprocessing(path) # (comment, label) : string, tuple
        self.tokenizer = None
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0

    def register_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_batch(self, batch_size=20, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            batch_size
        )

    def _create_data_generator(self, shuffle_examples=False):
        """
        Returns:A generator that iterates through all examples one by one.(Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        comments = []
        labels = []
        for idx in example_idxs:
            # Unpack data sample and tokenize comments.
            comment, label = self.samples[idx]

            # Convert words to tensor.
            comment_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(comment)
            )
            # 按理来说应该是一个6D vector
            label_ids = torch.tensor(label)

            # Store each part in an independent list.
            comments.append(comment_ids)
            labels.append(label_ids)

        return zip(comments, labels)

    def _create_batches(self, generator, batch_size):
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            comments = []
            labels = torch.zeros(bsz,6)
            max_comment_length = 0

            # Check max lengths for comments
            for ii in range(bsz):
                comments.append(current_batch[ii][0])
                labels[ii] = current_batch[ii][1]
                max_comment_length = max(max_comment_length, len(current_batch[ii][0]))

            # Assume pad token index is 0. Need to change here if pad token index is other than 0.
            padded_comments = torch.zeros(bsz, max_comment_length)
            # Pad comment
            for iii, comment in enumerate(comments):
                padded_comments[iii][:len(comment)] = comment


            # Create an input dictionary
            batch_dict = {
                'comments': cuda(self.args, padded_comments).long(),
                'labels': cuda(self.args, labels).long(),
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict



class InputDataset(Dataset):
    def __init__(self, data_x, data_y, max_len):
        self.data_x=data_x
        self.data_y=data_y
        self.tokenizer= AutoTokenizer.from_pretrained("albert-xxlarge-v2")
        self.MAX_LEN=max_len
        
    def __getitem__(self, idx):
        text=self.data_x[idx]
        label=torch.tensor(self.data_y[idx], dtype=torch.long)
#       label=torch.tensor(label, dtype=torch.long)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, # whether add the special token [cls] ... or not
            max_length=self.MAX_LEN, 
            return_token_type_ids=True, # segment_id
            pad_to_max_length=True, # whether to pad the segment_id, input_id to the max_len
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'text':text,
            'label':label,
            'input_ids':encoding['input_ids'].flatten(),
            'segment_ids':encoding['token_type_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten()
        }
    
    def __len__(self):
        return len(self.data_x)
    

    
train_dataset=InputDataset(train_X[:256], train_label[:256], tokenizer, 280)
train_dataloader=DataLoader(train_dataset, batch_size=BATCH_SIZE)

dev_dataset=InputDataset(dev_X, dev_label, tokenizer, 280)
dev_dataloader=DataLoader(dev_dataset, batch_size=BATCH_SIZE)


print('Data ready!')

print(train_X[5])