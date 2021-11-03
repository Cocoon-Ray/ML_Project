"""General utilities for training.

Author:
    Shrey Desai
"""

import os
import json
import gzip
import pickle

import torch
from tqdm import tqdm

from heapq import heappop, heappush, heapify
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt



def cuda(use_gpu, tensor):
    """
    Places tensor on CUDA device (by default, uses cuda:0).

    Args:
        tensor: PyTorch tensor.

    Returns:
        Tensor on CUDA device.
    """
    if use_gpu and torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def unpack(tensor):
    """
    Unpacks tensor into Python list.

    Args:
        tensor: PyTorch tensor.

    Returns:
        Python list with tensor contents.
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy().tolist()

def load_cached_embeddings(path):
    """
    Loads embedding from pickle cache, if it exists, otherwise embeddings
    are loaded into memory and cached for future accesses.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    bare_path = os.path.splitext(path)[0]
    cached_path = f'{bare_path}.pkl'
    if os.path.exists(cached_path):
        return pickle.load(open(cached_path, 'rb'))
    embedding_map = load_embeddings(path)
    pickle.dump(embedding_map, open(cached_path, 'wb'))
    return embedding_map

def load_embeddings(path="glove.6B.50d.txt"):
    """
    Loads GloVe-style embeddings into memory. This is *extremely slow* if used
    standalone -- `load_cached_embeddings` is almost always preferable.

    Args:
        path: Embedding path, e.g. "glove/glove.6B.300d.txt".

    Returns:
        Dictionary mapping words (strings) to vectors (list of floats).
    """
    embedding_map = {}
    with open(path) as f:
        next(f)  # Skip header.
        for line in f:
            try:
                pieces = line.rstrip().split()
                embedding_map[pieces[0]] = [float(weight) for weight in pieces[1:]]
            except:
                pass
    return embedding_map


# Data Cleaning

# 1. Clean html
def clean_html(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

# 2. remove punctuation marks
def clean_punctuation(sentence):
    cleaned = re.sub(r'[^a-zA-Z0-9.,!?\'\s]', r'', sentence)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

# 3. remove stop word (optional)
def clean_stopword(sentence):
    stop_words = set(stopwords.words('english'))
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    return re_stop_words.sub("", sentence)

# 4. Stemming (optional)
def word_stemming(sentence):
    ps = PorterStemmer()
    stemmed_sentence = ""
    for word in sentence.split():
        stemmed_sentence += ps.stem(word) + " "
    return stemmed_sentence.strip()

# 5. Remove the data where comment is empty and return max_length of all samples
def remove_empty_comment_sample(dataset):
    print(dataset.shape)
    index = []
    max_length = 0
    for i in range(dataset["comment_text"].shape[0]):
        l = len(dataset["comment_text"][i])
        if len(l) == 0:
            index.append(i)
        max_length = max(max_length,l)

    dataset.drop(index)
    print(dataset.shape)
    return max_length

# 6. transform all comments into lowercase and any comments longer than max_length will be truncated.
def format_raw_samples(sentence, max_length):
    comment = sentence.lower()
    comment = comment[:max_length]

    return comment

# investigation:

# 1. find distribution of number of labels:
def find_dist(dataset):
    rowSums = dataset.iloc[:, 2:].sum(axis=1)
    frequencies = list(rowSums)
    c = Counter(frequencies)

    num_of_label = [0, 1, 2, 3, 4, 5, 6]
    freq = [c[i] for i in range(7)]
    print(c, num_of_label, freq)
    plt.figure(figsize=(12, 8))
    plt.plot(num_of_label, freq, 'bo-')
    plt.xlabel('num_of_label')
    plt.ylabel('frequency')
    plt.show()

# 2. find all "good" comments and  "bad" comments
def identify_comments(dataset):
    rowSums = dataset.iloc[:, 2:].sum(axis=1)
    clean_comments_count = (rowSums == 0).sum(axis=0)
    total = len(dataset)
    good = clean_comments_count
    bad = total - good
    plt.figure(figsize=(10, 10), dpi=100)
    explode = (0, 0)
    plt.pie([good, bad], labels=["good", "bad"], explode=explode, autopct="%1.2f%%", colors=['c', 'm'],
            textprops={'fontsize': 24}, labeldistance=1.05, pctdistance=0.85, startangle=90)
    plt.pie([1], radius=0.7, colors='w')
    plt.legend(loc='upper right', fontsize=16)
    plt.axis('equal')
    plt.show()

# 3. find number of comments for each labels:
def comment_per_label(dataset):
    categories = list(dataset.columns.values)
    categories = categories[2:]
    print(categories)

    counts = []
    for category in categories:
        counts.append((category, dataset[category].sum()))  # data_raw.iloc[:,2:].sum().values[i]) 也可以

    plt.bar(range(len(counts)), [x[1] for x in counts], tick_label=[x[0] for x in counts])
    plt.show()

# 4. find the distribution of comments' length
def getLength(sentence):
    words = sentence.split(" ")
    return len(words)
def comment_distribtuion(dataset):
    length = dataset["comment_text"].apply(getLength)
    c = Counter(list(length))
    c = dict(c)
    print(c)
    temp = []
    for k, v in c.items():
        temp.append((k, v))

    temp.sort()
    num_of_words = [k[0] for k in temp]
    freq = [k[1] for k in temp]
    print(num_of_words)
    print(freq)
    plt.figure(figsize=(12, 8))
    plt.plot(num_of_words, freq, 'bo')
    plt.xlabel('num_of_words')
    plt.ylabel('frequency')
    plt.show()

def invistigation(dataset):
    find_dist(dataset)
    identify_comments(dataset)
    comment_per_label(dataset)
    comment_distribtuion(dataset)

# return samples
def preprocessing(path="train.csv"):
    # Load comment data
    dataset = pd.read_csv(path, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    # print(dataset.columns)
    # remove punctuation marks
    dataset["comment_text"] = dataset["comment_text"].apply(clean_punctuation)
    # Clean html
    dataset["comment_text"] = dataset["comment_text"].apply(clean_html)

    # dataset["comment_text"]  = dataset["comment_text"] .apply(clean_stopword)
    # dataset["comment_text"]  = dataset["comment_text"] .apply(word_stemming)  # optional
    max_length = remove_empty_comment_sample(dataset)

    # create samples by turn it to lowercase and truncate/短了的之后在变成vector的时候padding
    dataset["comment_text"] = dataset["comment_text"].apply(format_raw_samples, 150)
    # invistigation(dataset)

    comments = list(dataset["comment_text"])
    toxic = list(dataset["toxic"])
    severe_toxic = list(dataset["severe_toxic"])
    obscene = list(dataset["obscene"])
    threat = list(dataset["threat"])
    insult = list(dataset["insult"])
    identity_hate = list(dataset["identity_hate"])
    zipped = zip(comments, zip(toxic, severe_toxic, obscene, threat, insult,identity_hate))

    return list(zipped)