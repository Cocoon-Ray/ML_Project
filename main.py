
import argparse
import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data import CommentDataset, InputDataset, Tokenizer, Vocabulary
from utils import cuda,preprocessing

from model import Classifier, MyBert
from utils import cuda, unpack

_TQDM_BAR_SIZE = 75
_TQDM_LEAVE = False
_TQDM_UNIT = ' batches'
_TQDM_OPTIONS = {
    'ncols': _TQDM_BAR_SIZE, 'leave': _TQDM_LEAVE, 'unit': _TQDM_UNIT
}

"""
    选用哪种loss要看model是怎么设计的
    Returns: Loss value for a batch of samples.
"""
def calculate_loss():
    return

def train(args, dataset, model,epoch=50):
    # Set the model in "train" mode.
    model.train()

    # Cumulative loss and steps.
    train_loss = 0.
    train_steps = 0

    # Set up optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args["learning_rate"],
        weight_decay=args["weight_decay"],
    )

    # Set up training dataloader. Creates `args.batch_size`-sized
    # batches from available samples.
    train_dataloader = tqdm(
        dataset.get_batch(batch_size=20,shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    for batch in train_dataloader:
        # Zero gradients.
        optimizer.zero_grad()

        # Forward inputs, calculate loss, optimize model.
        logits = model(batch)

        loss = calculate_loss()
        loss.backward()
        if args["grad_clip"] > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
        optimizer.step()

        # Update tqdm bar.
        train_loss += loss.item()
        train_steps += 1
        train_dataloader.set_description(
            f'[train] epoch = {epoch}, loss = {train_loss / train_steps:.6f}'
        )

    return train_loss / train_steps

def evaluate(args, epoch, model, dataset):
    """
    Evaluates the model for a single epoch using the development dataset.

    Args:
        args: `argparse` object.
        epoch: Epoch number (used in the `tqdm` bar).
        model: Instance of the PyTorch model.
        dataset: Development dataset.

    Returns:
        Evaluation cross-entropy loss normalized across all samples.
    """
    # Set the model in "evaluation" mode.
    model.eval()

    # Cumulative loss and steps.
    eval_loss = 0.
    eval_steps = 0

    # Set up evaluation dataloader. Creates `args.batch_size`-sized
    # batches from available samples. Does not shuffle.
    eval_dataloader = tqdm(
        dataset.get_batch(shuffle_examples=False),
        **_TQDM_OPTIONS,
    )

    with torch.no_grad():
        for batch in eval_dataloader:
            # Forward inputs, calculate loss.
            start_logits, end_logits, entity_type = model(batch)
            loss = calculate_loss()

            # Update tqdm bar.
            eval_loss += loss.item()
            eval_steps += 1
            eval_dataloader.set_description(
                f'[eval] epoch = {epoch}, loss = {eval_loss / eval_steps:.6f}'
            )

    return eval_loss / eval_steps


def predict(args, model, dev_dataset):
    pass


def main(args):
    # Check if GPU is available.
    if not args["use_gpu"] and torch.cuda.is_available():
        print('warning: GPU is available but args.use_gpu = False')
        print()


    # Set up datasets.
    train_dataset = None
    dev_dataset = None
    if args["use_bert"]:
        comment,label = preprocessing(args["train_path"])
        train_dataset = InputDataset(comment,label)
        comment_dev,label_dev = preprocessing(args["dev_path"])
        dev_dataset = InputDataset(comment_dev,label_dev)
    else:
        train_dataset = CommentDataset(args, args["train_path"])
        dev_dataset = CommentDataset(args, args["dev_path"])

    # Create vocabulary and tokenizer.
    vocabulary = Vocabulary(train_dataset.samples, args["vocab_size"])
    tokenizer = Tokenizer(vocabulary)
    for dataset in (train_dataset, dev_dataset):
        dataset.register_tokenizer(tokenizer)
    args["vocab_size"] = len(vocabulary)
    args["pad_token_id"] = tokenizer.pad_token_id
    print(f'vocab words = {len(vocabulary)}')

    # Print number of samples.
    print(f'train samples = {len(train_dataset)}')
    print(f'dev samples = {len(dev_dataset)}')
    print()

    # Select model.
    model = Classifier(args)
    if args["use_bert"]:
        model = MyBert()
    

    num_pretrained = model.load_pretrained_embeddings(
        vocabulary, args["embedding_path"]
    )
    pct_pretrained = round(num_pretrained / len(vocabulary) * 100., 2)
    print(f'using pre-trained embeddings from \'{args["embedding_path"]}\'')
    print(
        f'initialized {num_pretrained}/{len(vocabulary)} '
        f'embeddings ({pct_pretrained}%)'
    )
    print()

    if args["use_gpu"]:
        model = cuda(True, model)

    if args["do_train"]:
        # Track training statistics for checkpointing.
        eval_history = []
        best_eval_loss = float('inf')

        # Begin training.
        for epoch in range(1, args.epochs + 1):
            # Perform training and evaluation steps.
            train_loss = train(args, epoch, model, train_dataset)
            eval_loss = evaluate(args, epoch, model, dev_dataset)

            # If the model's evaluation loss yields a global improvement,
            # checkpoint the model.
            eval_history.append(eval_loss < best_eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), args["model_path"])

            print(
                f'epoch = {epoch} | '
                f'train loss = {train_loss:.6f} | '
                f'eval loss = {eval_loss:.6f} | '
                f"{'saving model!' if eval_history[-1] else ''}"
            )

    if args.do_test:
        # Write predictions to the output file. Use the printed command
        # below to obtain official EM/F1 metrics.
        predict(args, model, dev_dataset)
        eval_cmd = (
            'python3 evaluate.py '
            f'--dataset_path {args["dev_path"]} '
            f'--output_path {args["output_path"]}'
        )
        print()
        print(f'predictions written to \'{args["output_path"]}\'')
        print(f'compute EM/F1 with: \'{eval_cmd}\'')
        print()

if __name__ == '__main__':
    args= {"vocab_size": None, \
            "dev_path": None, \
            "output_path": None, \
            "model_path":None ,\
            "use_gpu":None ,\
            "train_path":None ,\
            "dev_path":None ,\
            "vocab_size":None ,\
            "do_train":None ,\
            "do_test":None , \
            "embedding_path":None, \
           "pad_token_id":None, \
           "use_bert": True
            }
    main(args)

