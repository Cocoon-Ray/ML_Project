# changed something
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cuda, load_cached_embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Classifier(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.args = args
        self.embedding_dim = args["embedding_dim"]
        self.pad_token_id = args["pad_token_id"]

        # Initialize embedding layer (1)
        self.embedding = nn.Embedding(args["vocab_size"], args["embedding_dim"])
        """
        
        :param args: 
        """

    def load_pretrained_embeddings(self, vocabulary, path,use_gpu=False):
        """
        Loads GloVe vectors and initializes the embedding matrix.

        Args:
            vocabulary: `Vocabulary` object.
            path: Embedding path, e.g. "glove/glove.6B.300d.txt".
        """
        embedding_map = load_cached_embeddings(path)

        # Create embedding matrix. By default, embeddings are randomly
        # initialized from Uniform(-0.1, 0.1).
        embeddings = torch.zeros(
            (len(vocabulary), self.embedding_dim)
        ).uniform_(-0.1, 0.1)

        # Initialize pre-trained embeddings.
        num_pretrained = 0
        for (i, word) in enumerate(vocabulary.words):
            if word in embedding_map:
                embeddings[i] = torch.tensor(embedding_map[word])
                num_pretrained += 1

        # Place embedding matrix on GPU.
        self.embedding.weight.data = cuda(use_gpu, embeddings)

        return num_pretrained

    def forward(self, batch):

        return



class MyBert(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert=AutoModel.from_pretrained("albert-xxlarge-v2")
    self.drop = nn.Dropout(p=0.3)
    self.linear=nn.Linear(Hidden_size=4096,out_features=6)

  def forward(self, input_ids, segment_ids, attention_mask):
    pooler_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
    output=self.drop(pooler_output)
    output=self.linear(output)
    return output

model=Mymodel().to(device)