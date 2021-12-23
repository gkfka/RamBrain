import os
from argparse import Namespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from word2vec import get_bigram_sents

class Codec():
    def __init__(self, embed_model, tagset):
        self.emb = embed_model
        self.encode_tag_table = { t:i for i, t in enumerate(tagset) }
        self.decode_tag_table = tagset
        
    def encode_tag(self, sent, size=-1):
        if size > 0: sent = sent[:size]
        return [ self.encode_tag_table[t] for t in sent ]

    def encode_sent(self, sent, size=-1):
        if size > 0: sent = sent[:size]
        return [ self.emb.wv[w] for w in sent ]

    def decode_tag(self, sent, size=-1):
        if size > 0: sent = sent[:size]
        return [ self.decode_tag_table[t] for t in sent if t>0 ]

class ModuVectorizer(object):
    def __init__(self, codec):
        self.codec = codec

    def vectorize(self, sent, tag_sent, vector_length=-1):
        if vector_length < 0:
            vector_length = len(sent) - 1

        emb_sent = self.codec.encode_sent(sent)
        for i in range(vector_length-len(emb_sent)):
            emb_sent.append([ 0 for i in range(128)])
        from_vector = torch.FloatTensor(emb_sent)
        to_vector = np.zeros(vector_length, dtype=np.int64)
        to_vector[:len(tag_sent)] = self.codec.encode_tag(tag_sent)

        return from_vector, to_vector

class ModuDataset(Dataset):
    def __init__(self, fnm, vectorizer):
       self.word_sents, self.tag_sents = get_bigram_sents(fnm)

       self._vectorizer = vectorizer
       self._max_seq_length = max(map(len, self.tag_sents))

    def __len__(self):
        return len(self.tag_sents)

    def __getitem__(self, idx):
        from_vector, to_vector = \
            self._vectorizer.vectorize(self.word_sents[idx], self.tag_sents[idx], self._max_seq_length)
        return from_vector, to_vector

class RNNCRFTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_lid, pad_wid): 
        super().__init__()
        self.n_labels  = output_dim
        self.pad_wid   = pad_wid
        self.pad_lid   = pad_lid
        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn       = nn.LSTM(embedding_dim, hidden_dim, num_layers = n_layers, bidirectional = bidirectional, batch_first=True)
        self.fc        = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)        
        self.crf       = CRF(self.n_labels, batch_first=True)

    def compute_outputs(self, sentences, labels):
        #embedded   = self.embedding(sentences)
        rnn_out, _ = self.rnn(sentences)
        emissions  = self.fc(rnn_out)
        pad_mask = (labels == self.pad_wid).float()
        emissions[:, :, self.pad_lid] += pad_mask*10000
        return emissions

    def forward(self, sentences, labels):
        # Compute the outputs of the lower layers, which will be used as emission
        # scores for the CRF.
        emissions  = self.compute_outputs(sentences, labels)
        pad_mask   = (labels != self.pad_lid)
        #pad_mask   = (sentences != self.pad_wid)

        # We return the loss value. The CRF returns the log likelihood, but we return 
        # the *negative* log likelihood as the loss value.            
        # PyTorch's optimizers *minimize* the loss, while we want to *maximize* the
        # log likelihood.
        #print('pad_mask = ', pad_mask)
        return - self.crf(emissions, labels, pad_mask)

    def predict(self, sentences, labels):
        # Compute the emission scores, as above.
        emissions = self.compute_outputs(sentences, labels)
        pad_mask   = (labels != self.pad_lid)
        #pad_mask   = (sentences != self.pad_wid)

        # Apply the Viterbi algorithm to get the predictions. This implementation returns
        # the result as a list of lists (not a tensor), corresponding to a matrix
        # of shape (n_sentences, max_len).
        return self.crf.decode(emissions, pad_mask)

def pad_array(arr, len):
    for i in range(len(arr), len):
        arr.append(0)

def generate_batches(dataset, _batch_size=128, device="cpu"):
    dataloader = DataLoader(dataset, batch_size=_batch_size)#, shuffle=True)

    for x, y in dataloader:
        yield x.to(device), y.to(device)

def compute_accuracy(y_pred, y_true):
    n_correct, n_tot = 0, 0
    for yp_sent, yt_sent in zip(y_pred, y_true):
        for yp, yt in zip(yp_sent, yt_sent):
            if yt == 0: break
            n_tot += 1
            if yp == yt: n_correct += 1

    return n_correct / n_tot

def vector_len(vector, mask_idx=0):
    for i, v in enumerate(vector):
        if v == mask_idx: return i
    return -1