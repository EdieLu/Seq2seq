import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.config import PAD, EOS, BOS
from utils.dataset import load_pretrained_embedding
from utils.misc import check_device

import warnings
warnings.filterwarnings("ignore")

class EncRNN(nn.Module):

	""" encoder RNN """

	def __init__(self,
		vocab_size_enc,
		embedding_size_enc=200,
		embedding_dropout=0,
		hidden_size_enc=200,
		num_bilstm_enc=2,
		num_unilstm_enc=0,
		dropout=0.0,
		residual=False,
		batch_first=True,
		max_seq_len=32,
		load_embedding_src=None,
		src_word2id=None,
		src_id2word=None,
		use_gpu=False
		):

		super(EncRNN, self).__init__()
		device = check_device(use_gpu)

		# define embeddings
		self.vocab_size_enc = vocab_size_enc
		self.embedding_size_enc = embedding_size_enc

		self.load_embedding = load_embedding_src
		self.word2id = src_word2id
		self.id2word = src_id2word

		# define model param
		self.hidden_size_enc = hidden_size_enc
		self.num_bilstm_enc = num_bilstm_enc
		self.num_unilstm_enc= num_unilstm_enc
		self.residual = residual

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)

		# load embeddings
		if self.load_embedding:
			# import pdb; pdb.set_trace()
			embedding_matrix = np.random.rand(self.vocab_size_enc, self.embedding_size_enc)
			embedding_matrix = torch.FloatTensor(load_pretrained_embedding(
				self.word2id, embedding_matrix, self.load_embedding))
			self.embedder_enc = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.embedder_enc = nn.Embedding(self.vocab_size_enc,
				self.embedding_size_enc, sparse=False, padding_idx=PAD)

		# define enc
		# embedding_size_enc -> hidden_size_enc * 2
		self.enc = torch.nn.LSTM(self.embedding_size_enc, self.hidden_size_enc,
			num_layers=self.num_bilstm_enc, batch_first=batch_first,
			bias=True, dropout=dropout, bidirectional=True)

		if self.num_unilstm_enc != 0:
			if not self.residual:
				self.enc_uni = torch.nn.LSTM(
					self.hidden_size_enc * 2, self.hidden_size_enc * 2,
					num_layers=self.num_unilstm_enc, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=False)
			else:
				self.enc_uni = nn.Module()
				for i in range(self.num_unilstm_enc):
					self.enc_uni.add_module(
						'l'+str(i),
						torch.nn.LSTM(
							self.hidden_size_enc * 2, self.hidden_size_enc * 2,
							num_layers=1, batch_first=batch_first, bias=True,
							dropout=dropout,bidirectional=False
						)
					)


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None

			# set class attribute to default value
			setattr(self, var_name, var_val)


	def forward(self, src, hidden=None, use_gpu=True):

		"""
			Args:
				src: list of src word_ids [batch_size, seq_len, word_ids]
		"""

		device = check_device(use_gpu)

		# src mask
		mask_src = src.data.eq(PAD)
		batch_size = src.size(0)
		seq_len = src.size(1)

		# convert id to embedding
		emb_src = self.embedding_dropout(self.embedder_enc(src))

		# run enc
		enc_outputs, enc_hidden = self.enc(emb_src, hidden)
		enc_outputs = self.dropout(enc_outputs)\
			.view(batch_size, seq_len, enc_outputs.size(-1))

		if self.num_unilstm_enc != 0:
			if not self.residual:
				enc_hidden_uni_init = None
				enc_outputs, enc_hidden_uni = self.enc_uni(
					enc_outputs, enc_hidden_uni_init)
				enc_outputs = self.dropout(enc_outputs).view(
					batch_size, seq_len, enc_outputs.size(-1))
			else:
				enc_hidden_uni_init = None
				enc_hidden_uni_lis = []
				for i in range(self.num_unilstm_enc):
					enc_inputs = enc_outputs
					enc_func = getattr(self.enc_uni, 'l'+str(i))
					enc_outputs, enc_hidden_uni = enc_func(
						enc_inputs, enc_hidden_uni_init)
					enc_hidden_uni_lis.append(enc_hidden_uni)
					if i < self.num_unilstm_enc - 1: # no residual for last layer
						enc_outputs = enc_outputs + enc_inputs
					enc_outputs = self.dropout(enc_outputs).view(
						batch_size, seq_len, enc_outputs.size(-1))

		return enc_outputs
