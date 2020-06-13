import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.config import PAD, EOS, BOS
from utils.misc import check_device
from .EncRNN import EncRNN
from .DecRNN import DecRNN

import warnings
warnings.filterwarnings("ignore")

class Seq2seq(nn.Module):

	""" enc-dec model """

	def __init__(self,
		# add params
		vocab_size_enc,
		vocab_size_dec,
		share_embedder=False,
		embedding_size_enc=200,
		embedding_size_dec=200,
		embedding_dropout=0,
		hidden_size_enc=200,
		num_bilstm_enc=2,
		num_unilstm_enc=0,
		hidden_size_dec=200,
		num_unilstm_dec=2,
		hidden_size_att=10,
		hidden_size_shared=200,
		dropout=0.0,
		residual=False,
		batch_first=True,
		max_seq_len=32,
		load_embedding_src=None,
		load_embedding_tgt=None,
		src_word2id=None,
		src_id2word=None,
		tgt_word2id=None,
		tgt_id2word=None,
		att_mode='bahdanau',
		use_gpu=False
		):

		super(Seq2seq, self).__init__()

		self.encoder = EncRNN(
			vocab_size_enc,
			embedding_size_enc=embedding_size_enc,
			embedding_dropout=embedding_dropout,
			hidden_size_enc=hidden_size_enc,
			num_bilstm_enc=num_bilstm_enc,
			num_unilstm_enc=num_unilstm_enc,
			dropout=dropout,
			residual=residual,
			batch_first=batch_first,
			max_seq_len=max_seq_len,
			load_embedding_src=load_embedding_src,
			src_word2id=src_word2id,
			src_id2word=src_id2word,
			use_gpu=use_gpu
		)

		self.decoder = DecRNN(
			vocab_size_dec,
			embedding_size_dec=embedding_size_dec,
			embedding_dropout=embedding_dropout,
			hidden_size_enc=hidden_size_enc,
			hidden_size_dec=hidden_size_dec,
			num_unilstm_dec=num_unilstm_dec,
			att_mode=att_mode,
			hidden_size_att=hidden_size_att,
			hidden_size_shared=hidden_size_shared,
			dropout=dropout,
			residual=residual,
			batch_first=batch_first,
			max_seq_len=max_seq_len,
			load_embedding_tgt=load_embedding_tgt,
			tgt_word2id=tgt_word2id,
			tgt_id2word=tgt_id2word,
			use_gpu=use_gpu
		)

		# import pdb; pdb.set_trace()
		if share_embedder:
			assert vocab_size_enc == vocab_size_dec
			self.encoder.embedder_enc = self.decoder.embedder_dec


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None

			# set class attribute to default value
			setattr(self, var_name, var_val)


	def forward(self, src, tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=1.0,
		beam_width=1, use_gpu=True):

		"""
			Args:
				src: list of src word_ids [batch_size, max_seq_len, word_ids]
				tgt: list of tgt word_ids
				hidden: initial hidden state
				is_training: whether in eval or train mode
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output -
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
				ret_dict
		"""

		enc_outputs = self.encoder(src, use_gpu=use_gpu)
		decoder_outputs, dec_hidden, ret_dict = self.decoder(
			enc_outputs, src, tgt=tgt,
			is_training=is_training,
			teacher_forcing_ratio=teacher_forcing_ratio,
			beam_width=beam_width, use_gpu=use_gpu
		)

		return decoder_outputs, dec_hidden, ret_dict
