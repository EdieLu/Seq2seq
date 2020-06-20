# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import torch.utils.data
import collections
import codecs
import numpy as np
import random

from utils.config import PAD, UNK, BOS, EOS, SPC

import logging
logging.basicConfig(level=logging.INFO)

class IterDataset(torch.utils.data.Dataset):

	"""
		load features from
		'src_word_ids':src_word_ids[i_start:i_end],
		'src_sentence_lengths':src_sentence_lengths[i_start:i_end],
		'tgt_word_ids':tgt_word_ids[i_start:i_end],
		'tgt_sentence_lengths':tgt_sentence_lengths[i_start:i_end]
	"""

	def __init__(self, batches, max_seq_len):

		super(Dataset).__init__()

		self.batches = batches
		self.max_seq_len = max_seq_len

	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		srcid = self.batches[index]['src_word_ids'] # lis
		srcid = torch.nn.utils.rnn.pad_sequence(
			[torch.LongTensor(elem) for elem in srcid], batch_first=True) # tensor
		srclen = self.batches[index]['src_sentence_lengths'] # lis

		tgtid = self.batches[index]['tgt_word_ids'] # lis
		tgtid = torch.nn.utils.rnn.pad_sequence(
			[torch.LongTensor(elem) for elem in tgtid], batch_first=True) # tensor
		tgtlen = self.batches[index]['tgt_sentence_lengths'] # lis
		batch = {
			'srcid': srcid,
			'srclen': srclen,
			'tgtid': tgtid,
			'tgtlen': tgtlen,
		}

		return batch


class Dataset(object):

	""" load src-tgt from file """

	def __init__(self,
		# add params
		path_src,
		path_tgt,
		path_vocab_src,
		path_vocab_tgt,
		max_seq_len=32,
		batch_size=64,
		use_gpu=True,
		logger=None,
		seqrev=False,
		use_type='word'
		):

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.path_vocab_src = path_vocab_src
		self.path_vocab_tgt = path_vocab_tgt
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.logger = logger
		self.seqrev = seqrev
		self.use_type = use_type

		if type(self.logger) == type(None):
			self.logger = logging.getLogger(__name__)

		self.load_vocab()
		self.load_sentences()
		self.preprocess()


	def load_vocab(self):

		self.vocab_src = []
		self.vocab_tgt = []
		with codecs.open(self.path_vocab_src, encoding='UTF-8') as f:
			vocab_src_lines	= f.readlines()
		with codecs.open(self.path_vocab_tgt, encoding='UTF-8') as f:
			vocab_tgt_lines = f.readlines()

		self.src_word2id = collections.OrderedDict()
		self.tgt_word2id = collections.OrderedDict()
		self.src_id2word = collections.OrderedDict()
		self.tgt_id2word = collections.OrderedDict()

		for i, word in enumerate(vocab_src_lines):
			word = word.strip().split()[0] # remove \n
			self.vocab_src.append(word)
			self.src_word2id[word] = i
			self.src_id2word[i] = word

		for i, word in enumerate(vocab_tgt_lines):
			word = word.strip().split()[0] # remove \n
			self.vocab_tgt.append(word)
			self.tgt_word2id[word] = i
			self.tgt_id2word[i] = word


	def load_sentences(self):

		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()
		with codecs.open(self.path_tgt, encoding='UTF-8') as f:
			self.tgt_sentences = f.readlines()

		assert len(self.src_sentences) == len(self.tgt_sentences), \
			'Mismatch src:tgt - {}:{}'.format(len(self.src_sentences),len(self.tgt_sentences))

		if self.seqrev:
			for idx in range(len(self.src_sentences)):
				src_sent_rev = self.src_sentences[idx].strip().split()[::-1]
				tgt_sent_rev = self.tgt_sentences[idx].strip().split()[::-1]
				self.src_sentences[idx] = ' '.join(src_sent_rev)
				self.tgt_sentences[idx] = ' '.join(tgt_sent_rev)


	def preprocess(self):

		"""
			Use:
				map word2id once for all epoches (improved data loading efficiency)
				shuffling is done later
			Create:
				self.src_word_ids
				self.src_sentence_lengths
				self.tgt_word_ids
				self.tgt_sentence_lengths
		"""

		# src side
		vocab_size = {'src': len(self.src_word2id), 'tgt': len(self.tgt_word2id)}
		self.logger.info("num_vocab_src: {}".format(vocab_size['src']))
		self.logger.info("num_vocab_tgt: {}".format(vocab_size['tgt']))

		# declare temporary vars
		src_word_ids = []
		src_sentence_lengths = []
		tgt_word_ids = []
		tgt_sentence_lengths = []

		for idx in range(len(self.src_sentences)):
			src_sentence = self.src_sentences[idx]
			tgt_sentence = self.tgt_sentences[idx]

			src_words = src_sentence.strip().split()
			if self.use_type == 'char':
				tgt_words = tgt_sentence.strip()
			elif self.use_type == 'word':
				tgt_words = tgt_sentence.strip().split()

			# ignore long seq of words
			if len(src_words) > self.max_seq_len - 1 or len(tgt_words) > self.max_seq_len - 2:
				# src + EOS
				# BOS + tgt + EOS
				continue

			# emtry seq
			# if len(src_words) == 0 or len(tgt_words) == 0:
			# 	continue

			# source
			src_ids = []
			for i, word in enumerate(src_words):
				if word == ' ':
					assert self.use_type == 'char'
					src_ids.append(SPC)
				elif word in self.src_word2id:
					src_ids.append(self.src_word2id[word])
				else:
					src_ids.append(UNK)
			src_ids.append(EOS)

			# target
			tgt_ids = []
			tgt_ids.append(BOS)
			for i, word in enumerate(tgt_words):
				if word == ' ':
					assert self.use_type == 'char'
					tgt_ids.append(SPC)
				elif word in self.tgt_word2id:
					tgt_ids.append(self.tgt_word2id[word])
				else:
					tgt_ids.append(UNK)
			tgt_ids.append(EOS)

			src_word_ids.append(src_ids)
			src_sentence_lengths.append(len(src_words)+1) # include one EOS
			tgt_word_ids.append(tgt_ids)
			tgt_sentence_lengths.append(len(tgt_words)+2) # include BOS, EOS

		self.num_training_sentences = len(src_word_ids)
		self.logger.info("num_sentences: {}".format(self.num_training_sentences))

		# set class var to be used in batchify
		self.src_word_ids = src_word_ids
		self.src_sentence_lengths = src_sentence_lengths
		self.tgt_word_ids = tgt_word_ids
		self.tgt_sentence_lengths = tgt_sentence_lengths


	def construct_batches(self, is_train=False):

		"""
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src:
				a  SPC c a t SPC s a t SPC o n SPC t h e SPC m a t EOS PAD PAD ...
		"""

		# import pdb; pdb.set_trace()

		# organise by length
		_x = list(zip(self.src_word_ids, self.src_sentence_lengths,
			self.tgt_word_ids, self.tgt_sentence_lengths))
		if is_train:
			# _x = sorted(_x, key=lambda l:l[1])
			random.shuffle(_x)
		src_word_ids, src_sentence_lengths, tgt_word_ids, tgt_sentence_lengths = zip(*_x)

		# manual batching to allow shuffling by pt dataloader
		n_batches = int(self.num_training_sentences/self.batch_size +
			(self.num_training_sentences % self.batch_size > 0))
		batches = []
		for i in range(n_batches):
			i_start = i * self.batch_size
			i_end = min(i_start + self.batch_size, self.num_training_sentences)
			batch = {
				'src_word_ids':src_word_ids[i_start:i_end],
				'src_sentence_lengths': src_sentence_lengths[i_start:i_end],
				'tgt_word_ids':tgt_word_ids[i_start:i_end],
				'tgt_sentence_lengths': tgt_sentence_lengths[i_start:i_end],
			}
			batches.append(batch)

		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': is_train,
					'num_workers': 0}

		self.iter_set = IterDataset(batches, self.max_seq_len)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)


def load_pretrained_embedding(word2id, embedding_matrix, embedding_path):

	""" assign value to src_word_embeddings and tgt_word_embeddings """

	counter = 0
	with codecs.open(embedding_path, encoding="UTF-8") as f:
		for line in f:
			items = line.strip().split()
			if len(items) <= 2:
				continue
			word = items[0].lower()
			if word in word2id:
				id = word2id[word]
				vector = np.array(items[1:])
				embedding_matrix[id] = vector
				counter += 1

	print('loaded pre-trained embedding:', embedding_path)
	print('embedding vectors found:', counter)

	# import pdb; pdb.set_trace()
	return embedding_matrix
