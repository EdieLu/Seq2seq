import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset import Dataset
from utils.misc import save_config, validate_config
from utils.misc import get_memory_alloc, plot_alignment, check_device
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor
from utils.config import PAD, EOS
from modules.loss import NLLLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

import logging
logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" Seq2seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--seqrev', type=str, default='False', help='whether or not to reverse sequence')
	parser.add_argument('--use_type', type=str, default='word', help='word | char')

	return parser


def translate(test_set, load_dir, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.max_seq_len = max_seq_len
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))

	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()
				print(idx+1, len(evaliter))

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

				decoder_outputs, decoder_hidden, other = model(src=src_ids,
					is_training=False, beam_width=beam_width, use_gpu=use_gpu)

				# write to file
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
				for i in range(len(seqwords)):
					if src_lengths[i] == 0:
						continue
					words = []
					for word in seqwords[i]:
						if word == '<pad>':
							continue
						elif word == '<spc>':
							words.append(' ')
						elif word == '</s>':
							break
						else:
							words.append(word)
					if len(words) == 0:
						outline = ''
					else:
						if seqrev:
							words = words[::-1]
						if test_set.use_type == 'word':
							outline = ' '.join(words)
						elif test_set.use_type == 'char':
							outline = ''.join(words)
					f.write('{}\n'.format(outline))

				sys.stdout.flush()


def translate_batch(test_set, load_dir, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.max_seq_len = max_seq_len
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	print('batch_size: {}'.format(test_set.batch_size))

	model.eval()
	with torch.no_grad():

		# select batch
		n_total = len(evaliter)
		iter_idx = 0
		per_iter = 500 # 1892809 lines; 100/batch; 38 iterations
		st = iter_idx * per_iter
		ed = min((iter_idx + 1) * per_iter, n_total)
		f = open(os.path.join(test_path_out, '{:04d}.txt'.format(iter_idx)), 'w', encoding="utf8")

		for idx in range(len(evaliter)):
			batch_items = evaliter.next()
			if idx < st:
				continue
			elif idx >= ed:
				break
			print(idx, ed)

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

			decoder_outputs, decoder_hidden, other = model(src=src_ids,
				is_training=False, beam_width=beam_width, use_gpu=use_gpu)

			# memory usage
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))

			# write to file
			seqlist = other['sequence']
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			for i in range(len(seqwords)):
				if src_lengths[i] == 0:
					continue
				words = []
				for word in seqwords[i]:
					if word == '<pad>':
						continue
					elif word == '<spc>':
						words.append(' ')
					elif word == '</s>':
						break
					else:
						words.append(word)
				if len(words) == 0:
					outline = ''
				else:
					if seqrev:
						words = words[::-1]
					if test_set.use_type == 'word':
						outline = ' '.join(words)
					elif test_set.use_type == 'char':
						outline = ''.join(words)
				f.write('{}\n'.format(outline))

			sys.stdout.flush()


def att_plot(test_set, load_dir, plot_path, use_gpu, max_seq_len, beam_width, device):

	"""
		generate attention alignment plots
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
			max_seq_len
		Returns:

	"""

	# check devide
	print('cuda available: {}'.format(torch.cuda.is_available()))
	use_gpu = use_gpu and torch.cuda.is_available()

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	beam_width = 1
	model.max_seq_len = max_seq_len
	print('max seq len {}'.format(model.max_seq_len))


	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)

	# start eval
	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids[:,:tgt_len].to(device=device)

			decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids,
				is_training=False, beam_width=beam_width)

			# Evaluation
			# default batch_size = 1
			# attention: 31 * [1 x 1 x 32]
			# 	(tgt_len(query_len) * [ batch_size x 1 x src_len(key_len)]
			attention = other['attention_score']
			seqlist = other['sequence'] # traverse over time not batch
			bsize = test_set.batch_size
			max_seq = test_set.max_seq_len
			vocab_size = len(test_set.tgt_word2id)

			# Print sentence by sentence
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)

			n_q = len(attention)
			n_k = attention[0].size(2)
			b_size =  attention[0].size(0)
			att_score = torch.empty(n_q, n_k, dtype=torch.float)
			# att_score = np.empty([n_q, n_k])

			for i in range(len(seqwords)): # loop over sentences
				outline_src = ' '.join(srcwords[i])
				outline_ref = ' '.join(refwords[i])
				outline_gen = ' '.join(seqwords[i])
				print('SRC: {}'.format(outline_src))
				print('REF: {}'.format(outline_ref))
				print('GEN: {}'.format(outline_gen))
				for j in range(len(attention)):
					# i: idx of batch
					# j: idx of query
					gen = seqwords[i][j]
					ref = refwords[i][j]
					att = attention[j][i]
					# record att scores
					att_score[j] = att

				# plotting
				loc_eos_k = srcwords[i].index('</s>') + 1
				loc_eos_q = seqwords[i].index('</s>') + 1
				loc_eos_ref = refwords[i].index('</s>') + 1
				print('eos_k: {}, eos_q: {}'.format(loc_eos_k, loc_eos_q))
				att_score_trim = att_score[:loc_eos_q, :loc_eos_k]
				# each row (each query) sum up to 1
				print(att_score_trim)
				print('\n')
				# import pdb; pdb.set_trace()

				choice = input('Plot or not ? - y/n\n')
				if choice:
					if choice.lower()[0] == 'y':
						print('plotting ...')
						plot_dir = os.path.join(plot_path, '{}.png'.format(count))
						src = srcwords[i][:loc_eos_k]
						hyp = seqwords[i][:loc_eos_q]
						ref = refwords[i][:loc_eos_ref]
						# x-axis: src; y-axis: hyp
						# plot_alignment(att_score_trim.numpy(),
						# 	plot_dir, src=src, hyp=hyp, ref=ref)
						plot_alignment(att_score_trim.numpy(),
							plot_dir, src=src, hyp=hyp, ref=None) # no ref
						count += 1
						input('Press enter to continue ...')


def main():

	# load config
	parser = argparse.ArgumentParser(description='Seq2seq Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = test_path_src # dummy
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	test_path_out = config['test_path_out']
	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	use_type = config['use_type']

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)
	config_save_dir = os.path.join(test_path_out, 'eval.cfg')
	save_config(config, config_save_dir)

	# set test mode: 1 = translate; 2 = plot
	MODE = config['eval_mode']
	if MODE == 3:
		max_seq_len = 32
		batch_size = 1
		beam_width = 1
		use_gpu = False

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt,
						path_vocab_src, path_vocab_tgt,
						seqrev=seqrev,
						max_seq_len=max_seq_len,
						batch_size=batch_size,
						use_gpu=use_gpu,
						use_type=use_type)
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 1:
		translate(test_set, load_dir, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)

	if MODE == 2:
		translate_batch(test_set, load_dir, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev)

	elif MODE == 3:
		# plotting
		att_plot(test_set, load_dir, test_path_out, use_gpu,
			max_seq_len, beam_width, device)


if __name__ == '__main__':
	main()
