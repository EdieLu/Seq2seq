import torch
import random
import time
import os
import argparse
import sys
import numpy as np

from utils.misc import set_global_seeds, save_config, validate_config, check_device
from utils.dataset import Dataset
from models.Seq2seq import Seq2seq
from trainer.trainer import Trainer


def load_arguments(parser):

	""" Seq2seq model """

	# paths
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--train_path_tgt', type=str, required=True, help='train tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, required=True, help='vocab tgt dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--dev_path_tgt', type=str, default=None, help='dev tgt dir')
	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_embedding_src', type=str, default=None, help='pretrained src embedding')
	parser.add_argument('--load_embedding_tgt', type=str, default=None, help='pretrained tgt embedding')
	parser.add_argument('--use_type', type=str, default='word', help='word | char')


	# model
	parser.add_argument('--share_embedder', type=str, default='False', help='whether or not share embedder')
	parser.add_argument('--embedding_size_enc', type=int, default=200, help='encoder embedding size')
	parser.add_argument('--embedding_size_dec', type=int, default=200, help='decoder embedding size')
	parser.add_argument('--hidden_size_enc', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_bilstm_enc', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--num_unilstm_enc', type=int, default=0, help='number of encoder unilstm layers')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--num_unilstm_dec', type=int, default=2, help='number of encoder bilstm layers')
	parser.add_argument('--residual', type=str, default='False', help='residual connection')
	parser.add_argument('--att_mode', type=str, default='bahdanau',
		help='attention mechanism mode - bahdanau / hybrid / bilinear')
	parser.add_argument('--hidden_size_att', type=int, default=1,
		help='hidden size for bahdanau / hybrid attention')
	parser.add_argument('--hidden_size_shared', type=int, default=200,
		help='transformed att output hidden size (set as hidden_size_enc)')

	# data
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--batch_first', type=str, default='True', help='batch as the first dimension')
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')

	# train
	parser.add_argument('--random_seed', type=int, default=666, help='random seed')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--minibatch_split', type=int, default=1, help='split the batch to avoid OOM')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--normalise_loss', type=str, default='True', help='normalise loss or not')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0, help='ratio of teacher forcing')
	parser.add_argument('--scheduled_sampling', type=str, default='False',
		help='gradually turn off teacher forcing')
	parser.add_argument('--max_grad_norm', type=float, default=1.0,
		help='optimiser gradient norm clipping: max grad norm')

	# save and print
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')
	parser.add_argument('--max_count_no_improve', type=int, default=2,
		help='if meet max, operate roll back')
	parser.add_argument('--max_count_num_rollback', type=int, default=2,
		help='if meet max, reduce learning rate')
	parser.add_argument('--keep_num', type=int, default=1,
		help='number of models to keep')

	return parser


def main():

	# load config
	parser = argparse.ArgumentParser(description='Seq2seq Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# set random seed
	if config['random_seed'] is not None:
		set_global_seeds(config['random_seed'])

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# resume or not
	if config['load']:
		resume = True
		print('resuming {} ...'.format(config['load']))
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		resume = False
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					batch_size=config['batch_size'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					learning_rate=config['learning_rate'],
					eval_with_mask=config['eval_with_mask'],
					scheduled_sampling=config['scheduled_sampling'],
					teacher_forcing_ratio=config['teacher_forcing_ratio'],
					use_gpu=config['use_gpu'],
					max_grad_norm=config['max_grad_norm'],
					max_count_no_improve=config['max_count_no_improve'],
					max_count_num_rollback=config['max_count_num_rollback'],
					keep_num=config['keep_num'],
					normalise_loss=config['normalise_loss'],
					minibatch_split=config['minibatch_split'])

	# load train set
	train_path_src = config['train_path_src']
	train_path_tgt = config['train_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	train_set = Dataset(train_path_src, train_path_tgt,
		path_vocab_src=path_vocab_src, path_vocab_tgt=path_vocab_tgt,
		seqrev=config['seqrev'],
		max_seq_len=config['max_seq_len'],
		batch_size=config['batch_size'],
		use_gpu=config['use_gpu'],
		logger=t.logger,
		use_type=config['use_type'])

	vocab_size_enc = len(train_set.vocab_src)
	vocab_size_dec = len(train_set.vocab_tgt)

	# load dev set
	if config['dev_path_src'] and config['dev_path_tgt']:
		dev_path_src = config['dev_path_src']
		dev_path_tgt = config['dev_path_tgt']
		dev_set = Dataset(dev_path_src, dev_path_tgt,
			path_vocab_src=path_vocab_src, path_vocab_tgt=path_vocab_tgt,
			seqrev=config['seqrev'],
			max_seq_len=config['max_seq_len'],
			batch_size=config['batch_size'],
			use_gpu=config['use_gpu'],
			logger=t.logger,
			use_type=config['use_type'])
	else:
		dev_set = None

	# construct model
	seq2seq = Seq2seq(vocab_size_enc, vocab_size_dec,
					share_embedder=config['share_embedder'],
					embedding_size_enc=config['embedding_size_enc'],
					embedding_size_dec=config['embedding_size_dec'],
					embedding_dropout=config['embedding_dropout'],
					hidden_size_enc=config['hidden_size_enc'],
					num_bilstm_enc=config['num_bilstm_enc'],
					num_unilstm_enc=config['num_unilstm_enc'],
					hidden_size_dec=config['hidden_size_dec'],
					num_unilstm_dec=config['num_unilstm_dec'],
					hidden_size_att=config['hidden_size_att'],
					hidden_size_shared=config['hidden_size_shared'],
					dropout=config['dropout'],
					residual=config['residual'],
					batch_first=config['batch_first'],
					max_seq_len=config['max_seq_len'],
					load_embedding_src=config['load_embedding_src'],
					load_embedding_tgt=config['load_embedding_tgt'],
					src_word2id=train_set.src_word2id,
					tgt_word2id=train_set.tgt_word2id,
					src_id2word=train_set.src_id2word,
					tgt_id2word=train_set.tgt_id2word,
					att_mode=config['att_mode'],
					use_gpu=config['use_gpu'])


	device = check_device(config['use_gpu'])
	t.logger.info('device:{}'.format(device))
	seq2seq = seq2seq.to(device=device)

	# run training
	seq2seq = t.train(train_set, seq2seq,
		num_epochs=config['num_epochs'], resume=resume, dev_set=dev_set)


if __name__ == '__main__':
	main()
