import torch
import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.misc import get_memory_alloc, check_device
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.Seq2seq import Seq2seq

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)

class Trainer(object):

	def __init__(self,
		expt_dir='experiment',
		load_dir=None,
		checkpoint_every=100,
		print_every=100,
		batch_size=256,
		use_gpu=False,
		learning_rate=0.001,
		max_grad_norm=1.0,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=0.0,
		max_count_no_improve=2,
		max_count_num_rollback=2,
		keep_num=1
		):

		self.use_gpu = use_gpu
		self.device = check_device(self.use_gpu)

		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.learning_rate = learning_rate
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio

		self.max_count_no_improve = max_count_no_improve
		self.max_count_num_rollback = max_count_num_rollback
		self.keep_num = keep_num

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir

		self.logger = logging.getLogger(__name__)
		self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)

		self.batch_size = batch_size
		self.minibatch_size = self.batch_size # to be changed if OOM

	def _print_hyp(self,
		out_count, src_ids, tgt_ids, src_id2word, tgt_id2word, seqlist):

		if out_count < 3:
			srcwords = _convert_to_words_batchfirst(src_ids, src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], tgt_id2word)
			seqwords = _convert_to_words(seqlist, tgt_id2word)
			outsrc = 'SRC: {}\n'.format(' '.join(srcwords[0])).encode('utf-8')
			outref = 'REF: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
			outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
			sys.stdout.buffer.write(outsrc)
			sys.stdout.buffer.write(outref)
			sys.stdout.buffer.write(outline)
			out_count += 1
		return out_count


	def _evaluate_batches(self, model, dataset):

		model.eval()

		resloss = 0
		resloss_norm = 0

		match = 0
		total = 0

		evaliter = iter(dataset.iter_loader)
		out_count = 0

		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()

				# load data
				batch_src_ids = batch_items['srcid'][0]
				batch_src_lengths = batch_items['srclen']
				batch_tgt_ids = batch_items['tgtid'][0]
				batch_tgt_lengths = batch_items['tgtlen']

				# separate into minibatch
				batch_size = batch_src_ids.size(0)
				batch_seq_len = int(max(batch_src_lengths))

				n_minibatch = int(batch_size / self.minibatch_size)
				n_minibatch += int(batch_size % self.minibatch_size > 0)

				for bidx in range(n_minibatch):

					loss = NLLLoss()
					loss.reset()

					i_start = bidx * self.minibatch_size
					i_end = min(i_start + self.minibatch_size, batch_size)
					src_ids = batch_src_ids[i_start:i_end]
					src_lengths = batch_src_lengths[i_start:i_end]
					tgt_ids = batch_tgt_ids[i_start:i_end]
					tgt_lengths = batch_tgt_lengths[i_start:i_end]

					src_len = max(src_lengths)
					tgt_len = max(tgt_lengths)
					src_ids = src_ids[:,:src_len].to(device=self.device)
					tgt_ids = tgt_ids[:,:tgt_len].to(device=self.device)

					non_padding_mask_tgt = tgt_ids.data.ne(PAD)
					non_padding_mask_src = src_ids.data.ne(PAD)

					# run model
					decoder_outputs, decoder_hidden, other = model(
						src_ids, tgt_ids, is_training=False, use_gpu=self.use_gpu)

					# evaluation
					logps = torch.stack(decoder_outputs, dim=1).to(device=self.device)
					if not self.eval_with_mask:
						loss.eval_batch(logps.reshape(-1, logps.size(-1)),
							tgt_ids[:, 1:].reshape(-1))
						loss.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
					else:
						loss.eval_batch_with_mask(logps.reshape(-1, logps.size(-1)),
							tgt_ids[:, 1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))
						loss.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])
					loss.normalise()
					resloss += loss.get_loss()
					resloss_norm += 1

					seqlist = other['sequence']
					seqres = torch.stack(seqlist, dim=1).to(device=self.device)
					correct = seqres.view(-1).eq(tgt_ids[:,1:].reshape(-1))\
						.masked_select(non_padding_mask_tgt[:,1:].reshape(-1)).sum().item()
					match += correct
					total += non_padding_mask_tgt[:,1:].sum().item()

					out_count = self._print_hyp(out_count, src_ids, tgt_ids,
						dataset.src_id2word, dataset.tgt_id2word, seqlist)

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total

		resloss /= (1.0 * resloss_norm)
		torch.cuda.empty_cache()
		losses = {}
		losses['nll_loss'] = resloss

		return resloss, accuracy, losses


	def _train_batch(self,
		model, batch_items, dataset, step, total_steps):

		"""
			Args:
				src_ids 		=     w1 w2 w3 </s> <pad> <pad> <pad>
				tgt_ids 		= <s> w1 w2 w3 </s> <pad> <pad> <pad>

			Others:
				internal input 	= <s> w1 w2 w3 </s> <pad> <pad>
				decoder_outputs	= 	  w1 w2 w3 </s> <pad> <pad> <pad>
		"""

		# scheduled sampling
		progress = 1.0 * step / total_steps
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			teacher_forcing_ratio = 1.0 - progress

		# load data
		batch_src_ids = batch_items['srcid'][0]
		batch_src_lengths = batch_items['srclen']
		batch_tgt_ids = batch_items['tgtid'][0]
		batch_tgt_lengths = batch_items['tgtlen']

		# separate into minibatch
		batch_size = batch_src_ids.size(0)
		batch_seq_len = int(max(batch_src_lengths))
		n_minibatch = int(batch_size / self.minibatch_size)
		n_minibatch += int(batch_size % self.minibatch_size > 0)
		resloss = 0

		for bidx in range(n_minibatch):

			# debug
			# import pdb; pdb.set_trace()

			# define loss
			loss = NLLLoss()
			loss.reset()

			# load data
			i_start = bidx * self.minibatch_size
			i_end = min(i_start + self.minibatch_size, batch_size)
			src_ids = batch_src_ids[i_start:i_end]
			src_lengths = batch_src_lengths[i_start:i_end]
			tgt_ids = batch_tgt_ids[i_start:i_end]
			tgt_lengths = batch_tgt_lengths[i_start:i_end]

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			src_ids = src_ids[:,:src_len].to(device=self.device)
			tgt_ids = tgt_ids[:,:tgt_len].to(device=self.device)

			# get padding mask
			non_padding_mask_src = src_ids.data.ne(PAD)
			non_padding_mask_tgt = tgt_ids.data.ne(PAD)

			# Forward propagation
			decoder_outputs, decoder_hidden, ret_dict = model(src_ids, tgt_ids,
				is_training=True, teacher_forcing_ratio=teacher_forcing_ratio,
				use_gpu=self.use_gpu)

			# Get loss
			logps = torch.stack(decoder_outputs, dim=1).to(device=self.device)
			if not self.eval_with_mask:
				loss.eval_batch(logps.reshape(-1, logps.size(-1)),
					tgt_ids[:, 1:].reshape(-1))
				loss.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
			else:
				loss.eval_batch_with_mask(logps.reshape(-1, logps.size(-1)),
					tgt_ids[:,1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))
				loss.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])

			# Backward propagation: accumulate gradient
			loss.normalise()
			loss.acc_loss /= n_minibatch
			loss.backward()
			resloss += loss.get_loss()
			torch.cuda.empty_cache()

		# update weights
		self.optimizer.step()
		model.zero_grad()

		return resloss


	def _train_epoches(self,
		train_set, model, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		print_loss_total = 0  # Reset every print_every
		step = start_step
		step_elapsed = 0
		prev_acc = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# loop over epochs
		for epoch in range(start_epoch, n_epochs + 1):

			for param_group in self.optimizer.optimizer.param_groups:
				log.info('epoch:{} lr: {}'.format(epoch, param_group['lr']))
				lr_curr = param_group['lr']

			# construct batches - allow re-shuffling of data
			log.info('--- construct train set ---')
			train_set.construct_batches(is_train=True)
			if dev_set is not None:
				log.info('--- construct dev set ---')
				dev_set.construct_batches(is_train=False)

			# print info
			steps_per_epoch = len(train_set.iter_loader)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))

			log.info(" ---------- Epoch: %d, Step: %d ----------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			log.info('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# loop over batches
			model.train(True)
			trainiter = iter(train_set.iter_loader)
			for idx in range(steps_per_epoch):

				# load batch items
				batch_items = trainiter.next()

				# update macro count
				step += 1
				step_elapsed += 1

				# Get loss
				loss = self._train_batch(model, batch_items, train_set, step, total_steps)
				print_loss_total += loss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_avg = print_loss_total / self.print_every
					print_loss_total = 0

					log_msg = 'Progress: %d%%, Train nlll: %.4f' % (
						step / total_steps * 100, print_loss_avg)

					log.info(log_msg)
					self.writer.add_scalar('train_loss', print_loss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps:

					# save criteria
					if dev_set is not None:
						dev_loss, accuracy, _= self._evaluate_batches(model, dev_set)

						log_msg = 'Progress: %d%%, Dev loss: %.4f, accuracy: %.4f' % (
							step / total_steps * 100, dev_loss, accuracy)
						log.info(log_msg)
						self.writer.add_scalar('dev_loss', dev_loss, global_step=step)
						self.writer.add_scalar('dev_acc', accuracy, global_step=step)

						# save
						if prev_acc < accuracy:
							# save best model
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_tgt)

							saved_path = ckpt.save(self.expt_dir)
							log.info('saving at {} ... '.format(saved_path))
							# reset
							prev_acc = accuracy
							count_no_improve = 0
							count_num_rollback = 0
						else:
							count_no_improve += 1

						# roll back
						if count_no_improve > self.max_count_no_improve:
							# resuming
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								log.info('epoch:{} step: {} - rolling back {} ...'.format(
									epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim.__class__(
									model.parameters(), **defaults)

							# reset
							count_no_improve = 0
							count_num_rollback += 1

						# update learning rate
						if count_num_rollback > self.max_count_num_rollback:

							# roll back
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								log.info('epoch:{} step: {} - rolling back {} ...'.format(
									epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim.__class__(
									model.parameters(), **defaults)

							# decrease lr
							for param_group in self.optimizer.optimizer.param_groups:
								param_group['lr'] *= 0.5
								lr_curr = param_group['lr']
								log.info('reducing lr ...')
								log.info('step:{} - lr: {}'.format(step, param_group['lr']))

							# check early stop
							if lr_curr <= 0.125 * self.learning_rate :
								log.info('early stop ...')
								break

							# reset
							count_no_improve = 0
							count_num_rollback = 0

						model.train(mode=True)
						if ckpt is None:
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_tgt)
						ckpt.rm_old(self.expt_dir, keep_num=self.keep_num)
						log.info('n_no_improve {}, num_rollback {}'.format(
							count_no_improve, count_num_rollback))

					sys.stdout.flush()

			else:
				if dev_set is None:
					# save every epoch if no dev_set
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					log.info('saving at {} ... '.format(saved_path))
					continue

				else:
					continue

			# break nested for loop
			break


	def train(self, train_set, model, num_epochs=5, resume=False, optimizer=None, dev_set=None):

		"""
			Run training for a given model.
			Args:
				train_set: dataset
				dev_set: dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run
				resume(bool, optional): resume training with the latest checkpoint
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training

			Returns:
				model (seq2seq.models): trained model.
		"""

		torch.cuda.empty_cache()
		if resume:
			latest_checkpoint_path = self.load_dir
			self.logger.info('resuming {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			self.logger.info(model)
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			# start from prev
			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

			# just for the sake of finetuning
			# start_epoch = 1
			# step = 0

		else:
			start_epoch = 1
			step = 0
			self.logger.info(model)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(model.parameters(),
					lr=self.learning_rate), max_grad_norm=self.max_grad_norm)
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s" %
			(self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(train_set, model, num_epochs, start_epoch, step, dev_set=dev_set)

		return model
