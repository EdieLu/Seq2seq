#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3

# ------------------------ DIR --------------------------
savedir=models/en-de-v001/
train_path_src=../lib/mustc-en-de-proc/train/train.en
train_path_tgt=../lib/mustc-en-de-proc/train/train.de
dev_path_src=../lib/mustc-en-de-proc/dev/dev.en
dev_path_tgt=../lib/mustc-en-de-proc/dev/dev.de
# dev_path_src=None
# dev_path_tgt=None
path_vocab_src=../lib/mustc-en-de-proc/vocab/en-bpe-30000/vocab
path_vocab_tgt=../lib/mustc-en-de-proc/vocab/de-bpe-30000/vocab
load_embedding_src=None
load_embedding_tgt=None
use_type='word'

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=1000
print_every=200

batch_size=200
max_seq_len=50
num_epochs=30
random_seed=300
eval_with_mask=True
max_count_no_improve=2
max_count_num_rollback=2
keep_num=2

$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-base/train.py \
	--train_path_src $train_path_src \
	--train_path_tgt $train_path_tgt \
	--dev_path_src $dev_path_src \
	--dev_path_tgt $dev_path_tgt \
	--path_vocab_src $path_vocab_src \
	--path_vocab_tgt $path_vocab_tgt \
	--load_embedding_src $load_embedding_src \
	--load_embedding_tgt $load_embedding_tgt \
	--use_type $use_type \
	--save $savedir \
	--random_seed $random_seed \
	--embedding_size_enc 200 \
	--embedding_size_dec 200 \
	--hidden_size_enc 200 \
	--num_bilstm_enc 2 \
	--num_unilstm_enc 0 \
	--hidden_size_dec 200 \
	--num_unilstm_dec 4 \
	--hidden_size_att 10 \
	--att_mode bilinear \
	--residual True \
	--hidden_size_shared 200 \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--batch_first True \
	--seqrev False \
	--eval_with_mask $eval_with_mask \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--dropout 0.2 \
	--embedding_dropout 0.0 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--learning_rate 0.001 \
	--max_grad_norm 1.0 \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--max_count_no_improve $max_count_no_improve \
	--max_count_num_rollback $max_count_num_rollback \
	--keep_num $keep_num \
