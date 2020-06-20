#!/bin/bash
#$ -S /bin/bash

# ------------------------ ENV --------------------------
echo $HOSTNAME
unset LD_PRELOAD # overwrite env
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=0 # if using qsub
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE # if using dev machines
echo $CUDA_VISIBLE_DEVICES

# activate your conda env
# python 3.6
# pytorch 1.1/1.3
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3

# ------------------------ DIR --------------------------
savedir=models/gec-debug/
train_path_src=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/train.src.nodot
train_path_tgt=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/train.tgt.nodot
# dev_path_src=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/dev.src
# dev_path_tgt=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/dev.tgt
dev_path_src=None
dev_path_tgt=None
path_vocab_src=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
path_vocab_tgt=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
use_type='word'
load_embedding_src=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/embeddings/glove.6B.200d.txt
load_embedding_tgt=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/embeddings/glove.6B.200d.txt
share_embedder=True

# ------------------------ MODEL --------------------------
embedding_size_enc=200
embedding_size_dec=200
hidden_size_enc=200
hidden_size_dec=200
hidden_size_shared=200
num_bilstm_enc=2
num_unilstm_dec=4
att_mode=bilinear # bahdanau | bilinear

# ------------------------ TRAIN --------------------------
# checkpoint_every=5
# print_every=2
checkpoint_every=6000
print_every=1000

batch_size=256
max_seq_len=32
minibatch_split=1
num_epochs=20

random_seed=2020
eval_with_mask=True
max_count_no_improve=5
max_count_num_rollback=2
keep_num=2
normalise_loss=True

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
	--share_embedder $share_embedder \
	--embedding_size_enc $embedding_size_enc \
	--embedding_size_dec $embedding_size_dec \
	--hidden_size_enc $hidden_size_enc \
	--num_bilstm_enc $num_bilstm_enc \
	--num_unilstm_enc 0 \
	--hidden_size_dec $hidden_size_dec \
	--num_unilstm_dec $num_unilstm_dec \
	--hidden_size_att 10 \
	--att_mode $att_mode \
	--residual True \
	--hidden_size_shared $hidden_size_shared \
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
	--normalise_loss $normalise_loss \
	--minibatch_split $minibatch_split \
