#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
# source activate pt11-cuda9
# export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/pt11-cuda9/bin/python3
source activate py13-cuda9
export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3

# ----- dir ------
model=models/gec-v016
path_vocab_src=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
path_vocab_tgt=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/vocab/clctotal+swbd.min-count4.en
use_type='word'

libbase=/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe

# fname=test_fce_test
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/clc/test.src
# seqlen=100

# fname=test_clc
# ftst=$libbase/clc/nobpe/clc-test.src
# seqlen=125

# fname=test_nict
# ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-nict/nict.src
# seqlen=85

fname=test_nict_new
ftst=/home/alta/BLTSpeaking/exp-ytl28/projects/lib/gec-nict-new/nict.src
seqlen=85

# fname=test_dtal
# ftst=$libbase/dtal/nobpe/dtal.src
# seqlen=165

# fname=test_eval3 # default segauto
# ftst=$libbase/eval3/nobpe/eval3.src
# seqlen=145

# ----- models ------
beam_width=1
batch_size=500
use_gpu=True

for i in `seq 1 1 20`
do
    echo $i
    ckpt=$i

    $PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/nmt-base/translate.py \
        --test_path_src $ftst \
        --test_path_tgt $ftst \
        --seqrev False \
        --path_vocab_src $path_vocab_src \
        --path_vocab_tgt $path_vocab_tgt \
        --use_type $use_type \
        --load $model/checkpoints_epoch/$ckpt \
        --test_path_out $model/$fname/$ckpt/ \
        --max_seq_len $seqlen \
        --batch_size $batch_size \
        --use_gpu True \
        --beam_width 1 \
        --eval_mode 1
done
