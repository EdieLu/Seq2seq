#!/bin/bash

# evaluate bleu score

command="$0 $@"
cmddir=CMDs
echo "---------------------------------------------" >> $cmddir/eval_bleu.cmds
echo $command >> $cmddir/eval_bleu.cmds

fname=$1 #dev | tst-COMMON | tst-HE
ckpt=$2 #2020_05_28_07_27_34

fout=models/en-de-v001/eval_$fname/$ckpt
refdir=../lib/mustc-en-de-proc/$fname/$fname.de

python ./local/py-tools/bleu_scorer.py $fout/translate.txt $refdir > $fout/bleu.log
