#!/bin/tcsh

set ALLARGS = ($*)

# Check Number of Args
if ( $#argv != 1 ) then
   echo "Usage: $0 log"
   echo "  e.g: $0 LOGs/nmt-en1.train.txt"
   exit 100
endif

set LOG=$1
set TRAIN=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-rnn/run/train.sh

set CMD = `qsub -cwd -j yes -o $LOG -P esol -l hostname=air208.eng.cam.ac.uk -l qp=cuda-low -l gpuclass=pascal -l osrel='*' $TRAIN`
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass='*' -l osrel='*' $TRAIN`
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l hostname=air206.eng.cam.ac.uk -l qp=cuda-low -l gpuclass='*' -l osrel='*' $TRAIN`
echo $CMD
