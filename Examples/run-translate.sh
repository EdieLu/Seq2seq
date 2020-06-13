#!/bin/tcsh

set ALLARGS = ($*)

# Check Number of Args
if ( $#argv != 1 ) then
   echo "Usage: $0 log"
   echo "  e.g: $0 LOGs/translate.log.txt"
   exit 100
endif

set LOG=$1
set TRANSLATE=/home/alta/BLTSpeaking/exp-ytl28/projects/gec-rnn/run/translate.sh


# GPU jobs
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l hostname=air209.eng.cam.ac.uk -l qp=cuda-low -l gpuclass=pascal -l osrel='*' $TRANSLATE`
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l hostname=air207.eng.cam.ac.uk -l qp=cuda-low -l gpuclass='*' -l osrel='*' $TRANSLATE`
set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass='*' -l osrel='*' -l hostname='*' $TRANSLATE`

# CPU jobs
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=low -l hostname='*' -l mem_grab=8G -l mem_free=8G $TRANSLATE`

# CPU batch jobs
# set lst=/home/alta/BLTSpeaking/exp-ytl28/projects/nmt-transformer-en-de/run/lst/translate.lst
# set RUN_ARRAY_JOB=/home/alta/BLTSpeaking/exp-ytl28/projects/nmt-transformer-en-de/run/run-array-job.sh
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=low -l hostname='*' -l mem_grab=8G -l mem_free=8G -t 11-32 $RUN_ARRAY_JOB $lst $TRANSLATE SET`

echo $CMD
