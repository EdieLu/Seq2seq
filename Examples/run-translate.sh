#!/bin/tcsh

set ALLARGS = ($*)

# Check Number of Args
if ( $#argv != 1 ) then
   echo "Usage: $0 log"
   echo "  e.g: $0 LOGs/translate.log.txt"
   exit 100
endif

set LOG=$1
set TRANSLATE=/home/alta/BLTSpeaking/exp-ytl28/encdec/run-v6/translate.sh

# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l hostname=air209.eng.cam.ac.uk -l qp=cuda-low -l gpuclass=pascal -l osrel='*' $TRANSLATE`
set CMD = `qsub -cwd -j yes -o $LOG -P esol -l hostname=air207.eng.cam.ac.uk -l qp=cuda-low -l gpuclass='*' -l osrel='*' $TRANSLATE`
# set CMD = `qsub -cwd -j yes -o $LOG -P esol -l qp=cuda-low -l gpuclass='*' -l osrel='*' -l hostname='*' $TRANSLATE`
echo $CMD
