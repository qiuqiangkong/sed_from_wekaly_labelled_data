#!/bin/bash

# Create cross validation csv
python create_cv.py create_cv_csv

# Create mixutre yaml file
python create_cv.py create_mix_yaml

# Create mixture wav
python prepare_data.py mix

# Extract log Mel feature
python prepare_data.py calculate_logmel

# Train. 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python tmp01.py train --n_events=3 --snr=20

# Recognize. 
BGN_ITER=2000
FIN_ITER=3001
INTERVAL=200
while [ $BGN_ITER -lt $FIN_ITER ]
do
  echo $BGN_ITER
  THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python tmp01.py recognize --n_events=4 --snr=20 --model_name="md"$BGN_ITER"_iters.p"
  BGN_ITER=$[$BGN_ITER+$INTERVAL]
done

# Get stats. 
python tmp01.py get_stats --n_events=3 --snr=20

# Separate. 
python tmp01.py separate --n_events=3 --snr=20

# Evaluate separation. 
python tmp01.py evaluate_separation --n_events=3 --snr=20
