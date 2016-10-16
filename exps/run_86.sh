#!/usr/bin/env bash
set -ex
unset CUDA_VISIBLE_DEVICES  # ensure running on gpu

R=runs/86

mkdir -p $R/{naf,ddpg}

export ARGS="--max-run-time=14400 --action-force=100"

export RR=$R/naf
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/ddpg
./ddpg_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err
