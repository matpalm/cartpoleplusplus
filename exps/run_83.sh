#!/usr/bin/env bash
set -ex

R=runs/83

mkdir -p $R/mom{,_bn}

export ARGS="--use-raw-pixels --max-run-time=14400 --dont-do-rollouts --event-log-in=runs/80/events"

export RR=$R/mom
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/mom_bn
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--use-batch-norm \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err
