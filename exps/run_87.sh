#!/usr/bin/env bash
set -ex
export CUDA_VISIBLE_DEVICES=""  # ensure running on cpu

R=runs/87

mkdir -p $R/{repro1,repro2,bn,3repeats}

# exactly repros run_86
export RR=$R/repro1
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=2 --steps-per-repeat=5 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

# switchs to 6 repeats, instead of 5, so that's 12 total (like the 3x4 one below)
export RR=$R/repro2
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=2 --steps-per-repeat=6 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

# with batch norm
export RR=$R/bn
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=2 --steps-per-repeat=6 \
--use-batch-norm \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

# with 3 action repeats (but still 12 total steps)
export RR=$R/3repeats
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=3 --steps-per-repeat=4 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &
