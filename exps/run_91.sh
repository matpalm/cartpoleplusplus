#!/usr/bin/env bash
set -ex
export CUDA_VISIBLE_DEVICES=""  # ensure running on cpu

R=runs/91

export RR=$R/1_12
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=1 --steps-per-repeat=12 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

export RR=$R/2_6
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=2 --steps-per-repeat=6 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

export RR=$R/3_4
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=3 --steps-per-repeat=4 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

export RR=$R/4_3
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--action-repeats=4 --steps-per-repeat=3 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &
