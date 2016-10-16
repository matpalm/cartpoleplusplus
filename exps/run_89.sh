#!/usr/bin/env bash
set -ex
export CUDA_VISIBLE_DEVICES=""  # ensure running on cpu

R=runs/89

export RR=$R/action
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--reward-calc="action_norm" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &
sleep 1

export RR=$R/angle
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--reward-calc="angles" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &
sleep 1

export RR=$R/both
mkdir -p $RR
nice ./naf_cartpole.py --action-force=100 \
--reward-calc="both" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

