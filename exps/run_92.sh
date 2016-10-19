#!/usr/bin/env bash
set -ex
export CUDA_VISIBLE_DEVICES=""  # ensure running on cpu

R=runs/92

export RR=$R/a
mkdir -p $RR
nice ./lrpg_cartpole.py --ckpt-dir=$RR/ckpts >$RR/out 2>$RR/err &

export RR=$R/b
mkdir -p $RR
nice ./lrpg_cartpole.py --ckpt-dir=$RR/ckpts >$RR/out 2>$RR/err &


