#!/usr/bin/env bash

grep pre\ noise foo | perl -plne's/.*\[\[//;s/\]\]//;s/\s+/ /g;s/^\s*//;s/\s*$//' > /tmp/actions_pre_noise
grep post\ noise foo | perl -plne's/.*\[\[//;s/\]\]//;s/\s+/ /g;s/^\s*//;s/\s*$//' > /tmp/actions_post_noise
#grep ^BULLET /tmp/foo | sed -es/BULLET\ // > /tmp/forces &
grep ^Q\ LOSS /tmp/foo | sed -es/Q\ LOSS\ // > /tmp/q_loss &
grep ^EXPECTED /tmp/foo | perl -plne's/EXPECTED Q //;s/given state.*//' > /tmp/action_q_values &
grep ^STAT /tmp/foo | perl -plne's/.*episode_len\": //;s/,.*//' > /tmp/episode_len &
wait
wc -l /tmp/forces /tmp/q_loss /tmp/action_q_values /tmp/episode_len

