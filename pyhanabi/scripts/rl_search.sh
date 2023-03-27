#!/bin/bash

python rl_search.py \
    --save_dir game_data/6-7-splits/test/br/ \
    --weight1 ../models/sad_2p_models/sad_2.pthw \
    --weight2 ../models/my_models/br_sad_six_1_3_6_7_8_12/model_epoch1000.pthw \
    --sad_legacy 1,0 \
    --player_name sad_2,br_sad_six_1_3_6_7_8_12 \
    --data_type test \
    --split_type six \
    --game_seed 0 \
    --seed 0 \
    --burn_in_frames 5000 \
    --replay_buffer_size 100000 \
    --rl_rollout_device cuda:1 \
    --bp_rollout_device cuda:1 \
    --train_device cuda:0 \
    --belief_device cuda:0 \
    --rollout_batchsize 8000 \
    --num_thread 1 \
    --batchsize 128 \
    --num_epoch 1 \
    --epoch_len 5000 \
    --num_samples 50000 \
    --skip_search 0 \
    --ad_hoc 1 \
    --upload_gcloud 1 \
    --save_game 1 \
