#!/bin/bash

python sparta.py \
    --weight ../models/my_models/br_sad_six_1_3_6_7_8_12/model_epoch1000.pthw \
    --sad_legacy 0 \
    --test_partner_weight ../models/sad_2p_models/sad_2.pthw \
    --test_partner_sad_legacy 1 \
    --seed 0 \
    --game_seed 0 \
    --skip_search 0 \
