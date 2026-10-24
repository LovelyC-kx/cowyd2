#!/bin/bash
set -e
exec >> ./logs/F_chain_v2.log 2>&1
echo "[$(date)] F chain v2 started"

CUDA_VISIBLE_DEVICES=0 python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --backbone snn --bn_variant tdbn \
    --only_bn fedbn --only_scheme Gossip_Averaging \
    --partition_alpha 0.1 --num_epoch 200 \
    --train_batch_size 4 --num_workers 4 \
    --run_name F_snn --output_dir ./Outputs_v2 \
    > ./logs/F_snn.log 2>&1
echo "[$(date)] F_SNN done"

CUDA_VISIBLE_DEVICES=0 python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --backbone ann --bn_variant tdbn \
    --only_bn fedbn --only_scheme Gossip_Averaging \
    --partition_alpha 0.1 --num_epoch 200 \
    --train_batch_size 4 --num_workers 4 \
    --run_name F_ann --output_dir ./Outputs_v2 \
    > ./logs/F_ann.log 2>&1
echo "[$(date)] F_ANN done"

CUDA_VISIBLE_DEVICES=0 python -m cloud_removal_v2.run_smoke \
    --data_root /root/autodl-tmp/C-CUHK \
    --backbone plain --bn_variant bn2d \
    --only_bn fedbn --only_scheme Gossip_Averaging \
    --partition_alpha 0.1 --num_epoch 200 \
    --train_batch_size 4 --num_workers 4 \
    --run_name F_plain --output_dir ./Outputs_v2 \
    > ./logs/F_plain.log 2>&1
echo "[$(date)] F_plain done — ALL F SERIES COMPLETE"
