#!/bin/bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --need_mixture True --logs_dir /data/diffusion_logs/m_l_1000_f_1"
mpiexec -n 1 python3 scripts/image_train.py --data_dir /data/diffusion_logs/datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
