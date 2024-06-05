#!/bin/bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --need_mixture True --sample_mode removal_mu"
TEST_FLAGS="--out_filename /data/diffusion_logs/m_l_1000_f_1/test/240 --model_path /data/diffusion_logs/m_l_1000_f_1/ckpt/model2400000.pt"
mpiexec -n 8 python3 scripts/image_generate.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $TEST_FLAGS
