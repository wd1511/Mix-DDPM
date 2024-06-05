#!/bin/bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --schedule_sampler loss-second-moment --need_mixture False"
TEST_FLAGS="--out_filename /data/diffusion_logs/Imagenet-s/test/080 --model_path /data/diffusion_logs/Imagenet-s/ckpt/model0800000.pt"
mpiexec -n 1 python3 scripts/image_generate.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $TEST_FLAGS