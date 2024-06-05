#!/bin/bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 4 --need_mixture False"
TEST_FLAGS="--out_filename /data/diffusion_logs/lsun_s/test/200 --model_path /data/diffusion_logs/lsun_s/ckpt/model2000000.pt"
mpiexec -n 1 python3 scripts/image_generate.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $TEST_FLAGS