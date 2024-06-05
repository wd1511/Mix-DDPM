#!/bin/bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 4 --need_mixture True --logs_dir /home/zx/data/wd/diffusion_logs/lsun_m4 --resume_checkpoint /home/zx/data/wd/diffusion_logs/lsun_m4/ckpt/model1800000.pt"
CUDA_VISIBLE_DEVICES=2 mpiexec -n 1 python3 scripts/image_train.py --data_dir /home/zx/data/wd/lsun_train_output_dir $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS