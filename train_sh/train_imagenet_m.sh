#!/bin/bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --schedule_sampler loss-second-moment --need_mixture True --logs_dir ~/data/diffusion_logs/Imagenet-m --resume_checkpoint /home/whz/data/diffusion_logs/Imagenet-m/ckpt/model900000.pt"
CUDA_VISIBLE_DEVICES=1 mpiexec -n 1 python3 scripts/image_train.py --data_dir /home/whz/data/ImageNet/ImageNet_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS