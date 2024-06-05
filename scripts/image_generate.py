import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision.utils import make_grid,save_image
import datetime
import time
from improved_diffusion.score.both import get_inception_and_fid_score
import math

def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        need_mixture=False,
        mus=[0.0, 1.0, -1.0, math.sqrt(2), -math.sqrt(2), math.sqrt(3), -math.sqrt(3)],
        omegas=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05],
        logs_dir='./logs',
        eval_num=256,
        fid_cache='./stats/cifar10.train.npz',
        out_filename = './log/1.npy',
        model_path = './1',
        sample_mode = 'removal_mu'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def generate_image(model, diffusion, eval_num, batch_size, out_filename, image_size, need_mixture, sample_mode):
    batch = torch.ones((128,3,image_size,image_size))
    images = []
    with torch.no_grad():
        start_time = time.time()
        print("generating images......")
        for i in range(0, eval_num, batch_size):
            print(i, '{:.2f}'.format(time.time() - start_time))
            b_size = min(batch_size, eval_num - i)
            b_size = [b_size] + list(torch.tensor(batch.shape[1:]).numpy())
            if need_mixture:
                batch_images = diffusion.p_sample_loop(model, b_size, sample_mode=sample_mode).cpu()
            else:
                batch_images = diffusion.p_sample_loop(model, b_size).cpu()
            images.append((batch_images + 1) / 2)
            if i == 0 :
                print(batch_images.size())
        print(eval_num, '{:.2f}'.format(time.time() - start_time))
        #images = torch.cat(images, dim=0).numpy()
        #np.save(out_filename, images)
        images = torch.cat(images, dim=0)
        torch.save(images, out_filename)


args = create_argparser().parse_args()
dist_util.setup_dist()
if not os.path.exists(args.out_filename):
    os.mkdir(args.out_filename)
out_filename = os.path.join(
    args.out_filename,
    datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
)

print(out_filename)
print(args.model_path)

model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
model.to(dist_util.dev())
model.eval()

eval_num = 64

#out_filename = out_filename+'.npy'

generate_image(model, diffusion, eval_num, args.batch_size, out_filename,args.image_size,args.need_mixture, args.sample_mode)


