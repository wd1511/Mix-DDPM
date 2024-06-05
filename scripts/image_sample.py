"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
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

def evaluate(model, diffusion):
    eval_num = 50000
    batch_size = 128
    batch = th.ones((128,3,32,32))
    fid_cache = './stats/cifar10.train.npz'
    with th.no_grad():
        images = []
        start_time = time.time()
        print("generating images......")
        for i in range(0, eval_num, batch_size):
            print(i, '{:.2f}'.format(time.time() - start_time))
            b_size = min(batch_size, eval_num - i)
            b_size = [b_size] + list(th.tensor(batch.shape[1:]).numpy())
            batch_images = diffusion.p_sample_loop(model, b_size).cpu()
            images.append((batch_images + 1) / 2)
        images = th.cat(images, dim=0).numpy()
        print(eval_num, '{:.2f}'.format(time.time() - start_time))
    (IS, IS_std), FID = get_inception_and_fid_score(images, fid_cache, eval_num,
                                                    use_torch=False, verbose=True)
    print(IS,IS_std,FID)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = os.path.join(
        args.logs_dir,
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
    )
    logger.configure(log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    evaluate(model=model,diffusion=diffusion)
    #batchshape = [2,3,32,32]
    #p_sample(batchshape, model, diffusion)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
        logs_dir='./logs'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()