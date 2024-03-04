"""
Generate samples from the forward and backward diffusion process for a given dataset.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import load_data, _list_images_per_classes
import datetime
import pickle

from PIL import Image

import time


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    output_images = os.path.join(
        args.output, f"t_{args.step_reverse}_{args.timestep_respacing}_images"
    )
    logger.configure(dir=output_images)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    list_images = _list_images_per_classes(
        args.data_dir, args.num_per_class, args.num_classes, output_images
    )
    num_samples = len(list_images)
    data_start = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        class_cond=True,
        random_crop=False,
        random_flip=False,
        list_images=list_images,
        drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
    )

    logger.log(f"creating {num_samples} samples...")

    logger.log("sampling...")
    generated_samples = 0
    time_start = time.time()
    while generated_samples < num_samples:
        batch_start, extra = next(data_start)

        labels_start = extra["y"].to(dist_util.dev())
        batch_start = batch_start.to(dist_util.dev())
        img_names = extra["img_name"]
        # Sample noisy images from the diffusion process at time t_reverse given by the step_reverse argument
        t_reverse = diffusion._scale_timesteps(th.tensor([args.step_reverse])).to(
            dist_util.dev()
        )
        batch_noisy = (
            diffusion.q_sample(batch_start, t_reverse)
            if args.step_reverse < int(args.timestep_respacing)
            else th.randn(batch_start.shape, device=dist_util.dev())
        )
        logger.log("completed forward diffusion...")

        model_kwargs = {}
        if args.class_cond:
            classes = labels_start  # Condition the diffusion on the labels of the original images
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop_forw_back
            if not args.use_ddim
            else diffusion.ddim_sample_loop_forw_back
        )
        sample = sample_fn(
            model,
            (len(batch_start), 3, args.image_size, args.image_size),
            step_reverse=args.step_reverse,  # Step when to reverse the diffusion process
            noise=batch_noisy,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        logger.log("completed backward diffusion...")
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # Save the images
        real_t_reverse = (
            int(t_reverse.item())
            if diffusion.rescale_timesteps
            else int(t_reverse.item() * (1000.0 / float(args.timestep_respacing)))
        )
        for ii in range(len(sample)):
            name = (
                img_names[ii].split(".")[0]
                + "_t"
                + "{:04d}".format(real_t_reverse)
                + ".JPEG"
            )
            img = Image.fromarray(np.array(sample[ii].cpu()).astype(np.uint8))
            img.save(os.path.join(output_images, name))

        sample_size = th.tensor(len(sample)).to(dist_util.dev())
        dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)

        sample_size = th.tensor(len(sample)).to(dist_util.dev())
        dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)
        generated_samples += sample_size.item()
        logger.log(
            f"created {generated_samples} samples in {time.time() - time_start:.1f} seconds"
        )

    if dist.get_rank() == 0:
        # Save the arguments of the run
        date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        out_args = os.path.join(
            args.output,
            f"t_{args.step_reverse}_{args.timestep_respacing}_args_{date_time}.pk",
        )
        logger.log(f"saving args to {out_args}")
        with open(out_args, "wb") as handle:
            pickle.dump(args, handle)

        # Save the time it took to generate the samples
        out_time = os.path.join(
            args.output, f"t_{args.step_reverse}_{args.timestep_respacing}_timing.txt"
        )
        with open(out_time, "a") as f:
            f.write(f"{generated_samples} \t {time.time() - time_start:.3f}\n")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(
        dict(
            step_reverse=10,
            data_dir="datasets/ILSVRC2012/validation",
            output=os.path.join(
                os.getcwd(), "results", "diffused_ILSVRC2012_validation"
            ),
            num_per_class=10,
            num_classes=10,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    assert parser.parse_args().step_reverse >= 0, "step_reverse must be positive"
    assert parser.parse_args().step_reverse <= int(
        parser.parse_args().timestep_respacing
    ), "step_reverse must be smaller than or equal to timestep_respacing"

    return parser


if __name__ == "__main__":
    main()
