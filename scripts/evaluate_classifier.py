import argparse
import os
import torch as th
import pickle
import copy
import numpy as np
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import (
    load_data,
    _list_starting_images,
    _list_image_files_recursively,
)
from guided_diffusion.torch_classifiers import load_classifier
import time


def check_same_images(list_sample, list_start):
    for i in range(len(list_sample)):
        name_sample = list_sample[i].split(".")[0][:-6]
        name_start = list_start[i].split(".")[0]
        if name_sample != name_start:
            return False
    return True


def main():
    args = create_argparser().parse_args()
    args.output = os.path.join(args.output, args.classifier_name)

    dist_util.setup_dist()
    logger.configure(dir=args.output)

    classifier, preprocess, module_names = load_classifier(args.classifier_name)
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def class_eval(x):
        with th.no_grad():
            logits = classifier(preprocess(x))
            return logits

    # Upload activations statistics
    file = os.path.join(args.output, f"act_stat_{args.classifier_name}.pk")

    with open(file, "rb") as f:
        act_stat = pickle.load(f)
    activations_mean = act_stat["activations_mean"]
    activations_var = act_stat["activations_var"]
    for key in activations_mean.keys():
        activations_mean[key] = th.tensor(activations_mean[key]).to(dist_util.dev())
        activations_var[key] = th.tensor(activations_var[key]).to(dist_util.dev())

    def whiten_act(aa):
        for key in activations_mean.keys():
            aa[key] = (aa[key] - activations_mean[key]) / th.sqrt(
                activations_var[key] + 1e-8
            )
        return aa

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = []
    for layer_name in module_names:
        layer = dict([*classifier.named_modules()])[layer_name]
        hook = layer.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)

    # Time steps to evaluate
    for time_step in args.time_series:

        # Load starting data
        name_at_time_step = f"t_{time_step}_{args.timestep_respacing}_images"
        sample_data_dir = os.path.join(args.data_dir, name_at_time_step)

        logger.log("creating data loader...")
        list_sample_imgs = _list_image_files_recursively(sample_data_dir)
        list_start_imgs = _list_starting_images(args.starting_data_dir, sample_data_dir)

        num_samples = len(list_sample_imgs)
        num_start = len(list_start_imgs)
        assert (
            num_samples == num_start
        ), "Number of samples and starting images must be the same"

        data_sample = load_data(
            data_dir=sample_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            deterministic=True,
            class_cond=True,
            random_crop=False,
            random_flip=False,
            list_images=list_sample_imgs,
            drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
        )

        data_start = load_data(
            data_dir=args.starting_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            deterministic=True,
            class_cond=True,
            random_crop=False,
            random_flip=False,
            list_images=list_start_imgs,
            drop_last=False,  # It is important when batch_size < num_samples, otherwise it doesn't yield
        )

        evaluated_samples = 0
        dict_list = []
        all_logits_start = []
        all_logits_sample = []
        time_start = time.time()
        while evaluated_samples < num_samples:
            batch_start, extra_start = next(data_start)
            batch_sample, extra_sample = next(data_sample)

            # labels_start = extra["y"].to(dist_util.dev())
            batch_start = batch_start.to(dist_util.dev())
            batch_sample = batch_sample.to(dist_util.dev())
            start_names = extra_start["img_name"]
            sample_names = extra_sample["img_name"]
            assert check_same_images(
                sample_names, start_names
            ), "Images in sample and starting batches must be the same"

            class_eval_sample = class_eval(batch_sample)
            activations_sample = copy.deepcopy(whiten_act(activations))
            class_eval_start = class_eval(batch_start)
            activations_start = copy.deepcopy(whiten_act(activations))

            diff_activations = {}
            cosine_sim = th.nn.CosineSimilarity(dim=1, eps=1e-8)
            for key in activations_start.keys():
                diff_activations[key] = {}
                activations_sample[key] = activations_sample[key].flatten(start_dim=1)
                activations_start[key] = activations_start[key].flatten(start_dim=1)
                diff_activations[key]["L2"] = (
                    th.linalg.norm(
                        activations_sample[key] - activations_start[key], dim=1
                    )
                    ** 2
                )
                diff_activations[key]["L2_normalized"] = diff_activations[key]["L2"] / (
                    th.linalg.norm(activations_sample[key], dim=1)
                    * th.linalg.norm(activations_start[key], dim=1)
                )
                diff_activations[key]["cosine"] = cosine_sim(
                    activations_sample[key], activations_start[key]
                )

            dict_list.append(diff_activations)

            gathered_logits_start = [
                th.zeros_like(class_eval_start) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_logits_start, class_eval_start)
            all_logits_start.extend(
                [logits.cpu().numpy() for logits in gathered_logits_start]
            )

            gathered_logits_sample = [
                th.zeros_like(class_eval_sample) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_logits_sample, class_eval_sample)
            all_logits_sample.extend(
                [logits.cpu().numpy() for logits in gathered_logits_sample]
            )

            sample_size = th.tensor(len(class_eval_sample)).to(dist_util.dev())
            dist.all_reduce(sample_size, op=dist.ReduceOp.SUM)
            evaluated_samples += sample_size.item()
            logger.log(
                f"evaluated {evaluated_samples} samples in {time.time() - time_start:.1f} seconds"
            )

        # Conatenate batches
        dictionary_act = {}
        for key in dict_list[0].keys():
            dictionary_act[key] = {}
            for key2 in dict_list[0][key].keys():
                dictionary_act[key][key2] = (
                    th.cat(
                        [dict_list[i][key][key2] for i in range(len(dict_list))], dim=0
                    )
                    .cpu()
                    .numpy()
                )

        all_logits_start = np.concatenate(all_logits_start, axis=0)
        all_logits_sample = np.concatenate(all_logits_sample, axis=0)

        # Save activations statistics
        outfile = os.path.join(
            args.output,
            f"act_diff_{args.classifier_name}-t_{time_step}_{args.timestep_respacing}.pk",
        )

        logger.log(f"saving activations to {outfile}")
        with open(outfile, "wb") as handle:
            pickle.dump(dictionary_act, handle)

        # Save logits
        outfile = os.path.join(
            args.output,
            f"logits_{args.classifier_name}-t_{time_step}_{args.timestep_respacing}.pk",
        )
        logger.log(f"saving logits to {outfile}")
        with open(outfile, "wb") as handle:
            pickle.dump(
                {
                    "logits_start": all_logits_start,
                    "logits_sample": all_logits_sample,
                },
                handle,
            )
    ## End of time steps loop

    # Clean up
    for hook in hooks:
        hook.remove()

    dist.barrier()
    logger.log("Done!")


def create_argparser():
    defaults = dict(
        classifier_name="convnext_base",
        classifier_use_fp16=False,
        batch_size=128,
        image_size=256,
        timestep_respacing="250",
        starting_data_dir="datasets/ILSVRC2012/validation",
        data_dir=os.path.join(os.getcwd(), "results", "diffused_ILSVRC2012_validation"),
        output=os.path.join(os.getcwd(), "classifier_statistics"),
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--time_series",
        nargs="+",
        type=int,
        default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250],
        help="Time steps to evaluate. Pass like: --time_series 25 50 100",
    )
    return parser


if __name__ == "__main__":
    main()
